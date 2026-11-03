#!/bin/bash
#SBATCH --partition=rtx
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=03:00:00
#SBATCH --job-name=edge-bank-onecase
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/one_case-%j.log
set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHON=/mnt/beegfs/home/wangchunyun25/.conda/envs/csle/bin/python
RUN_DIR="/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/one_case_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/source"
cp -r edge_rca "$RUN_DIR/source/edge_rca"
export PYTHONPATH="$RUN_DIR/source"
cd "$RUN_DIR"
PREPARED=/mnt/beegfs/home/wangchunyun25/Documents/OpsAgent/results/edge_stream/Bank_train0000_20261015/prepared
QUERIES=/mnt/beegfs/home/wangchunyun25/Documents/OpsAgent/dataset/Bank/train/60_query.csv
NODE_MODEL=/mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3___5-4B
COORD_MODEL=/mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3___5-9B
NODE_MODE=${EDGERCA_NODE_MODE:-prose}
REPLAY_PID=""
if [ -n "${EDGERCA_FRAMES_SOURCE:-}" ]; then
    test -f "$EDGERCA_FRAMES_SOURCE/complete.json"
    ln -s "$EDGERCA_FRAMES_SOURCE" "$RUN_DIR/frames"
else
    "$PYTHON" -u -m edge_rca.case_runtime prepare --prepared "$PREPARED" --output "$RUN_DIR/frames" --query-file "$QUERIES" --row 0 > "$RUN_DIR/prepare.log" 2>&1 &
    REPLAY_PID=$!
fi
trap 'if [ -n "$REPLAY_PID" ]; then kill "$REPLAY_PID" 2>/dev/null || true; fi' EXIT
for SECONDS_POINT in 600 1200 1800; do
    until [ -f "$RUN_DIR/frames/c${SECONDS_POINT}/ready.json" ]; do
        if [ -z "$REPLAY_PID" ]; then
            exit 1
        elif ! kill -0 "$REPLAY_PID" 2>/dev/null; then
            wait "$REPLAY_PID"
            test -f "$RUN_DIR/frames/c${SECONDS_POINT}/ready.json"
        fi
        sleep 5
    done
    "$PYTHON" -u -m edge_rca.case_runtime nodes --root "$RUN_DIR" --prepared "$PREPARED" --seconds "$SECONDS_POINT" --node-model "$NODE_MODEL" --node-mode "$NODE_MODE"
    if [ "$SECONDS_POINT" = 1800 ]; then
        "$PYTHON" -u -m edge_rca.case_runtime coordinate --root "$RUN_DIR" --prepared "$PREPARED" --seconds "$SECONDS_POINT" --node-model "$NODE_MODEL" --coordinator-model "$COORD_MODEL" --max-queries 3 --final --node-mode "$NODE_MODE"
    else
        "$PYTHON" -u -m edge_rca.case_runtime coordinate --root "$RUN_DIR" --prepared "$PREPARED" --seconds "$SECONDS_POINT" --node-model "$NODE_MODEL" --coordinator-model "$COORD_MODEL" --max-queries 1 --node-mode "$NODE_MODE"
    fi
done
if [ -n "$REPLAY_PID" ]; then wait "$REPLAY_PID"; fi
"$PYTHON" -u -m edge_rca.case_eval --root "$RUN_DIR" --scorer /mnt/beegfs/home/wangchunyun25/Documents/OpenRCA/main/evaluate.py
