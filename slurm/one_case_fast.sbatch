#!/bin/bash
#SBATCH --partition=rtx
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=01:15:00
#SBATCH --job-name=edge-bank-fast1800
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/one_case_fast-%j.log
set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHON=/mnt/beegfs/home/wangchunyun25/.conda/envs/csle/bin/python
RUN_DIR="/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/one_case_fast_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/source"
cp -r edge_rca "$RUN_DIR/source/edge_rca"
export PYTHONPATH="$RUN_DIR/source"
cd "$RUN_DIR"
PREPARED=/mnt/beegfs/home/wangchunyun25/Documents/OpsAgent/results/edge_stream/Bank_train0000_20261015/prepared
QUERIES=/mnt/beegfs/home/wangchunyun25/Documents/OpsAgent/dataset/Bank/train/60_query.csv
NODE_MODEL=/mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3___5-4B
COORD_MODEL=/mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3___5-9B
NODE_MODE=${EDGERCA_NODE_MODE:-prose}
SCREEN_LIMIT=${EDGERCA_SCREEN_LIMIT:-3}
FRAMES_SOURCE=${EDGERCA_FRAMES_SOURCE:-/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/one_case_141859/frames}
REPLAY_PID=""
if [ -f "$FRAMES_SOURCE/complete.json" ] && [ -f "$FRAMES_SOURCE/c1800/ready.json" ]; then
    ln -s "$FRAMES_SOURCE" "$RUN_DIR/frames"
else
    "$PYTHON" -u -m edge_rca.case_runtime prepare --prepared "$PREPARED" --output "$RUN_DIR/frames" \
        --query-file "$QUERIES" --row 0 --checkpoints 1800 > "$RUN_DIR/prepare.log" 2>&1 &
    REPLAY_PID=$!
fi
trap 'if [ -n "$REPLAY_PID" ]; then kill "$REPLAY_PID" 2>/dev/null || true; fi' EXIT
until [ -f "$RUN_DIR/frames/c1800/ready.json" ]; do
    if [ -z "$REPLAY_PID" ]; then
        exit 1
    elif ! kill -0 "$REPLAY_PID" 2>/dev/null; then
        wait "$REPLAY_PID"
        test -f "$RUN_DIR/frames/c1800/ready.json"
    fi
    sleep 5
done
"$PYTHON" -u -m edge_rca.case_runtime nodes --root "$RUN_DIR" --prepared "$PREPARED" \
    --seconds 1800 --node-model "$NODE_MODEL" --node-mode "$NODE_MODE" --screen-limit "$SCREEN_LIMIT"
"$PYTHON" -u -m edge_rca.case_runtime coordinate --root "$RUN_DIR" --prepared "$PREPARED" \
    --seconds 1800 --node-model "$NODE_MODEL" --coordinator-model "$COORD_MODEL" --max-queries 3 --final --node-mode "$NODE_MODE"
"$PYTHON" -u -m edge_rca.case_eval --root "$RUN_DIR" --scorer /mnt/beegfs/home/wangchunyun25/Documents/OpenRCA/main/evaluate.py
