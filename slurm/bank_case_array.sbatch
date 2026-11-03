#!/bin/bash
#SBATCH --partition=rtx
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:40:00
#SBATCH --job-name=edge-bank-case
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/bank_case-%A_%a.log
set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHON=/mnt/beegfs/home/wangchunyun25/.conda/envs/csle/bin/python
ROW=${SLURM_ARRAY_TASK_ID:?array task id required}
JOB=${SLURM_ARRAY_JOB_ID:-manual}
RUN_DIR="/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/bank_case_${JOB}_${ROW}"
QUERY_FILE=${BANK_QUERY_FILE:-/mnt/beegfs/home/wangchunyun25/Documents/OpenRCA/dataset/Bank/query.csv}
NODE_MODEL=/mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3___5-4B
COORD_MODEL=/mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3___5-9B
mkdir -p "$RUN_DIR/source"
cp -r edge_rca "$RUN_DIR/source/edge_rca"
export PYTHONPATH="$RUN_DIR/source"
cd "$RUN_DIR"
PREPARED="$RUN_DIR/prepared"
TELEMETRY=/mnt/beegfs/home/wangchunyun25/Documents/OpenRCA/dataset/Bank/telemetry
"$PYTHON" -u -m edge_rca prepare --telemetry-root "$TELEMETRY" --query-file "$QUERY_FILE" --query-row "$ROW" --output "$PREPARED"
"$PYTHON" -u -m edge_rca.case_runtime prepare --prepared "$PREPARED" --output "$RUN_DIR/frames" \
    --query-file "$QUERY_FILE" --row "$ROW" --checkpoints 1800 --expected-nodes 0
"$PYTHON" -u -m edge_rca.case_runtime nodes --root "$RUN_DIR" --prepared "$PREPARED" \
    --seconds 1800 --node-model "$NODE_MODEL" --node-mode "${EDGERCA_NODE_MODE:-catalogue}" ${EDGERCA_DECISIONS_ONLY:+--decisions-only}
"$PYTHON" -u -m edge_rca.case_runtime coordinate --root "$RUN_DIR" --prepared "$PREPARED" \
    --seconds 1800 --node-model "$NODE_MODEL" --coordinator-model "$COORD_MODEL" \
    --max-queries 3 --final --node-mode "${EDGERCA_NODE_MODE:-catalogue}"
"$PYTHON" -u -m edge_rca.case_eval --root "$RUN_DIR" \
    --scorer /mnt/beegfs/home/wangchunyun25/Documents/OpenRCA/main/evaluate.py
# The raw per-window stream is no longer needed; frames/nodes/evaluation remain.
rm -rf "$PREPARED"
