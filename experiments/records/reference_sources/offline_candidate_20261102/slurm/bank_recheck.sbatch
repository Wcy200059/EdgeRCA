#!/bin/bash
#SBATCH --partition=rtx
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH --job-name=edge-bank-recheck
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/bank_recheck-%A_%a.log
#SBATCH --array=100,103%2
set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHON=/mnt/beegfs/home/wangchunyun25/.conda/envs/csle/bin/python
ROW=${SLURM_ARRAY_TASK_ID:?}
JOB=${SLURM_ARRAY_JOB_ID:-manual}
OLD="/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/bank_pilot_143758_${ROW}"
RUN_DIR="/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/bank_recheck_${JOB}_${ROW}"
test -d "$OLD/frames/c1800"
test -d "$OLD/nodes_1800"
test -d "$OLD/prepared"
mkdir -p "$RUN_DIR/source"
cp -r edge_rca "$RUN_DIR/source/edge_rca"
ln -sfn "$OLD/frames" "$RUN_DIR/frames"
ln -sfn "$OLD/nodes_1800" "$RUN_DIR/nodes_1800"
ln -sfn "$OLD/prepared" "$RUN_DIR/prepared"
export PYTHONPATH="$RUN_DIR/source"
cd "$RUN_DIR"
"$PYTHON" -u -m edge_rca.case_runtime coordinate --root "$RUN_DIR" --prepared "$RUN_DIR/prepared" \
    --seconds 1800 --node-model /mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3___5-4B \
    --coordinator-model /mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3___5-9B \
    --max-queries 3 --final --node-mode catalogue
"$PYTHON" -u -m edge_rca.case_eval --root "$RUN_DIR" \
    --scorer /mnt/beegfs/home/wangchunyun25/Documents/OpenRCA/main/evaluate.py
