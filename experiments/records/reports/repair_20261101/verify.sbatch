#!/bin/bash
#SBATCH --partition=rtx
#SBATCH --gres=gpu:3090:1
#SBATCH --array=20,36%1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:45:00
#SBATCH --job-name=edge-repair-verify
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/repair_20261101/logs/verify-%A_%a.log
set -euo pipefail
EDGERCA_RUN=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/repair_20261101
EDGERCA_SOURCE="$EDGERCA_RUN/frozen_source_v2"
EDGERCA_CASE="$EDGERCA_RUN/cases/${SLURM_ARRAY_TASK_ID:?row required}"
EDGERCA_PY=/mnt/beegfs/home/wangchunyun25/.conda/envs/csle/bin/python
EDGERCA_MODELS=/mnt/beegfs/home/wangchunyun25/models/Qwen
export PYTHONPATH="$EDGERCA_SOURCE" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TZ=Asia/Shanghai
cd "$EDGERCA_SOURCE"
"$EDGERCA_PY" rollback_check.py verify --source "$EDGERCA_SOURCE" --manifest "$EDGERCA_RUN/source_manifest.json"
test -f "$EDGERCA_CASE/frames/complete.json"
test ! -e "$EDGERCA_CASE/evaluation/result.json"
"$EDGERCA_PY" -u -m edge_rca.case_runtime nodes --root "$EDGERCA_CASE" --prepared "$EDGERCA_CASE/prepared" \
    --seconds 1800 --node-model "$EDGERCA_MODELS/Qwen3___5-4B" --node-mode catalogue
"$EDGERCA_PY" -u -m edge_rca.case_runtime coordinate --root "$EDGERCA_CASE" --prepared "$EDGERCA_CASE/prepared" \
    --seconds 1800 --node-model "$EDGERCA_MODELS/Qwen3___5-4B" --coordinator-model "$EDGERCA_MODELS/Qwen3___5-9B" \
    --max-queries 3 --final --node-mode catalogue
"$EDGERCA_PY" -u -m edge_rca.case_eval --root "$EDGERCA_CASE" \
    --scorer /mnt/beegfs/home/wangchunyun25/Documents/OpenRCA/main/evaluate.py --time-tolerance-seconds 300
"$EDGERCA_PY" rollback_check.py rescore --case "$EDGERCA_CASE"
"$EDGERCA_PY" "$EDGERCA_RUN/summarize.py"
