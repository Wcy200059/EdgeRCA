#!/bin/bash
#SBATCH --partition=rtx
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=edge-reference-reload
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/score_repair_20261101/logs/reload-%j.log
set -euo pipefail
EDGERCA_CHECK=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/score_repair_20261101
EDGERCA_PY=/mnt/beegfs/home/wangchunyun25/.conda/envs/csle/bin/python
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$EDGERCA_CHECK/baseline_source"
cd "$EDGERCA_CHECK"
"$EDGERCA_PY" -u reload_run.py infer
"$EDGERCA_PY" reload_run.py evaluate
