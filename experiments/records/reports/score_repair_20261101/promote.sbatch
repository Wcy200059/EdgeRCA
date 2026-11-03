#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:05:00
#SBATCH --job-name=edge-apply-verified
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/score_repair_20261101/logs/promote-%j.log
set -euo pipefail
EDGERCA_CHECK=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/score_repair_20261101
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
cd "$EDGERCA_CHECK"
python3 promote.py retrieval
PYTHONPATH="$EDGERCA_CHECK/baseline_source" python3 live_summary.py
