#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:05:00
#SBATCH --job-name=edge-bank-combine
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/bank_combine-%j.log
set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
PYTHON=/mnt/beegfs/home/wangchunyun25/.conda/envs/csle/bin/python
A_JOB=${A_JOB:?}
B_JOB=${B_JOB:?}
export PYTHONDONTWRITEBYTECODE=1
OUT_DIR=${OUT_DIR:-"$PWD/outputs/bank_combined_${SLURM_JOB_ID:?}"}
"$PYTHON" -m tools.bank_combine --root outputs --a-job "$A_JOB" --b-job "$B_JOB" --output "$OUT_DIR"
