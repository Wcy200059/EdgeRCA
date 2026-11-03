#!/bin/bash
#SBATCH --partition=rtx
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=edge-bank-summary
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/bank_summary-%j.log
set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
PYTHON=/mnt/beegfs/home/wangchunyun25/.conda/envs/csle/bin/python
"$PYTHON" tools/bank_summary.py --root outputs --job "${BANK_JOB:?BANK_JOB is required}"
