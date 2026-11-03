#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:10:00
#SBATCH --job-name=edge-bank-summary
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/bank_summary-%j.log
set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
PYTHON=/mnt/beegfs/home/wangchunyun25/.conda/envs/csle/bin/python
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -m tools.bank_summary --root outputs --job "${BANK_JOB:?BANK_JOB is required}" \
    --time-tolerance-seconds "${TIME_TOLERANCE_SECONDS:-60}" --expected-rows "${EXPECTED_ROWS:-136}"
