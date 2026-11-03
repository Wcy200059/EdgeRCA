#!/bin/bash
#SBATCH --partition=rtx
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=edge-bank-diagnose
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/bank_diagnose-%j.log
set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
export PYTHONPATH=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
PYTHON=/mnt/beegfs/home/wangchunyun25/.conda/envs/csle/bin/python
"$PYTHON" tools/bank_diagnose.py --root outputs --job "${BANK_JOB:?BANK_JOB required}" \
    --time-tolerance-seconds 300
"$PYTHON" tools/candidate_recall.py --job "${BANK_JOB}" | tee "outputs/bank_candidate_recall_${BANK_JOB}.json"
"$PYTHON" tools/stage_recall.py --job "${BANK_JOB}" > "outputs/bank_stage_recall_${BANK_JOB}.json"
cat "outputs/bank_stage_recall_${BANK_JOB}.json"
