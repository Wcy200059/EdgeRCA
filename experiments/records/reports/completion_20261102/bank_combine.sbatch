#!/bin/bash
#SBATCH --partition=rtx
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=edge-bank-combine
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/bank_combine-%j.log
set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
PYTHON=/mnt/beegfs/home/wangchunyun25/.conda/envs/csle/bin/python
A_JOB=${A_JOB:?}
B_JOB=${B_JOB:?}
OUT_JOB=${OUT_JOB:?}
for row in $(seq 0 45); do
    src="$PWD/outputs/bank_case_${A_JOB}_${row}"
    rm -rf "outputs/bank_case_${OUT_JOB}_${row}"
    [ -d "$src" ] && ln -sfn "$src" "outputs/bank_case_${OUT_JOB}_${row}"
done
for row in $(seq 46 135); do
    src="$PWD/outputs/bank_case_${B_JOB}_${row}"
    rm -rf "outputs/bank_case_${OUT_JOB}_${row}"
    [ -d "$src" ] && ln -sfn "$src" "outputs/bank_case_${OUT_JOB}_${row}"
done
"$PYTHON" tools/bank_summary.py --root outputs --job "$OUT_JOB"
"$PYTHON" tools/bank_diagnose.py --root outputs --job "$OUT_JOB" --time-tolerance-seconds 300
"$PYTHON" tools/candidate_recall.py --job "$OUT_JOB" > "outputs/bank_candidate_recall_${OUT_JOB}.json"
"$PYTHON" tools/stage_recall.py --job "$OUT_JOB" > "outputs/bank_stage_recall_${OUT_JOB}.json"
cat "outputs/bank_stage_recall_${OUT_JOB}.json"
