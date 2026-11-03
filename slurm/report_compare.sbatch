#!/bin/bash
#SBATCH --partition=rtx
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=edge-report-compare
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/report_compare-%j.log
set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHON=/mnt/beegfs/home/wangchunyun25/.conda/envs/csle/bin/python
for MODEL in Qwen3-0.6B Qwen3___5-2B Qwen3___5-4B; do
  "$PYTHON" -u -m edge_rca.report_probe run --inputs "$1" \
    --model "/mnt/beegfs/home/wangchunyun25/models/Qwen/$MODEL" \
    --output "outputs/report_compare_${SLURM_JOB_ID}/$MODEL"
done
