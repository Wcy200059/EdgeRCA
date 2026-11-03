#!/bin/bash
#SBATCH --partition=rtx
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=edge-grounded
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/report_grounded-%j.log
set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHON=/mnt/beegfs/home/wangchunyun25/.conda/envs/csle/bin/python
"$PYTHON" -u -m edge_rca.report_probe run \
  --inputs outputs/report_inputs_141576 \
  --model /mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3___5-4B \
  --output "outputs/report_grounded_${SLURM_JOB_ID}" --coordinator
