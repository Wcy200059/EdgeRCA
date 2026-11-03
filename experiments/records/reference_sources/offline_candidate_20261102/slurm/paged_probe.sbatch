#!/bin/bash
#SBATCH --partition=rtx
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --job-name=edgerca-alert-06b
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/paged_probe-%j.log
set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHON=/mnt/beegfs/home/wangchunyun25/.conda/envs/csle/bin/python
"$PYTHON" -u -m edge_rca.paged_probe \
  --inputs "$1" --output "outputs/paged_probe_${SLURM_JOB_ID}" \
  --model /mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3-0.6B --device cuda:0
