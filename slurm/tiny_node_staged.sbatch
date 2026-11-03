#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=edgerca-06b-stages
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/tiny_node_staged-%j.log
set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHON=/mnt/beegfs/home/wangchunyun25/.conda/envs/csle/bin/python
"$PYTHON" -u -m edge_rca.staged_probe \
  --inputs outputs/tiny_node_inputs_20261016 --updates 2 \
  --output "outputs/tiny_node_staged_${SLURM_JOB_ID}" \
  --model /mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3-0.6B --device cpu "$@"
