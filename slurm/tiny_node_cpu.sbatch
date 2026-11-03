#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=edgerca-06b-cpu
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/tiny_node_cpu-%j.log

set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

PYTHON=/mnt/beegfs/home/wangchunyun25/.conda/envs/csle/bin/python
"$PYTHON" -u -m edge_rca.node_probe run \
  --inputs outputs/tiny_node_inputs_20261016 \
  --output "outputs/tiny_node_qwen06b_cpu_${SLURM_JOB_ID}" \
  --model /mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3-0.6B --device cpu
