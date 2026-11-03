#!/bin/bash
#SBATCH --partition=rtx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=edgerca-06b-probe
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/tiny_node_slurm-%j.log

set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

PYTHON=/mnt/beegfs/home/wangchunyun25/.conda/envs/csle/bin/python
MODEL=/mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3-0.6B
INPUTS=${1:-outputs/tiny_node_inputs_20261016}
OUTPUT=outputs/tiny_node_qwen06b_${SLURM_JOB_ID}

nvidia-smi --query-gpu=name,memory.total --format=csv
"$PYTHON" -u -m edge_rca.node_probe run --inputs "$INPUTS" --output "$OUTPUT" --model "$MODEL"
