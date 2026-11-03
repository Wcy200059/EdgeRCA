#!/bin/bash
#SBATCH --partition=rtx
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:20:00
#SBATCH --job-name=edge-final-replay
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/final_replay-%j.log
set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHON=/mnt/beegfs/home/wangchunyun25/.conda/envs/csle/bin/python
OUT="/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/final_replay_${SLURM_JOB_ID}"
cp -r edge_rca "$OUT/source/edge_rca" 2>/dev/null || { mkdir -p "$OUT/source"; cp -r edge_rca "$OUT/source/edge_rca"; }
export PYTHONPATH="$OUT/source"
"$PYTHON" tools/replay_final.py --root outputs/bank_case_144445_1 \
    --model /mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3___5-9B \
    --output "$OUT/result" --time-tolerance-seconds 300
