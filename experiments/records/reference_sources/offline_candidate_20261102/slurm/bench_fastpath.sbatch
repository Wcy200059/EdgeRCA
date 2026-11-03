#!/bin/bash
#SBATCH --partition=rtx
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:15:00
#SBATCH --job-name=edge-fastpath-bench
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/fastpath-%j.log
set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
BASE_PY=/mnt/beegfs/home/wangchunyun25/.conda/envs/csle/bin/python
FAST_PY=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/.venv_fast/bin/python
MODEL=/mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3___5-4B
OUT=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/fastpath_bench_${SLURM_JOB_ID}
mkdir -p "$OUT"
"$BASE_PY" tools/bench_fastpath.py --model "$MODEL" --output "$OUT/base.json" || true
"$FAST_PY" tools/bench_fastpath.py --model "$MODEL" --output "$OUT/fast.json" || true
cat "$OUT/base.json" "$OUT/fast.json" 2>/dev/null || true
