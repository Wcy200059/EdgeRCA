#!/bin/bash
#SBATCH --partition=rtx
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=edge-build-conv1d
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/build_conv1d-%j.log
set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
which nvcc || true
nvcc --version 2>/dev/null | tail -2 || true
FAST_PY=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/.venv_fast/bin/python
if ! "$FAST_PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "WARNING: torch.cuda.is_available() is false on this CPU-only build node" >&2
fi
"$FAST_PY" -m pip install --no-cache-dir -q ninja packaging
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDA_HOME
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=2
"$FAST_PY" -m pip install --no-cache-dir --no-build-isolation causal-conv1d
"$FAST_PY" - <<'PY'
import causal_conv1d
print('causal_conv1d import ok', getattr(causal_conv1d, '__version__', '?'))
PY
