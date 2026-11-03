#!/bin/bash
#SBATCH --partition=rtx
#SBATCH --gres=gpu:3090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:50:00
#SBATCH --job-name=edge-offline-restore
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/rollback_20261030/bank-%A_%a.log
set -euo pipefail

EDGERCA_PROJECT=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
EDGERCA_REPEAT="$EDGERCA_PROJECT/outputs/rollback_20261030"
EDGERCA_SOURCE="$EDGERCA_REPEAT/frozen_source"
EDGERCA_ROW=${SLURM_ARRAY_TASK_ID:?array row required}
EDGERCA_CASE="$EDGERCA_REPEAT/cases/$EDGERCA_ROW"
EDGERCA_QUERY=/mnt/beegfs/home/wangchunyun25/Documents/OpenRCA/dataset/Bank/query.csv
EDGERCA_TELEMETRY=/mnt/beegfs/home/wangchunyun25/Documents/OpenRCA/dataset/Bank/telemetry
EDGERCA_4B=/mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3___5-4B
EDGERCA_9B=/mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3___5-9B
EDGERCA_PY=/mnt/beegfs/home/wangchunyun25/.conda/envs/csle/bin/python
export PYTHONPATH="$EDGERCA_SOURCE"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TZ=Asia/Shanghai
cd "$EDGERCA_SOURCE"
"$EDGERCA_PY" "$EDGERCA_SOURCE/rollback_check.py" verify --source "$EDGERCA_SOURCE" --manifest "$EDGERCA_REPEAT/restore_manifest.json"
mkdir -p "$EDGERCA_CASE"
test ! -e "$EDGERCA_CASE/evaluation/result.json"
"$EDGERCA_PY" -u - <<'PY'
import torch, transformers
print('ENV', torch.__version__, transformers.__version__, 'GPU', torch.cuda.get_device_name(0), flush=True)
assert torch.cuda.is_available()
PY
"$EDGERCA_PY" -u -m edge_rca prepare --telemetry-root "$EDGERCA_TELEMETRY" --query-file "$EDGERCA_QUERY" --query-row "$EDGERCA_ROW" --output "$EDGERCA_CASE/prepared"
"$EDGERCA_PY" -u -m edge_rca.case_runtime prepare --prepared "$EDGERCA_CASE/prepared" --output "$EDGERCA_CASE/frames" --query-file "$EDGERCA_QUERY" --row "$EDGERCA_ROW" --checkpoints 1800 --expected-nodes 0
"$EDGERCA_PY" -u -m edge_rca.case_runtime nodes --root "$EDGERCA_CASE" --prepared "$EDGERCA_CASE/prepared" --seconds 1800 --node-model "$EDGERCA_4B" --node-mode catalogue
"$EDGERCA_PY" -u -m edge_rca.case_runtime coordinate --root "$EDGERCA_CASE" --prepared "$EDGERCA_CASE/prepared" --seconds 1800 --node-model "$EDGERCA_4B" --coordinator-model "$EDGERCA_9B" --max-queries 3 --final --node-mode catalogue
"$EDGERCA_PY" -u -m edge_rca.case_eval --root "$EDGERCA_CASE" --scorer /mnt/beegfs/home/wangchunyun25/Documents/OpenRCA/main/evaluate.py --time-tolerance-seconds 300
"$EDGERCA_PY" "$EDGERCA_SOURCE/rollback_check.py" rescore --case "$EDGERCA_CASE"
"$EDGERCA_PY" - "$EDGERCA_CASE/prepared" <<'PY'
from pathlib import Path
import shutil,sys
p=Path(sys.argv[1]).resolve()
assert p.name=='prepared' and p.parent.parent.name=='cases'
shutil.rmtree(p)
PY
echo "OFFLINE_CASE_COMPLETE row=$EDGERCA_ROW"
