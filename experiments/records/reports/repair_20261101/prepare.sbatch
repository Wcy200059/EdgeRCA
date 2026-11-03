#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:35:00
#SBATCH --job-name=edge-repair-prepare
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/repair_20261101/logs/prepare-%j.log
set -euo pipefail
EDGERCA_RUN=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/repair_20261101
EDGERCA_SOURCE="$EDGERCA_RUN/frozen_source"
EDGERCA_BANK=/mnt/beegfs/home/wangchunyun25/Documents/OpenRCA/dataset/Bank
export PYTHONPATH="$EDGERCA_SOURCE" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TZ=Asia/Shanghai
cd "$EDGERCA_SOURCE"
python3 rollback_check.py verify --source "$EDGERCA_SOURCE" --manifest "$EDGERCA_RUN/source_manifest.json"
for EDGERCA_ROW in 20 36; do
    EDGERCA_CASE="$EDGERCA_RUN/cases/$EDGERCA_ROW"
    python3 -u -m edge_rca prepare --telemetry-root "$EDGERCA_BANK/telemetry" \
        --query-file "$EDGERCA_BANK/query.csv" --query-row "$EDGERCA_ROW" --output "$EDGERCA_CASE/prepared"
    python3 -u -m edge_rca.case_runtime prepare --prepared "$EDGERCA_CASE/prepared" \
        --output "$EDGERCA_CASE/frames" --query-file "$EDGERCA_BANK/query.csv" --row "$EDGERCA_ROW" \
        --checkpoints 1800 --expected-nodes 0
done
