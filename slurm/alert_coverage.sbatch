#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --job-name=edgerca-18node-audit
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/alert_coverage-%j.log
set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 -u -m edge_rca.alert_coverage \
  --prepared /mnt/beegfs/home/wangchunyun25/Documents/OpsAgent/results/edge_stream/Bank_train0000_20261015/prepared \
  --output "outputs/alert_coverage_${SLURM_JOB_ID}"
