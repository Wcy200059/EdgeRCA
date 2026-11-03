#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --job-name=edge-report-input
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/report_prepare-%j.log
set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
export PYTHONDONTWRITEBYTECODE=1
python3 -u -m edge_rca.report_probe prepare \
  --source outputs/memory_run_141454 \
  --prepared /mnt/beegfs/home/wangchunyun25/Documents/OpsAgent/results/edge_stream/Bank_train0000_20261015/prepared \
  --output "outputs/report_inputs_${SLURM_JOB_ID}"
