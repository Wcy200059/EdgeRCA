#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --job-name=edgerca-memory
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/memory_run-%j.log
set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
export PYTHONDONTWRITEBYTECODE=1
python3 -u -m edge_rca.memory_run \
  --prepared /mnt/beegfs/home/wangchunyun25/Documents/OpsAgent/results/edge_stream/Bank_train0000_20261015/prepared \
  --output "outputs/memory_run_${SLURM_JOB_ID}"
