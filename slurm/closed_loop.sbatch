#!/bin/bash
#SBATCH --partition=rtx
#SBATCH --gres=gpu:4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=edge-9b-followup
#SBATCH --output=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/closed_loop-%j.log
set -euo pipefail
cd /mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHON=/mnt/beegfs/home/wangchunyun25/.conda/envs/csle/bin/python
RUN_DIR="/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/closed_loop_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR/source"
cp -r edge_rca "$RUN_DIR/source/edge_rca"
export PYTHONPATH="$RUN_DIR/source"
# Run outside the checkout so the frozen package wins module discovery.
cd "$RUN_DIR"
COORD_MODEL=/mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3___5-9B
NODE_MODEL=/mnt/beegfs/home/wangchunyun25/models/Qwen/Qwen3___5-4B
INBOX=/mnt/beegfs/home/wangchunyun25/Documents/EdgeRCA/outputs/report_grounded_141723/coordinator_inbox.json
PREPARED=/mnt/beegfs/home/wangchunyun25/Documents/OpsAgent/results/edge_stream/Bank_train0000_20261015/prepared
"$PYTHON" -u -m edge_rca.closed_loop plan --inbox "$INBOX" --model "$COORD_MODEL" --overview-only --output "$RUN_DIR/plan"
"$PYTHON" -u -m edge_rca.closed_loop tools --plan "$RUN_DIR/plan" --prepared "$PREPARED" --max-requests 1 --output "$RUN_DIR/tools"
"$PYTHON" -u -m edge_rca.closed_loop node --inputs "$RUN_DIR/tools" --prepared "$PREPARED" --max-self-requests 2 --model "$NODE_MODEL" --output "$RUN_DIR/answers"
"$PYTHON" -u -m edge_rca.closed_loop update --inbox "$INBOX" --plan "$RUN_DIR/plan" --answers "$RUN_DIR/answers" --model "$COORD_MODEL" --overview-only --output "$RUN_DIR/update"
