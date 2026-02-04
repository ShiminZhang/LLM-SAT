#!/bin/bash
#SBATCH --job-name=qwen8b_calib
#SBATCH --account=def-vganesh
#SBATCH --output=logs/qwen8b_calib_%j.log
#SBATCH --error=logs/qwen8b_calib_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=01:00:00
#SBATCH --mail-user=pnguyen337@gatech.edu
#SBATCH --mail-type=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ $# -lt 2 ]; then
  echo "Usage: sbatch scripts/slurm_calibrate_qwen8b_leaders.sh <TEAM_BATCH_DIR> <TAG> [extra args...]"
  echo "Example: sbatch scripts/slurm_calibrate_qwen8b_leaders.sh outputs/team_large_jan27/batch_batch_... team_large_jan27 --top-k 10"
  exit 1
fi

TEAM_BATCH_DIR="$1"
TAG="$2"
shift 2

# Activate your env
source ~/.venvs/llmsat312/bin/activate
export PYTHONPATH="./src:${PYTHONPATH:-}"

# Optional: use a shared HF cache on cluster
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"

python scripts/calibrate_leader_similarity.py \
  --team-batch-dir "$TEAM_BATCH_DIR" \
  --qwen-model Qwen/Qwen3-Embedding-8B \
  --qwen-dtype float16 \
  --qwen-max-length 512 \
  --qwen-batch-size 2 \
  --out "outputs/$TAG/calibration/qwen8b_leaders.json" \
  "$@"
