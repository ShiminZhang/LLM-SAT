#!/bin/bash
#SBATCH --job-name=qwen8b_div
#SBATCH --account=def-vganesh
#SBATCH --output=logs/qwen8b_div_%j.log
#SBATCH --error=logs/qwen8b_div_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=02:00:00
#SBATCH --mail-user=pnguyen337@gatech.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --export=ALL


set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ $# -lt 2 ]; then
  echo "Usage: sbatch scripts/slurm_analyze_diversity_qwen8b.sh <TEAM_BATCH_DIR> <OUT_DIR> [extra args...]"
  echo "Example: sbatch scripts/slurm_analyze_diversity_qwen8b.sh outputs/controlled_mutation/batch_batch_... outputs/controlled_mutation/diversity_qwen8b --embedding qwen3"
  exit 1
fi

TEAM_BATCH_DIR="$1"
OUT_DIR="$2"
shift 2

source ~/.venvs/llmsat312/bin/activate
export PYTHONPATH="./src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"

python scripts/analyze_diversity.py \
  --team-batch-dir "$TEAM_BATCH_DIR" \
  --out-dir "$OUT_DIR" \
  --embedding qwen3 \
  --qwen-model Qwen/Qwen3-Embedding-8B \
  --qwen-dtype float16 \
  --qwen-require-cuda \
  --qwen-max-length 512 \
  --qwen-batch-size 2 \
  "$@"
