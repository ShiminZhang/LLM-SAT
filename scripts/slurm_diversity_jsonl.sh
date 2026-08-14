#!/bin/bash
#SBATCH --job-name=qwen3_div
#SBATCH --account=def-vganesh
#SBATCH --output=logs/qwen3_div_%j.log
#SBATCH --error=logs/qwen3_div_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=00:45:00
#SBATCH --export=ALL

# Run analyze_diversity.py with --in <jsonl> --embedding qwen3 on a GPU.
# Usage: sbatch scripts/slurm_diversity_jsonl.sh <strategies.jsonl> <out_dir>

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ $# -lt 2 ]; then
  echo "Usage: sbatch scripts/slurm_diversity_jsonl.sh <strategies.jsonl> <out_dir>"
  exit 1
fi

JSONL="$1"
OUT_DIR="$2"

module load scipy-stack/2024b
source ~/general/bin/activate
export PYTHONPATH=src

mkdir -p "$OUT_DIR"
python scripts/analyze_diversity.py \
  --in "$JSONL" \
  --out-dir "$OUT_DIR" \
  --embedding qwen3 \
  --qwen-batch-size 8 \
  --qwen-max-length 1024
