#!/bin/bash
#SBATCH --job-name=bench-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/benchmark_generate_%j.out
#SBATCH --error=logs/benchmark_generate_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mgopalan6@gatech.edu

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

# Activate conda environment
source ~/.bashrc 2>/dev/null || true
conda activate dpo-training

set -eo pipefail

export PYTHONPATH="./src:${PYTHONPATH:-}"

# Source database credentials
source export_aws_db_pw.sh

echo "Starting benchmark code generation at $(date)"
echo "Using GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Generation tag (can be overridden by passing argument to sbatch)
TAG="${1:-dpo_testing}"
OUTPUT_DIR="${2:-outputs/benchmark}"

echo "Generation tag: $TAG"
echo "Output directory: $OUTPUT_DIR"

python scripts/benchmark_generate_codes.py \
    --tag "$TAG" \
    --output "$OUTPUT_DIR" \
    --temperature 0.7 \
    --max-tokens 2048

echo "Code generation completed at $(date)"
