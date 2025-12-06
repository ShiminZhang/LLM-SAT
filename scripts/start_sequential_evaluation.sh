#!/bin/bash
#SBATCH --job-name=seq_eval
#SBATCH --account:def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sequential_eval_%j.log
#SBATCH --mail-user=pnguyen337@gatech.edu
#SBATCH --mail-type=ALL

# Usage: sbatch scripts/start_sequential_evaluation.sh <generation_tag>
# Example: sbatch scripts/start_sequential_evaluation.sh kissatmab_experiment7

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ -z "${1:-}" ]; then
    echo "Usage: sbatch scripts/start_sequential_evaluation.sh <generation_tag> [options]"
    echo "Example: sbatch scripts/start_sequential_evaluation.sh kissatmab_experiment7"
    echo "Example: sbatch scripts/start_sequential_evaluation.sh kissatmab_experiment7 --timeout 600 --penalty 1200"
    exit 1
fi

GENERATION_TAG=$1
shift  # Remove first argument, keep the rest

# Activate venv
source ~/.venvs/llmsat312/bin/activate

echo "Starting sequential evaluation for tag: $GENERATION_TAG"
echo "Additional args: $@"
echo "Start time: $(date)"

PYTHONPATH=src python scripts/evaluate_sequential.py --tag "$GENERATION_TAG" "$@"

echo "End time: $(date)"
echo "Sequential evaluation complete!"