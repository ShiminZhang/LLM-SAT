#!/bin/bash
#SBATCH --time=0-10:0:00
#SBATCH --account=def-vganesh
#SBATCH --mem=16G
#SBATCH -o logs/evaluation_%j.log

# Usage: sbatch scripts/start_evaluation_with_tag.sh <generation_tag> [additional args]
# Example: sbatch scripts/start_evaluation_with_tag.sh chatgpt_data_generation_gpt4o_1
# Example: sbatch scripts/start_evaluation_with_tag.sh chatgpt_data_generation_gpt4o_1 --first_n 5

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [ -z "$1" ]; then
    echo "Error: generation_tag not provided"
    echo "Usage: sbatch scripts/start_evaluation_with_tag.sh <generation_tag> [additional args]"
    exit 1
fi

GENERATION_TAG=$1
shift  # Remove first argument, keep the rest

# activate your existing venv (path relative to submit dir)
source ~/general/bin/activate

export PYTHONPATH="./src:${PYTHONPATH:-}"

# Run evaluation with the generation tag and any additional arguments
