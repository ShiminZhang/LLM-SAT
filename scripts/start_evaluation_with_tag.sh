#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=0-10:00:00
#SBATCH --qos=coc-ice
#SBATCH --output=logs/evaluation_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgopalan6@gatech.edu

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
python src/llmsat/pipelines/evaluation.py --run_all --generation_tag "$GENERATION_TAG" "$@"
