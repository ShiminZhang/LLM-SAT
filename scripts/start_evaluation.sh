#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --time=0-10:00:00
#SBATCH --account=def-vganesh
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=h100:2
#SBATCH -o logs/evaluation_%j.log
#SBATCH --mail-user=pnguyen337@gatech.edu
#SBATCH --mail-type=ALL

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source ~/.venvs/llmsat312/bin/activate
export PYTHONPATH="./src:${PYTHONPATH:-}"

# Usage:
#   sbatch start_evaluation.sh                     # run all (uses default tag)
#   sbatch start_evaluation.sh --first_n 2         # run first 2 algorithms
#   sbatch start_evaluation.sh --algorithm_id XYZ  # run specific algorithm
#   sbatch start_evaluation.sh --generation_tag TAG --run_all  # specify tag

if [ $# -eq 0 ]; then
    python src/llmsat/pipelines/evaluation.py --run_all
else
    python src/llmsat/pipelines/evaluation.py "$@"
fi