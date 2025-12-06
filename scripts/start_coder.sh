#!/bin/bash
#SBATCH --job-name=coder
#SBATCH --time=0-5:00:00
#SBATCH --account=def-vganesh
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -o logs/coder_%j.log
#SBATCH --gpus-per-node=h100:2

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source ~/.venvs/llmsat312/bin/activate
export PYTHONPATH="./src:${PYTHONPATH:-}"
module load arrow
pip install --quiet trl
python src/llmsat/data/algorithm_parse.py
python src/llmsat/evaluation/coder.py --first_n "$1"