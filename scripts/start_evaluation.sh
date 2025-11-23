#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/evaluation_%j.log
#SBATCH -A gts-vganesh3
#SBATCH -q inferno

cd "$SLURM_SUBMIT_DIR"

source "$HOME/.bashrc"
conda activate llmsat

export PYTHONPATH="./src:${PYTHONPATH:-}"

python src/llmsat/pipelines/evaluation.py --run_all
