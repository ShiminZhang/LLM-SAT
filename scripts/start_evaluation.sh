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

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# activate your existing venv (path relative to submit dir)
source ~/general/bin/activate

export PYTHONPATH="./src:${PYTHONPATH:-}"

python src/llmsat/pipelines/evaluation.py --run_all
