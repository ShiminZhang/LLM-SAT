#!/bin/bash
#SBATCH --job-name=coder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-05:00:00
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:a100:2
#SBATCH --output=logs/coder_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mgopalan6@gatech.edu

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# activate your existing venv (path relative to submit dir)
source ~/general/bin/activate

# any modules you rely on
# module load arrow

export PYTHONPATH="./src:${PYTHONPATH:-}"

# If trl is not already in your env, this will install it for the job.
# (Better to pre-install in the env if possible.)
pip install --quiet trl

python src/llmsat/data/algorithm_parse.py
python src/llmsat/evaluation/coder.py --first_n "$1"