#!/bin/bash
#SBATCH --time=0-10:0:00
#SBATCH --account=def-vganesh
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH -o logs/dumb_script_%j.log

source ../../general/bin/activate
PYTHONPATH=./src:$PYTHONPATH
python scripts/run_baseline.py