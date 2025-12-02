#!/bin/bash
#SBATCH --time=0-10:0:00
#SBATCH --account=def-vganesh
#SBATCH --mem=16G
#SBATCH -o logs/evaluation_%j.log

source ../../general/bin/activate
PYTHONPATH=./src:$PYTHONPATH
python src/llmsat/pipelines/evaluation.py --run_all --generation_tag $1