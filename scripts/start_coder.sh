#!/bin/bash
#SBATCH --time=0-10:0:00
#SBATCH --account=def-vganesh
#SBATCH --mem=64G
#SBATCH -o logs/coder_%j.log
#SBATCH --gpus=h100:1

source ../../general/bin/activate
PYTHONPATH=./src:$PYTHONPATH
module load arrow
pip install trl
python src/llmsat/data/algorithm_parse.py
python src/llmsat/evaluation/coder.py --first_n $1