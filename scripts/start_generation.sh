#!/bin/bash
#SBATCH --time=0-10:0:00
#SBATCH --account=def-vganesh
#SBATCH --mem=16G
#SBATCH -o logs/generation_%j.log

source ../../general/bin/activate
PYTHONPATH=./src:$PYTHONPATH
source .env
python src/llmsat/pipelines/chatgpt_data_generation.py