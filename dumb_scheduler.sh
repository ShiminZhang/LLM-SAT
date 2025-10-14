#!/bin/bash
#SBATCH --nodes=1
#SBATCH --qos=interactive
#SBATCH --time=00:30:00
#SBATCH --constraint=gpu&hbm80g
#SBATCH --gpus=1
#SBATCH --account=m5029

conda activate llmsat
accelerate launch src/dpo.py
