#!/bin/bash
#SBATCH --job-name=sat-dpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/dpo_%j.out
#SBATCH --error=logs/dpo_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mgopalan6@gatech.edu

cd "$SLURM_SUBMIT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment (do this before set -u to avoid bashrc errors)
source ~/.bashrc 2>/dev/null || true
conda activate dpo-training

# Enable strict error checking AFTER conda activation
set -eo pipefail

# Upgrade packages to compatible versions for Llama 3.1
echo "Upgrading transformers, trl, and bitsandbytes..."
pip install --upgrade transformers>=4.43.0 trl>=0.7.0 bitsandbytes>=0.43.0 --quiet

# Set PYTHONPATH
export PYTHONPATH="./src:${PYTHONPATH:-}"

echo "Starting DPO training at $(date)"
echo "Using GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run training
python scripts/train_dpo.py \
    --model_name "meta-llama/Llama-3.1-8B-Instruct" \
    --train_data "data/dpo_formatted" \
    --output_dir "outputs/dpo1/dpo_training" \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --num_epochs 3 \
    --beta 0.1 \
    --max_length 2048 \
    --max_prompt_length 1024 \
    --run_name "sat-solver-dpo"

echo "Job completed at $(date)"
