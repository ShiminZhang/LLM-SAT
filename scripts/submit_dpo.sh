#!/bin/bash
#SBATCH --job-name=sat-dpo
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --output=logs/dpo_%j.out
#SBATCH --error=logs/dpo_%j.err
#SBATCH --mail-user=pnguyen337@gatech.edu
#SBATCH --mail-type=ALL

cd "$SLURM_SUBMIT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Prefer conda if available; otherwise fall back to venv
set -eo pipefail

if command -v conda >/dev/null 2>&1; then
    # Source conda.sh explicitly; bashrc may not be loaded in batch jobs
    if [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda/etc/profile.d/conda.sh"
    elif [ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]; then
        source "$HOME/mambaforge/etc/profile.d/conda.sh"
    else
        source ~/.bashrc 2>/dev/null || true
    fi
    conda activate dpo-training || echo "[WARN] conda env 'dpo-training' not found; continuing without conda"
else
    echo "[INFO] 'conda' not found; using Python venv at ~/.venvs/llmsat312"
    if [ -f "$HOME/.venvs/llmsat312/bin/activate" ]; then
        source "$HOME/.venvs/llmsat312/bin/activate"
    else
        echo "[ERROR] venv ~/.venvs/llmsat312 not found. Create it first: python3.12 -m venv ~/.venvs/llmsat312" >&2
        exit 1
    fi
fi

pip list | grep -E "transformers|trl|bitsandbytes"

# Set PYTHONPATH
export PYTHONPATH="./src:${PYTHONPATH:-}"

# Optional: OpenAI API setup if any downstream uses it
if [ -z "${OPENAI_API_KEY}" ]; then
    echo "[WARN] OPENAI_API_KEY not set; set it if required"
fi

echo "Starting DPO training at $(date)"
echo "Using GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run training
python scripts/train_dpo.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --train_data "data/dpo_formatted" \
    --output_dir "outputs/dpo1/dpo_training" \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --no_qlora \
    --no_lora \
    --learning_rate 5e-6 \
    --num_epochs 3 \
    --beta 0.1 \
    --max_length 2048 \
    --max_prompt_length 1024 \
    --run_name "sat-solver-dpo"

echo "Job completed at $(date)"