#!/bin/bash
#SBATCH --job-name=bench-gen
#SBATCH --account=def-vganesh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --output=logs/benchmark_generate_%j.out
#SBATCH --error=logs/benchmark_generate_%j.err
#SBATCH --mail-user=pnguyen337@gatech.edu
#SBATCH --mail-type=ALL

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

# Prefer conda if available; otherwise fall back to venv
set -eo pipefail

if command -v conda >/dev/null 2>&1; then
    # On some clusters, ~/.bashrc isn't sourced in batch jobs; source conda.sh explicitly
    if [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda/etc/profile.d/conda.sh"
    elif [ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]; then
        source "$HOME/mambaforge/etc/profile.d/conda.sh"
    else
        # Fallback: try bashrc to load conda function
        source ~/.bashrc 2>/dev/null || true
    fi
    conda activate dpo-training || echo "[WARN] conda env 'dpo-training' not found; continuing without conda"
else
    echo "[INFO] 'conda' not found; using Python venv at ~/.venvs/llmsat312"
    # Activate Python venv if present
    if [ -f "$HOME/.venvs/llmsat312/bin/activate" ]; then
        source "$HOME/.venvs/llmsat312/bin/activate"
    else
        echo "[ERROR] venv ~/.venvs/llmsat312 not found. Create it first: python3.12 -m venv ~/.venvs/llmsat312" >&2
        exit 1
    fi
fi

export PYTHONPATH="./src:${PYTHONPATH:-}"

# Source database credentials
source export_aws_db_pw.sh

# Optional: OpenAI API setup if required by generator
if [ -z "${OPENAI_API_KEY}" ]; then
    echo "[WARN] OPENAI_API_KEY not set; set it if generation uses OpenAI"
fi

echo "Starting benchmark code generation at $(date)"
echo "Using GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Generation tag (can be overridden by passing argument to sbatch)
TAG="${1:-dpo_testing}"
OUTPUT_DIR="${2:-outputs/benchmark}"

echo "Generation tag: $TAG"
echo "Output directory: $OUTPUT_DIR"

python scripts/benchmark_generate_codes.py \
    --tag "$TAG" \
    --output "$OUTPUT_DIR" \
    --temperature 0.7 \
    --max-tokens 2048

echo "Code generation completed at $(date)"