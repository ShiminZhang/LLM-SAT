#!/bin/bash
# run_loop_a.sh — Leader Refinement Loop (Loop A)
#
# Iterates N times: generate mutants → evaluate → promote
# Leaders are NOT regenerated; only new mutant variants are created each iteration.
#
# Usage:
#   ./run_loop_a.sh <cc|nersc> <base_tag> <n_iterations> [source_tag] [--init]
#
# Examples:
#   # Start from initial leaders on Compute Canada
#   ./run_loop_a.sh cc gemini_trial5 3
#
#   # Start from GE offspring on NERSC
#   ./run_loop_a.sh nersc gemini_trial5_ge1 3 gemini_trial5_ge1_iter0
#
#   # Initialize from scratch: generate leaders, evaluate, then run iterations
#   ./run_loop_a.sh cc gemini_trial5 3 --init
#
# Arguments:
#   cc|nersc       - Cluster to run on (selects script variants)
#   base_tag       - Base name for iteration tags ({base_tag}_iter1, _iter2, ...)
#   n_iterations   - Number of mutant→evaluate→promote cycles to run
#   source_tag     - (Optional) Tag to load initial leaders from.
#                    Defaults to {base_tag}_iter0
#   --init         - (Optional) Generate initial leaders + members + code first
#
# Init mode env vars:
#   N_LEADERS        - Number of leaders to generate (default: 5)
#   DESIGNER_PROMPT  - Path to leader prompt (default: data/prompts/leader_prompt_testing.txt)

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 <cc|nersc> <base_tag> <n_iterations> [source_tag] [--init]"
    exit 1
fi

CLUSTER="$1"
BASE_TAG="$2"
N_ITERATIONS="$3"

# Parse --init from args (can appear as $4 or $5)
INIT=false
for arg in "$@"; do
    if [ "$arg" = "--init" ]; then
        INIT=true
    fi
done

# If $4 is --init, don't treat it as source_tag
if [ "${4:-}" = "--init" ]; then
    SOURCE_TAG="${BASE_TAG}_iter0"
else
    SOURCE_TAG="${4:-${BASE_TAG}_iter0}"
fi

DATAGEN_SCRIPT="src/llmsat/pipelines/gemini_data_generation.py"

case "$CLUSTER" in
    cc)
        EVAL_SCRIPT="src/llmsat/pipelines/evaluation.py"
        ;;
    nersc)
        EVAL_SCRIPT="src/llmsat/pipelines/evaluation_nersc.py"
        ;;
    *)
        echo "ERROR: cluster must be 'cc' or 'nersc', got '$CLUSTER'"
        exit 1
        ;;
esac
POLL_INTERVAL="${POLL_INTERVAL:-120}"  # seconds between squeue checks
M_VARIANTS="${M_VARIANTS:-3}"
_CFG_MODEL=$(grep '^default_model:' path_config.yaml 2>/dev/null | sed 's/^default_model:[[:space:]]*//' | tr -d '"' || true)
MODEL="${MODEL:-${_CFG_MODEL:-gemini-3-flash-preview}}"

echo "============================================"
echo "Loop A: Leader Refinement"
echo "  Cluster:      $CLUSTER"
echo "  Base tag:     $BASE_TAG"
echo "  Iterations:   $N_ITERATIONS"
echo "  Source tag:   $SOURCE_TAG"
echo "  Init mode:    $INIT"
echo "  Variants/leader: $M_VARIANTS"
echo "  Model:        $MODEL"
echo "  Poll interval: ${POLL_INTERVAL}s"
if [ "$INIT" = true ]; then
    echo "  N_LEADERS:    ${N_LEADERS:-5}"
    echo "  Designer prompt: ${DESIGNER_PROMPT:-data/prompts/leader_prompt_testing.txt}"
fi
echo "============================================"

# Init block: generate initial population if --init is passed
if [ "$INIT" = true ]; then
    INIT_TAG="${BASE_TAG}_iter0"
    echo ""
    echo "=== Init: Generating initial population under $INIT_TAG ==="
    echo ""

    # Step 0a: Generate initial leaders + members + code
    echo "[Init Step 1] Generating leaders, members, and code..."
    python "$DATAGEN_SCRIPT" \
        --generation_tag "$INIT_TAG" \
        --designer_prompt_path "${DESIGNER_PROMPT:-data/prompts/leader_prompt_testing.txt}" \
        --variant_prompt_path data/prompts/variant_prompt.txt \
        --code_prompt_path data/prompts/coder_prompt_testing.txt \
        --n_leaders "${N_LEADERS:-5}" \
        --m_variants "$M_VARIANTS" \
        --model "$MODEL" \
        --sync

    # Step 0b: Build & submit SLURM evaluation (no --skip-evaluated)
    echo "[Init Step 2] Building and submitting evaluation..."
    python "$EVAL_SCRIPT" \
        --run_all --generation_tag "$INIT_TAG" \
        --quick-eval --batch-mode

    # Step 0c: Poll SLURM until all jobs complete
    JOB_IDS_FILE="outputs/${INIT_TAG}/submitted_job_ids.json"
    echo "[Init Step 3] Polling SLURM jobs..."
    if [ ! -f "$JOB_IDS_FILE" ]; then
        echo "  No job IDs file found at $JOB_IDS_FILE, skipping poll"
    else
        while true; do
            RUNNING=$(python3 -c "
import json, subprocess, sys
try:
    ids = json.load(open('$JOB_IDS_FILE'))['job_ids']
    if not ids:
        print(0)
        sys.exit(0)
    result = subprocess.run(
        ['squeue', '-j', ','.join(str(j) for j in ids), '-h'],
        capture_output=True, text=True
    )
    lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
    print(len(lines))
except Exception as e:
    print(0, file=sys.stderr)
    print(0)
" 2>/dev/null)

            if [ "$RUNNING" -eq 0 ] 2>/dev/null; then
                echo "  All SLURM jobs completed"
                break
            fi
            echo "  $RUNNING jobs still running/pending, waiting ${POLL_INTERVAL}s..."
            sleep "$POLL_INTERVAL"
        done
    fi

    # Step 0d: Collect PAR2 results
    echo "[Init Step 4] Collecting results..."
    python "$EVAL_SCRIPT" \
        --collect_all_results --generation_tag "$INIT_TAG" --quick-eval

    # Step 0e: Promote best member in each team
    echo "[Init Step 5] Promoting leaders..."
    python "$EVAL_SCRIPT" \
        --promote-leaders --generation_tag "$INIT_TAG"

    SOURCE_TAG="$INIT_TAG"
    echo "=== Init complete. Leaders ready in $INIT_TAG ==="
fi

for i in $(seq 1 "$N_ITERATIONS"); do
    ITER_TAG="${BASE_TAG}_iter${i}"
    echo ""
    echo "=== Iteration $i/$N_ITERATIONS: $SOURCE_TAG -> $ITER_TAG ==="
    echo ""

    # Step 1: Generate mutants for existing leaders
    echo "[Step 1] Generating mutants..."
    python "$DATAGEN_SCRIPT" \
        --mutants-only \
        --source_tag "$SOURCE_TAG" \
        --output_tag "$ITER_TAG" \
        --variant_prompt_path data/prompts/variant_prompt.txt \
        --code_prompt_path data/prompts/coder_prompt_testing.txt \
        --m_variants "$M_VARIANTS" \
        --model "$MODEL" \
        --sync

    # Step 2: Build & submit SLURM evaluation (skip already-evaluated leaders)
    echo "[Step 2] Building and submitting evaluation..."
    python "$EVAL_SCRIPT" \
        --run_all --generation_tag "$ITER_TAG" \
        --quick-eval --batch-mode --skip-evaluated

    # Step 3: Poll SLURM until all jobs complete
    JOB_IDS_FILE="outputs/${ITER_TAG}/submitted_job_ids.json"
    echo "[Step 3] Polling SLURM jobs..."

    if [ ! -f "$JOB_IDS_FILE" ]; then
        echo "  No job IDs file found at $JOB_IDS_FILE, skipping poll"
    else
        while true; do
            RUNNING=$(python3 -c "
import json, subprocess, sys
try:
    ids = json.load(open('$JOB_IDS_FILE'))['job_ids']
    if not ids:
        print(0)
        sys.exit(0)
    result = subprocess.run(
        ['squeue', '-j', ','.join(str(j) for j in ids), '-h'],
        capture_output=True, text=True
    )
    lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
    print(len(lines))
except Exception as e:
    print(0, file=sys.stderr)
    print(0)
" 2>/dev/null)

            if [ "$RUNNING" -eq 0 ] 2>/dev/null; then
                echo "  All SLURM jobs completed"
                break
            fi
            echo "  $RUNNING jobs still running/pending, waiting ${POLL_INTERVAL}s..."
            sleep "$POLL_INTERVAL"
        done
    fi

    # Step 4: Collect PAR2 results
    echo "[Step 4] Collecting results..."
    python "$EVAL_SCRIPT" \
        --collect_all_results --generation_tag "$ITER_TAG" --quick-eval

    # Step 5: Promote best member in each team
    echo "[Step 5] Promoting leaders..."
    python "$EVAL_SCRIPT" \
        --promote-leaders --generation_tag "$ITER_TAG"

    # Next iteration reads from this iteration's promoted leaders
    SOURCE_TAG="$ITER_TAG"
    echo "=== Iteration $i complete ==="
done

echo ""
echo "============================================"
echo "Loop A complete after $N_ITERATIONS iterations"
echo "Refined leaders are in: $SOURCE_TAG"
echo "============================================"
