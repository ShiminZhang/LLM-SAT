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
#   # Start from GE offspring on NERSC (source_tag defaults to base_tag)
#   ./run_loop_a.sh nersc gemini_trial5_ge1 3
#
#   # Initialize from scratch: generate leaders, evaluate, then run iterations
#   ./run_loop_a.sh cc gemini_trial5 3 --init
#
# Arguments:
#   cc|nersc       - Cluster to run on (selects script variants)
#   base_tag       - Base name for iteration tags ({base_tag}_iter1, _iter2, ...)
#   n_iterations   - Number of mutant→evaluate→promote cycles to run
#   source_tag     - (Optional) Tag to load initial leaders from.
#                    Defaults to {base_tag}
#   --init         - (Optional) Generate initial leaders + members + code first
#
# Init mode env vars:
#   N_LEADERS        - Number of leaders to generate (default: 5)
#   DESIGNER_PROMPT  - Path to leader prompt (default: data/prompts/leader_prompt_testing.txt)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/loop_common.sh
source "${SCRIPT_DIR}/scripts/lib/loop_common.sh"

DATAGEN_MAX_RETRIES="${DATAGEN_MAX_RETRIES:-10}"
DATAGEN_RETRY_DELAY="${DATAGEN_RETRY_DELAY:-60}"

# Run a datagen command with retry logic.
# The Python datagen functions are resumable: they skip leaders that already
# have enough members and skip code generation for algorithms that already
# have code. So on failure we just wait and re-run the same command.
#
# Usage: run_datagen_with_retry <command...>
run_datagen_with_retry() {
    for attempt in $(seq 1 "$DATAGEN_MAX_RETRIES"); do
        echo "  [datagen attempt $attempt/$DATAGEN_MAX_RETRIES]"
        if "$@"; then
            return 0
        fi

        if [ "$attempt" -eq "$DATAGEN_MAX_RETRIES" ]; then
            echo "  [datagen] All $DATAGEN_MAX_RETRIES attempts failed"
            return 1
        fi

        echo "  [datagen] Failed, retrying in ${DATAGEN_RETRY_DELAY}s..."
        sleep "$DATAGEN_RETRY_DELAY"
    done
}

# Record how long we waited on SLURM eval jobs in the tag's timing log.
# Usage: record_eval_wait <tag> <eval_wait_seconds>
record_eval_wait() {
    local tag="$1"
    local eval_wait_s="$2"
    python3 - "outputs/${tag}/timing_log.json" "$eval_wait_s" <<'PY'
import json, os, sys
p = sys.argv[1]
wait_s = int(sys.argv[2])
if os.path.exists(p):
    d = json.load(open(p))
    rec = d[-1] if isinstance(d, list) else d
    rec['eval_wait_s'] = wait_s
    rec['total_s'] = round(rec.get('pipeline_wall_s', 0) + wait_s, 2)
    json.dump(d, open(p, 'w'), indent=2)
PY
}

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
    SOURCE_TAG="${BASE_TAG}"
else
    SOURCE_TAG="${4:-${BASE_TAG}}"
fi

DATAGEN_SCRIPT="src/llmsat/pipelines/gemini_data_generation.py"

case "$CLUSTER" in
    cc)
        EVAL_SCRIPT="src/llmsat/pipelines/evaluation.py"
        module load cuda/12.2 faiss/1.8.0 2>/dev/null || true
        # Re-activate venv after module load (module load overrides PATH/python)
        _PY_ACTIVATE=$(grep '^python_activate:' path_config.yaml 2>/dev/null | sed 's/^python_activate:[[:space:]]*//' | tr -d '"' || true)
        if [ -n "$_PY_ACTIVATE" ]; then
            # shellcheck disable=SC1090
            source "$(eval echo "$_PY_ACTIVATE")"
        fi
        ;;
    nb)
        EVAL_SCRIPT="src/llmsat/pipelines/evaluation_nb.py"
        ;;
    nersc)
        EVAL_SCRIPT="src/llmsat/pipelines/evaluation_nersc.py"
        ;;
    *)
        echo "ERROR: cluster must be 'cc', 'nb', or 'nersc', got '$CLUSTER'"
        exit 1
        ;;
esac
NERSC_FLAG=$([ "$CLUSTER" = "nersc" ] && echo "--nersc" || echo "")
POLL_INTERVAL="${POLL_INTERVAL:-120}"  # seconds between squeue checks
M_VARIANTS="${M_VARIANTS:-3}"
MODEL="${MODEL:-$(cfg_default_model)}"
PARALLEL="${PARALLEL:-0}"
QUICK_EVAL="${QUICK_EVAL:-1}"
VERIFY_PROOFS="${VERIFY_PROOFS:-1}"
PROOF_VERIFY_TIME="${PROOF_VERIFY_TIME:-01:00:00}"
PROOF_VERIFY_MEM="${PROOF_VERIFY_MEM:-10G}"
# Shared defaults + cluster-aware SLURM account/qos/constraint (loop_common.sh)
init_proof_verify_defaults

export_proof_candidates() {
    local generation_tag="$1"
    python "$EVAL_SCRIPT" \
        --generation_tag "$generation_tag" \
        --export-proof-candidates
}

if [ "$VERIFY_PROOFS" = "1" ]; then
    ensure_verifiers_available
fi

# Controlled retrieval: set TARGET_SUBCATEGORY=easy|hard|sat|unsat to steer
# mutation-exemplar selection toward that subcategory (paper §3.2).
if [ -n "${TARGET_SUBCATEGORY:-}" ]; then
    export LLMSAT_TARGET_SUBCATEGORY="$TARGET_SUBCATEGORY"
fi

echo "============================================"
echo "Loop A: Leader Refinement"
echo "  Cluster:      $CLUSTER"
echo "  Base tag:     $BASE_TAG"
echo "  Iterations:   $N_ITERATIONS"
echo "  Source tag:   $SOURCE_TAG"
echo "  Init mode:    $INIT"
echo "  Variants/leader: $M_VARIANTS"
echo "  Model:        $MODEL"
echo "  Parallel:     $PARALLEL"
echo "  Quick eval:   $QUICK_EVAL"
echo "  Verify proofs: $VERIFY_PROOFS"
echo "  drat-trim:    $DRAT_TRIM_CMD"
echo "  Proof acct:   $PROOF_VERIFY_ACCOUNT"
if [ "$CLUSTER" = "nersc" ]; then
    echo "  Proof qos:    $PROOF_VERIFY_QOS"
    echo "  Proof constr: $PROOF_VERIFY_CONSTRAINT"
fi
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

    if [ "$PARALLEL" = "1" ]; then
        # Streaming init: leaders + variants + code + build + submit all in one
        echo "[Init Step 1] Generating leaders + members + code + build + submit (parallel streaming)..."
        run_datagen_with_retry \
            python "$DATAGEN_SCRIPT" \
                --generation_tag "$INIT_TAG" \
                --designer_prompt_path "${DESIGNER_PROMPT:-data/prompts/leader_prompt_testing.txt}" \
                --variant_prompt_path data/prompts/variant_prompt.txt \
                --code_prompt_path data/prompts/coder_prompt_testing.txt \
                --n_leaders "${N_LEADERS:-5}" \
                --m_variants "$M_VARIANTS" \
                --model "$MODEL" \
                --parallel \
                $( [ "$QUICK_EVAL" = "1" ] && printf '%s' "--quick-eval" || printf '%s' "--no-quick-eval" ) \
                ${NERSC_FLAG}
        # Steps 0b (build+submit) handled by --parallel
    else
        # Sequential init
        echo "[Init Step 1] Generating leaders, members, and code..."
        run_datagen_with_retry \
            python "$DATAGEN_SCRIPT" \
                --generation_tag "$INIT_TAG" \
                --designer_prompt_path "${DESIGNER_PROMPT:-data/prompts/leader_prompt_testing.txt}" \
                --variant_prompt_path data/prompts/variant_prompt.txt \
                --code_prompt_path data/prompts/coder_prompt_testing.txt \
                --n_leaders "${N_LEADERS:-5}" \
                --m_variants "$M_VARIANTS" \
                --model "$MODEL" \
                --sync \
                ${NERSC_FLAG}

        echo "[Init Step 2] Building and submitting evaluation..."
        python "$EVAL_SCRIPT" \
            --run_all --generation_tag "$INIT_TAG" \
            $( [ "$QUICK_EVAL" = "1" ] && printf '%s' "--quick-eval " )--batch-mode
    fi

    # Poll SLURM until all jobs complete (job IDs merged from
    # submitted_job_ids.json/.jsonl by poll_slurm_job_ids in loop_common.sh)
    echo "[Init Step 3] Polling SLURM jobs..."
    EVAL_POLL_START=$SECONDS
    poll_slurm_job_ids "$INIT_TAG" "$POLL_INTERVAL"
    EVAL_WAIT=$(( SECONDS - EVAL_POLL_START ))
    echo "  Eval wait: ${EVAL_WAIT}s"
    record_eval_wait "$INIT_TAG" "$EVAL_WAIT"

    # Step 0d: Collect PAR2 results
    echo "[Init Step 4] Collecting results..."
    python "$EVAL_SCRIPT" \
        --collect_all_results --generation_tag "$INIT_TAG" \
        $( [ "$QUICK_EVAL" = "1" ] && printf '%s' "--quick-eval" )

    # Step 0e: Update mutation experience pool
    echo "[Init Step 5] Updating mutation experience pool for ${INIT_TAG}..."
    python scripts/update_experience_pool.py --generation_tag "${INIT_TAG}" \
        || echo "  [exp_pool] WARNING: pool update failed (non-fatal)"

    # Step 0f: Export promotable members and verify only those challengers.
    if [ "$VERIFY_PROOFS" = "1" ]; then
        echo "[Init Step 6] Exporting proof candidates..."
        export_proof_candidates "$INIT_TAG"

        echo "[Init Step 7] Submitting proof verification..."
        submit_proof_verification_job "$INIT_TAG"

        echo "[Init Step 8] Waiting for proof verification..."
        wait_for_proof_verification_job "$INIT_TAG"
    fi

    # Step 0g: Promote best member in each team, but only if proof verification passed.
    echo "[Init Step 9] Promoting leaders..."
    python "$EVAL_SCRIPT" \
        --promote-leaders --generation_tag "$INIT_TAG" \
        $( [ "$VERIFY_PROOFS" = "1" ] && printf '%s' "--require-valid-proof" )

    SOURCE_TAG="$INIT_TAG"
    echo "=== Init complete. Leaders ready in $INIT_TAG ==="
fi

for i in $(seq 1 "$N_ITERATIONS"); do
    ITER_TAG="${BASE_TAG}_iter${i}"
    echo ""
    echo "=== Iteration $i/$N_ITERATIONS: $SOURCE_TAG -> $ITER_TAG ==="
    echo ""

    if [ "$PARALLEL" = "1" ]; then
        # Streaming mode: generate + build + submit in one step
        echo "[Step 1] Generating mutants + building + submitting (parallel streaming)..."
        run_datagen_with_retry \
            python "$DATAGEN_SCRIPT" \
                --mutants-only \
                --source_tag "$SOURCE_TAG" \
                --output_tag "$ITER_TAG" \
                --variant_prompt_path data/prompts/variant_prompt.txt \
                --code_prompt_path data/prompts/coder_prompt_testing.txt \
                --m_variants "$M_VARIANTS" \
                --model "$MODEL" \
                --parallel \
                $( [ "$QUICK_EVAL" = "1" ] && printf '%s' "--quick-eval" || printf '%s' "--no-quick-eval" ) \
                ${NERSC_FLAG}
        # Step 2 is handled by --parallel (build + submit is part of the streaming pipeline)
    else
        # Sequential mode: generate, then build + submit separately
        echo "[Step 1] Generating mutants..."
        run_datagen_with_retry \
            python "$DATAGEN_SCRIPT" \
                --mutants-only \
                --source_tag "$SOURCE_TAG" \
                --output_tag "$ITER_TAG" \
                --variant_prompt_path data/prompts/variant_prompt.txt \
                --code_prompt_path data/prompts/coder_prompt_testing.txt \
                --m_variants "$M_VARIANTS" \
                --model "$MODEL" \
                --sync \
                ${NERSC_FLAG}

        echo "[Step 2] Building and submitting evaluation..."
        python "$EVAL_SCRIPT" \
            --run_all --generation_tag "$ITER_TAG" \
            $( [ "$QUICK_EVAL" = "1" ] && printf '%s' "--quick-eval " )--batch-mode --skip-evaluated
    fi

    # Step 3: Poll SLURM until all jobs complete (job IDs merged from
    # submitted_job_ids.json/.jsonl by poll_slurm_job_ids in loop_common.sh)
    echo "[Step 3] Polling SLURM jobs..."
    EVAL_POLL_START=$SECONDS
    poll_slurm_job_ids "$ITER_TAG" "$POLL_INTERVAL"
    EVAL_WAIT=$(( SECONDS - EVAL_POLL_START ))
    echo "  Eval wait: ${EVAL_WAIT}s"
    record_eval_wait "$ITER_TAG" "$EVAL_WAIT"

    # Step 4: Collect PAR2 results
    echo "[Step 4] Collecting results..."
    python "$EVAL_SCRIPT" \
        --collect_all_results --generation_tag "$ITER_TAG" \
        $( [ "$QUICK_EVAL" = "1" ] && printf '%s' "--quick-eval" )

    # Step 4b: Update mutation experience pool
    echo "[Step 4b] Updating mutation experience pool for ${ITER_TAG}..."
    python scripts/update_experience_pool.py --generation_tag "${ITER_TAG}" \
        || echo "  [exp_pool] WARNING: pool update failed (non-fatal)"

    # Step 5: Export promotable members and verify only those challengers.
    if [ "$VERIFY_PROOFS" = "1" ]; then
        echo "[Step 5] Exporting proof candidates..."
        export_proof_candidates "$ITER_TAG"

        echo "[Step 6] Submitting proof verification..."
        submit_proof_verification_job "$ITER_TAG"

        echo "[Step 7] Waiting for proof verification..."
        wait_for_proof_verification_job "$ITER_TAG"
    fi

    # Step 6: Promote best member in each team, but only if proof verification passed.
    echo "[Step 8] Promoting leaders..."
    python "$EVAL_SCRIPT" \
        --promote-leaders --generation_tag "$ITER_TAG" \
        $( [ "$VERIFY_PROOFS" = "1" ] && printf '%s' "--require-valid-proof" )

    # Next iteration reads from this iteration's promoted leaders
    SOURCE_TAG="$ITER_TAG"
    echo "=== Iteration $i complete ==="
done

echo ""
echo "============================================"
echo "Loop A complete after $N_ITERATIONS iterations"
echo "Refined leaders are in: $SOURCE_TAG"
echo "============================================"
