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
#   INIT_M_VARIANTS  - Members per leader in the initial population
#                      (default: M_VARIANTS)
#   CANDIDATE_BUDGET - Exact number of LLM-generated algorithms, including
#                      init leaders/members and later mutants (0 disables)
#   DESIGNER_PROMPT  - Path to leader prompt (default: data/prompts/leader_prompt_testing.txt)
#   VARIANT_PROMPT   - Path to mutation prompt (default: data/prompts/variant_prompt.txt)
#   CODER_PROMPT     - Path to target-specific coder prompt
#                      (default: data/prompts/coder_prompt_testing.txt)
#   RESUME_EVAL      - auto: reuse solver binaries when an evaluation checkpoint exists
#                      1: always reuse binaries; 0: always rebuild (default: auto)

set -euo pipefail

# Make all relative paths and Python imports independent of the caller's cwd
# and shell environment.
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

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

evaluation_checkpoint_complete() {
    local generation_tag="$1"
    python3 - "$generation_tag" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path("outputs") / sys.argv[1] / "evaluation_submission_state.json"
try:
    state = json.loads(path.read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
batches = state.get("batches", [])
raise SystemExit(0 if batches and all(b.get("status") == "completed" for b in batches) else 1)
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
        source scripts/activate_rorqual.sh
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
INIT_M_VARIANTS="${INIT_M_VARIANTS:-$M_VARIANTS}"
N_LEADERS_VALUE="${N_LEADERS:-5}"
CANDIDATE_BUDGET="${CANDIDATE_BUDGET:-0}"
DESIGNER_PROMPT_VALUE="${DESIGNER_PROMPT:-data/prompts/leader_prompt_testing.txt}"
VARIANT_PROMPT_VALUE="${VARIANT_PROMPT:-data/prompts/variant_prompt.txt}"
CODER_PROMPT_VALUE="${CODER_PROMPT:-data/prompts/coder_prompt_testing.txt}"
_CFG_MODEL=$(grep '^default_model:' path_config.yaml 2>/dev/null | sed 's/^default_model:[[:space:]]*//' | tr -d '"' || true)
MODEL="${MODEL:-${_CFG_MODEL:-gpt-5.6-luna}}"
PARALLEL="${PARALLEL:-0}"
QUICK_EVAL="${QUICK_EVAL:-0}"
VERIFY_PROOFS="${VERIFY_PROOFS:-0}"
RESUME_EVAL="${RESUME_EVAL:-auto}"
DRAT_TRIM_CMD="${DRAT_TRIM_CMD:-external/drat-trim/drat-trim}"
PROOF_CHECK_TIMEOUT="${PROOF_CHECK_TIMEOUT:-7200}"
PROOF_VERIFY_TIME="${PROOF_VERIFY_TIME:-01:00:00}"
PROOF_VERIFY_MEM="${PROOF_VERIFY_MEM:-10G}"
PROOF_VERIFY_MAX_CONCURRENT="${PROOF_VERIFY_MAX_CONCURRENT:-200}"
if [ "$CLUSTER" = "nersc" ]; then
    PROOF_VERIFY_ACCOUNT="${PROOF_VERIFY_ACCOUNT:-m4831}"
    PROOF_VERIFY_QOS="${PROOF_VERIFY_QOS:-regular}"
    PROOF_VERIFY_CONSTRAINT="${PROOF_VERIFY_CONSTRAINT:-cpu}"
else
    PROOF_VERIFY_ACCOUNT="${PROOF_VERIFY_ACCOUNT:-def-vganesh}"
fi

for value_name in N_ITERATIONS M_VARIANTS INIT_M_VARIANTS N_LEADERS_VALUE CANDIDATE_BUDGET; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: $value_name must be a non-negative integer, got '$value'" >&2
        exit 2
    fi
done
if [ "$M_VARIANTS" -eq 0 ] || [ "$INIT_M_VARIANTS" -eq 0 ] || [ "$N_LEADERS_VALUE" -eq 0 ]; then
    echo "ERROR: M_VARIANTS, INIT_M_VARIANTS, and N_LEADERS must be positive" >&2
    exit 2
fi

REQUIRED_ITERATIONS="$N_ITERATIONS"
if [ "$CANDIDATE_BUDGET" -gt 0 ]; then
    if [ "$INIT" != true ]; then
        echo "ERROR: CANDIDATE_BUDGET currently requires --init so the initial population is countable" >&2
        exit 2
    fi
    INIT_CANDIDATES=$((N_LEADERS_VALUE * (INIT_M_VARIANTS + 1)))
    if [ "$CANDIDATE_BUDGET" -lt "$INIT_CANDIDATES" ]; then
        echo "ERROR: CANDIDATE_BUDGET=$CANDIDATE_BUDGET is smaller than the $INIT_CANDIDATES-candidate initial population" >&2
        exit 2
    fi
    REMAINING_CANDIDATES=$((CANDIDATE_BUDGET - INIT_CANDIDATES))
    if [ $((REMAINING_CANDIDATES % N_LEADERS_VALUE)) -ne 0 ]; then
        echo "ERROR: post-init budget must be divisible by N_LEADERS=$N_LEADERS_VALUE to preserve equal team sizes" >&2
        exit 2
    fi
    REMAINING_GROUPS=$((REMAINING_CANDIDATES / N_LEADERS_VALUE))
    REQUIRED_ITERATIONS=$(((REMAINING_GROUPS + M_VARIANTS - 1) / M_VARIANTS))
    if [ "$N_ITERATIONS" -ne "$REQUIRED_ITERATIONS" ]; then
        echo "ERROR: CANDIDATE_BUDGET=$CANDIDATE_BUDGET requires n_iterations=$REQUIRED_ITERATIONS with current N_LEADERS/M_VARIANTS; got $N_ITERATIONS" >&2
        exit 2
    fi
fi

resolve_drat_trim() {
    local cmd="$1"
    if [[ "$cmd" == */* ]]; then
        if [ -x "$cmd" ]; then
            printf '%s\n' "$cmd"
            return 0
        fi
    else
        if command -v "$cmd" >/dev/null 2>&1; then
            command -v "$cmd"
            return 0
        fi
    fi
    return 1
}

ensure_drat_trim_available() {
    local resolved
    if resolved="$(resolve_drat_trim "$DRAT_TRIM_CMD")"; then
        DRAT_TRIM_CMD="$resolved"
        return 0
    fi

    echo "drat-trim not available at '$DRAT_TRIM_CMD'; attempting local build..."
    if ! make -C external/drat-trim drat-trim; then
        echo "ERROR: failed to build drat-trim in external/drat-trim" >&2
        return 1
    fi

    if resolved="$(resolve_drat_trim "$DRAT_TRIM_CMD")"; then
        DRAT_TRIM_CMD="$resolved"
        return 0
    fi

    if resolved="$(resolve_drat_trim "external/drat-trim/drat-trim")"; then
        DRAT_TRIM_CMD="$resolved"
        return 0
    fi

    echo "ERROR: drat-trim is still unavailable after build attempt" >&2
    return 1
}


submit_proof_verification_job() {
    local generation_tag="$1"
    local slurm_args=(
        --slurm-account "$PROOF_VERIFY_ACCOUNT"
        --slurm-mem "$PROOF_VERIFY_MEM"
        --slurm-time "$PROOF_VERIFY_TIME"
        --slurm-max-concurrent "$PROOF_VERIFY_MAX_CONCURRENT"
    )

    if [ "$CLUSTER" = "nersc" ]; then
        slurm_args+=(
            --nersc
            --slurm-qos "$PROOF_VERIFY_QOS"
            --slurm-constraint "$PROOF_VERIFY_CONSTRAINT"
        )
    fi

    python scripts/verify_iteration_proofs.py \
        --submit-slurm \
        --generation_tag "$generation_tag" \
        --benchmark_path data/benchmarks/formula-families/cryptography-ascon \
        --drat_trim "$DRAT_TRIM_CMD" \
        --check_timeout "$PROOF_CHECK_TIMEOUT" \
        "${slurm_args[@]}"
}

export_proof_candidates() {
    local generation_tag="$1"
    python "$EVAL_SCRIPT" \
        --generation_tag "$generation_tag" \
        --export-proof-candidates
}

wait_for_proof_verification_job() {
    local generation_tag="$1"
    local metadata_path="outputs/${generation_tag}/proof_verification_job.json"

    _parse_proof_job_state() {
        python3 -c "
import json, subprocess, sys
path = '$metadata_path'
try:
    with open(path) as f:
        meta = json.load(f)
except FileNotFoundError:
    print(-2)
    sys.exit(0)
status = meta.get('status')
if status in ('no_tasks', 'all_tasks_already_collected'):
    print(0)
    sys.exit(0)
job_id = meta.get('collector_job_id')
if not job_id:
    print(-3)
    sys.exit(0)
result = subprocess.run(
    ['squeue', '-j', str(job_id), '-h'],
    capture_output=True, text=True
)
if result.returncode != 0:
    print(-1)
    sys.exit(0)
lines = [l for l in result.stdout.strip().split('\\n') if l.strip()]
print(len(lines))
" 2>/dev/null
    }

    while true; do
        RUNNING=$(_parse_proof_job_state)

        if [ "$RUNNING" = "-1" ]; then
            echo "  proof squeue query failed, retrying in ${POLL_INTERVAL}s..."
            sleep "$POLL_INTERVAL"
            continue
        fi

        if [ "$RUNNING" = "-2" ]; then
            echo "ERROR: proof verification metadata missing at $metadata_path" >&2
            return 1
        fi

        if [ "$RUNNING" = "-3" ]; then
            echo "ERROR: proof verification collector job ID missing in $metadata_path" >&2
            return 1
        fi

        if [ "$RUNNING" -eq 0 ] 2>/dev/null; then
            echo "  Proof verification completed"
            break
        fi

        echo "  Proof verification still pending/running, waiting ${POLL_INTERVAL}s..."
        sleep "$POLL_INTERVAL"
    done
}

if [ "$VERIFY_PROOFS" = "1" ]; then
    ensure_drat_trim_available
fi

echo "============================================"
echo "Loop A: Leader Refinement"
echo "  Cluster:      $CLUSTER"
echo "  Base tag:     $BASE_TAG"
echo "  Iterations:   $N_ITERATIONS"
echo "  Source tag:   $SOURCE_TAG"
echo "  Init mode:    $INIT"
echo "  Variants/leader: $M_VARIANTS"
echo "  Init variants/leader: $INIT_M_VARIANTS"
echo "  Candidate budget: $CANDIDATE_BUDGET"
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
echo "  Resume eval:  ${RESUME_EVAL}"
if [ "$INIT" = true ]; then
    echo "  N_LEADERS:    $N_LEADERS_VALUE"
    echo "  Designer prompt: $DESIGNER_PROMPT_VALUE"
    echo "  Variant prompt:  $VARIANT_PROMPT_VALUE"
    echo "  Coder prompt:    $CODER_PROMPT_VALUE"
fi
echo "============================================"

# Init block: generate initial population if --init is passed
if [ "$INIT" = true ]; then
    INIT_TAG="${BASE_TAG}_iter0"
    echo ""
    echo "=== Init: Generating initial population under $INIT_TAG ==="
    echo ""

    INIT_EVAL_RESUME=false
    if [ "$RESUME_EVAL" = "1" ] || {
        [ "$RESUME_EVAL" = "auto" ] &&
        [ -f "outputs/${INIT_TAG}/evaluation_submission_state.json" ]
    }; then
        INIT_EVAL_RESUME=true
    fi

    if [ "$INIT_EVAL_RESUME" = true ]; then
        echo "[Init Step 1] Evaluation checkpoint found; skipping data generation"
        if evaluation_checkpoint_complete "$INIT_TAG"; then
            echo "[Init Step 2] Evaluation checkpoint is complete; skipping evaluation"
        else
            echo "[Init Step 2] Resuming evaluation with existing solver binaries..."
            python "$EVAL_SCRIPT" \
                --run_all --generation_tag "$INIT_TAG" \
                $( [ "$QUICK_EVAL" = "1" ] && printf '%s' "--quick-eval " )--batch-mode \
                --skip-build
        fi
    elif [ "$PARALLEL" = "1" ]; then
        # Streaming init: leaders + variants + code + build + submit all in one
        echo "[Init Step 1] Generating leaders + members + code + build + submit (parallel streaming)..."
        run_datagen_with_retry \
            python "$DATAGEN_SCRIPT" \
                --generation_tag "$INIT_TAG" \
                --designer_prompt_path "$DESIGNER_PROMPT_VALUE" \
                --variant_prompt_path "$VARIANT_PROMPT_VALUE" \
                --code_prompt_path "$CODER_PROMPT_VALUE" \
                --n_leaders "$N_LEADERS_VALUE" \
                --m_variants "$INIT_M_VARIANTS" \
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
                --designer_prompt_path "$DESIGNER_PROMPT_VALUE" \
                --variant_prompt_path "$VARIANT_PROMPT_VALUE" \
                --code_prompt_path "$CODER_PROMPT_VALUE" \
                --n_leaders "$N_LEADERS_VALUE" \
                --m_variants "$INIT_M_VARIANTS" \
                --model "$MODEL" \
                --sync \
                ${NERSC_FLAG}

        echo "[Init Step 2] Building and submitting evaluation..."
        python "$EVAL_SCRIPT" \
            --run_all --generation_tag "$INIT_TAG" \
            $( [ "$QUICK_EVAL" = "1" ] && printf '%s' "--quick-eval " )--batch-mode
    fi

    # Poll SLURM until all jobs complete
    JOB_IDS_FILE="outputs/${INIT_TAG}/submitted_job_ids.json"
    JOB_IDS_JSONL="outputs/${INIT_TAG}/submitted_job_ids.jsonl"
    echo "[Init Step 3] Polling SLURM jobs..."

    _parse_init_job_ids() {
        python3 -c "
import json, subprocess, sys
ids = []
try:
    with open('$JOB_IDS_JSONL') as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                ids.extend(record.get('job_ids', []))
except FileNotFoundError:
    pass
# Always merge JSON IDs too: in NERSC parallel mode the final accumulator flush
# writes IDs only to the consolidated JSON (not JSONL), so JSONL can be non-empty
# but incomplete.  Using JSONL as a fallback-only source would miss those IDs.
try:
    json_ids = json.load(open('$JOB_IDS_FILE')).get('job_ids', [])
    ids = list(dict.fromkeys(ids + json_ids))
except (FileNotFoundError, json.JSONDecodeError):
    pass
if not ids:
    print(0)
    sys.exit(0)
result = subprocess.run(
    ['squeue', '-j', ','.join(str(j) for j in ids), '-h'],
    capture_output=True, text=True
)
if result.returncode != 0:
    # Signal transient query failure to caller; do not treat as completion.
    print(-1)
    sys.exit(0)
lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
print(len(lines))
" 2>/dev/null
    }

    EVAL_POLL_START=$SECONDS
    if [ ! -f "$JOB_IDS_FILE" ] && [ ! -f "$JOB_IDS_JSONL" ]; then
        echo "  No job IDs file found, skipping poll"
    else
        while true; do
            RUNNING=$(_parse_init_job_ids)

            if [ "$RUNNING" = "-1" ]; then
                echo "  squeue query failed, retrying in ${POLL_INTERVAL}s..."
                sleep "$POLL_INTERVAL"
                continue
            fi

            if [ "$RUNNING" -eq 0 ] 2>/dev/null; then
                echo "  All SLURM jobs completed"
                break
            fi
            echo "  $RUNNING jobs still running/pending, waiting ${POLL_INTERVAL}s..."
            sleep "$POLL_INTERVAL"
        done
    fi
    EVAL_WAIT=$(( SECONDS - EVAL_POLL_START ))
    echo "  Eval wait: ${EVAL_WAIT}s"
    python3 -c "
import json, os
p = 'outputs/${INIT_TAG}/timing_log.json'
if os.path.exists(p):
    d = json.load(open(p))
    rec = d[-1] if isinstance(d, list) else d
    rec['eval_wait_s'] = $EVAL_WAIT
    rec['total_s'] = round(rec.get('pipeline_wall_s', 0) + $EVAL_WAIT, 2)
    json.dump(d, open(p, 'w'), indent=2)
"

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

CANDIDATES_GENERATED=0
if [ "$CANDIDATE_BUDGET" -gt 0 ]; then
    CANDIDATES_GENERATED=$((N_LEADERS_VALUE * (INIT_M_VARIANTS + 1)))
    echo "Budget progress after init: ${CANDIDATES_GENERATED}/${CANDIDATE_BUDGET} candidates"
fi

for i in $(seq 1 "$N_ITERATIONS"); do
    ITER_M_VARIANTS="$M_VARIANTS"
    if [ "$CANDIDATE_BUDGET" -gt 0 ]; then
        REMAINING_GROUPS=$(((CANDIDATE_BUDGET - CANDIDATES_GENERATED) / N_LEADERS_VALUE))
        if [ "$REMAINING_GROUPS" -lt "$ITER_M_VARIANTS" ]; then
            ITER_M_VARIANTS="$REMAINING_GROUPS"
        fi
    fi
    ITER_TAG="${BASE_TAG}_iter${i}"
    echo ""
    echo "=== Iteration $i/$N_ITERATIONS: $SOURCE_TAG -> $ITER_TAG ==="
    echo ""

    ITER_EVAL_RESUME=false
    if [ "$RESUME_EVAL" = "1" ] || {
        [ "$RESUME_EVAL" = "auto" ] &&
        [ -f "outputs/${ITER_TAG}/evaluation_submission_state.json" ]
    }; then
        ITER_EVAL_RESUME=true
    fi

    if [ "$ITER_EVAL_RESUME" = true ]; then
        echo "[Step 1] Evaluation checkpoint found; skipping mutant generation"
        if evaluation_checkpoint_complete "$ITER_TAG"; then
            echo "[Step 2] Evaluation checkpoint is complete; skipping evaluation"
        else
            echo "[Step 2] Resuming evaluation with existing solver binaries..."
            python "$EVAL_SCRIPT" \
                --run_all --generation_tag "$ITER_TAG" \
                $( [ "$QUICK_EVAL" = "1" ] && printf '%s' "--quick-eval " )--batch-mode --skip-evaluated \
                --skip-build
        fi
    elif [ "$PARALLEL" = "1" ]; then
        # Streaming mode: generate + build + submit in one step
        echo "[Step 1] Generating mutants + building + submitting (parallel streaming)..."
        run_datagen_with_retry \
            python "$DATAGEN_SCRIPT" \
                --mutants-only \
                --source_tag "$SOURCE_TAG" \
                --output_tag "$ITER_TAG" \
                --variant_prompt_path "$VARIANT_PROMPT_VALUE" \
                --code_prompt_path "$CODER_PROMPT_VALUE" \
                --m_variants "$ITER_M_VARIANTS" \
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
                --variant_prompt_path "$VARIANT_PROMPT_VALUE" \
                --code_prompt_path "$CODER_PROMPT_VALUE" \
                --m_variants "$ITER_M_VARIANTS" \
                --model "$MODEL" \
                --sync \
                ${NERSC_FLAG}

        echo "[Step 2] Building and submitting evaluation..."
        python "$EVAL_SCRIPT" \
            --run_all --generation_tag "$ITER_TAG" \
            $( [ "$QUICK_EVAL" = "1" ] && printf '%s' "--quick-eval " )--batch-mode --skip-evaluated
    fi

    # Step 3: Poll SLURM until all jobs complete
    JOB_IDS_FILE="outputs/${ITER_TAG}/submitted_job_ids.json"
    JOB_IDS_JSONL="outputs/${ITER_TAG}/submitted_job_ids.jsonl"
    echo "[Step 3] Polling SLURM jobs..."

    # Parse job IDs from either JSONL (parallel) or JSON (sequential) format
    _parse_job_ids() {
        python3 -c "
import json, subprocess, sys
ids = []
# Read JSONL first (parallel mode: threshold-triggered batch IDs)
try:
    with open('$JOB_IDS_JSONL') as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                ids.extend(record.get('job_ids', []))
except FileNotFoundError:
    pass
# Always merge JSON IDs too: in NERSC parallel mode the final accumulator flush
# writes IDs only to the consolidated JSON (not JSONL), so JSONL can be non-empty
# but incomplete.  Using JSONL as a fallback-only source would miss those IDs.
try:
    json_ids = json.load(open('$JOB_IDS_FILE')).get('job_ids', [])
    ids = list(dict.fromkeys(ids + json_ids))
except (FileNotFoundError, json.JSONDecodeError):
    pass
if not ids:
    print(0)
    sys.exit(0)
result = subprocess.run(
    ['squeue', '-j', ','.join(str(j) for j in ids), '-h'],
    capture_output=True, text=True
)
if result.returncode != 0:
    # Signal transient query failure to caller; do not treat as completion.
    print(-1)
    sys.exit(0)
lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
print(len(lines))
" 2>/dev/null
    }

    EVAL_POLL_START=$SECONDS
    if [ ! -f "$JOB_IDS_FILE" ] && [ ! -f "$JOB_IDS_JSONL" ]; then
        echo "  No job IDs file found, skipping poll"
    else
        while true; do
            RUNNING=$(_parse_job_ids)

            if [ "$RUNNING" = "-1" ]; then
                echo "  squeue query failed, retrying in ${POLL_INTERVAL}s..."
                sleep "$POLL_INTERVAL"
                continue
            fi

            if [ "$RUNNING" -eq 0 ] 2>/dev/null; then
                echo "  All SLURM jobs completed"
                break
            fi
            echo "  $RUNNING jobs still running/pending, waiting ${POLL_INTERVAL}s..."
            sleep "$POLL_INTERVAL"
        done
    fi
    EVAL_WAIT=$(( SECONDS - EVAL_POLL_START ))
    echo "  Eval wait: ${EVAL_WAIT}s"
    python3 -c "
import json, os
p = 'outputs/${ITER_TAG}/timing_log.json'
if os.path.exists(p):
    d = json.load(open(p))
    rec = d[-1] if isinstance(d, list) else d
    rec['eval_wait_s'] = $EVAL_WAIT
    rec['total_s'] = round(rec.get('pipeline_wall_s', 0) + $EVAL_WAIT, 2)
    json.dump(d, open(p, 'w'), indent=2)
"

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
    if [ "$CANDIDATE_BUDGET" -gt 0 ]; then
        CANDIDATES_GENERATED=$((CANDIDATES_GENERATED + N_LEADERS_VALUE * ITER_M_VARIANTS))
        echo "Budget progress: ${CANDIDATES_GENERATED}/${CANDIDATE_BUDGET} candidates"
    fi
    echo "=== Iteration $i complete ==="
done

if [ "$CANDIDATE_BUDGET" -gt 0 ] && [ "$CANDIDATES_GENERATED" -ne "$CANDIDATE_BUDGET" ]; then
    echo "ERROR: generated $CANDIDATES_GENERATED candidates, expected $CANDIDATE_BUDGET" >&2
    exit 2
fi

echo ""
echo "============================================"
echo "Loop A complete after $N_ITERATIONS iterations"
echo "Refined leaders are in: $SOURCE_TAG"
echo "============================================"
