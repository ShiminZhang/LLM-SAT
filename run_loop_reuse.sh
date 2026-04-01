#!/bin/bash
# run_loop_reuse.sh — Re-evaluate an existing generation tag in place.
#
# Reuses already generated leaders/mutants under an existing generation tag,
# optionally clears prior evaluation artifacts, then rebuilds/re-evaluates,
# collects results, and optionally promotes leaders.
#
# Usage:
#   ./run_loop_reuse.sh <cc|nb|nersc> <generation_tag> [--no-clean] [--no-promote] [--skip-build]
#
# Examples:
#   # Re-run evaluation for an existing generation and promote using fresh PAR2
#   ./run_loop_reuse.sh nb rescale_7_iter0
#
#   # Keep existing logs/results and only submit missing tasks
#   ./run_loop_reuse.sh nb rescale_7_iter0 --no-clean
#
# Env vars:
#   POLL_INTERVAL    Seconds between squeue checks (default: 120)
#   QUICK_EVAL       1 to pass --quick-eval, 0 for full eval (default: 1)
#   VERIFY_PROOFS    1 to submit async proof verification, 0 to skip (default: 1)
#   DRAT_TRIM_CMD    drat-trim path or command (default: external/drat-trim/drat-trim)
#   PROOF_CHECK_TIMEOUT        Default: 7200
#   PROOF_VERIFY_TIME          Default: 02:00:00
#   PROOF_VERIFY_MEM           Default: 8G
#   PROOF_VERIFY_MAX_CONCURRENT Default: 200

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <cc|nb|nersc> <generation_tag> [--no-clean] [--no-promote] [--skip-build]"
    exit 1
fi

CLUSTER="$1"
GENERATION_TAG="$2"
shift 2

CLEAN_RESULTS=true
PROMOTE_LEADERS=true
SKIP_BUILD=false

for arg in "$@"; do
    case "$arg" in
        --no-clean)
            CLEAN_RESULTS=false
            ;;
        --no-promote)
            PROMOTE_LEADERS=false
            ;;
        --skip-build)
            SKIP_BUILD=true
            ;;
        *)
            echo "ERROR: unknown argument '$arg'"
            exit 1
            ;;
    esac
done

case "$CLUSTER" in
    cc)
        EVAL_SCRIPT="src/llmsat/pipelines/evaluation.py"
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

POLL_INTERVAL="${POLL_INTERVAL:-120}"
QUICK_EVAL="${QUICK_EVAL:-1}"
VERIFY_PROOFS="${VERIFY_PROOFS:-1}"
DRAT_TRIM_CMD="${DRAT_TRIM_CMD:-external/drat-trim/drat-trim}"
PROOF_CHECK_TIMEOUT="${PROOF_CHECK_TIMEOUT:-7200}"
PROOF_VERIFY_TIME="${PROOF_VERIFY_TIME:-02:00:00}"
PROOF_VERIFY_MEM="${PROOF_VERIFY_MEM:-8G}"
PROOF_VERIFY_MAX_CONCURRENT="${PROOF_VERIFY_MAX_CONCURRENT:-200}"

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
    python scripts/verify_iteration_proofs.py \
        --submit-slurm \
        --generation_tag "$generation_tag" \
        --benchmark_path data/benchmarks/satcomp2025 \
        --drat_trim "$DRAT_TRIM_CMD" \
        --check_timeout "$PROOF_CHECK_TIMEOUT" \
        --slurm-account def-vganesh \
        --slurm-mem "$PROOF_VERIFY_MEM" \
        --slurm-time "$PROOF_VERIFY_TIME" \
        --slurm-max-concurrent "$PROOF_VERIFY_MAX_CONCURRENT"
}

poll_slurm_jobs() {
    local job_ids_file="$1"
    echo "[Step 2] Polling SLURM jobs..."

    if [ ! -f "$job_ids_file" ]; then
        echo "  No job IDs file found at $job_ids_file, skipping poll"
        return 0
    fi

    while true; do
        RUNNING=$(python3 -c "
import json, subprocess, sys
try:
    ids = json.load(open('$job_ids_file'))['job_ids']
    if not ids:
        print(0)
        sys.exit(0)
    result = subprocess.run(
        ['squeue', '-j', ','.join(str(j) for j in ids), '-h'],
        capture_output=True, text=True
    )
    lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
    print(len(lines))
except Exception:
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
}

clean_generation_results() {
    local generation_tag="$1"
    local generation_dir="solvers/$generation_tag"
    local output_dir="outputs/$generation_tag"

    if [ ! -d "$generation_dir" ]; then
        echo "ERROR: generation directory not found: $generation_dir" >&2
        exit 1
    fi

    echo "[Step 0] Cleaning previous evaluation artifacts under $generation_tag..."
    find "$generation_dir" -type d -name result -prune -exec rm -rf {} +
    find "$generation_dir" -type f \( -name 'solving_times_*.json' -o -name 'solver_stats_*.json' -o -name 'par2_breakdown_*.json' \) -delete
    rm -f "$generation_dir/submitted_job_ids.json"
    rm -f "$output_dir/submitted_job_ids.json"
    rm -f "$output_dir/par2_scores.txt"
}

if [ "$VERIFY_PROOFS" = "1" ]; then
    ensure_drat_trim_available
fi

echo "============================================"
echo "Loop Reuse: Re-evaluate Existing Generation"
echo "  Cluster:        $CLUSTER"
echo "  Generation tag: $GENERATION_TAG"
echo "  Clean results:  $CLEAN_RESULTS"
echo "  Promote leaders: $PROMOTE_LEADERS"
echo "  Skip build:     $SKIP_BUILD"
echo "  Quick eval:     $QUICK_EVAL"
echo "  Verify proofs:  $VERIFY_PROOFS"
echo "  drat-trim:      $DRAT_TRIM_CMD"
echo "  Poll interval:  ${POLL_INTERVAL}s"
echo "============================================"

if [ "$CLEAN_RESULTS" = true ]; then
    clean_generation_results "$GENERATION_TAG"
fi

RUN_ARGS=(python "$EVAL_SCRIPT" --run_all --generation_tag "$GENERATION_TAG" --batch-mode)
if [ "$QUICK_EVAL" = "1" ]; then
    RUN_ARGS+=(--quick-eval)
fi
if [ "$SKIP_BUILD" = true ]; then
    RUN_ARGS+=(--skip-build)
fi

echo "[Step 1] Building and submitting evaluation..."
"${RUN_ARGS[@]}"

poll_slurm_jobs "outputs/${GENERATION_TAG}/submitted_job_ids.json"

COLLECT_ARGS=(python "$EVAL_SCRIPT" --collect_all_results --generation_tag "$GENERATION_TAG")
if [ "$QUICK_EVAL" = "1" ]; then
    COLLECT_ARGS+=(--quick-eval)
fi

echo "[Step 3] Collecting results..."
"${COLLECT_ARGS[@]}"

if [ "$VERIFY_PROOFS" = "1" ]; then
    echo "[Step 4] Submitting async UNSAT proof verification..."
    submit_proof_verification_job "$GENERATION_TAG"
fi

if [ "$PROMOTE_LEADERS" = true ]; then
    echo "[Step 5] Promoting leaders..."
    python "$EVAL_SCRIPT" --promote-leaders --generation_tag "$GENERATION_TAG"
fi

echo ""
echo "============================================"
echo "Reuse loop complete for $GENERATION_TAG"
echo "============================================"
