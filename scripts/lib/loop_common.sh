#!/bin/bash
# scripts/lib/loop_common.sh — shared helpers for the orchestration scripts
# (run_loop_a.sh, run_loop_reuse.sh, run_loop_eval_success.sh, run_bridge.sh,
# run_ge_collect.sh).
#
# This file is meant to be SOURCED (from the repo root), never executed:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "${SCRIPT_DIR}/scripts/lib/loop_common.sh"
#
# Sourcing has no side effects: only functions (and the double-source guard
# variable) are defined. Functions read/write the caller globals documented
# on each function.

# Guard against double-sourcing. The `return ... || exit` form also makes a
# direct execution a harmless no-op.
if [ -n "${_LLMSAT_LOOP_COMMON_SOURCED:-}" ]; then
    return 0 2>/dev/null || exit 0
fi
_LLMSAT_LOOP_COMMON_SOURCED=1

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

# cfg_default_model [fallback]
# Print the default LLM model: `default_model` from path_config.yaml if set,
# else the fallback (default: gemini-3-flash-preview). Typical use:
#   MODEL="${MODEL:-$(cfg_default_model)}"
cfg_default_model() {
    local fallback="${1:-gemini-3-flash-preview}"
    local cfg_model
    cfg_model=$(grep '^default_model:' path_config.yaml 2>/dev/null | sed 's/^default_model:[[:space:]]*//' | tr -d '"' || true)
    printf '%s\n' "${cfg_model:-$fallback}"
}

# ---------------------------------------------------------------------------
# Proof-verification tooling (drat-trim + checkmodel)
# ---------------------------------------------------------------------------

# resolve_drat_trim <cmd>
# Print the resolved drat-trim path for <cmd> (path or bare command name).
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

# ensure_drat_trim_available
# Resolve the caller's DRAT_TRIM_CMD global, building external/drat-trim if
# needed. Updates DRAT_TRIM_CMD in place.
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

# ensure_checkmodel_available
# Build the SAT model checker (validates satisfying assignments, not just
# UNSAT proofs) if missing; verify_iteration_proofs.py degrades SAT results
# to "unverified" without it. A failed build is a warning, not an error.
ensure_checkmodel_available() {
    if [ ! -x tools/checkmodel/checkmodel ]; then
        make -s -C tools/checkmodel || echo "WARNING: checkmodel build failed; SAT models will be unverified" >&2
    fi
}

# ensure_verifiers_available
# Make both proof-verification tools (drat-trim + checkmodel) available.
ensure_verifiers_available() {
    ensure_drat_trim_available
    ensure_checkmodel_available
}

# ---------------------------------------------------------------------------
# Proof-verification SLURM submission
# ---------------------------------------------------------------------------

# init_proof_verify_defaults
# Fill in the shared proof-verification defaults and the cluster-aware SLURM
# account/qos/constraint (reads the caller's CLUSTER global). Callers keep
# their own PROOF_VERIFY_TIME / PROOF_VERIFY_MEM defaults.
init_proof_verify_defaults() {
    DRAT_TRIM_CMD="${DRAT_TRIM_CMD:-external/drat-trim/drat-trim}"
    PROOF_CHECK_TIMEOUT="${PROOF_CHECK_TIMEOUT:-7200}"
    PROOF_VERIFY_MAX_CONCURRENT="${PROOF_VERIFY_MAX_CONCURRENT:-200}"
    if [ "${CLUSTER:-}" = "nersc" ]; then
        PROOF_VERIFY_ACCOUNT="${PROOF_VERIFY_ACCOUNT:-m4831}"
        PROOF_VERIFY_QOS="${PROOF_VERIFY_QOS:-regular}"
        PROOF_VERIFY_CONSTRAINT="${PROOF_VERIFY_CONSTRAINT:-cpu}"
    else
        PROOF_VERIFY_ACCOUNT="${PROOF_VERIFY_ACCOUNT:-def-vganesh}"
    fi
}

# submit_proof_verification_job <generation_tag>
# Submit the async proof-verification SLURM array for a generation tag.
# Cluster-aware: on NERSC adds --nersc plus qos/constraint. Reads globals
# CLUSTER, DRAT_TRIM_CMD, PROOF_CHECK_TIMEOUT and PROOF_VERIFY_* (call
# init_proof_verify_defaults first).
submit_proof_verification_job() {
    local generation_tag="$1"
    local slurm_args=(
        --slurm-account "$PROOF_VERIFY_ACCOUNT"
        --slurm-mem "$PROOF_VERIFY_MEM"
        --slurm-time "$PROOF_VERIFY_TIME"
        --slurm-max-concurrent "$PROOF_VERIFY_MAX_CONCURRENT"
    )

    if [ "${CLUSTER:-}" = "nersc" ]; then
        slurm_args+=(
            --nersc
            --slurm-qos "$PROOF_VERIFY_QOS"
            --slurm-constraint "$PROOF_VERIFY_CONSTRAINT"
        )
    fi

    python scripts/verify_iteration_proofs.py \
        --submit-slurm \
        --generation_tag "$generation_tag" \
        --benchmark_path data/benchmarks/satcomp2025 \
        --drat_trim "$DRAT_TRIM_CMD" \
        --check_timeout "$PROOF_CHECK_TIMEOUT" \
        "${slurm_args[@]}"
}

# ---------------------------------------------------------------------------
# SLURM polling
# ---------------------------------------------------------------------------

# _count_queued_slurm_jobs <json_path> <jsonl_path> <summary_path>
# Print the number of recorded jobs still in the SLURM queue, or a sentinel:
#   -1  squeue query failed (transient scheduler error; caller should retry)
_count_queued_slurm_jobs() {
    python3 - "$1" "$2" "$3" <<'PY' 2>/dev/null
import json, subprocess, sys

json_path, jsonl_path, summary_path = sys.argv[1], sys.argv[2], sys.argv[3]
ids = []
# Read JSONL first (parallel mode: threshold-triggered batch IDs)
try:
    with open(jsonl_path) as f:
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
    json_ids = json.load(open(json_path)).get('job_ids', [])
    ids = list(dict.fromkeys(ids + json_ids))
except (FileNotFoundError, json.JSONDecodeError):
    pass
# GE runs (genetic_evolution.py --evaluate) persist their SLURM array IDs only
# in evolution_summary.json under 'slurm_job_ids'; merge those too so bridge
# runs can poll their own jobs instead of every job the user owns.
try:
    summary_ids = json.load(open(summary_path)).get('slurm_job_ids', [])
    ids = list(dict.fromkeys(ids + summary_ids))
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
    # squeue returns non-zero once all passed job IDs have aged out of the
    # queue. Treat empty stdout as "0 running"; surface real query failures
    # (e.g. scheduler outage) as -1.
    if not result.stdout.strip():
        print(0)
    else:
        print(-1)
    sys.exit(0)
lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
print(len(lines))
PY
}

# poll_slurm_job_ids <tag> [poll_interval]
# Block until every SLURM job recorded for <tag> has left the queue.
# Job IDs are merged (de-duplicated) from:
#   outputs/<tag>/submitted_job_ids.json    (sequential submissions)
#   outputs/<tag>/submitted_job_ids.jsonl   (parallel/streaming submissions)
#   outputs/<tag>/evolution_summary.json    (GE runs: 'slurm_job_ids')
# Transient squeue failures are retried, never treated as completion.
# poll_interval defaults to $POLL_INTERVAL, then 120s.
poll_slurm_job_ids() {
    local tag="$1"
    local poll_interval="${2:-${POLL_INTERVAL:-120}}"
    local json_file="outputs/${tag}/submitted_job_ids.json"
    local jsonl_file="outputs/${tag}/submitted_job_ids.jsonl"
    local summary_file="outputs/${tag}/evolution_summary.json"
    local running

    if [ ! -f "$json_file" ] && [ ! -f "$jsonl_file" ] && [ ! -f "$summary_file" ]; then
        echo "  No job IDs file found, skipping poll"
        return 0
    fi

    while true; do
        running=$(_count_queued_slurm_jobs "$json_file" "$jsonl_file" "$summary_file")

        if [ "$running" = "-1" ]; then
            echo "  squeue query failed, retrying in ${poll_interval}s..."
            sleep "$poll_interval"
            continue
        fi

        if [ "$running" -eq 0 ] 2>/dev/null; then
            echo "  All SLURM jobs completed"
            break
        fi
        echo "  $running jobs still running/pending, waiting ${poll_interval}s..."
        sleep "$poll_interval"
    done
}

# _proof_verification_job_state <metadata_path>
# Print the number of queued proof-verification collector jobs, or a sentinel:
#   -1  squeue query failed (retry)
#   -2  metadata file missing
#   -3  collector job ID missing from metadata
_proof_verification_job_state() {
    python3 - "$1" <<'PY' 2>/dev/null
import json, subprocess, sys

path = sys.argv[1]
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
    # squeue returns non-zero when the job ID is no longer in queue
    # (completed and aged out). Treat empty stdout as "0 running".
    if not result.stdout.strip():
        print(0)
    else:
        print(-1)
    sys.exit(0)
lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
print(len(lines))
PY
}

# wait_for_proof_verification_job <generation_tag> [poll_interval]
# Block until the proof-verification collector job submitted by
# submit_proof_verification_job for <generation_tag> has finished.
# Returns 1 if the job metadata is missing or has no collector job ID.
wait_for_proof_verification_job() {
    local generation_tag="$1"
    local poll_interval="${2:-${POLL_INTERVAL:-120}}"
    local metadata_path="outputs/${generation_tag}/proof_verification_job.json"
    local running

    while true; do
        running=$(_proof_verification_job_state "$metadata_path")

        if [ "$running" = "-1" ]; then
            echo "  proof squeue query failed, retrying in ${poll_interval}s..."
            sleep "$poll_interval"
            continue
        fi

        if [ "$running" = "-2" ]; then
            echo "ERROR: proof verification metadata missing at $metadata_path" >&2
            return 1
        fi

        if [ "$running" = "-3" ]; then
            echo "ERROR: proof verification collector job ID missing in $metadata_path" >&2
            return 1
        fi

        if [ "$running" -eq 0 ] 2>/dev/null; then
            echo "  Proof verification completed"
            break
        fi

        echo "  Proof verification still pending/running, waiting ${poll_interval}s..."
        sleep "$poll_interval"
    done
}
