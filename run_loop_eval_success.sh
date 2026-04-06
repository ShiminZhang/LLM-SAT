#!/bin/bash
# run_loop_eval_success.sh — Full benchmark eval for team-best solvers in one tag.
#
# Selects only the `BEST` solver from outputs/<tag>/par2_scores.txt for each team,
# clears their old evaluation artifacts, runs full benchmark evaluation without
# rebuilding by default, collects results, then submits proof validation.
#
# Usage:
#   ./run_loop_eval_success.sh <generation_tag> [cc|nb|nersc] [--no-clean] [--build] [--no-validation]
#
# Examples:
#   ./run_loop_eval_success.sh rescale_3_iter1
#   ./run_loop_eval_success.sh rescale_3_iter1 nb
#   ./run_loop_eval_success.sh rescale_3_iter1 nb --build

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <generation_tag> [cc|nb|nersc] [--no-clean] [--build] [--no-validation]"
    exit 1
fi

GENERATION_TAG="$1"
shift

CLUSTER="${CLUSTER:-cc}"
CLEAN_RESULTS=true
SKIP_BUILD=true
VERIFY_PROOFS=true

for arg in "$@"; do
    case "$arg" in
        cc|nb|nersc)
            CLUSTER="$arg"
            ;;
        --no-clean)
            CLEAN_RESULTS=false
            ;;
        --build)
            SKIP_BUILD=false
            ;;
        --skip-build)
            SKIP_BUILD=true
            ;;
        --no-validation)
            VERIFY_PROOFS=false
            ;;
        *)
            echo "ERROR: unknown argument '$arg'"
            exit 1
            ;;
    esac
done

case "$CLUSTER" in
    cc)
        EVAL_MODULE="llmsat.pipelines.evaluation"
        PIPELINE_CLASS="EvaluationPipeline"
        ;;
    nb)
        EVAL_MODULE="llmsat.pipelines.evaluation_nb"
        PIPELINE_CLASS="EvaluationPipelineNB"
        ;;
    nersc)
        EVAL_MODULE="llmsat.pipelines.evaluation_nersc"
        PIPELINE_CLASS="EvaluationPipelineNERSC"
        ;;
    *)
        echo "ERROR: cluster must be 'cc', 'nb', or 'nersc', got '$CLUSTER'"
        exit 1
        ;;
esac

POLL_INTERVAL="${POLL_INTERVAL:-120}"
DRAT_TRIM_CMD="${DRAT_TRIM_CMD:-external/drat-trim/drat-trim}"
PROOF_CHECK_TIMEOUT="${PROOF_CHECK_TIMEOUT:-7200}"
PROOF_VERIFY_TIME="${PROOF_VERIFY_TIME:-03:00:00}"
PROOF_VERIFY_MEM="${PROOF_VERIFY_MEM:-10G}"
PROOF_VERIFY_MAX_CONCURRENT="${PROOF_VERIFY_MAX_CONCURRENT:-200}"
SUCCESS_PAIRS_PATH="outputs/${GENERATION_TAG}/best_solver_pairs.json"

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

export_best_pairs() {
    mkdir -p "outputs/${GENERATION_TAG}"
    python - "$GENERATION_TAG" "$SUCCESS_PAIRS_PATH" <<'PY'
import json
import re
import sys
from pathlib import Path

generation_tag = sys.argv[1]
output_path = Path(sys.argv[2])
par2_path = Path(f"outputs/{generation_tag}/par2_scores.txt")
solvers_root = Path(f"solvers/{generation_tag}")

if not par2_path.exists():
    raise SystemExit(f"ERROR: missing {par2_path}; cannot locate BEST solvers")

line_re = re.compile(r"^\s+\[(?P<role>[LM])\]\s+(?P<algorithm_id>\S+)\s+(?P<code_id>\S+)\s+PAR2:")


def resolve_pair(algorithm_prefix: str, code_prefix: str, reported_role: str) -> dict:
    matches = []
    role_candidates = ["leaders", "members"]
    preferred_roles = [reported_role] + [role for role in role_candidates if role != reported_role]

    for role_dir in preferred_roles:
        role_root = solvers_root / role_dir
        if not role_root.exists():
            continue
        for algorithm_dir in sorted(role_root.glob(f"algorithm_{algorithm_prefix}*")):
            if not algorithm_dir.is_dir():
                continue
            for solver_dir in sorted(algorithm_dir.glob(f"code_{code_prefix}*")):
                if not solver_dir.is_dir():
                    continue
                matches.append(
                    {
                        "algorithm_id": algorithm_dir.name.removeprefix("algorithm_"),
                        "code_id": solver_dir.name.removeprefix("code_"),
                        "role": role_dir,
                    }
                )

    unique_matches = []
    seen = set()
    for match in matches:
        key = (match["algorithm_id"], match["code_id"], match["role"])
        if key in seen:
            continue
        seen.add(key)
        unique_matches.append(match)

    if len(unique_matches) == 1:
        return unique_matches[0]

    if not unique_matches:
        raise SystemExit(
            "ERROR: could not resolve BEST solver prefixes "
            f"algorithm={algorithm_prefix} code={code_prefix} under {solvers_root}"
        )

    raise SystemExit(
        "ERROR: ambiguous BEST solver prefixes "
        f"algorithm={algorithm_prefix} code={code_prefix}: {unique_matches}"
    )


pairs = []
seen = set()

for raw_line in par2_path.read_text().splitlines():
    if "<-- BEST" not in raw_line:
        continue
    match = line_re.match(raw_line)
    if not match:
        continue

   resolved = resolve_pair(
        match.group("algorithm_id"),
        match.group("code_id"),
        "leaders" if match.group("role") == "L" else "members",
    )
    key = (resolved["algorithm_id"], resolved["code_id"], resolved["role"])
    if key in seen:
        continue
    seen.add(key)
    pairs.append(resolved)

output_path.write_text(json.dumps(pairs, indent=2))
print(f"Selected {len(pairs)} team-best code entries from {par2_path}")
PY
}

clean_successful_results() {
    python - "$GENERATION_TAG" "$SUCCESS_PAIRS_PATH" <<'PY'
import json
import shutil
import sys
from pathlib import Path

repo_root = Path.cwd()
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

generation_tag = sys.argv[1]
pairs_path = Path(sys.argv[2])
pairs = json.loads(pairs_path.read_text())

removed_dirs = 0
removed_files = 0

for pair in pairs:
    algorithm_id = pair["algorithm_id"]
    code_id = pair["code_id"]
    role = pair["role"]

    result_dir = Path(f"solvers/{generation_tag}/{role}/algorithm_{algorithm_id}/result/code_{code_id}")
    if result_dir.exists():
        shutil.rmtree(result_dir)
        removed_dirs += 1

    solver_dir = Path(f"solvers/{generation_tag}/{role}/algorithm_{algorithm_id}")
    solving_times = solver_dir / f"solving_times_{code_id}.json"
    for extra_path in (
        solving_times,
        solver_dir / f"solver_stats_{code_id}.json",
        solver_dir / f"par2_breakdown_{code_id}.json",
    ):
        if extra_path.exists():
            extra_path.unlink()
            removed_files += 1

output_dir = Path(f"outputs/{generation_tag}")
output_dir.mkdir(parents=True, exist_ok=True)
for extra_name in (
    "submitted_job_ids.json",
    "proof_verification.json",
    "proof_verification_job.json",
    "proof_verification_valid.json",
    "proof_verification_invalid.json",
    "proof_verification_timeout.json",
    "proof_verification_tasks.json",
    "run_proof_verification_array.sh",
):
    extra_path = output_dir / extra_name
    if extra_path.exists():
        extra_path.unlink()
        removed_files += 1

parts_dir = output_dir / "proof_verification_parts"
if parts_dir.exists():
    shutil.rmtree(parts_dir)
    removed_dirs += 1

print(
    f"Cleaned {removed_dirs} directories and {removed_files} files "
    f"for successful solvers under {generation_tag}"
)
PY
}

run_full_eval_for_successful_pairs() {
    python - "$GENERATION_TAG" "$EVAL_MODULE" "$PIPELINE_CLASS" "$SUCCESS_PAIRS_PATH" "$SKIP_BUILD" <<'PY'
import importlib
import json
import sys
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

repo_root = Path.cwd()
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from llmsat.llmsat import setup_logging
from llmsat.utils.aws import get_algorithm_result, get_code_result

generation_tag = sys.argv[1]
module_name = sys.argv[2]
class_name = sys.argv[3]
pairs_path = Path(sys.argv[4])
skip_build = sys.argv[5].lower() == "true"

setup_logging()

module = importlib.import_module(module_name)
pipeline_cls = getattr(module, class_name)
pipeline = pipeline_cls(generation_tag=generation_tag)

pairs = json.loads(pairs_path.read_text())
grouped = OrderedDict()
for pair in pairs:
    code_result = get_code_result(pair["code_id"])
    if code_result is None:
        print(f"WARNING: missing code result for {pair['code_id']}, skipping")
        continue
    grouped.setdefault(code_result.algorithm_id, []).append(code_result.id)

algorithms = []
for algorithm_id, code_ids in grouped.items():
    algorithm = get_algorithm_result(algorithm_id)
    if algorithm is None:
       print(f"WARNING: missing algorithm result for {algorithm_id}, skipping")
        continue
    unique_code_ids = list(OrderedDict.fromkeys(code_ids))
    algorithms.append(replace(algorithm, code_id_list=unique_code_ids))

print(
    f"Launching full evaluation for {len(algorithms)} algorithms / "
    f"{sum(len(a.code_id_list) for a in algorithms)} successful code entries"
)
pipeline.run_all_solvers_batch(
    algorithms,
    build_only=False,
    dry_run=False,
    skip_build=skip_build,
    skip_evaluated=False,
)
PY
}

collect_full_eval_results() {
    python - "$GENERATION_TAG" "$EVAL_MODULE" "$PIPELINE_CLASS" "$SUCCESS_PAIRS_PATH" <<'PY'
import importlib
import json
import sys
from pathlib import Path

repo_root = Path.cwd()
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from llmsat.llmsat import setup_logging

generation_tag = sys.argv[1]
module_name = sys.argv[2]
class_name = sys.argv[3]
pairs_path = Path(sys.argv[4])

setup_logging()

module = importlib.import_module(module_name)
pipeline_cls = getattr(module, class_name)
pipeline = pipeline_cls(generation_tag=generation_tag)

pairs = json.loads(pairs_path.read_text())
print(f"Collecting full-eval results for {len(pairs)} successful code entries")
for pair in pairs:
    pipeline.collect_results(
        pair["algorithm_id"],
        pair["code_id"],
        force_recollect=True,
    )
PY
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
    echo "[Step 3] Polling SLURM jobs..."

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

export_best_pairs

SUCCESS_COUNT=$(python - "$SUCCESS_PAIRS_PATH" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print(0)
else:
    print(len(json.loads(path.read_text())))
PY
)

if [ "$SUCCESS_COUNT" -eq 0 ]; then
    echo "No BEST solvers found under tag '$GENERATION_TAG'"
    exit 0
fi

if [ "$VERIFY_PROOFS" = true ]; then
    ensure_drat_trim_available
fi

echo "============================================"
echo "Full Eval for Successful Solvers"
echo "  Cluster:         $CLUSTER"
echo "  Generation tag:  $GENERATION_TAG"
echo "  Team-best codes: $SUCCESS_COUNT"
echo "  Clean results:   $CLEAN_RESULTS"
echo "  Skip build:      $SKIP_BUILD"
echo "  Validation:      $VERIFY_PROOFS"
echo "  Poll interval:   ${POLL_INTERVAL}s"
echo "============================================"

if [ "$CLEAN_RESULTS" = true ]; then
    echo "[Step 1] Cleaning previous evaluation artifacts..."
    clean_successful_results
fi

echo "[Step 2] Launching full benchmark evaluation..."
run_full_eval_for_successful_pairs

poll_slurm_jobs "outputs/${GENERATION_TAG}/submitted_job_ids.json"

echo "[Step 4] Collecting evaluation results..."
collect_full_eval_results

if [ "$VERIFY_PROOFS" = true ]; then
    echo "[Step 5] Submitting proof validation..."
    submit_proof_verification_job "$GENERATION_TAG"
fi

echo ""
echo "============================================"
echo "Full success-eval loop complete for $GENERATION_TAG"
echo "============================================"
