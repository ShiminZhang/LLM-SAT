#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${SAT_REPO_ROOT:-/scratch/s568zhan/LLM-SAT}"
OPENEVOLVE_ROOT="${OPENEVOLVE_ROOT:-/scratch/s568zhan/OpenEvolve}"
source "$REPO_ROOT/experiment/common/protocol.sh"
export_comparison_protocol
TARGET="${1:-}"
BENCHMARK_FAMILY="${2:-cryptography-ascon}"
BUDGET="${3:-100}"
BENCHMARK_DIR="${OE_BENCHMARK_DIR:-$REPO_ROOT/data/benchmarks/formula-families/$BENCHMARK_FAMILY}"
RUN_ID="${OE_RUN_ID:-oe_${TARGET}_${BENCHMARK_FAMILY}_${BUDGET}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="$REPO_ROOT/experiment/comparison_runs/openevolve/$RUN_ID"

case "$TARGET" in
  decide|kissat_decide_phase)
    TARGET_SLUG=decide
    TARGET_FUNCTION=kissat_decide_phase
    TARGET_SOURCE=src/decide.c
    INITIAL_PROGRAM="$REPO_ROOT/experiment/openevolve/initial_program.c"
    if [ "$BENCHMARK_FAMILY" = "cryptography-ascon" ]; then
      CONFIG="$REPO_ROOT/experiment/openevolve/config.yaml"
    else
      CONFIG="$REPO_ROOT/experiment/comparison_configs/openevolve_edge_decide.yaml"
    fi
    ;;
  restart|restarting|kissat_restarting)
    TARGET_SLUG=restarting
    TARGET_FUNCTION=kissat_restarting
    TARGET_SOURCE=src/restart.c
    INITIAL_PROGRAM="$REPO_ROOT/experiment/openevolve_restart_ascon/initial_program.c"
    if [ "$BENCHMARK_FAMILY" = "cryptography-ascon" ]; then
      CONFIG="$REPO_ROOT/experiment/openevolve_restart_ascon/config.yaml"
    else
      CONFIG="$REPO_ROOT/experiment/comparison_configs/openevolve_edge_restart.yaml"
    fi
    ;;
  *)
    echo "Usage: $0 <decide|restarting> <benchmark-family> <offspring-budget>" >&2
    exit 2
    ;;
esac

if ! [[ "$BUDGET" =~ ^[1-9][0-9]*$ ]]; then
  echo "Offspring budget must be a positive integer" >&2
  exit 2
fi
if [ -e "$OUTPUT_DIR" ]; then
  echo "Refusing to reuse existing output directory: $OUTPUT_DIR" >&2
  exit 2
fi
if [ ! -d "$BENCHMARK_DIR" ]; then
  echo "Benchmark directory not found: $BENCHMARK_DIR" >&2
  exit 2
fi

curl -sS --connect-timeout 10 --max-time 20 -o /dev/null https://api.openai.com/v1/models
mkdir -p "$OUTPUT_DIR"
printf '%s\n' "$$" > "$OUTPUT_DIR/orchestrator.pid"
RESOLVED_CONFIG="$OUTPUT_DIR/config.resolved.yaml"
python3 "$REPO_ROOT/experiment/openevolve/render_config.py" \
  "$CONFIG" "$RESOLVED_CONFIG" \
  --model "$COMPARISON_MODEL" \
  --reasoning-effort "$COMPARISON_REASONING_EFFORT" \
  --parallel-evaluations "$COMPARISON_MAX_CANDIDATE_JOBS"

cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/activate_rorqual.sh"
set -a
source "$REPO_ROOT/.env"
set +a

export PYTHONPATH="$OPENEVOLVE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OE_REPO_ROOT="$REPO_ROOT"
export OE_TARGET_FUNCTION="$TARGET_FUNCTION"
export OE_TARGET_SOURCE="$TARGET_SOURCE"
export OE_BENCHMARK_DIR="$BENCHMARK_DIR"
export OE_WORK_DIR="${OE_WORK_DIR:-$REPO_ROOT/experiment/comparison_work/$BENCHMARK_FAMILY/$TARGET_SLUG}"

echo "host=$(hostname) pid=$$ run_id=$RUN_ID target=$TARGET_FUNCTION benchmark=$BENCHMARK_FAMILY budget=$BUDGET"
exec python "$OPENEVOLVE_ROOT/openevolve-run.py" \
  "$INITIAL_PROGRAM" \
  "$REPO_ROOT/experiment/openevolve/evaluator.py" \
  --config "$RESOLVED_CONFIG" \
  --output "$OUTPUT_DIR" \
  --iterations "$BUDGET"
