#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${SAT_REPO_ROOT:-/scratch/s568zhan/LLM-SAT}"
OPENEVOLVE_ROOT="${OPENEVOLVE_ROOT:-/scratch/s568zhan/OpenEvolve}"
SHINKA_ROOT="${SHINKA_ROOT:-$REPO_ROOT/experiments/ShinkaEvolve}"
source "$REPO_ROOT/experiment/common/protocol.sh"
export_comparison_protocol
BENCHMARK_FAMILY="${BENCHMARK_FAMILY:?BENCHMARK_FAMILY is required}"
BENCHMARK_DIR="${OE_BENCHMARK_DIR:-$REPO_ROOT/data/benchmarks/formula-families/$BENCHMARK_FAMILY}"
RESULTS_DIR="${BASELINE_RESULTS_DIR:?BASELINE_RESULTS_DIR is required}"

cd "$REPO_ROOT"
source "$REPO_ROOT/scripts/activate_rorqual.sh"
set -a
source "$REPO_ROOT/.env"
set +a

export SAT_REPO_ROOT="$REPO_ROOT"
export OE_REPO_ROOT="$REPO_ROOT"
export OE_TARGET_FUNCTION=kissat_decide_phase
export OE_TARGET_SOURCE=src/decide.c
export OE_BENCHMARK_DIR="$BENCHMARK_DIR"
export OE_WORK_DIR="${OE_WORK_DIR:-$REPO_ROOT/experiment/comparison_work/$BENCHMARK_FAMILY/decide}"
export PYTHONPATH="$REPO_ROOT:$SHINKA_ROOT:$OPENEVOLVE_ROOT:${PYTHONPATH:-}"

exec python "$REPO_ROOT/experiment/shinka/evaluate.py" \
  --program_path "$REPO_ROOT/experiment/shinka/initial.c" \
  --results_dir "$RESULTS_DIR"
