#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${SAT_REPO_ROOT:-/scratch/s568zhan/LLM-SAT}"
source "$REPO_ROOT/experiment/common/protocol.sh"
export_comparison_protocol
TARGET="${1:-}"
BUDGET="${LLMSAT_CANDIDATE_BUDGET:-100}"
BENCHMARK_FAMILY="${LLMSAT_BENCHMARK_FAMILY:-cryptography-ascon}"
BENCHMARK_DIR="${LLMSAT_BENCHMARK_DIR:-$REPO_ROOT/data/benchmarks/formula-families/$BENCHMARK_FAMILY}"

case "$TARGET" in
  decide|kissat_decide_phase)
    TARGET_SLUG=decide
    LEADER_PROMPT="${LLMSAT_LEADER_PROMPT:-$REPO_ROOT/experiment/llmsat/prompts/leader_decide.txt}"
    CODER_PROMPT="$REPO_ROOT/experiment/llmsat/prompts/coder_decide.txt"
    ;;
  restart|restarting|kissat_restarting)
    TARGET_SLUG=restarting
    LEADER_PROMPT="${LLMSAT_LEADER_PROMPT:-$REPO_ROOT/experiment/llmsat/prompts/leader_restart.txt}"
    CODER_PROMPT="$REPO_ROOT/experiment/llmsat/prompts/coder_restart.txt"
    ;;
  *)
    echo "Usage: $0 <decide|restarting>" >&2
    exit 2
    ;;
esac

if ! [[ "$BUDGET" =~ ^[0-9]+$ ]] || [ "$BUDGET" -lt 30 ]; then
  echo "LLMSAT_CANDIDATE_BUDGET must be an integer >= 30" >&2
  exit 2
fi
if [ $(((BUDGET - 30) % 5)) -ne 0 ]; then
  echo "Budget minus the 30-candidate initial population must be divisible by 5" >&2
  exit 2
fi
# Five leaders with five initial members each give 30 initial candidates. Later
# iterations generate three members per leader (15 candidates), with the final
# iteration automatically shortened when required. 100 -> 5 iterations;
# 500 -> 32 iterations.
CANDIDATE_GROUPS=$(((BUDGET - 30) / 5))
ITERATIONS=$(((CANDIDATE_GROUPS + 2) / 3))

if [ "${LLMSAT_PLAN_ONLY:-0}" = "1" ]; then
  printf 'budget=%s initial_candidates=30 iterations=%s\n' "$BUDGET" "$ITERATIONS"
  exit 0
fi
if [ ! -d "$BENCHMARK_DIR" ]; then
  echo "Benchmark directory not found: $BENCHMARK_DIR" >&2
  exit 2
fi
if ! find "$BENCHMARK_DIR" -maxdepth 1 -type f -name '*.cnf' -print -quit | grep -q .; then
  echo "No CNFs found in benchmark directory: $BENCHMARK_DIR" >&2
  exit 2
fi

export CANDIDATE_BUDGET="$BUDGET"
export MODEL="${MODEL:-$COMPARISON_MODEL}"
export N_LEADERS=5
export INIT_M_VARIANTS=5
export M_VARIANTS=3
export DESIGNER_PROMPT="$LEADER_PROMPT"
export CODER_PROMPT
export VARIANT_PROMPT="$REPO_ROOT/data/prompts/variant_prompt.txt"
export QUICK_EVAL=0
export LLMSAT_BENCHMARK_DIR="$BENCHMARK_DIR"

BASE_TAG="${LLMSAT_RUN_ID:-llmsat_${TARGET_SLUG}_${BUDGET}_${BENCHMARK_FAMILY}_$(date +%Y%m%d_%H%M%S)}"

# The generated function and durable score files live beside each algorithm.
# Full copied Kissat trees are only build artifacts; prune completed iterations
# while long runs continue so a 500-candidate run cannot exhaust inode quota.
bash "$REPO_ROOT/run_loop_a.sh" cc "$BASE_TAG" "$ITERATIONS" --init &
LOOP_PID=$!
cleanup() {
  kill "$LOOP_PID" 2>/dev/null || true
}
trap cleanup INT TERM
while kill -0 "$LOOP_PID" 2>/dev/null; do
  python3 "$REPO_ROOT/scripts/prune_llmsat_solver_trees.py" \
    --repo-root "$REPO_ROOT" --tag-prefix "$BASE_TAG"
  sleep 60
done
set +e
wait "$LOOP_PID"
STATUS=$?
set -e
python3 "$REPO_ROOT/scripts/prune_llmsat_solver_trees.py" \
  --repo-root "$REPO_ROOT" --tag-prefix "$BASE_TAG"
exit "$STATUS"
