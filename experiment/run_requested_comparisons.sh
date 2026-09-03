#!/usr/bin/env bash
set -u -o pipefail

REPO_ROOT="${SAT_REPO_ROOT:-/scratch/s568zhan/LLM-SAT}"
source "$REPO_ROOT/experiment/common/protocol.sh"
export_comparison_protocol
CAMPAIGN_ID="${CAMPAIGN_ID:-comparison_queue_$(date +%Y%m%d_%H%M%S)}"
CAMPAIGN_DIR="$REPO_ROOT/experiment/campaigns/$CAMPAIGN_ID"
PREREQUISITE_SUITE="${PREREQUISITE_SUITE:-suite100_cont_20260830_233746}"
EDGE_SUITE_ID="${EDGE_SUITE_ID:-edge100_$(date +%Y%m%d_%H%M%S)}"
ASCON_SUITE_ID="${ASCON_SUITE_ID:-ascon_decide500_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${CAMPAIGN_DRY_RUN:-0}"
ALLOW_QUEUE_CONTENTION="${ALLOW_QUEUE_CONTENTION:-0}"

mkdir -p "$CAMPAIGN_DIR"
printf '%s\n' "$$" > "$CAMPAIGN_DIR/campaign.pid"

wait_for_idle() {
  if [ "$ALLOW_QUEUE_CONTENTION" = "1" ]; then
    return 0
  fi
  while squeue -h -u "$USER" | grep -q .; do
    echo "$(date --iso-8601=seconds) waiting for existing jobs to leave the queue"
    squeue -h -u "$USER" -o '%A|%j|%T|%R' | sort -u
    sleep 60
  done
}

wait_for_prerequisite() {
  local status="$REPO_ROOT/experiment/suites/$PREREQUISITE_SUITE/status.tsv"
  while true; do
    if [ -f "$status" ] && awk -F '\t' \
      '$1 == "llmsat_restarting" && ($2 == "completed" || $2 == "failed") { found=1 } END { exit !found }' \
      "$status"; then
      echo "$(date --iso-8601=seconds) prerequisite suite reached a terminal state"
      break
    fi
    echo "$(date --iso-8601=seconds) waiting for prerequisite suite $PREREQUISITE_SUITE"
    sleep 60
  done
  wait_for_idle
  python3 "$REPO_ROOT/scripts/prune_llmsat_solver_trees.py" \
    --repo-root "$REPO_ROOT" --tag-prefix "${PREREQUISITE_SUITE}_llmsat_restarting"
}

new_suite() {
  local suite_id="$1" benchmark="$2" budget="$3" instances="$4" scope="$5"
  local suite_dir="$REPO_ROOT/experiment/suites/$suite_id"
  mkdir -p "$suite_dir"
  printf 'run\tstatus\texit_code\tstarted_at\tfinished_at\n' > "$suite_dir/status.tsv"
  printf '%s\n' \
    "suite_id=$suite_id" \
    "scope=$scope" \
    "candidate_budget=$budget" \
    "benchmark=$benchmark" \
    "instances=$instances" \
    'benchmark_source=Global Benchmark Database, SAT Competition 2025 main track' \
    "model=$COMPARISON_MODEL" \
    "candidate_job_cpus=$OE_SLURM_CPUS" \
    "cpu_constraint=$OE_SLURM_CONSTRAINT" \
    "candidate_job_walltime=$OE_WALL_TIME" \
    "instance_timeout_seconds=$OE_TIMEOUT" \
    "par2_penalty_seconds=$OE_PAR2_PENALTY" \
    "max_concurrent_candidate_jobs=$LLMSAT_MAX_CANDIDATE_JOBS" \
    "keep_proofs=$OE_KEEP_PROOFS" > "$suite_dir/protocol.txt"
}

run_one() {
  local suite_id="$1" name="$2"
  shift 2
  local suite_dir="$REPO_ROOT/experiment/suites/$suite_id"
  local log="$suite_dir/$name.log"
  local started finished status
  if [ "$DRY_RUN" = "1" ]; then
    started="$(date --iso-8601=seconds)"
    printf 'DRY RUN %s/%s:' "$suite_id" "$name"
    printf ' %q' "$@"
    printf '\n'
    printf '%s\tdry_run\t0\t%s\t%s\n' "$name" "$started" "$started" >> "$suite_dir/status.tsv"
    return 0
  fi
  wait_for_idle
  started="$(date --iso-8601=seconds)"
  echo "$started START $suite_id/$name"
  "$@" > "$log" 2>&1
  status=$?
  wait_for_idle
  finished="$(date --iso-8601=seconds)"
  if [ "$status" -eq 0 ]; then
    printf '%s\tcompleted\t0\t%s\t%s\n' "$name" "$started" "$finished" >> "$suite_dir/status.tsv"
    echo "$finished COMPLETE $suite_id/$name"
  else
    printf '%s\tfailed\t%s\t%s\t%s\n' "$name" "$status" "$started" "$finished" >> "$suite_dir/status.tsv"
    echo "$finished FAILED $suite_id/$name exit=$status; continuing"
  fi
}

validate_shinka() {
  local run_id="$1" budget="$2"
  python3 - "$REPO_ROOT/experiment/shinka/runs/$run_id/programs.sqlite" "$budget" <<'PY'
import sqlite3
import sys

db, budget = sys.argv[1], int(sys.argv[2])
connection = sqlite3.connect(f"file:{db}?immutable=1", uri=True)
distinct, minimum, maximum = connection.execute(
    "select count(distinct generation), min(generation), max(generation) from programs"
).fetchone()
expected = budget + 1
if (distinct, minimum, maximum) != (expected, 0, budget):
    raise SystemExit(
        f"incomplete Shinka budget: distinct={distinct}, min={minimum}, max={maximum}, expected={expected}"
    )
print(f"validated Shinka generations 0..{budget}")
PY
}

run_shinka() {
  local suite_id="$1" target="$2" benchmark="$3" budget="$4" instances="$5"
  local slug run_id benchmark_dir source
  local wait_args=()
  benchmark_dir="$REPO_ROOT/data/benchmarks/formula-families/$benchmark"
  if [ "$target" = "kissat_decide_phase" ]; then
    slug=decide
    source=src/decide.c
  else
    slug=restarting
    source=src/restart.c
  fi
  run_id="${suite_id}_shinka_${slug}"
  if [ "$ALLOW_QUEUE_CONTENTION" != "1" ]; then
    wait_args+=(--wait-for-idle)
  fi
  env SHINKA_RUN_ID="$run_id" \
      SHINKA_TARGET="$target" \
      SHINKA_OFFSPRING_BUDGET="$budget" \
      SHINKA_NUM_GENERATIONS="$((budget + 1))" \
      SHINKA_BENCHMARK_FAMILY="$benchmark" \
      SHINKA_BENCHMARK_INSTANCES="$instances" \
      OE_TARGET_FUNCTION="$target" \
      OE_TARGET_SOURCE="$source" \
      OE_BENCHMARK_DIR="$benchmark_dir" \
      OE_WORK_DIR="$REPO_ROOT/experiment/comparison_work/$benchmark/$slug" \
    bash "$REPO_ROOT/experiment/shinka/run_shinka.sh" "${wait_args[@]}" && \
    validate_shinka "$run_id" "$budget"
}

run_llmsat() {
  local suite_id="$1" target="$2" benchmark="$3" budget="$4"
  local slug prompt
  if [ "$target" = "decide" ]; then
    slug=decide
    prompt="$REPO_ROOT/experiment/llmsat/prompts/leader_decide.txt"
  else
    slug=restarting
    prompt="$REPO_ROOT/experiment/llmsat/prompts/leader_restart.txt"
  fi
  if [ "$benchmark" = "edge-matching" ]; then
    if [ "$target" = "decide" ]; then
      prompt="$REPO_ROOT/experiment/llmsat/prompts/leader_decide_edge_matching.txt"
    else
      prompt="$REPO_ROOT/experiment/llmsat/prompts/leader_restart_edge_matching.txt"
    fi
  fi
  env LLMSAT_RUN_ID="${suite_id}_llmsat_${slug}" \
      LLMSAT_CANDIDATE_BUDGET="$budget" \
      LLMSAT_BENCHMARK_FAMILY="$benchmark" \
      LLMSAT_BENCHMARK_DIR="$REPO_ROOT/data/benchmarks/formula-families/$benchmark" \
      LLMSAT_LEADER_PROMPT="$prompt" \
    bash "$REPO_ROOT/experiment/llmsat/run_candidates.sh" "$target"
}

run_edge_suite() {
  local suite_id="$EDGE_SUITE_ID" benchmark=edge-matching budget=100 instances=2
  local dir="$REPO_ROOT/data/benchmarks/formula-families/$benchmark"
  [ "$(find "$dir" -maxdepth 1 -type f -name '*.cnf' | wc -l)" -eq "$instances" ] || return 4
  new_suite "$suite_id" "$benchmark" "$budget" "$instances" decide_and_restarting

  run_one "$suite_id" kissat_baseline \
    env BENCHMARK_FAMILY="$benchmark" OE_BENCHMARK_DIR="$dir" \
        BASELINE_RESULTS_DIR="$REPO_ROOT/experiment/suites/$suite_id/kissat_baseline" \
        OE_WORK_DIR="$REPO_ROOT/experiment/comparison_work/$benchmark/decide" \
      bash "$REPO_ROOT/experiment/run_baseline_comparison.sh"
  for target in decide restarting; do
    run_one "$suite_id" "oe_$target" \
      env OE_RUN_ID="${suite_id}_oe_$target" \
        bash "$REPO_ROOT/experiment/run_openevolve_comparison.sh" "$target" "$benchmark" "$budget"
    if [ "$target" = decide ]; then function_name=kissat_decide_phase; else function_name=kissat_restarting; fi
    run_one "$suite_id" "shinka_$target" run_shinka \
      "$suite_id" "$function_name" "$benchmark" "$budget" "$instances"
    run_one "$suite_id" "llmsat_$target" run_llmsat \
      "$suite_id" "$target" "$benchmark" "$budget"
  done
  echo "$(date --iso-8601=seconds) SUITE FINISHED $suite_id"
}

run_ascon_500_suite() {
  local suite_id="$ASCON_SUITE_ID" benchmark=cryptography-ascon budget=500 instances=26
  local dir="$REPO_ROOT/data/benchmarks/formula-families/$benchmark"
  [ "$(find "$dir" -maxdepth 1 -type f -name '*.cnf' | wc -l)" -eq "$instances" ] || return 4
  new_suite "$suite_id" "$benchmark" "$budget" "$instances" decide

  run_one "$suite_id" kissat_baseline \
    env BENCHMARK_FAMILY="$benchmark" OE_BENCHMARK_DIR="$dir" \
        BASELINE_RESULTS_DIR="$REPO_ROOT/experiment/suites/$suite_id/kissat_baseline" \
        OE_WORK_DIR="$REPO_ROOT/experiment/comparison_work/$benchmark/decide" \
      bash "$REPO_ROOT/experiment/run_baseline_comparison.sh"

  run_one "$suite_id" oe_decide \
    env OE_RUN_ID="${suite_id}_oe_decide" \
      bash "$REPO_ROOT/experiment/run_openevolve_comparison.sh" decide "$benchmark" "$budget"
  run_one "$suite_id" shinka_decide run_shinka \
    "$suite_id" kissat_decide_phase "$benchmark" "$budget" "$instances"
  run_one "$suite_id" llmsat_decide run_llmsat \
    "$suite_id" decide "$benchmark" "$budget"
  echo "$(date --iso-8601=seconds) SUITE FINISHED $suite_id"
}

cd "$REPO_ROOT"
printf '%s\n' \
  "campaign_id=$CAMPAIGN_ID" \
  "prerequisite_suite=$PREREQUISITE_SUITE" \
  "first_suite=$EDGE_SUITE_ID" \
  "second_suite=$ASCON_SUITE_ID" > "$CAMPAIGN_DIR/protocol.txt"
printf '%s\n' \
  "cpu_constraint=$OE_SLURM_CONSTRAINT" \
  "candidate_job_cpus=$OE_SLURM_CPUS" \
  "instance_timeout_seconds=$OE_TIMEOUT" \
  "par2_penalty_seconds=$OE_PAR2_PENALTY" >> "$CAMPAIGN_DIR/protocol.txt"

if [ "$DRY_RUN" != "1" ] && [ "$ALLOW_QUEUE_CONTENTION" != "1" ]; then
  wait_for_prerequisite
elif [ "$ALLOW_QUEUE_CONTENTION" = "1" ]; then
  echo "$(date --iso-8601=seconds) queue contention enabled; starting immediately"
fi
# The two-instance edge suite goes first so it can produce a complete new-family
# comparison before the much longer 500-offspring ASCON suite monopolizes the queue.
run_edge_suite
run_ascon_500_suite
echo "$(date --iso-8601=seconds) CAMPAIGN FINISHED $CAMPAIGN_ID"
