#!/bin/bash
# Watch restart_mab_iter0 full-eval SLURM jobs; collect + report PAR2 when done.
# Output: outputs/restart_mab_iter0/full_eval_par2.txt

set -uo pipefail

JOB_IDS=(37565431 37565452 37565528)

declare -A SOLVER_PATH=(
  [37565431]="solvers/restart_mab_iter0/members/algorithm_f9d75c8ec541aa1e5dac9a68a8d7a1cd7ea39d9a1858d0b97dc1e6237322c965/code_27b38abdadabad20b036a753f751995d73faa8eb18205a170f96e27c387359bf"
  [37565452]="solvers/restart_mab_iter0/members/algorithm_7f38c0060ffa108fd3003547025e448ce94a50d9dab1e6cb0f784dc1be288795/code_da0c42357827ff819b05e6946e5f149db071f95572c7f823264e374d4225061f"
  [37565528]="solvers/restart_mab_iter0/members/algorithm_86b5b1f142c210044c759c0f27ca650ba9eccb94971511db7998244101ef8edd/code_49aae3237a6260c5105f3ae7bdada24b4a660016e44ec8f1315b88c2505e0823"
)

declare -A LABEL=(
  [37565431]="f9d75c8ec541 / 27b38abdadab  (quick 455.33)"
  [37565452]="7f38c0060ffa / da0c42357827  (quick 462.46)"
  [37565528]="86b5b1f142c2 / 49aae3237a62  (quick 466.51)"
)

cd /scratch/meru/LLM-SAT
source ~/general/bin/activate
export PYTHONPATH=src

mkdir -p outputs/restart_mab_iter0
OUT=outputs/restart_mab_iter0/full_eval_par2.txt
: > "$OUT"
echo "Watching jobs: ${JOB_IDS[*]}" | tee -a "$OUT"
echo "Started: $(date)" | tee -a "$OUT"

# Poll every 90 s until all three jobs are gone from the queue.
while true; do
  remaining=0
  for jid in "${JOB_IDS[@]}"; do
    n=$(squeue -u "$USER" -h -j "$jid" 2>/dev/null | wc -l)
    remaining=$((remaining + n))
  done
  if [ "$remaining" -eq 0 ]; then
    break
  fi
  sleep 90
done

echo "" | tee -a "$OUT"
echo "All jobs cleared queue at $(date)" | tee -a "$OUT"
echo "" | tee -a "$OUT"

for jid in "${JOB_IDS[@]}"; do
  sp="${SOLVER_PATH[$jid]}"
  echo "=== Job $jid : ${LABEL[$jid]} ===" | tee -a "$OUT"
  python scripts/ice_scripts/evaluate_single_solver.py \
    --solver-path "$sp" \
    --result-dir "$sp/result_full" \
    --collect --timeout 5000 --penalty 10000 \
    --output "$sp/solving_times_full.json" 2>&1 \
    | grep -E "PAR2|Completed|Timeouts" | tee -a "$OUT"
  echo "" | tee -a "$OUT"
done

echo "Finished: $(date)" | tee -a "$OUT"
