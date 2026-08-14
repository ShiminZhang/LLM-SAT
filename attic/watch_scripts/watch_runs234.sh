#!/bin/bash
# Watch the 6 new AE+B1 simul-pair evaluations (jobs 38083396-38083401).
# When ALL drain, collect each and append to outputs/full_eval_log.csv.
set -uo pipefail
cd /scratch/meru/LLM-SAT
source ~/general/bin/activate
export PYTHONPATH=src

JOBS=(38083396 38083397 38083398 38083399 38083400 38083401)
LABELS=("AE" "B1" "AE" "B1" "AE" "B1")
RUNS=(run2 run2 run3 run3 run4 run4)
PATHS=(
  "solvers/AE_kissat2025_MAB_clean"
  "solvers/kissat_evolve_iter1/members/algorithm_cee194034a6f78b6552b5d246e855e66f1ff2cd2fe039f1bfc4008d987a25bce/code_462fc8f76750f22b957b62158221d53faaddba8c8c543457a1fc802c482d21c6"
  "solvers/AE_kissat2025_MAB_clean"
  "solvers/kissat_evolve_iter1/members/algorithm_cee194034a6f78b6552b5d246e855e66f1ff2cd2fe039f1bfc4008d987a25bce/code_462fc8f76750f22b957b62158221d53faaddba8c8c543457a1fc802c482d21c6"
  "solvers/AE_kissat2025_MAB_clean"
  "solvers/kissat_evolve_iter1/members/algorithm_cee194034a6f78b6552b5d246e855e66f1ff2cd2fe039f1bfc4008d987a25bce/code_462fc8f76750f22b957b62158221d53faaddba8c8c543457a1fc802c482d21c6"
)

OUT=outputs/runs234_summary.txt
: > "$OUT"
echo "Watching jobs: ${JOBS[*]}" | tee -a "$OUT"
echo "Started: $(date)" | tee -a "$OUT"

while true; do
  remaining=0
  for jid in "${JOBS[@]}"; do
    n=$(squeue -u "$USER" -h -j "$jid" 2>/dev/null | wc -l)
    remaining=$((remaining + n))
  done
  [ "$remaining" -eq 0 ] && break
  sleep 90
done

echo "" | tee -a "$OUT"
echo "All jobs cleared queue at $(date)" | tee -a "$OUT"
echo "" | tee -a "$OUT"

# AE has no algo/code IDs; B1 does
B1_AID="cee194034a6f"
B1_CID="462fc8f76750"
B1_TAG="kissat_evolve_iter1"

for i in 0 1 2 3 4 5; do
  label="${LABELS[$i]}"
  run="${RUNS[$i]}"
  jid="${JOBS[$i]}"
  sp="${PATHS[$i]}"
  rd="$sp/result_simul_$run"
  echo "=== $label $run (job $jid) ===" | tee -a "$OUT"

  # Collect (writes solving_times.json into rd)
  python scripts/evaluate_solver.py "$sp" --collect --result-dir "$rd" 2>&1 \
    | grep -E "Instances|Solved|Timeouts|PAR-2" | tee -a "$OUT"

  # Append to log
  if [ "$label" = "AE" ]; then
    python scripts/append_eval_to_log.py \
      --solver "AE_kissat2025_MAB" \
      --run-label "simul-2026-05-01-$run" \
      --result-dir "$rd" \
      --job-id "$jid" >> "$OUT"
  else
    python scripts/append_eval_to_log.py \
      --solver "B1" \
      --run-label "simul-2026-05-01-$run" \
      --result-dir "$rd" \
      --job-id "$jid" \
      --generation-tag "$B1_TAG" \
      --algorithm-id-short "$B1_AID" \
      --code-id-short "$B1_CID" >> "$OUT"
  fi
  echo "" | tee -a "$OUT"
done

echo "Finished: $(date)" | tee -a "$OUT"
echo "Updated outputs/full_eval_log.csv" | tee -a "$OUT"
