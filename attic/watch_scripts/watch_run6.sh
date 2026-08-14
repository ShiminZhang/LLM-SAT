#!/bin/bash
# Wait for run6 evals (AE+B1, jobs 38672866 + 38672887) to drain,
# then collect each + append to outputs/full_eval_log.csv.
set -uo pipefail
cd /scratch/meru/LLM-SAT
source ~/general/bin/activate
export PYTHONPATH=src

JOBS=(38672866 38672887)
LABELS=("AE" "B1")
PATHS=(
  "solvers/AE_kissat2025_MAB_clean"
  "solvers/kissat_evolve_iter1/members/algorithm_cee194034a6f78b6552b5d246e855e66f1ff2cd2fe039f1bfc4008d987a25bce/code_462fc8f76750f22b957b62158221d53faaddba8c8c543457a1fc802c482d21c6"
)
declare -A AID=([AE]="—" [B1]="cee194034a6f")
declare -A CID=([AE]="—" [B1]="462fc8f76750")
declare -A TAG=([AE]="—" [B1]="kissat_evolve_iter1")
declare -A FULL_LABEL=([AE]="AE_kissat2025_MAB" [B1]="B1")

OUT=outputs/run6_summary.txt
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

for i in 0 1; do
  label="${LABELS[$i]}"
  jid="${JOBS[$i]}"
  sp="${PATHS[$i]}"
  rd="$sp/result_simul_run6"
  echo "=== $label (job $jid) ===" | tee -a "$OUT"
  python scripts/evaluate_solver.py "$sp" --collect --result-dir "$rd" 2>&1 \
    | grep -E "Instances|Solved|Timeouts|PAR-2" | tee -a "$OUT"
  python scripts/append_eval_to_log.py \
    --solver "${FULL_LABEL[$label]}" \
    --run-label "simul-2026-05-03-run6" \
    --result-dir "$rd" \
    --job-id "$jid" \
    --generation-tag "${TAG[$label]}" \
    --algorithm-id-short "${AID[$label]}" \
    --code-id-short "${CID[$label]}" >> "$OUT"
  echo "" | tee -a "$OUT"
done

echo "Finished: $(date)" | tee -a "$OUT"
echo "Updated outputs/full_eval_log.csv" | tee -a "$OUT"
