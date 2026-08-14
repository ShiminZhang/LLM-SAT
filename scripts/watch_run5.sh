#!/bin/bash
# Wait for run5 evals (5 solvers, jobs 38333452-38333456) to drain,
# then collect each + append to outputs/full_eval_log.csv.
set -uo pipefail
cd /scratch/meru/LLM-SAT
source ~/general/bin/activate
export PYTHONPATH=src

JOBS=(38333452 38333453 38333454 38333455 38333456)
LABELS=("AE" "B1" "B2" "compo_R1_B1" "R1")
PATHS=(
  "solvers/AE_kissat2025_MAB_clean"
  "solvers/kissat_evolve_iter1/members/algorithm_cee194034a6f78b6552b5d246e855e66f1ff2cd2fe039f1bfc4008d987a25bce/code_462fc8f76750f22b957b62158221d53faaddba8c8c543457a1fc802c482d21c6"
  "solvers/kissat_evolve_iter0/members/algorithm_32fb7a62e3041b4a86fcf532ccbfdb3fbcd6afec2549db7d8f6872ba63dcd7a0/code_5998cbf41f91e1168c9054a5681f04057b8ac2825097e96c9aeaa0106de429ad"
  "solvers/compositions/compo_R1_B1"
  "solvers/restart_mab_iter0/members/algorithm_f9d75c8ec541aa1e5dac9a68a8d7a1cd7ea39d9a1858d0b97dc1e6237322c965/code_27b38abdadabad20b036a753f751995d73faa8eb18205a170f96e27c387359bf"
)
# Per-solver metadata for the log
declare -A AID=(
  [AE]="—" [B1]="cee194034a6f" [B2]="32fb7a62e304" [compo_R1_B1]="—" [R1]="f9d75c8ec541"
)
declare -A CID=(
  [AE]="—" [B1]="462fc8f76750" [B2]="5998cbf41f91" [compo_R1_B1]="—" [R1]="27b38abdadab"
)
declare -A TAG=(
  [AE]="—" [B1]="kissat_evolve_iter1" [B2]="kissat_evolve_iter0" [compo_R1_B1]="compositions" [R1]="restart_mab_iter0"
)
declare -A FULL_LABEL=(
  [AE]="AE_kissat2025_MAB" [B1]="B1" [B2]="B2" [compo_R1_B1]="compo_R1_B1" [R1]="R1"
)

OUT=outputs/run5_summary.txt
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

for i in 0 1 2 3 4; do
  label="${LABELS[$i]}"
  jid="${JOBS[$i]}"
  sp="${PATHS[$i]}"
  rd="$sp/result_simul_run5"
  echo "=== $label (job $jid) ===" | tee -a "$OUT"
  python scripts/evaluate_solver.py "$sp" --collect --result-dir "$rd" 2>&1 \
    | grep -E "Instances|Solved|Timeouts|PAR-2" | tee -a "$OUT"
  python scripts/append_eval_to_log.py \
    --solver "${FULL_LABEL[$label]}" \
    --run-label "simul-2026-05-02-run5" \
    --result-dir "$rd" \
    --job-id "$jid" \
    --generation-tag "${TAG[$label]}" \
    --algorithm-id-short "${AID[$label]}" \
    --code-id-short "${CID[$label]}" >> "$OUT"
  echo "" | tee -a "$OUT"
done

echo "Finished: $(date)" | tee -a "$OUT"
echo "Updated outputs/full_eval_log.csv" | tee -a "$OUT"
