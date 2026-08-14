#!/bin/bash
# Watch the 6-solver simultaneous full-eval batch (B1, AE, B2, compo_R1_B1, R1, R2).
set -uo pipefail
cd /scratch/meru/LLM-SAT
source ~/general/bin/activate
export PYTHONPATH=src

JOBS=(38052695 38052696 38052697 38052698 38052699 38052700)
LABELS=("B1" "AE" "B2" "compo_R1_B1" "R1" "R2")
PATHS=(
  "solvers/kissat_evolve_iter1/members/algorithm_cee194034a6f78b6552b5d246e855e66f1ff2cd2fe039f1bfc4008d987a25bce/code_462fc8f76750f22b957b62158221d53faaddba8c8c543457a1fc802c482d21c6"
  "solvers/AE_kissat2025_MAB_clean"
  "solvers/kissat_evolve_iter0/members/algorithm_32fb7a62e3041b4a86fcf532ccbfdb3fbcd6afec2549db7d8f6872ba63dcd7a0/code_5998cbf41f91e1168c9054a5681f04057b8ac2825097e96c9aeaa0106de429ad"
  "solvers/compositions/compo_R1_B1"
  "solvers/restart_mab_iter0/members/algorithm_f9d75c8ec541aa1e5dac9a68a8d7a1cd7ea39d9a1858d0b97dc1e6237322c965/code_27b38abdadabad20b036a753f751995d73faa8eb18205a170f96e27c387359bf"
  "solvers/restart_mab_iter0/members/algorithm_7f38c0060ffa108fd3003547025e448ce94a50d9dab1e6cb0f784dc1be288795/code_da0c42357827ff819b05e6946e5f149db071f95572c7f823264e374d4225061f"
)

OUT=outputs/simul_full_eval_par2.txt
mkdir -p outputs
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

declare -a P2
for i in 0 1 2 3 4 5; do
  echo "=== ${LABELS[$i]} (job ${JOBS[$i]}) ===" | tee -a "$OUT"
  out=$(python scripts/evaluate_solver.py "${PATHS[$i]}" --collect --result-dir "${PATHS[$i]}/result_simul" 2>&1)
  echo "$out" | grep -E "Instances|Solved|Timeouts|PAR-2" | tee -a "$OUT"
  par2=$(echo "$out" | grep "PAR-2:" | awk '{print $NF}')
  P2[$i]="$par2"
  echo "" | tee -a "$OUT"
done

{
  echo "=== Side-by-side comparison (simultaneous-cluster-conditions full eval) ==="
  printf "  %-15s %12s\n" "Solver" "PAR2"
  for i in 0 1 2 3 4 5; do
    printf "  %-15s %12s\n" "${LABELS[$i]}" "${P2[$i]}"
  done
  echo ""
  echo "Finished: $(date)"
} | tee -a "$OUT"
