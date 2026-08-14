#!/bin/bash
# Wait for the 3 verification full-evals (AE base, B1 fresh, phase_iter1 best)
# to drain SLURM, then collect each and write a comparison report.
set -uo pipefail
cd /scratch/meru/LLM-SAT
source ~/general/bin/activate
export PYTHONPATH=src

JOBS=(37923909 37923950 37923955)
PATHS=(
  "solvers/AE_kissat2025_MAB_clean"
  "solvers/kissat_evolve_iter1/members/algorithm_cee194034a6f78b6552b5d246e855e66f1ff2cd2fe039f1bfc4008d987a25bce/code_462fc8f76750f22b957b62158221d53faaddba8c8c543457a1fc802c482d21c6"
  "solvers/phase_iter1/members/algorithm_4ca95fb7f81b53bf0caeb649b9653beafc7ef9998fdde729bc9387ebeda28cae/code_03ad558da1d4dd2dba981307a17262a8d210a3ecc8bc53f1a3e7ba82aaefa172"
)
LABELS=("AE_kissat2025_MAB (fresh baseline)" "B1 cee194034a6f / 462fc8f76750 (fresh)" "phase_iter1 best 4ca95fb7f81b / 03ad558da1d4")
RESULT_DIRS=(
  "solvers/AE_kissat2025_MAB_clean/result"
  "solvers/kissat_evolve_iter1/members/algorithm_cee194034a6f78b6552b5d246e855e66f1ff2cd2fe039f1bfc4008d987a25bce/code_462fc8f76750f22b957b62158221d53faaddba8c8c543457a1fc802c482d21c6/result_full_v2"
  "solvers/phase_iter1/members/algorithm_4ca95fb7f81b53bf0caeb649b9653beafc7ef9998fdde729bc9387ebeda28cae/code_03ad558da1d4dd2dba981307a17262a8d210a3ecc8bc53f1a3e7ba82aaefa172/result"
)

OUT=outputs/verification_evals.txt
mkdir -p outputs
: > "$OUT"
echo "Watching jobs: ${JOBS[*]}" | tee -a "$OUT"
echo "Started: $(date)" | tee -a "$OUT"

# Poll every 90 s until all three jobs are gone from the queue
while true; do
  remaining=0
  for jid in "${JOBS[@]}"; do
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

# Collect each — capture the PAR2 line from evaluate_solver.py output
declare -a PAR2_VALS
for i in 0 1 2; do
  echo "=== ${LABELS[$i]} ===" | tee -a "$OUT"
  set +e
  out=$(python scripts/evaluate_solver.py "${PATHS[$i]}" --collect --result-dir "${RESULT_DIRS[$i]}" 2>&1)
  echo "$out" | grep -E "Instances|Solved|Timeouts|PAR-2" | tee -a "$OUT"
  par2=$(echo "$out" | grep "PAR-2:" | awk '{print $NF}')
  PAR2_VALS[$i]="$par2"
  set -e
  echo "" | tee -a "$OUT"
done

# Side-by-side comparison
{
  echo "=== Comparison ==="
  printf "  %-50s %12s %12s\n" "Solver" "Prior PAR2" "Fresh PAR2"
  printf "  %-50s %12s %12s\n" "AE_kissat2025_MAB" "1925.52" "${PAR2_VALS[0]:-?}"
  printf "  %-50s %12s %12s\n" "B1 (cee194034a6f / 462fc8f76750)" "1868.43" "${PAR2_VALS[1]:-?}"
  printf "  %-50s %12s %12s\n" "phase_iter1 best (4ca95fb7f81b / 03ad558da1d4)" "—" "${PAR2_VALS[2]:-?}"
  echo ""
  echo "Reference: Kissat-CURE baseline full PAR2 = 2165.94 (different solver — only relevant for phase_iter1 best)"
  echo "Finished: $(date)"
} | tee -a "$OUT"
