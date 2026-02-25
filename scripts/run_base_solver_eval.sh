#!/bin/bash
# One-time script: evaluate base (unmodified) Kissat on all 400 CNFs.
# After SLURM jobs finish, run again with --collect to build solving_times.json.
#
# Usage:
#   bash scripts/run_base_solver_eval.sh            # submit SLURM array
#   bash scripts/run_base_solver_eval.sh --collect   # collect results into JSON

set -euo pipefail

BASE_SOLVER="solvers/base"
BENCHMARK_PATH="data/benchmarks/satcomp2025"
RESULT_DIR="solvers/base/result"
TIMEOUT=5000
WALL_TIME="01:30:00"
ACCOUNT="def-vganesh"
MEM="4G"
MAX_CONCURRENT=1000

# ---- Collect mode: parse logs into solving_times.json ----
if [[ "${1:-}" == "--collect" ]]; then
    echo "Collecting results from $RESULT_DIR ..."
    python3 -c "
import os, re, json

result_dir = '$RESULT_DIR'
par2_penalty = 10000
times = {}

for f in sorted(os.listdir(result_dir)):
    if not f.endswith('.solving.log'):
        continue
    instance = f.rsplit('.cnf.solving.log', 1)[0]
    path = os.path.join(result_dir, f)
    t = par2_penalty
    for line in reversed(open(path).readlines()):
        if 'process-time' in line:
            m = re.search(r'(\d*\.?\d+)\s+seconds', line)
            if m:
                t = float(m.group(1))
            break
        if 'TIMEOUT' in line or 'ERROR' in line or 'CANCELLED' in line:
            break
    times[instance] = t

out_path = os.path.join('$BASE_SOLVER', 'solving_times.json')
with open(out_path, 'w') as fh:
    json.dump(times, fh, indent=2)
print(f'Wrote {len(times)} entries to {out_path}')
"
    exit 0
fi

# ---- Submit mode: launch SLURM array ----
if [ ! -x "$BASE_SOLVER/kissat" ]; then
    echo "ERROR: base solver not found at $BASE_SOLVER/kissat"
    echo "Build it first: cd $BASE_SOLVER && ./configure && make -j4"
    exit 1
fi

mkdir -p "$RESULT_DIR"

# Build CNF file list (skip already completed)
CNF_LIST="$RESULT_DIR/cnf_file_list.txt"
> "$CNF_LIST"
SKIPPED=0
TOTAL=0
for cnf in $(ls "$BENCHMARK_PATH"/*.cnf 2>/dev/null | sort); do
    fname=$(basename "$cnf")
    if [ -f "$RESULT_DIR/${fname}.solving.log" ]; then
        SKIPPED=$((SKIPPED + 1))
    else
        echo "$fname" >> "$CNF_LIST"
    fi
    TOTAL=$((TOTAL + 1))
done

NUM_TASKS=$(wc -l < "$CNF_LIST")
if [ "$NUM_TASKS" -eq 0 ]; then
    echo "All $TOTAL benchmarks already evaluated ($SKIPPED skipped)."
    echo "Run with --collect to generate solving_times.json"
    exit 0
fi

echo "Submitting $NUM_TASKS tasks ($SKIPPED already done out of $TOTAL)"

# Create wrapper script
SCRIPT_PATH="$RESULT_DIR/run_solver_array.sh"
cat > "$SCRIPT_PATH" << 'ENDSCRIPT'
#!/bin/bash
CNF_LIST="__CNF_LIST__"
SOLVER="__SOLVER__"
BENCHMARK_PATH="__BENCHMARK_PATH__"
RESULT_DIR="__RESULT_DIR__"
TIMEOUT=__TIMEOUT__

GNU_TIME=$(which time 2>/dev/null)
if [ -z "$GNU_TIME" ]; then
    echo "ERROR: GNU time not found in PATH" >&2
    exit 1
fi

CNF_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CNF_LIST")
OUTPUT_FILE="${RESULT_DIR}/${CNF_FILE}.solving.log"

if [ -z "$CNF_FILE" ]; then
    echo "ERROR: No CNF file found for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

if [ -f "$OUTPUT_FILE" ]; then
    echo "Already completed: $CNF_FILE"
    exit 0
fi

echo "Running solver on $CNF_FILE (array task $SLURM_ARRAY_TASK_ID)"

"$GNU_TIME" -f "%U %S" -o "${OUTPUT_FILE}.time" \
    timeout ${TIMEOUT}s "$SOLVER" "$BENCHMARK_PATH/$CNF_FILE" > "$OUTPUT_FILE" 2>&1
EXIT_CODE=$?

if [ -f "${OUTPUT_FILE}.time" ]; then
    CPU_TIME=$(awk '{printf "%.6f", $1 + $2}' "${OUTPUT_FILE}.time")
    rm -f "${OUTPUT_FILE}.time"
else
    CPU_TIME=""
fi

if [ $EXIT_CODE -eq 124 ]; then
    echo "TIMEOUT after ${TIMEOUT}s" >> "$OUTPUT_FILE"
elif [ -z "$CPU_TIME" ]; then
    echo "ERROR: process killed (OOM or SLURM limit)" >> "$OUTPUT_FILE"
else
    echo "c process-time: $CPU_TIME seconds" >> "$OUTPUT_FILE"
fi

echo "Solver finished with exit code $EXIT_CODE"
exit $EXIT_CODE
ENDSCRIPT

# Fill in placeholders
sed -i "s|__CNF_LIST__|$(realpath "$CNF_LIST")|g" "$SCRIPT_PATH"
sed -i "s|__SOLVER__|$(realpath "$BASE_SOLVER/kissat")|g" "$SCRIPT_PATH"
sed -i "s|__BENCHMARK_PATH__|$(realpath "$BENCHMARK_PATH")|g" "$SCRIPT_PATH"
sed -i "s|__RESULT_DIR__|$(realpath "$RESULT_DIR")|g" "$SCRIPT_PATH"
sed -i "s|__TIMEOUT__|$TIMEOUT|g" "$SCRIPT_PATH"
chmod +x "$SCRIPT_PATH"

# Submit
ARRAY_RANGE="0-$((NUM_TASKS - 1))%${MAX_CONCURRENT}"
sbatch \
    --account="$ACCOUNT" \
    --job-name=base_eval \
    --output="$RESULT_DIR/slurm_array_%a.log" \
    --mem="$MEM" \
    --time="$WALL_TIME" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --array="$ARRAY_RANGE" \
    "$SCRIPT_PATH"

echo ""
echo "After jobs complete, run:  bash scripts/run_base_solver_eval.sh --collect"
