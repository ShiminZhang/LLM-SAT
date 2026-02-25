#!/bin/bash
CNF_LIST="solvers/gemini_trial5_gen1_iter1/members/algorithm_9b5447f5bbd74b58a712f4b8a9c1e9f4e703a876ff6c5a50b6f90b5a2428c5e3/result/code_ac9e1d323d4d961c7bb075de8a3427349458184e5d5b37d60f1a801fb347d960//cnf_file_list.txt"
SOLVER="solvers/gemini_trial5_gen1_iter1/members/algorithm_9b5447f5bbd74b58a712f4b8a9c1e9f4e703a876ff6c5a50b6f90b5a2428c5e3/code_ac9e1d323d4d961c7bb075de8a3427349458184e5d5b37d60f1a801fb347d960//kissat"
BENCHMARK_PATH="data/benchmarks/satcomp2025"
RESULT_DIR="solvers/gemini_trial5_gen1_iter1/members/algorithm_9b5447f5bbd74b58a712f4b8a9c1e9f4e703a876ff6c5a50b6f90b5a2428c5e3/result/code_ac9e1d323d4d961c7bb075de8a3427349458184e5d5b37d60f1a801fb347d960/"
TIMEOUT=5000

CNF_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CNF_LIST")
OUTPUT_FILE="${RESULT_DIR}/${CNF_FILE}.solving.log"

if [ -z "$CNF_FILE" ]; then
    echo "ERROR: No CNF file found for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Skip if already done
if [ -f "$OUTPUT_FILE" ]; then
    echo "Already completed: $CNF_FILE"
    exit 0
fi

echo "Running solver on $CNF_FILE (array task $SLURM_ARRAY_TASK_ID)"

# Run solver with timeout, capturing wall-clock time
START_TIME=$(date +%s.%N)
timeout ${TIMEOUT}s "$SOLVER" "$BENCHMARK_PATH/$CNF_FILE" > "$OUTPUT_FILE" 2>&1
EXIT_CODE=$?
END_TIME=$(date +%s.%N)
ELAPSED=$(awk "BEGIN {printf \"%.6f\", $END_TIME - $START_TIME}")

if [ $EXIT_CODE -eq 124 ]; then
    echo "TIMEOUT after ${TIMEOUT}s" >> "$OUTPUT_FILE"
else
    echo "c process-time: $ELAPSED seconds" >> "$OUTPUT_FILE"
fi

echo "Solver finished with exit code $EXIT_CODE"
exit $EXIT_CODE
