#!/bin/bash
CNF_LIST="solvers/gemini_trial5_gen1_iter1/members/algorithm_14747fd180ef447582c5b32c571c56ed7dcdb36082f85aad7e0fdf12034f1494/result/code_93212fd9f08edb8d995dbcd0b3127998117a34e39162aa43847498562c16a099//cnf_file_list.txt"
SOLVER="solvers/gemini_trial5_gen1_iter1/members/algorithm_14747fd180ef447582c5b32c571c56ed7dcdb36082f85aad7e0fdf12034f1494/code_93212fd9f08edb8d995dbcd0b3127998117a34e39162aa43847498562c16a099//kissat"
BENCHMARK_PATH="data/benchmarks/satcomp2025"
RESULT_DIR="solvers/gemini_trial5_gen1_iter1/members/algorithm_14747fd180ef447582c5b32c571c56ed7dcdb36082f85aad7e0fdf12034f1494/result/code_93212fd9f08edb8d995dbcd0b3127998117a34e39162aa43847498562c16a099/"
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
