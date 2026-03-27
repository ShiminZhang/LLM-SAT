#!/bin/bash
# Check solving logs of best solvers for segmentation faults.
# Usage: ./scripts/check_segfault.sh <tag>
#   e.g. ./scripts/check_segfault.sh rescale_3_iter0

set -euo pipefail

TAG="${1:?Usage: $0 <tag>}"
BASEDIR="$(cd "$(dirname "$0")/.." && pwd)"
PAR2_FILE="$BASEDIR/outputs/${TAG}/par2_scores.txt"
SOLVERS_DIR="$BASEDIR/solvers/${TAG}"

if [[ ! -f "$PAR2_FILE" ]]; then
    echo "ERROR: par2_scores.txt not found at $PAR2_FILE"
    exit 1
fi

SEGFAULT_PATTERN='[Ss]eg(mentation)?\s*[Ff]ault|SIGSEGV|signal 11'

found_any_segfault=0
best_count=0

while IFS= read -r line; do
    # Extract role [L] or [M], short algo hash, short code hash
    role=$(echo "$line" | grep -oP '\[[LM]\]')
    algo_short=$(echo "$line" | awk '{print $2}')
    code_short=$(echo "$line" | awk '{print $3}')

    # Search both leaders and members for the algorithm directory
    algo_dir=""
    subdir=""
    for candidate in leaders members; do
        match=$(find "$SOLVERS_DIR/$candidate" -maxdepth 1 -type d -name "algorithm_${algo_short}*" 2>/dev/null | head -1)
        if [[ -n "$match" ]]; then
            algo_dir="$match"
            subdir="$candidate"
            break
        fi
    done
    if [[ -z "$algo_dir" ]]; then
        echo "WARNING: No algorithm dir found for ${algo_short} in leaders or members"
        continue
    fi

    # Find the full code result directory by prefix match
    result_code_dir=$(find "$algo_dir/result" -maxdepth 1 -type d -name "code_${code_short}*" 2>/dev/null | head -1)
    if [[ -z "$result_code_dir" ]]; then
        echo "WARNING: No result/code dir found for ${code_short} in $algo_dir/result"
        continue
    fi

    best_count=$((best_count + 1))
    algo_full=$(basename "$algo_dir" | sed 's/^algorithm_//')
    code_full=$(basename "$result_code_dir" | sed 's/^code_//')

    echo "========================================"
    echo "BEST solver #${best_count}: [$subdir] algo=${algo_short}... code=${code_short}..."
    echo "  Dir: $result_code_dir"

    log_count=0
    segfault_logs=0
    for logfile in "$result_code_dir"/*.solving.log; do
        [[ -f "$logfile" ]] || continue
        log_count=$((log_count + 1))
        if grep -qEi "$SEGFAULT_PATTERN" "$logfile" 2>/dev/null; then
            segfault_logs=$((segfault_logs + 1))
            found_any_segfault=1
            logname=$(basename "$logfile")
            echo "  ** SEGFAULT in: $logname"
            grep -Eni "$SEGFAULT_PATTERN" "$logfile" | head -5 | sed 's/^/     /'
        fi
    done

    if [[ $segfault_logs -eq 0 ]]; then
        echo "  OK: No segfaults found in $log_count log(s)"
    else
        echo "  FOUND segfaults in $segfault_logs / $log_count log(s)"
    fi

done < <(grep '<-- BEST' "$PAR2_FILE")

echo ""
echo "========================================"
echo "Summary: checked $best_count best solver(s)"
if [[ $found_any_segfault -eq 1 ]]; then
    echo "Result: SEGFAULTS DETECTED (see above)"
    exit 1
else
    echo "Result: All clean, no segfaults found"
    exit 0
fi
