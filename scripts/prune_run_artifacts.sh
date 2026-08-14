#!/bin/bash
# Prune reclaimable build artifacts from past evolution-run archives.
#
# Every evaluated candidate under solvers/<tag>/ carries a full build/ object
# tree (and often duplicated source) that is dead weight once the binary is
# built and results are collected. This removes, per candidate solver copy:
#   - build/*.o, build/libkissat.a, build/tissat, build/kitten  (objects)
#   - build/kissat IF a root-level ./kissat binary exists       (dup binary)
# It never touches: root ./kissat binaries (needed for --skip-build re-eval),
# result/ logs, proofs, src/, or any JSON artifacts.
#
# Usage:
#   scripts/prune_run_artifacts.sh <solvers-dir> [<solvers-dir> ...]   # dry run
#   scripts/prune_run_artifacts.sh --execute <solvers-dir> [...]       # delete
#
# Example:
#   scripts/prune_run_artifacts.sh solvers/kissat_evolve_iter1
#   scripts/prune_run_artifacts.sh --execute solvers/kissat_evolve_iter1

set -euo pipefail

EXECUTE=0
if [ "${1:-}" = "--execute" ]; then
    EXECUTE=1
    shift
fi

if [ $# -lt 1 ]; then
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -20
    exit 1
fi

total_bytes=0
total_files=0

for root in "$@"; do
    if [ ! -d "$root" ]; then
        echo "skip (not a directory): $root" >&2
        continue
    fi
    echo "Scanning $root ..."
    while IFS= read -r -d '' build_dir; do
        solver_dir=$(dirname "$build_dir")
        # objects and secondary binaries are always prunable
        while IFS= read -r -d '' f; do
            sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
            total_bytes=$((total_bytes + sz))
            total_files=$((total_files + 1))
            [ "$EXECUTE" = "1" ] && rm -f "$f"
        done < <(find "$build_dir" -maxdepth 1 \( -name '*.o' -o -name 'libkissat.a' -o -name 'tissat' -o -name 'kitten' \) -type f -print0)
        # build/kissat only if the root binary duplicate exists
        if [ -f "$build_dir/kissat" ] && [ -f "$solver_dir/kissat" ]; then
            sz=$(stat -c%s "$build_dir/kissat" 2>/dev/null || echo 0)
            total_bytes=$((total_bytes + sz))
            total_files=$((total_files + 1))
            [ "$EXECUTE" = "1" ] && rm -f "$build_dir/kissat"
        fi
    done < <(find "$root" -type d -name build -print0)
done

human=$(numfmt --to=iec "$total_bytes" 2>/dev/null || echo "${total_bytes} bytes")
if [ "$EXECUTE" = "1" ]; then
    echo "Deleted $total_files files, reclaimed $human"
else
    echo "DRY RUN: would delete $total_files files, reclaiming $human"
    echo "Re-run with --execute to delete."
fi
