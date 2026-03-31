#!/bin/bash
# Setup a kissat-based solver as the base solver for LLM-SAT.
#
# Extracts the tarball, moves contents to the configured base_solver path,
# copies the function registry, builds the solver.
#
# Usage:
#   bash setup_solver.sh <solver.tar.xz>
#
# Examples:
#   bash setup_solver.sh Kissat-CURE.tar.xz
#   bash setup_solver.sh AE_kissat2025_MAB.tar.xz

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <solver.tar.xz>"
    exit 1
fi

TARBALL="$1"
if [ ! -f "$TARBALL" ]; then
    echo "Error: $TARBALL not found"
    exit 1
fi

BASE_SOLVER=$(python3 -c "import yaml, os; c=yaml.safe_load(open('path_config.yaml')); print(os.path.expanduser(c['base_solver']))")
PARENT_DIR="$(dirname "$BASE_SOLVER")"

# Clean out old base solver
if [ -d "$BASE_SOLVER" ]; then
    echo "Removing old base solver at $BASE_SOLVER"
    rm -rf "$BASE_SOLVER"
fi
mkdir -p "$BASE_SOLVER"

# Extract to a temp dir, then move contents regardless of inner directory name
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

tar -xf "$TARBALL" -C "$TMPDIR"

# Find the extracted top-level directory (or files if tarball has no wrapper dir)
ENTRIES=("$TMPDIR"/*)
if [ ${#ENTRIES[@]} -eq 1 ] && [ -d "${ENTRIES[0]}" ]; then
    # Single directory inside tarball — move its contents
    mv "${ENTRIES[0]}"/* "$BASE_SOLVER"/
else
    # Files directly in tarball root
    mv "$TMPDIR"/* "$BASE_SOLVER"/
fi

cd "$BASE_SOLVER"

# Copy function registry from repo root
REPO_ROOT="$(cd "$PARENT_DIR/.." && pwd)"
if [ -f "$REPO_ROOT/function_registry.yaml" ]; then
    cp "$REPO_ROOT/function_registry.yaml" "$BASE_SOLVER/"
fi

# Clean up stale makefiles/symlinks that may point to the original author's machine
rm -f makefile src/makefile

./configure

# Replace the symlink configure creates with a regular file copy.
# copytree(symlinks=True) would otherwise propagate an absolute symlink that
# points back to the base solver, breaking generated solver builds.
rm -f "$BASE_SOLVER/src/makefile"
cp "$BASE_SOLVER/makefile" "$BASE_SOLVER/src/makefile"

make -j"$(nproc)"
cp build/kissat kissat

echo ""
echo "Base solver ready at $BASE_SOLVER"
echo "  Binary: $BASE_SOLVER/kissat"
echo ""
echo "Next: re-index your target function(s):"
echo "  python scripts/configure_target.py <function_name>"
