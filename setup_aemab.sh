#!/bin/bash
# Setup AE_kissatMAB solver

set -euo pipefail

BASE_SOLVER=$(python3 -c "import yaml, os; c=yaml.safe_load(open('path_config.yaml')); print(os.path.expanduser(c['base_solver']))")

tar -xf AE_kissat2025_MAB.tar.xz -C "$(dirname "$BASE_SOLVER")"
mv "$(dirname "$BASE_SOLVER")/AE_kissat2025_MAB"/* "$BASE_SOLVER"/
rm -rf "$(dirname "$BASE_SOLVER")/AE_kissat2025_MAB"
cd "$BASE_SOLVER"
rm makefile
cp "$(dirname "$BASE_SOLVER")/../function_registry.yaml" "$BASE_SOLVER/"

# Remove broken symlink in src/makefile (points to /home/iser/... from original machine).
# [ -f ] returns false for broken symlinks, so configure's own removal check misses it,
# causing its final `ln -s` to fail with "File exists".
rm -f "$BASE_SOLVER/src/makefile"

./configure

# Replace the symlink configure just created with a regular file copy.
# copytree(symlinks=True) would otherwise propagate an absolute symlink that
# points back to the base solver, breaking generated solver builds.
rm -f "$BASE_SOLVER/src/makefile"
cp "$BASE_SOLVER/makefile" "$BASE_SOLVER/src/makefile"
