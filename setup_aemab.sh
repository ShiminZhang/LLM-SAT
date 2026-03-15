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
./configure

# Fix broken symlink in src/makefile (points to non-existent path from original machine)
rm -f "$BASE_SOLVER/src/makefile"
cp "$BASE_SOLVER/makefile" "$BASE_SOLVER/src/makefile"
