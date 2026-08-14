#!/bin/bash
# Build colleague-supplied solvers from the DB against an alt base solver,
# then submit full evals. Mirrors the approach used for ReKCCkr1_iter0.
#
# Edit the ALT_BASE / TAG / IDS section below for each new batch.
#
# Approach:
#   1. Temporarily swap path_config.yaml's base_solver to the alt base.
#   2. evaluation.py --algorithm_id ... --build-only  (per solver)
#   3. Restore path_config.yaml.
#   4. scripts/evaluate_solver.py <built_path>        (per solver)
#
set -euo pipefail
cd /scratch/meru/LLM-SAT
source ~/general/bin/activate
export PYTHONPATH=src

# ---- batch config ----
ALT_BASE_REL="solvers/kissat_corephase_coreward"
TAG="ReKCClsuv1_iter0"
# (algo_id, code_id) pairs
ALGOS=(
  "02b3cc13addcb2c330db7ec84efb7f384fc10d579150f64277e44298d9064741:286856fe0c78bc9965ab660c3aa55472b2ff0bbc67cde1af84c02bae47d2f5cf"
  "96e80276516987099b2f6623700f4be82568510cc1a5395028109cebd03824e9:073b6e7531c82e593f63b21af5837991345b86b0514c61a09f23c93e6a6feaa3"
)
# ----------------------

# 1. Swap path_config.yaml + function_registry.yaml
#    (evaluation.py loads the registry from solvers/base/function_registry.yaml
#     regardless of base_solver, so we need to swap that too.)
ORIG_BASE=$(grep '^base_solver:' path_config.yaml | sed 's/^base_solver:[[:space:]]*//')
ABS_ALT="$HOME/scratch/LLM-SAT/${ALT_BASE_REL}"
echo "Original base_solver: $ORIG_BASE"
echo "Swapping to alt base: $ABS_ALT"

cp path_config.yaml path_config.yaml.bak
sed -i "s|^base_solver:.*|base_solver: $ABS_ALT|" path_config.yaml

cp solvers/base/function_registry.yaml solvers/base/function_registry.yaml.bak
cp "${ALT_BASE_REL}/function_registry.yaml" solvers/base/function_registry.yaml

# Restore both files on exit (success OR failure)
trap 'mv path_config.yaml.bak path_config.yaml 2>/dev/null; mv solvers/base/function_registry.yaml.bak solvers/base/function_registry.yaml 2>/dev/null; echo "Restored path_config.yaml + solvers/base/function_registry.yaml"' EXIT

# 2. Build each solver
for pair in "${ALGOS[@]}"; do
  algo_id="${pair%%:*}"
  code_id="${pair##*:}"
  echo ""
  echo "=== Building algo=${algo_id:0:12} code=${code_id:0:12} ==="
  python src/llmsat/pipelines/evaluation.py \
    --algorithm_id "$algo_id" \
    --code_id "$code_id" \
    --build-only \
    --generation_tag "$TAG" 2>&1 | tail -5
done

# 3. Restore path_config (handled by trap)

# 4. Submit full evals
echo ""
echo "=== Submitting full evals ==="
for pair in "${ALGOS[@]}"; do
  algo_id="${pair%%:*}"
  code_id="${pair##*:}"
  # Built solver path: solvers/$TAG/{leaders|members}/algorithm_<algo_id>/code_<code_id>/
  built=$(find "solvers/$TAG" -maxdepth 4 -type d -name "code_${code_id}" 2>/dev/null | head -1)
  if [ -z "$built" ]; then
    echo "  [SKIP] no built dir for code_${code_id:0:12}"
    continue
  fi
  if [ ! -f "$built/build/kissat" ]; then
    echo "  [SKIP] $built/build/kissat missing — build failed"
    continue
  fi
  echo "  Submitting $built ..."
  python scripts/evaluate_solver.py "$built" 2>&1 | grep -E "Submitted job array" | tail -1
done
