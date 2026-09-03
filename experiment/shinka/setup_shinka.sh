#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${SAT_REPO_ROOT:-/scratch/s568zhan/LLM-SAT}"
SHINKA_ROOT="${SHINKA_ROOT:-${REPO_ROOT}/experiments/ShinkaEvolve}"
SHINKA_VENV="${SHINKA_VENV:-${SHINKA_ROOT}/.venv}"
OPENEVOLVE_ROOT="${OPENEVOLVE_ROOT:-/scratch/s568zhan/OpenEvolve}"

source "${REPO_ROOT}/scripts/activate_rorqual.sh"
if [[ ! -x "${SHINKA_VENV}/bin/python" ]]; then
  python -m venv "${SHINKA_VENV}"
fi
"${SHINKA_VENV}/bin/python" -m pip install -e "${SHINKA_ROOT}"
"${SHINKA_VENV}/bin/python" -m pip install -e "${OPENEVOLVE_ROOT}"
"${SHINKA_VENV}/bin/python" -c \
  'import openevolve, shinka; from shinka.core import ShinkaEvolveRunner; print("Shinka/OpenEvolve imports OK")'
