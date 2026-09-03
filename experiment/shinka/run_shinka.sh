#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${SAT_REPO_ROOT:-/scratch/s568zhan/LLM-SAT}"
SHINKA_ROOT="${SHINKA_ROOT:-${REPO_ROOT}/experiments/ShinkaEvolve}"
SHINKA_VENV="${SHINKA_VENV:-${SHINKA_ROOT}/.venv}"
OPENEVOLVE_ROOT="${OPENEVOLVE_ROOT:-/scratch/s568zhan/OpenEvolve}"
EXPERIMENT_DIR="${REPO_ROOT}/experiment/shinka"
RUN_ID="${SHINKA_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${SHINKA_RESULTS_DIR:-${EXPERIMENT_DIR}/runs/${RUN_ID}}"
source "$REPO_ROOT/experiment/common/protocol.sh"
export_comparison_protocol

if [[ "${1:-}" == "--wait-for-idle" ]]; then
  while squeue -h -u "${USER}" | grep -q .; do
    echo "Waiting for existing ${USER} jobs to leave the queue..."
    sleep 60
  done
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Results directory already exists: ${OUTPUT_DIR}" >&2
  exit 2
fi

HTTP_STATUS="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 \
  https://api.openai.com/v1/models || true)"
if [[ "${HTTP_STATUS}" == "000" ]]; then
  echo "Warning: API connectivity check did not succeed; continuing in case the" >&2
  echo "request merely lacked credentials. Run this controller on a login node." >&2
fi

mkdir -p "${OUTPUT_DIR}"
echo "$$" > "${OUTPUT_DIR}/controller.pid"

source "${REPO_ROOT}/scripts/activate_rorqual.sh"
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi

export SAT_REPO_ROOT="${REPO_ROOT}"
export OE_REPO_ROOT="${REPO_ROOT}"
export SHINKA_RESULTS_DIR="${OUTPUT_DIR}"
export PYTHONPATH="${REPO_ROOT}:${SHINKA_ROOT}:${OPENEVOLVE_ROOT}:${PYTHONPATH:-}"

if [[ ! -x "${SHINKA_VENV}/bin/python" ]]; then
  echo "Shinka is not installed. Run ${EXPERIMENT_DIR}/setup_shinka.sh first." >&2
  exit 3
fi
"${SHINKA_VENV}/bin/python" -c 'import shinka, openevolve' || {
  echo "Shinka/OpenEvolve imports failed; rerun setup_shinka.sh." >&2
  exit 3
}

cd "${REPO_ROOT}"
exec "${SHINKA_VENV}/bin/python" "${EXPERIMENT_DIR}/run_evo.py"
