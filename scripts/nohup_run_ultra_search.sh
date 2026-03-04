#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/venv/bin/python}"
DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/baron_human_pancreas.h5ad}"
PRESET="${PRESET:-baron_best}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
N_TRIALS="${N_TRIALS:-300}"
N_SEEDS="${N_SEEDS:-1}"
DRY_RUN="${DRY_RUN:-0}"
TIMEOUT="${TIMEOUT:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DATA_STEM="$(basename "${DATA_PATH}")"
DATA_STEM="${DATA_STEM%.h5ad}"

RUN_NAME="${PRESET}_${DATA_STEM}_ultra_search_${TIMESTAMP}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/results/ultra_hparam_search/${RUN_NAME}}"
LOG_FILE="${OUTPUT_ROOT}/nohup.log"

mkdir -p "${OUTPUT_ROOT}"

CMD=(
  "env"
  "PYTHONUNBUFFERED=1"
  "${PYTHON_BIN}"
  "${ROOT_DIR}/scripts/run_hyperparam_search.py"
  "--preset" "${PRESET}"
  "--data" "${DATA_PATH}"
  "--output-root" "${OUTPUT_ROOT}"
  "--device" "${DEVICE}"
  "--seed" "${SEED}"
  "--n-trials" "${N_TRIALS}"
  "--n-seeds" "${N_SEEDS}"
)

if [[ -n "${TIMEOUT}" ]]; then
  CMD+=("--timeout" "${TIMEOUT}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=("--dry-run")
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

{
  echo "Launch time: $(date)"
  echo "Output root: ${OUTPUT_ROOT}"
  echo "N trials: ${N_TRIALS}"
  echo "N seeds: ${N_SEEDS}"
  echo "Device: ${DEVICE}"
  echo "Command:"
  printf ' %q' "${CMD[@]}"
  echo
} > "${OUTPUT_ROOT}/launch_info.txt"

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "Ultra hyperparameter search started in background."
echo "PID: ${PID}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Log file: ${LOG_FILE}"
echo "Follow logs with:"
echo "  tail -f \"${LOG_FILE}\""
