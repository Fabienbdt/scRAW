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
REFINE_TOP_K="${REFINE_TOP_K:-24}"
MAX_RUNS_SELECTION="${MAX_RUNS_SELECTION:-random}"
SEARCH_GROUPS="${SEARCH_GROUPS:-baseline,single,pairwise,batch,dann}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DANN_CONTROLS="${DANN_CONTROLS:-0}"
DRY_RUN="${DRY_RUN:-0}"
TIMEOUT="${TIMEOUT:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "Error: no Python interpreter found (PYTHON_BIN='${PYTHON_BIN}')." >&2
    exit 1
  fi
fi

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Error: DATA_PATH not found: ${DATA_PATH}" >&2
  exit 1
fi

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
  "CUBLAS_WORKSPACE_CONFIG=:4096:8"
  "${PYTHON_BIN}"
  "${ROOT_DIR}/scripts/run_hyperparam_search.py"
  "--preset" "${PRESET}"
  "--data" "${DATA_PATH}"
  "--output-root" "${OUTPUT_ROOT}"
  "--device" "${DEVICE}"
  "--seed" "${SEED}"
  "--groups" "${SEARCH_GROUPS}"
  "--max-runs-selection" "${MAX_RUNS_SELECTION}"
  "--n-seeds" "${N_SEEDS}"
)

if [[ "${N_TRIALS}" != "0" ]]; then
  CMD+=("--max-runs" "${N_TRIALS}")
fi

if [[ "${REFINE_TOP_K}" != "0" ]]; then
  CMD+=("--refine-top-k" "${REFINE_TOP_K}")
fi

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  CMD+=("--skip-existing")
else
  CMD+=("--no-skip-existing")
fi

if [[ "${DANN_CONTROLS}" == "1" ]]; then
  CMD+=("--dann-controls")
else
  CMD+=("--no-dann-controls")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=("--dry-run")
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

# Ultra search is metrics-only by design: never run figure-generating ablation.
CMD+=("--no-loss-ablation")

{
  echo "Launch time: $(date)"
  echo "Output root: ${OUTPUT_ROOT}"
  echo "N trials: ${N_TRIALS}"
  echo "N seeds: ${N_SEEDS}"
  echo "Refine top-k: ${REFINE_TOP_K}"
  echo "Max-runs selection: ${MAX_RUNS_SELECTION}"
  echo "Search groups: ${SEARCH_GROUPS}"
  echo "Device: ${DEVICE}"
  if [[ -n "${TIMEOUT}" ]]; then
    echo "Timeout requested: ${TIMEOUT}"
  fi
  echo "Command:"
  printf ' %q' "${CMD[@]}"
  echo
} > "${OUTPUT_ROOT}/launch_info.txt"

NOHUP_CMD=("${CMD[@]}")
if [[ -n "${TIMEOUT}" ]]; then
  if command -v timeout >/dev/null 2>&1; then
    NOHUP_CMD=("timeout" "${TIMEOUT}" "${NOHUP_CMD[@]}")
  else
    echo "Warning: 'timeout' command not found; TIMEOUT ignored." >> "${OUTPUT_ROOT}/launch_info.txt"
  fi
fi

nohup "${NOHUP_CMD[@]}" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "Ultra hyperparameter search started in background."
echo "PID: ${PID}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Log file: ${LOG_FILE}"
echo "Follow logs with:"
echo "  tail -f \"${LOG_FILE}\""
