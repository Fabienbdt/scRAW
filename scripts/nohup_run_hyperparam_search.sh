#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/Users/fabienbidet/miniforge3/envs/scrbenchmark/bin/python}"
DATA_PATH="${DATA_PATH:-/Users/fabienbidet/Documents/MASTER 2/STAGE/SCRBenchmark/data/baron_human_pancreas.h5ad}"
PRESET="${PRESET:-baron_best}"
DEVICE="${DEVICE:-cpu}"
SEED="${SEED:-42}"
SEARCH_GROUPS="${SEARCH_GROUPS:-baseline,single,pairwise,batch}"
MAX_RUNS="${MAX_RUNS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DATA_STEM="$(basename "${DATA_PATH}")"
DATA_STEM="${DATA_STEM%.h5ad}"

RUN_NAME="${PRESET}_${DATA_STEM}_metrics_only_${TIMESTAMP}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/results/hparam_search/${RUN_NAME}}"
LOG_FILE="${OUTPUT_ROOT}/nohup.log"

mkdir -p "${OUTPUT_ROOT}"

CMD=(
  "${PYTHON_BIN}"
  "${ROOT_DIR}/scripts/run_hyperparam_search.py"
  "--preset" "${PRESET}"
  "--data" "${DATA_PATH}"
  "--output-root" "${OUTPUT_ROOT}"
  "--device" "${DEVICE}"
  "--seed" "${SEED}"
  "--groups" "${SEARCH_GROUPS}"
)

if [[ "${MAX_RUNS}" != "0" ]]; then
  CMD+=("--max-runs" "${MAX_RUNS}")
fi

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  CMD+=("--skip-existing")
else
  CMD+=("--no-skip-existing")
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
  echo "Command:"
  printf ' %q' "${CMD[@]}"
  echo
} > "${OUTPUT_ROOT}/launch_info.txt"

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "Hyperparameter search started in background."
echo "PID: ${PID}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Log file: ${LOG_FILE}"
echo "Follow logs with:"
echo "  tail -f \"${LOG_FILE}\""
