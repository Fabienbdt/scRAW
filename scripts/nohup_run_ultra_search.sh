#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/venv/bin/python}"
DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/baron_human_pancreas.h5ad}"
PRESET="${PRESET:-baron_best}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
SEED_STEP="${SEED_STEP:-97}"
N_TRIALS="${N_TRIALS:-400}"
N_SEEDS="${N_SEEDS:-1}"
SAMPLER="${SAMPLER:-tpe}"
N_STARTUP_TRIALS="${N_STARTUP_TRIALS:-32}"
DANN_MODE="${DANN_MODE:-on}"
FINAL_CLUSTERING_MODE="${FINAL_CLUSTERING_MODE:-mixed}"
TARGET_CLUSTERS="${TARGET_CLUSTERS:-14}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
TIMEOUT="${TIMEOUT:-}"
STUDY_NAME="${STUDY_NAME:-}"

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
  "${ROOT_DIR}/scripts/run_optuna_ultra_search.py"
  "--preset" "${PRESET}"
  "--data" "${DATA_PATH}"
  "--output-root" "${OUTPUT_ROOT}"
  "--device" "${DEVICE}"
  "--seed" "${SEED}"
  "--seed-step" "${SEED_STEP}"
  "--n-seeds" "${N_SEEDS}"
  "--n-trials" "${N_TRIALS}"
  "--sampler" "${SAMPLER}"
  "--n-startup-trials" "${N_STARTUP_TRIALS}"
  "--dann-mode" "${DANN_MODE}"
  "--final-clustering-mode" "${FINAL_CLUSTERING_MODE}"
  "--target-clusters" "${TARGET_CLUSTERS}"
)

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  CMD+=("--skip-existing")
else
  CMD+=("--no-skip-existing")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=("--dry-run")
fi

if [[ -n "${TIMEOUT}" ]]; then
  CMD+=("--timeout" "${TIMEOUT}")
fi

if [[ -n "${STUDY_NAME}" ]]; then
  CMD+=("--study-name" "${STUDY_NAME}")
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

{
  echo "Launch time: $(date)"
  echo "Output root: ${OUTPUT_ROOT}"
  echo "N trials: ${N_TRIALS}"
  echo "N seeds: ${N_SEEDS}"
  echo "Seed step: ${SEED_STEP}"
  echo "Sampler: ${SAMPLER}"
  echo "TPE startup trials: ${N_STARTUP_TRIALS}"
  echo "DANN mode: ${DANN_MODE}"
  echo "Final clustering mode: ${FINAL_CLUSTERING_MODE}"
  echo "Target clusters (Leiden): ${TARGET_CLUSTERS}"
  echo "Device: ${DEVICE}"
  if [[ -n "${TIMEOUT}" ]]; then
    echo "Timeout requested: ${TIMEOUT} s"
  fi
  echo "Command:"
  printf ' %q' "${CMD[@]}"
  echo
} > "${OUTPUT_ROOT}/launch_info.txt"

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "Ultra Optuna hyperparameter search started in background."
echo "PID: ${PID}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Log file: ${LOG_FILE}"
echo "Follow logs with:"
echo "  tail -f \"${LOG_FILE}\""
