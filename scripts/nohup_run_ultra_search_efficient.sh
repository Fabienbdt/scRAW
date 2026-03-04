#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/venv/bin/python}"
DATA_PATH="${DATA_PATH:-${ROOT_DIR}/data/baron_human_pancreas.h5ad}"
PRESET="${PRESET:-baron_best}"
DEVICE="${DEVICE:-cuda}"
BASE_SEED="${BASE_SEED:-42}"
SEED_STEP="${SEED_STEP:-97}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

# Stage 1 (exploration)
N_TRIALS_STAGE1="${N_TRIALS_STAGE1:-450}"
N_SEEDS_STAGE1="${N_SEEDS_STAGE1:-1}"
SAMPLER_STAGE1="${SAMPLER_STAGE1:-tpe}"
N_STARTUP_TRIALS_STAGE1="${N_STARTUP_TRIALS_STAGE1:-32}"
DANN_MODE_STAGE1="${DANN_MODE_STAGE1:-on}"
FINAL_CLUSTERING_MODE_STAGE1="${FINAL_CLUSTERING_MODE_STAGE1:-mixed}"
TARGET_CLUSTERS="${TARGET_CLUSTERS:-14}"
TIMEOUT_STAGE1="${TIMEOUT_STAGE1:-}"

# Stage 2 (robust refinement)
RUN_STAGE2="${RUN_STAGE2:-1}"
TOP_K_STAGE2="${TOP_K_STAGE2:-10}"
N_SEEDS_STAGE2="${N_SEEDS_STAGE2:-3}"
TIMEOUT_STAGE2="${TIMEOUT_STAGE2:-}"

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

RUN_NAME="${PRESET}_${DATA_STEM}_ultra_efficient_${TIMESTAMP}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/results/ultra_hparam_search/${RUN_NAME}}"
LOG_FILE="${OUTPUT_ROOT}/nohup.log"
LAUNCH_INFO="${OUTPUT_ROOT}/launch_info.txt"

mkdir -p "${OUTPUT_ROOT}"

STAGE1_OUT="${OUTPUT_ROOT}/stage1_optuna"
STAGE2_OUT="${OUTPUT_ROOT}/stage2_refine_topk"

CMD_STAGE1=(
  "env"
  "PYTHONUNBUFFERED=1"
  "${PYTHON_BIN}"
  "${ROOT_DIR}/scripts/run_optuna_ultra_search.py"
  "--preset" "${PRESET}"
  "--data" "${DATA_PATH}"
  "--output-root" "${STAGE1_OUT}"
  "--device" "${DEVICE}"
  "--seed" "${BASE_SEED}"
  "--seed-step" "${SEED_STEP}"
  "--n-seeds" "${N_SEEDS_STAGE1}"
  "--n-trials" "${N_TRIALS_STAGE1}"
  "--sampler" "${SAMPLER_STAGE1}"
  "--n-startup-trials" "${N_STARTUP_TRIALS_STAGE1}"
  "--dann-mode" "${DANN_MODE_STAGE1}"
  "--final-clustering-mode" "${FINAL_CLUSTERING_MODE_STAGE1}"
  "--target-clusters" "${TARGET_CLUSTERS}"
  "--study-name" "scraw_ultra_stage1_${TIMESTAMP}"
)

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  CMD_STAGE1+=("--skip-existing")
else
  CMD_STAGE1+=("--no-skip-existing")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  CMD_STAGE1+=("--dry-run")
fi

if [[ -n "${TIMEOUT_STAGE1}" ]]; then
  CMD_STAGE1+=("--timeout" "${TIMEOUT_STAGE1}")
fi

CMD_STAGE2=(
  "env"
  "PYTHONUNBUFFERED=1"
  "${PYTHON_BIN}"
  "${ROOT_DIR}/scripts/run_optuna_refine_topk.py"
  "--search-root" "${STAGE1_OUT}"
  "--preset" "${PRESET}"
  "--data" "${DATA_PATH}"
  "--output-root" "${STAGE2_OUT}"
  "--python-bin" "${PYTHON_BIN}"
  "--device" "${DEVICE}"
  "--base-seed" "${BASE_SEED}"
  "--seed-step" "${SEED_STEP}"
  "--n-seeds" "${N_SEEDS_STAGE2}"
  "--top-k" "${TOP_K_STAGE2}"
  "--target-clusters" "${TARGET_CLUSTERS}"
)

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  CMD_STAGE2+=("--skip-existing")
else
  CMD_STAGE2+=("--no-skip-existing")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  CMD_STAGE2+=("--dry-run")
fi

{
  echo "Launch time: $(date)"
  echo "Output root: ${OUTPUT_ROOT}"
  echo "Device: ${DEVICE}"
  echo "Base seed: ${BASE_SEED}"
  echo "Seed step: ${SEED_STEP}"
  echo "Skip existing: ${SKIP_EXISTING}"
  echo "Dry run: ${DRY_RUN}"
  echo ""
  echo "[Stage 1] Optuna exploration"
  echo "  N trials: ${N_TRIALS_STAGE1}"
  echo "  N seeds/trial: ${N_SEEDS_STAGE1}"
  echo "  Sampler: ${SAMPLER_STAGE1}"
  echo "  TPE startup: ${N_STARTUP_TRIALS_STAGE1}"
  echo "  DANN mode: ${DANN_MODE_STAGE1}"
  echo "  Final clustering mode: ${FINAL_CLUSTERING_MODE_STAGE1}"
  echo "  Target clusters: ${TARGET_CLUSTERS}"
  if [[ -n "${TIMEOUT_STAGE1}" ]]; then
    echo "  Timeout stage1: ${TIMEOUT_STAGE1} s"
  fi
  echo "  Command:"
  printf ' %q' "${CMD_STAGE1[@]}"
  echo ""
  echo ""
  echo "[Stage 2] Top-k multi-seed refinement"
  echo "  Run stage2: ${RUN_STAGE2}"
  echo "  Top-k: ${TOP_K_STAGE2}"
  echo "  N seeds/candidate: ${N_SEEDS_STAGE2}"
  if [[ -n "${TIMEOUT_STAGE2}" ]]; then
    echo "  Timeout stage2: ${TIMEOUT_STAGE2} s"
  fi
  echo "  Command:"
  printf ' %q' "${CMD_STAGE2[@]}"
  echo ""
} > "${LAUNCH_INFO}"

cat > "${OUTPUT_ROOT}/run_pipeline.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
EOF

{
  echo "cd \"${ROOT_DIR}\""
  printf ' %q' "${CMD_STAGE1[@]}"
  echo ""
  if [[ "${RUN_STAGE2}" == "1" ]]; then
    if [[ -n "${TIMEOUT_STAGE2}" ]]; then
      if command -v timeout >/dev/null 2>&1; then
        printf ' timeout %q' "${TIMEOUT_STAGE2}"
        printf ' %q' "${CMD_STAGE2[@]}"
        echo ""
      else
        printf ' %q' "${CMD_STAGE2[@]}"
        echo ""
      fi
    else
      printf ' %q' "${CMD_STAGE2[@]}"
      echo ""
    fi
  fi
} >> "${OUTPUT_ROOT}/run_pipeline.sh"

chmod +x "${OUTPUT_ROOT}/run_pipeline.sh"

nohup bash "${OUTPUT_ROOT}/run_pipeline.sh" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "Efficient ultra-search pipeline started in background."
echo "PID: ${PID}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Log file: ${LOG_FILE}"
echo "Follow logs with:"
echo "  tail -f \"${LOG_FILE}\""
