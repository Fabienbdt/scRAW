#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/Users/fabienbidet/miniforge3/envs/scrbenchmark/bin/python}"
SEARCH_ROOT="${SEARCH_ROOT:-${ROOT_DIR}/results/hparam_search/baron_best_baron_human_pancreas_metrics_only_20260304_005801}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SEARCH_ROOT}/ablation_loss_impact}"
DEVICE="${DEVICE:-}"
RESUME_EPOCH="${RESUME_EPOCH:-30}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-10}"
OVERWRITE="${OVERWRITE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"

LOG_FILE="${LOG_FILE:-${SEARCH_ROOT}/nohup_ablation_four_losses_$(date +%Y%m%d_%H%M%S).log}"

CMD=(
  "env"
  "PYTHONUNBUFFERED=1"
  "${PYTHON_BIN}"
  "${ROOT_DIR}/scripts/run_ablation_four_losses.py"
  "--search-root" "${SEARCH_ROOT}"
  "--output-root" "${OUTPUT_ROOT}"
  "--resume-epoch" "${RESUME_EPOCH}"
  "--snapshot-interval" "${SNAPSHOT_INTERVAL}"
)

if [[ -n "${DEVICE}" ]]; then
  CMD+=("--device" "${DEVICE}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=("--overwrite")
else
  CMD+=("--no-overwrite")
fi

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  CMD+=("--skip-existing")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=("--dry-run")
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "4-loss ablation started in background."
echo "PID: ${PID}"
echo "Search root: ${SEARCH_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Log file: ${LOG_FILE}"
echo "Follow logs with:"
echo "  tail -f \"${LOG_FILE}\""
