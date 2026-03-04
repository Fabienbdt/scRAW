#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/Users/fabienbidet/miniforge3/envs/scrbenchmark/bin/python}"
SEARCH_ROOT="${SEARCH_ROOT:-${ROOT_DIR}/results/hparam_search/baron_best_baron_human_pancreas_metrics_only_20260304_005801}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SEARCH_ROOT}/ablation_loss_impact}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-10}"
OVERWRITE="${OVERWRITE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
MAX_RUNS="${MAX_RUNS:-0}"
DRY_RUN="${DRY_RUN:-0}"

LOG_FILE="${LOG_FILE:-${SEARCH_ROOT}/nohup_regenerate_ablation_figures_$(date +%Y%m%d_%H%M%S).log}"

CMD=(
  "${PYTHON_BIN}"
  "${ROOT_DIR}/scripts/run_regenerate_ablation_figures.py"
  "--search-root" "${SEARCH_ROOT}"
  "--output-root" "${OUTPUT_ROOT}"
  "--snapshot-interval" "${SNAPSHOT_INTERVAL}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=("--overwrite")
else
  CMD+=("--no-overwrite")
fi

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  CMD+=("--skip-existing")
fi

if [[ "${MAX_RUNS}" != "0" ]]; then
  CMD+=("--max-runs" "${MAX_RUNS}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=("--dry-run")
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "Ablation regeneration started in background."
echo "PID: ${PID}"
echo "Search root: ${SEARCH_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Log file: ${LOG_FILE}"
echo "Follow logs with:"
echo "  tail -f \"${LOG_FILE}\""
