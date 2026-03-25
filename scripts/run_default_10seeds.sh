#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
HOST_TAG="${SCRAW_MACHINE_TAG:-$(hostname -s 2>/dev/null || echo machine)}"

PYTHON_BIN="${SCRAW_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
CONFIG_PATH="${SCRAW_CONFIG:-${REPO_ROOT}/configs/default_scraw.json}"
DATA_PATH="${SCRAW_DATA_PATH:-${REPO_ROOT}/data/baron_human_pancreas.h5ad}"
DEVICE="${SCRAW_DEVICE:-cuda}"
SEEDS_CSV="${SCRAW_SEEDS_CSV:-1,42,43,44,45,46,47,48,49,50}"
OUTPUT_ROOT="${SCRAW_OUTPUT_ROOT:-${REPO_ROOT}/results/default_10seeds_${HOST_TAG}}"
LOG_DIR="${SCRAW_LOG_DIR:-${REPO_ROOT}/logs_10seeds}"
LOG_FILE="${SCRAW_LOG_FILE:-${LOG_DIR}/run_default_10seeds_${HOST_TAG}.log}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

export SCRAW_REPO_ROOT="${REPO_ROOT}"
export SCRAW_CONFIG_PATH="${CONFIG_PATH}"
export SCRAW_DATA_PATH="${DATA_PATH}"
export SCRAW_DEVICE="${DEVICE}"
export SCRAW_SEEDS_CSV="${SEEDS_CSV}"
export SCRAW_OUTPUT_ROOT="${OUTPUT_ROOT}"
export SCRAW_MACHINE_TAG="${HOST_TAG}"

{
  echo "=========================================="
  echo "  scRAW default 10-seed run"
  echo "  machine: ${HOST_TAG}"
  echo "  start: $(date)"
  echo "  config: ${CONFIG_PATH}"
  echo "  data: ${DATA_PATH}"
  echo "  device: ${DEVICE}"
  echo "  output_root: ${OUTPUT_ROOT}"
  echo "  seeds: ${SEEDS_CSV}"
  echo "=========================================="

  "${PYTHON_BIN}" - <<'PY'
import json
import os
import sys
from pathlib import Path

repo_root = Path(os.environ["SCRAW_REPO_ROOT"]).resolve()
sys.path.insert(0, str(repo_root / "src"))

from scraw import load_config, run_pipeline

config_path = Path(os.environ["SCRAW_CONFIG_PATH"]).resolve()
data_path = Path(os.environ["SCRAW_DATA_PATH"]).resolve()
device = str(os.environ["SCRAW_DEVICE"])
machine_tag = str(os.environ["SCRAW_MACHINE_TAG"])
output_root = Path(os.environ["SCRAW_OUTPUT_ROOT"]).resolve()
seeds = [int(token.strip()) for token in os.environ["SCRAW_SEEDS_CSV"].split(",") if token.strip()]

rows = []
for seed in seeds:
    seed_output = output_root / f"seed_{seed}"
    results_json = seed_output / "results" / "results.json"

    if results_json.exists():
        payload = json.loads(results_json.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        row = {
            "machine": machine_tag,
            "seed": seed,
            "status": "reused_existing",
            "strict_repro": None,
            "NMI": metrics.get("NMI"),
            "ARI": metrics.get("ARI"),
            "ACC": metrics.get("ACC"),
            "RareACC": metrics.get("RareACC"),
            "Silhouette": metrics.get("Silhouette"),
            "n_clusters_found": metrics.get("n_clusters_found"),
            "output_dir": str(seed_output),
        }
        rows.append(row)
        print(f"SEED={seed} {json.dumps(row, sort_keys=True)}", flush=True)
        continue

    config = load_config(config_path)
    config.runtime.seed = seed
    config.runtime.device = device
    config.data.data_path = str(data_path)
    config.data.output_dir = str(seed_output)

    result = run_pipeline(config)
    metrics = result["metrics"]
    row = {
        "machine": machine_tag,
        "seed": seed,
        "status": "completed",
        "strict_repro": result["config"]["runtime"]["strict_repro"],
        "NMI": metrics["NMI"],
        "ARI": metrics["ARI"],
        "ACC": metrics["ACC"],
        "RareACC": metrics["RareACC"],
        "Silhouette": metrics["Silhouette"],
        "n_clusters_found": metrics["n_clusters_found"],
        "output_dir": result["output_dir"],
    }
    rows.append(row)
    print(f"SEED={seed} {json.dumps(row, sort_keys=True)}", flush=True)

print("SUMMARY_JSON_START", flush=True)
print(json.dumps(rows, indent=2, sort_keys=True), flush=True)
print("SUMMARY_JSON_END", flush=True)
PY

  echo "=========================================="
  echo "  completed: $(date)"
  echo "  log_file: ${LOG_FILE}"
  echo "=========================================="
} 2>&1 | tee "${LOG_FILE}"
