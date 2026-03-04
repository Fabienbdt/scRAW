#!/usr/bin/env python3
"""Ultra-comprehensive Optuna hyperparameter search for standalone scRAW.

This runner executes scRAW through the public CLI for each trial so that:
1) preprocessing/training/evaluation stay identical to standard project runs,
2) results are reproducible and directly comparable to existing outputs.

Design goals:
- broad search space (architecture, reconstruction, weighted loss, triplet, DANN),
- conditional sampling (DANN params only when DANN is enabled),
- final clustering choice for scoring: HDBSCAN or Leiden target-k optimized in CLI,
- robust persistence (SQLite + per-trial artifacts + global CSV summaries),
- metrics-only search (no figure generation) for large remote GPU sweeps.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cli import _detect_label_key
from .presets import PRESETS

logger = logging.getLogger("scraw_optuna_ultra")


SCORE_WEIGHTS: Dict[str, float] = {
    "ARI": 0.30,
    "NMI": 0.25,
    "F1_Macro": 0.20,
    "BalancedACC": 0.15,
    "RareACC": 0.10,
}

SCORE_KEYS: Tuple[str, ...] = ("ARI", "NMI", "F1_Macro", "BalancedACC", "RareACC")

TRIAL_SUMMARY_FIELDS: List[str] = [
    "trial",
    "status",
    "score_mean",
    "score_std",
    "n_seed_runs",
    "n_seed_ok",
    "elapsed_seconds",
    "final_clustering_requested",
    "final_clustering_used",
    "leiden_fallback_count",
    "dann_enabled",
    "ARI_mean",
    "NMI_mean",
    "ACC_mean",
    "F1_Macro_mean",
    "BalancedACC_mean",
    "RareACC_mean",
    "n_clusters_found_mean",
    "resolution_mean",
    "runtime_mean",
    "overrides_json",
]


@dataclass
class TrialConfig:
    """One sampled Optuna trial configuration."""

    final_clustering_requested: str
    param_overrides: Dict[str, Any]
    dann_enabled: bool


def _as_cli_value(value: Any) -> str:
    """Convert Python value to CLI scalar."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _safe_float(value: Any) -> float:
    """Parse a float or return NaN."""
    try:
        out = float(value)
    except Exception:
        return float("nan")
    if not np.isfinite(out):
        return float("nan")
    return float(out)


def _safe_int(value: Any) -> Optional[int]:
    """Parse an int or return None."""
    try:
        return int(value)
    except Exception:
        return None


def _composite_score(metrics: Dict[str, Any]) -> float:
    """Compute weighted score from clustering metrics."""
    score = 0.0
    for key, w in SCORE_WEIGHTS.items():
        v = _safe_float(metrics.get(key))
        if np.isnan(v):
            v = 0.0
        score += float(w) * float(v)
    return float(score)


def _score_from_rows(rows: List[Dict[str, Any]]) -> float:
    """Compute mean score from per-seed rows."""
    vals = [float(r.get("score", np.nan)) for r in rows if np.isfinite(float(r.get("score", np.nan)))]
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _build_env(output_root: Path) -> Dict[str, str]:
    """Create subprocess env (PYTHONPATH + writable cache dirs)."""
    env = os.environ.copy()
    src_dir = Path(__file__).resolve().parents[1]
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_dir) if not prev else f"{src_dir}:{prev}"

    cache_root = output_root / ".cache"
    numba_dir = cache_root / "numba"
    mpl_dir = cache_root / "mpl"
    xdg_dir = cache_root / "xdg_cache"
    numba_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("NUMBA_CACHE_DIR", str(numba_dir))
    env.setdefault("MPLCONFIGDIR", str(mpl_dir))
    env.setdefault("XDG_CACHE_HOME", str(xdg_dir))
    env.setdefault("NUMBA_DISABLE_JIT", "1")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return env


def _csv_read_rows(path: Path) -> List[Dict[str, str]]:
    """Read CSV rows or return empty list."""
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _read_analysis_metrics(run_dir: Path) -> Dict[str, Any]:
    """Read standard metrics from analysis_results.csv."""
    csv_path = run_dir / "results" / "analysis_results.csv"
    rows = _csv_read_rows(csv_path)
    if not rows:
        return {}
    row = rows[0]
    out: Dict[str, Any] = {
        "ARI": _safe_float(row.get("ARI")),
        "NMI": _safe_float(row.get("NMI")),
        "ACC": _safe_float(row.get("ACC")),
        "F1_Macro": _safe_float(row.get("F1_Macro")),
        "BalancedACC": _safe_float(row.get("BalancedACC")),
        "RareACC": _safe_float(row.get("RareACC")),
        "n_clusters_found": _safe_float(row.get("n_clusters_found")),
        "runtime": _safe_float(row.get("runtime")),
    }
    return out


def _read_final_clustering_table(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Read final clustering comparison table keyed by `method`."""
    csv_path = run_dir / "results" / "clustering_final" / "final_clustering_comparison.csv"
    rows = _csv_read_rows(csv_path)
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        method = str(row.get("method", "")).strip()
        if not method:
            continue
        out[method] = {
            "ARI": _safe_float(row.get("ARI")),
            "NMI": _safe_float(row.get("NMI")),
            "ACC": _safe_float(row.get("ACC")),
            "F1_Macro": _safe_float(row.get("F1_Macro")),
            "BalancedACC": _safe_float(row.get("BalancedACC")),
            "RareACC": _safe_float(row.get("RareACC")),
            "n_clusters_found": _safe_float(row.get("n_clusters_found")),
            "resolution": _safe_float(row.get("resolution")),
        }
    return out


def _selected_clustering_method(requested: str, target_clusters: int) -> str:
    """Map requested final clustering to CSV method name."""
    req = str(requested).strip().lower()
    if req == "leiden":
        return f"leiden_target{int(target_clusters)}_final"
    return "hdbscan_final"


def _mean_of(rows: List[Dict[str, Any]], key: str) -> float:
    """Mean of finite values for `key` over rows."""
    vals = [float(r.get(key, np.nan)) for r in rows if np.isfinite(float(r.get(key, np.nan)))]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _seed_sequence(base_seed: int, n_seeds: int, seed_step: int) -> List[int]:
    """Deterministic sequence of trial seeds."""
    n = max(1, int(n_seeds))
    step = max(1, int(seed_step))
    start = int(base_seed)
    return [start + i * step for i in range(n)]


def _append_summary_row(path: Path, row: Dict[str, Any]) -> None:
    """Append one row to global trial summary CSV with fixed schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_SUMMARY_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in TRIAL_SUMMARY_FIELDS})


def _log_cmd(cmd: Sequence[str]) -> str:
    """Render command for logs/files."""
    return " ".join(shlex.quote(c) for c in cmd)


def _build_cli_cmd(
    python_bin: str,
    preset: str,
    data_path: Path,
    output_dir: Path,
    device: str,
    seed: int,
    batch_key: Optional[str],
    overrides: Dict[str, Any],
) -> List[str]:
    """Build one scRAW CLI command for metrics-only execution."""
    cmd: List[str] = [
        python_bin,
        "-m",
        "scraw_dedicated.cli",
        "--preset",
        preset,
        "--data",
        str(data_path),
        "--output",
        str(output_dir),
        "--seed",
        str(int(seed)),
        "--device",
        str(device),
        "--metrics-only",
        "--save-processed-data",
        "off",
        "--capture-snapshots",
        "off",
        "--dann",
        "auto",
    ]
    if batch_key:
        cmd.extend(["--batch-key", str(batch_key)])
    for key, value in sorted(overrides.items()):
        cmd.extend(["--param", f"{key}={_as_cli_value(value)}"])
    return cmd


def _sample_trial_config(trial: Any, args: argparse.Namespace) -> TrialConfig:
    """Sample full scRAW hyperparameter config for one Optuna trial."""
    # Network / optimization
    hidden_layers = trial.suggest_categorical(
        "hidden_layers",
        ["256,128,64", "512,256,128", "512,256", "1024,512,256,128"],
    )
    z_dim = trial.suggest_categorical("z_dim", [64, 96, 128, 192, 256])
    dropout = trial.suggest_float("dropout", 0.05, 0.30, step=0.05)
    epochs = trial.suggest_int("epochs", 80, 220, step=10)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.15, 0.45, step=0.05)
    warmup_epochs = int(max(5, min(epochs - 1, round(epochs * warmup_ratio))))
    lr = trial.suggest_float("lr", 2e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 192, 256, 384, 512])

    # Reconstruction / masking
    reconstruction_distribution = trial.suggest_categorical(
        "reconstruction_distribution",
        ["nb", "mse"],
    )
    nb_input_transform = trial.suggest_categorical(
        "nb_input_transform",
        ["log1p", "pearson_residuals"],
    )
    nb_theta = trial.suggest_float("nb_theta", 1.0, 120.0, log=True)
    masking_rate = trial.suggest_float("masking_rate", 0.10, 0.40, step=0.05)
    masked_recon_weight = trial.suggest_float("masked_recon_weight", 0.50, 1.00, step=0.05)
    masking_apply_weighted = trial.suggest_categorical("masking_apply_weighted", [False, True])

    # Dynamic cell weighting
    weight_exponent = trial.suggest_float("weight_exponent", 0.1, 0.8, step=0.1)
    cluster_density_alpha = trial.suggest_float("cluster_density_alpha", 0.0, 1.0, step=0.1)
    density_knn_k = trial.suggest_categorical("density_knn_k", [10, 15, 20, 30])
    density_weight_clip = trial.suggest_categorical("density_weight_clip", [3.0, 5.0, 8.0, 10.0])
    dynamic_weight_momentum = trial.suggest_float("dynamic_weight_momentum", 0.5, 0.95, step=0.05)
    dynamic_weight_update_interval = trial.suggest_categorical(
        "dynamic_weight_update_interval",
        [5, 10, 15, 20],
    )
    min_cell_weight = trial.suggest_float("min_cell_weight", 0.10, 0.60, step=0.05)
    max_cell_weight = trial.suggest_categorical("max_cell_weight", [5.0, 8.0, 10.0, 15.0, 20.0])
    weight_fusion_mode = trial.suggest_categorical("weight_fusion_mode", ["additive", "multiplicative"])

    # Rare triplet loss
    rare_triplet_weight = trial.suggest_float("rare_triplet_weight", 0.01, 0.30, log=True)
    triplet_start_offset = trial.suggest_int("triplet_start_offset", 0, 25, step=5)
    rare_triplet_start_epoch = int(min(epochs - 1, warmup_epochs + triplet_start_offset))
    rare_triplet_margin = trial.suggest_float("rare_triplet_margin", 0.2, 1.0, step=0.1)
    rare_triplet_min_weight = trial.suggest_float("rare_triplet_min_weight", 0.8, 2.0, step=0.2)
    max_triplet_anchors = trial.suggest_categorical("max_triplet_anchors_per_batch", [32, 64, 96, 128])

    # Pseudo-label / final HDBSCAN controls
    pseudo_label_method = trial.suggest_categorical("pseudo_label_method", ["leiden", "kmeans"])
    hdbscan_min_cluster_size = trial.suggest_int("hdbscan_min_cluster_size", 2, 20, step=1)
    hdbscan_min_samples = trial.suggest_int("hdbscan_min_samples", 1, 10, step=1)
    hdbscan_min_samples = int(min(hdbscan_min_samples, hdbscan_min_cluster_size))
    hdbscan_cluster_selection_method = trial.suggest_categorical(
        "hdbscan_cluster_selection_method",
        ["eom", "leaf"],
    )
    hdbscan_reassign_noise = trial.suggest_categorical("hdbscan_reassign_noise", [True, False])

    # DANN sampling policy
    if args.dann_mode == "on":
        use_dann = True
    elif args.dann_mode == "off":
        use_dann = False
    else:
        use_dann = bool(trial.suggest_categorical("use_batch_conditioning", [False, True]))

    adversarial_batch_weight = 0.0
    adversarial_lambda = 1.0
    adversarial_start_epoch = 0
    adversarial_ramp_epochs = 0
    mmd_batch_weight = 0.0
    if use_dann:
        adversarial_batch_weight = trial.suggest_float(
            "adversarial_batch_weight",
            0.01,
            0.50,
            log=True,
        )
        adversarial_lambda = trial.suggest_float("adversarial_lambda", 0.25, 2.5, step=0.25)
        max_start = int(max(0, min(60, epochs - 1)))
        adversarial_start_epoch = trial.suggest_int("adversarial_start_epoch", 0, max_start, step=5)
        adversarial_ramp_epochs = trial.suggest_int("adversarial_ramp_epochs", 0, 60, step=5)
        mmd_batch_weight = trial.suggest_categorical("mmd_batch_weight", [0.0, 0.02, 0.05, 0.10])

    # Final clustering method used for trial scoring
    if args.final_clustering_mode == "hdbscan":
        final_clustering_requested = "hdbscan"
    elif args.final_clustering_mode == "leiden":
        final_clustering_requested = "leiden"
    else:
        final_clustering_requested = str(
            trial.suggest_categorical("final_clustering", ["hdbscan", "leiden"])
        )

    overrides: Dict[str, Any] = {
        "hidden_layers": hidden_layers,
        "z_dim": z_dim,
        "dropout": dropout,
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "lr": lr,
        "batch_size": batch_size,
        "reconstruction_distribution": reconstruction_distribution,
        "nb_input_transform": nb_input_transform,
        "nb_theta": nb_theta,
        "masking_rate": masking_rate,
        "masked_recon_weight": masked_recon_weight,
        "masking_apply_weighted": masking_apply_weighted,
        "weight_exponent": weight_exponent,
        "cluster_density_alpha": cluster_density_alpha,
        "density_knn_k": density_knn_k,
        "density_weight_clip": density_weight_clip,
        "dynamic_weight_momentum": dynamic_weight_momentum,
        "dynamic_weight_update_interval": dynamic_weight_update_interval,
        "weight_fusion_mode": weight_fusion_mode,
        "min_cell_weight": min_cell_weight,
        "max_cell_weight": max_cell_weight,
        "rare_triplet_weight": rare_triplet_weight,
        "rare_triplet_start_epoch": rare_triplet_start_epoch,
        "rare_triplet_margin": rare_triplet_margin,
        "rare_triplet_min_weight": rare_triplet_min_weight,
        "max_triplet_anchors_per_batch": max_triplet_anchors,
        "pseudo_label_method": pseudo_label_method,
        "hdbscan_min_cluster_size": hdbscan_min_cluster_size,
        "hdbscan_min_samples": hdbscan_min_samples,
        "hdbscan_cluster_selection_method": hdbscan_cluster_selection_method,
        "hdbscan_reassign_noise": hdbscan_reassign_noise,
        "use_batch_conditioning": bool(use_dann),
        "adversarial_batch_weight": adversarial_batch_weight,
        "adversarial_lambda": adversarial_lambda,
        "adversarial_start_epoch": adversarial_start_epoch,
        "adversarial_ramp_epochs": adversarial_ramp_epochs,
        "mmd_batch_weight": mmd_batch_weight,
        "capture_embedding_snapshots": False,
    }

    if use_dann:
        overrides["batch_correction_key"] = str(args.batch_key or "batch")
    else:
        overrides["batch_correction_key"] = "auto"

    return TrialConfig(
        final_clustering_requested=final_clustering_requested,
        param_overrides=overrides,
        dann_enabled=bool(use_dann),
    )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _build_search_space_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    """Human-readable definition of the sampled search space."""
    return {
        "algorithm": "scraw",
        "search_type": "optuna_tpe",
        "objective": "maximize weighted clustering score",
        "score_formula": SCORE_WEIGHTS,
        "cluster_count_penalty": "none",
        "dann_mode": args.dann_mode,
        "final_clustering_mode": args.final_clustering_mode,
        "target_clusters_for_leiden": int(args.target_clusters),
        "spaces": {
            "architecture": {
                "hidden_layers": ["256,128,64", "512,256,128", "512,256", "1024,512,256,128"],
                "z_dim": [64, 96, 128, 192, 256],
                "dropout": "0.05..0.30 step=0.05",
            },
            "optimization": {
                "epochs": "80..220 step=10",
                "warmup_ratio": "0.15..0.45 step=0.05 (converted to warmup_epochs)",
                "lr": "2e-4..5e-3 log-uniform",
                "batch_size": [128, 192, 256, 384, 512],
            },
            "reconstruction": {
                "reconstruction_distribution": ["nb", "mse"],
                "nb_input_transform": ["log1p", "pearson_residuals"],
                "nb_theta": "1..120 log-uniform",
                "masking_rate": "0.10..0.40 step=0.05",
                "masked_recon_weight": "0.50..1.00 step=0.05",
                "masking_apply_weighted": [False, True],
            },
            "dynamic_weighting": {
                "weight_exponent": "0.1..0.8 step=0.1",
                "cluster_density_alpha": "0.0..1.0 step=0.1",
                "density_knn_k": [10, 15, 20, 30],
                "density_weight_clip": [3.0, 5.0, 8.0, 10.0],
                "dynamic_weight_momentum": "0.5..0.95 step=0.05",
                "dynamic_weight_update_interval": [5, 10, 15, 20],
                "weight_fusion_mode": ["additive", "multiplicative"],
                "min_cell_weight": "0.10..0.60 step=0.05",
                "max_cell_weight": [5.0, 8.0, 10.0, 15.0, 20.0],
            },
            "triplet": {
                "rare_triplet_weight": "0.01..0.30 log-uniform",
                "triplet_start_offset": "0..25 step=5 (added to warmup_epochs)",
                "rare_triplet_margin": "0.2..1.0 step=0.1",
                "rare_triplet_min_weight": "0.8..2.0 step=0.2",
                "max_triplet_anchors_per_batch": [32, 64, 96, 128],
            },
            "pseudo_labels_and_hdbscan": {
                "pseudo_label_method": ["leiden", "kmeans"],
                "hdbscan_min_cluster_size": "2..20 step=1",
                "hdbscan_min_samples": "1..10 step=1 (clamped <= min_cluster_size)",
                "hdbscan_cluster_selection_method": ["eom", "leaf"],
                "hdbscan_reassign_noise": [True, False],
            },
            "dann_conditional": {
                "enabled_when": "dann_mode=on or sampled use_batch_conditioning=True",
                "adversarial_batch_weight": "0.01..0.50 log-uniform",
                "adversarial_lambda": "0.25..2.5 step=0.25",
                "adversarial_start_epoch": "0..min(60, epochs-1) step=5",
                "adversarial_ramp_epochs": "0..60 step=5",
                "mmd_batch_weight": [0.0, 0.02, 0.05, 0.10],
                "batch_correction_key": "batch (or --batch-key override)",
            },
            "final_scoring_clustering": {
                "method": {
                    "hdbscan": "score from final_clustering_comparison.csv -> hdbscan_final",
                    "leiden": (
                        "score from final_clustering_comparison.csv -> "
                        f"leiden_target{int(args.target_clusters)}_final"
                    ),
                },
                "leiden_resolution_search": "performed by scraw_dedicated.cli (0.05..3.0, step 0.05)",
            },
        },
    }


def _study_sampler(args: argparse.Namespace, optuna_mod: Any) -> Any:
    """Create Optuna sampler from CLI args."""
    if args.sampler == "random":
        return optuna_mod.samplers.RandomSampler(seed=int(args.seed))
    return optuna_mod.samplers.TPESampler(
        seed=int(args.seed),
        multivariate=True,
        group=True,
        constant_liar=False,
        n_startup_trials=max(5, int(args.n_startup_trials)),
    )


def _configure_logging(verbose: bool) -> None:
    """Setup process logger."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_parser() -> argparse.ArgumentParser:
    """CLI parser for Optuna ultra search."""
    p = argparse.ArgumentParser(
        description="Optuna ultra-search for scRAW (DANN + Leiden/HDBSCAN final scoring)."
    )
    p.add_argument("--preset", required=True, choices=sorted(PRESETS.keys()))
    p.add_argument("--data", required=True, help="Input .h5ad path")
    p.add_argument("--output-root", required=True, help="Output root directory")
    p.add_argument("--device", default="auto", help="auto|cuda|cpu|mps")
    p.add_argument("--python-bin", default=sys.executable, help="Python interpreter for CLI runs")
    p.add_argument("--seed", type=int, default=42, help="Base seed")
    p.add_argument("--seed-step", type=int, default=97, help="Step between per-trial seeds")
    p.add_argument("--n-seeds", type=int, default=1, help="Number of seeds per trial")
    p.add_argument("--n-trials", type=int, default=400, help="Total Optuna trials")
    p.add_argument("--timeout", type=int, default=None, help="Global timeout in seconds")
    p.add_argument("--study-name", default=None, help="Optuna study name")
    p.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    p.add_argument("--n-startup-trials", type=int, default=32, help="TPE startup random trials")
    p.add_argument("--dann-mode", choices=["on", "off", "mixed"], default="on")
    p.add_argument("--batch-key", default=None, help="Batch key override when DANN is active")
    p.add_argument(
        "--final-clustering-mode",
        choices=["hdbscan", "leiden", "mixed"],
        default="mixed",
        help="Final clustering method used for trial scoring",
    )
    p.add_argument(
        "--target-clusters",
        type=int,
        default=14,
        help="Target clusters for Leiden scoring row (CLI currently exports target14).",
    )
    p.add_argument("--dry-run", action="store_true", help="Sample/search without launching training")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip seed runs where analysis_results.csv already exists (default: on).",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Force rerun even if metrics file already exists.",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entrypoint."""
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    if int(args.target_clusters) != 14:
        raise ValueError(
            "Current CLI exports Leiden final metrics as method 'leiden_target14_final'. "
            "Use --target-clusters 14 for now."
        )

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    trials_root = output_root / "trials"
    logs_root = output_root / "logs"
    summaries_root = output_root / "summaries"
    config_root = output_root / "config"
    for d in (trials_root, logs_root, summaries_root, config_root):
        d.mkdir(parents=True, exist_ok=True)

    env = _build_env(output_root)

    # Minimal up-front label check for metric-based optimization clarity.
    try:
        import anndata as ad

        adata_backed = ad.read_h5ad(data_path, backed="r")
        try:
            obs_cols = list(adata_backed.obs.columns)
            label_key = _detect_label_key(obs_cols)
        finally:
            if getattr(adata_backed, "file", None) is not None:
                adata_backed.file.close()
        if label_key is None:
            logger.warning(
                "No recognized label column found in obs. ARI/NMI/ACC-based scoring may be NaN."
            )
    except Exception as exc:
        logger.warning("Label-column precheck skipped: %s", exc)
        label_key = None

    search_space_manifest = _build_search_space_manifest(args)
    _write_json(config_root / "search_space.json", search_space_manifest)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = str(args.study_name or f"scraw_ultra_{args.preset}_{timestamp}")
    study_db = output_root / "optuna_study.db"

    search_config = {
        "timestamp": timestamp,
        "study_name": study_name,
        "preset": args.preset,
        "data": str(data_path),
        "device": args.device,
        "python_bin": args.python_bin,
        "seed": int(args.seed),
        "seed_step": int(args.seed_step),
        "n_seeds": int(args.n_seeds),
        "n_trials": int(args.n_trials),
        "timeout": args.timeout,
        "sampler": args.sampler,
        "n_startup_trials": int(args.n_startup_trials),
        "dann_mode": args.dann_mode,
        "batch_key": args.batch_key,
        "final_clustering_mode": args.final_clustering_mode,
        "target_clusters": int(args.target_clusters),
        "score_weights": SCORE_WEIGHTS,
        "dry_run": bool(args.dry_run),
        "skip_existing": bool(args.skip_existing),
        "label_key_detected": label_key,
        "metrics_only": True,
        "save_processed_data": False,
    }
    _write_json(config_root / "search_config.json", search_config)

    try:
        import optuna
    except Exception as exc:
        raise RuntimeError(
            "optuna is required for this script. Install it with: pip install optuna"
        ) from exc

    sampler = _study_sampler(args, optuna)
    storage = f"sqlite:///{study_db}"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    trial_summary_csv = summaries_root / "all_trials.csv"
    seeds = _seed_sequence(args.seed, args.n_seeds, args.seed_step)

    logger.info("Study: %s", study_name)
    logger.info("DB: %s", study_db)
    logger.info("Trials target: %d", int(args.n_trials))
    logger.info("Seeds/trial: %s", seeds)
    logger.info("DANN mode: %s | Final clustering mode: %s", args.dann_mode, args.final_clustering_mode)

    def objective(trial: Any) -> float:
        t0 = time.time()
        cfg = _sample_trial_config(trial, args)
        trial_name = f"trial_{int(trial.number):04d}"
        trial_dir = trials_root / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        _write_json(
            trial_dir / "trial_config.json",
            {
                "trial_number": int(trial.number),
                "final_clustering_requested": cfg.final_clustering_requested,
                "dann_enabled": bool(cfg.dann_enabled),
                "param_overrides": cfg.param_overrides,
            },
        )

        per_seed_rows: List[Dict[str, Any]] = []
        selected_method = _selected_clustering_method(
            cfg.final_clustering_requested,
            target_clusters=int(args.target_clusters),
        )
        fallback_count = 0
        seed_ok = 0

        for seed in seeds:
            run_dir = trial_dir / f"seed_{int(seed)}"
            log_file = logs_root / trial_name / f"seed_{int(seed)}.log"
            run_dir.mkdir(parents=True, exist_ok=True)
            log_file.parent.mkdir(parents=True, exist_ok=True)

            cmd = _build_cli_cmd(
                python_bin=args.python_bin,
                preset=args.preset,
                data_path=data_path,
                output_dir=run_dir,
                device=args.device,
                seed=int(seed),
                batch_key=args.batch_key,
                overrides=cfg.param_overrides,
            )

            run_status = "ok"
            if args.dry_run:
                run_status = "dry_run"
                log_file.write_text(_log_cmd(cmd) + "\n", encoding="utf-8")
            else:
                analysis_csv = run_dir / "results" / "analysis_results.csv"
                if args.skip_existing and analysis_csv.exists():
                    run_status = "existing"
                else:
                    with log_file.open("w") as fh:
                        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
                    if int(proc.returncode) != 0:
                        run_status = f"failed_{int(proc.returncode)}"

            analysis_metrics = _read_analysis_metrics(run_dir)
            cluster_rows = _read_final_clustering_table(run_dir)

            used_method = selected_method
            metric_source = cluster_rows.get(selected_method)
            if metric_source is None and selected_method != "hdbscan_final":
                metric_source = cluster_rows.get("hdbscan_final")
                used_method = "hdbscan_final"
                if metric_source is not None:
                    fallback_count += 1

            metrics: Dict[str, Any] = {}
            if metric_source is not None:
                metrics.update(metric_source)
            metrics["runtime"] = analysis_metrics.get("runtime", float("nan"))
            metrics["ACC"] = metrics.get("ACC", analysis_metrics.get("ACC", float("nan")))
            score = _composite_score(metrics)
            if run_status.startswith("failed"):
                score = 0.0
            elif run_status in {"dry_run"}:
                score = 0.0

            if run_status in {"ok", "existing"}:
                seed_ok += 1

            row = {
                "seed": int(seed),
                "status": run_status,
                "requested_method": selected_method,
                "used_method": used_method,
                "score": float(score),
                "ARI": _safe_float(metrics.get("ARI")),
                "NMI": _safe_float(metrics.get("NMI")),
                "ACC": _safe_float(metrics.get("ACC")),
                "F1_Macro": _safe_float(metrics.get("F1_Macro")),
                "BalancedACC": _safe_float(metrics.get("BalancedACC")),
                "RareACC": _safe_float(metrics.get("RareACC")),
                "n_clusters_found": _safe_float(metrics.get("n_clusters_found")),
                "resolution": _safe_float(metrics.get("resolution")),
                "runtime": _safe_float(metrics.get("runtime")),
                "run_dir": str(run_dir),
                "log_file": str(log_file),
                "command": _log_cmd(cmd),
            }
            per_seed_rows.append(row)

            _write_json(run_dir / "trial_seed_summary.json", row)

        score_mean = _score_from_rows(per_seed_rows)
        score_std = (
            float(np.std([float(r.get("score", 0.0)) for r in per_seed_rows]))
            if per_seed_rows
            else float("nan")
        )
        elapsed = float(time.time() - t0)

        trial_status = "ok"
        if not per_seed_rows:
            trial_status = "empty"
        elif all(str(r.get("status", "")).startswith("failed") for r in per_seed_rows):
            trial_status = "failed"
            score_mean = 0.0
        elif any(str(r.get("status", "")).startswith("failed") for r in per_seed_rows):
            trial_status = "partial"

        trial_summary = {
            "trial": int(trial.number),
            "status": trial_status,
            "score_mean": float(score_mean),
            "score_std": float(score_std),
            "n_seed_runs": int(len(per_seed_rows)),
            "n_seed_ok": int(seed_ok),
            "elapsed_seconds": round(elapsed, 1),
            "final_clustering_requested": cfg.final_clustering_requested,
            "final_clustering_used": (
                str(per_seed_rows[0].get("used_method")) if per_seed_rows else selected_method
            ),
            "leiden_fallback_count": int(fallback_count),
            "dann_enabled": int(cfg.dann_enabled),
            "ARI_mean": _mean_of(per_seed_rows, "ARI"),
            "NMI_mean": _mean_of(per_seed_rows, "NMI"),
            "ACC_mean": _mean_of(per_seed_rows, "ACC"),
            "F1_Macro_mean": _mean_of(per_seed_rows, "F1_Macro"),
            "BalancedACC_mean": _mean_of(per_seed_rows, "BalancedACC"),
            "RareACC_mean": _mean_of(per_seed_rows, "RareACC"),
            "n_clusters_found_mean": _mean_of(per_seed_rows, "n_clusters_found"),
            "resolution_mean": _mean_of(per_seed_rows, "resolution"),
            "runtime_mean": _mean_of(per_seed_rows, "runtime"),
            "overrides_json": json.dumps(cfg.param_overrides, sort_keys=True),
        }
        _append_summary_row(trial_summary_csv, trial_summary)

        _write_json(
            trial_dir / "trial_summary.json",
            {
                "trial_summary": trial_summary,
                "seed_rows": per_seed_rows,
            },
        )

        trial.set_user_attr("trial_status", trial_status)
        trial.set_user_attr("final_clustering_requested", cfg.final_clustering_requested)
        trial.set_user_attr("dann_enabled", bool(cfg.dann_enabled))
        trial.set_user_attr("ari_mean", trial_summary["ARI_mean"])
        trial.set_user_attr("nmi_mean", trial_summary["NMI_mean"])
        trial.set_user_attr("f1_macro_mean", trial_summary["F1_Macro_mean"])
        trial.set_user_attr("balanced_acc_mean", trial_summary["BalancedACC_mean"])
        trial.set_user_attr("rare_acc_mean", trial_summary["RareACC_mean"])
        trial.set_user_attr("n_clusters_found_mean", trial_summary["n_clusters_found_mean"])
        trial.set_user_attr("runtime_mean", trial_summary["runtime_mean"])

        return float(score_mean)

    n_completed = len([t for t in study.trials if t.state.name == "COMPLETE"])
    n_remaining = max(0, int(args.n_trials) - n_completed)
    if n_completed > 0:
        logger.info("Resuming existing study: completed=%d remaining=%d", n_completed, n_remaining)

    if n_remaining > 0:
        study.optimize(
            objective,
            n_trials=n_remaining,
            timeout=args.timeout,
            show_progress_bar=True,
        )

    complete_trials = [t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None]
    ranked = sorted(complete_trials, key=lambda t: float(t.value), reverse=True)

    top_rows: List[Dict[str, Any]] = []
    for t in ranked[:20]:
        top_rows.append(
            {
                "trial": int(t.number),
                "score": float(t.value),
                "dann_enabled": t.user_attrs.get("dann_enabled"),
                "final_clustering_requested": t.user_attrs.get("final_clustering_requested"),
                "ARI_mean": t.user_attrs.get("ari_mean"),
                "NMI_mean": t.user_attrs.get("nmi_mean"),
                "F1_Macro_mean": t.user_attrs.get("f1_macro_mean"),
                "BalancedACC_mean": t.user_attrs.get("balanced_acc_mean"),
                "RareACC_mean": t.user_attrs.get("rare_acc_mean"),
                "n_clusters_found_mean": t.user_attrs.get("n_clusters_found_mean"),
                "runtime_mean": t.user_attrs.get("runtime_mean"),
                "params_json": json.dumps(t.params, sort_keys=True),
            }
        )

    top_csv = summaries_root / "top_trials.csv"
    if top_rows:
        with top_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(top_rows[0].keys()))
            writer.writeheader()
            for row in top_rows:
                writer.writerow(row)

    summary: Dict[str, Any] = {
        "study_name": study_name,
        "storage": str(study_db),
        "n_trials_total": len(study.trials),
        "n_trials_complete": len(complete_trials),
        "score_weights": SCORE_WEIGHTS,
        "dann_mode": args.dann_mode,
        "final_clustering_mode": args.final_clustering_mode,
        "target_clusters": int(args.target_clusters),
        "search_space_file": str(config_root / "search_space.json"),
        "all_trials_csv": str(trial_summary_csv),
        "top_trials_csv": str(top_csv),
        "timestamp": datetime.now().isoformat(),
    }

    if ranked:
        best = ranked[0]
        summary["best_trial"] = {
            "trial": int(best.number),
            "score": float(best.value),
            "params": best.params,
            "user_attrs": best.user_attrs,
        }
    else:
        summary["best_trial"] = None

    try:
        if len(complete_trials) >= 20:
            import optuna.importance as optuna_importance

            imp = optuna_importance.get_param_importances(study)
            summary["param_importances"] = {str(k): float(v) for k, v in imp.items()}
    except Exception as exc:
        summary["param_importances_error"] = str(exc)

    _write_json(summaries_root / "search_summary.json", summary)

    if ranked:
        best = ranked[0]
        _write_json(
            summaries_root / "best_trial.json",
            {
                "trial": int(best.number),
                "score": float(best.value),
                "params": best.params,
                "user_attrs": best.user_attrs,
            },
        )

    logger.info("Search complete. Output root: %s", output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
