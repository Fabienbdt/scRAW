#!/usr/bin/env python3
"""Ultra-comprehensive Optuna-based hyperparameter search for scRAW + DANN.

Features:
- TPE sampler with multivariate correlations
- Conditional search spaces (DANN params only when DANN is active)
- Leiden final clustering with resolution optimization as alternative to HDBSCAN
- Composite scoring: 0.30*ARI + 0.25*NMI + 0.20*F1_Macro + 0.15*BalancedACC + 0.10*RareACC
- SQLite persistence for crash recovery
- CSV/JSON reports per trial
- Dry-run mode for validation
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Resolve project root and make sure src/ is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logger = logging.getLogger("hparam_search")

# ---------------------------------------------------------------------------
# Composite score weights
# ---------------------------------------------------------------------------
SCORE_WEIGHTS = {
    "ARI": 0.30,
    "NMI": 0.25,
    "F1_Macro": 0.20,
    "BalancedACC": 0.15,
    "RareACC": 0.10,
}


def _composite_score(metrics: Dict[str, Any]) -> float:
    """Compute weighted composite score from metrics dict."""
    score = 0.0
    for key, weight in SCORE_WEIGHTS.items():
        val = metrics.get(key, float("nan"))
        if val is None or not np.isfinite(float(val)):
            val = 0.0
        score += weight * float(val)
    return float(score)


# ---------------------------------------------------------------------------
# Leiden optimized final clustering (alternative to HDBSCAN)
# ---------------------------------------------------------------------------
def _leiden_final_clustering(
    embeddings: np.ndarray,
    seed: int,
    target_clusters: int = 14,
    resolution_range: Tuple[float, float] = (0.05, 3.0),
    n_neighbors: int = 15,
) -> np.ndarray:
    """Apply Leiden clustering with resolution optimization on embeddings."""
    import anndata as ad
    import scanpy as sc

    emb = np.asarray(embeddings, dtype=np.float32)
    n_cells = emb.shape[0]
    if n_cells < 3:
        return np.zeros(n_cells, dtype=np.int64)

    adata = ad.AnnData(X=emb)
    nn = max(2, min(n_neighbors, n_cells - 1))
    sc.pp.neighbors(
        adata,
        n_neighbors=nn,
        use_rep="X",
        method="gauss",
        transformer="sklearn",
        random_state=seed,
    )

    best_res, best_diff = 1.0, n_cells
    for res in np.arange(resolution_range[0], resolution_range[1] + 0.01, 0.05):
        sc.tl.leiden(adata, resolution=float(res), random_state=seed, key_added="_tmp")
        n_found = len(np.unique(adata.obs["_tmp"].astype(int).values))
        diff = abs(n_found - target_clusters)
        if diff < best_diff:
            best_diff = diff
            best_res = float(res)
        if n_found == target_clusters:
            break

    sc.tl.leiden(adata, resolution=best_res, random_state=seed, key_added="_final")
    labels = adata.obs["_final"].astype(int).values.astype(np.int64)

    # Remap to contiguous
    unique_labels = np.unique(labels)
    mapping = {old: new for new, old in enumerate(unique_labels)}
    return np.array([mapping[l] for l in labels], dtype=np.int64)


# ---------------------------------------------------------------------------
# Single trial runner
# ---------------------------------------------------------------------------
def _run_single_trial(
    preset_name: str,
    data_path: Path,
    output_dir: Path,
    device: str,
    seed: int,
    param_overrides: Dict[str, Any],
    metrics_only: bool = True,
    final_clustering: str = "hdbscan",
    leiden_resolution_range: Tuple[float, float] = (0.05, 3.0),
    leiden_n_neighbors: int = 15,
    leiden_target_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a single scRAW trial and return metrics + timing."""
    from scraw_dedicated.cli import (
        _build_scraw_params,
        _detect_batch_key_in_file,
        _detect_label_key,
    )
    from scraw_dedicated.metrics import compute_metrics
    from scraw_dedicated.preprocessing import preprocess_adata

    import anndata as ad

    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Load data
    adata = ad.read_h5ad(str(data_path))

    # Detect labels
    label_key = _detect_label_key(adata.obs.columns.tolist())
    labels_true = None
    if label_key:
        labels_true = np.asarray(adata.obs[label_key].values)
        n_true_clusters = len(np.unique(labels_true))
    else:
        n_true_clusters = 14  # fallback

    # Detect batch key
    batch_key = _detect_batch_key_in_file(data_path, preferred=param_overrides.get("batch_correction_key"))

    # Build params
    dann_mode = "on" if param_overrides.get("use_batch_conditioning", False) else "off"
    params = _build_scraw_params(
        preset_name=preset_name,
        seed=seed,
        device=device,
        dann_mode=dann_mode,
        batch_key_override=param_overrides.get("batch_correction_key"),
        capture_snapshots="off",
        snapshot_interval=None,
        param_overrides=param_overrides,
    )

    # Get preprocessing config from preset
    from scraw_dedicated.presets import get_preset
    preset = get_preset(preset_name)
    preprocessing_config = dict(preset.preprocessing)

    # Preprocess (preprocess_adata takes a params dict, not **kwargs)
    adata = preprocess_adata(adata, params=preprocessing_config)

    # Build & train algorithm
    from scraw_dedicated.algorithms.scraw_algorithm import ScRAWAlgorithm
    algo = ScRAWAlgorithm(params=params)
    algo.fit(adata, labels=labels_true)

    # Get embeddings for final clustering
    embeddings = algo._embeddings  # noqa: SLF001

    # Final clustering: HDBSCAN (default in algo) or Leiden
    if final_clustering == "leiden":
        target_k = leiden_target_k or n_true_clusters
        labels_pred = _leiden_final_clustering(
            embeddings=embeddings,
            seed=seed,
            target_clusters=target_k,
            resolution_range=leiden_resolution_range,
            n_neighbors=leiden_n_neighbors,
        )
    else:
        # Use the HDBSCAN labels already computed by algo.fit()
        labels_pred = algo.predict()

    labels_pred = np.asarray(labels_pred)

    # Compute metrics
    metrics = compute_metrics(
        labels_true=labels_true,
        labels_pred=labels_pred,
        embeddings=embeddings,
    )

    elapsed = time.time() - t0
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["final_clustering"] = final_clustering
    metrics["composite_score"] = _composite_score(metrics)

    # Save trial results
    results = {
        "params": {k: v for k, v in params.items() if not k.startswith("_")},
        "metrics": metrics,
        "final_clustering": final_clustering,
        "preset": preset_name,
        "seed": seed,
        "device": device,
    }
    with open(output_dir / "trial_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return metrics


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def _create_objective(
    preset_name: str,
    data_path: Path,
    output_root: Path,
    device: str,
    base_seed: int,
    n_seeds: int,
    metrics_only: bool,
    dry_run: bool,
):
    """Return an Optuna objective closure."""

    def objective(trial) -> float:
        import optuna

        # ===================================================================
        # 1. Architecture
        # ===================================================================
        architecture = trial.suggest_categorical(
            "architecture",
            ["512,256,128", "256,128,64", "512,256", "1024,512,256,128"],
        )
        z_dim = trial.suggest_categorical("z_dim", [64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.05, 0.30, step=0.05)

        # ===================================================================
        # 2. Training schedule
        # ===================================================================
        epochs = trial.suggest_int("epochs", 80, 200, step=10)
        warmup_ratio = trial.suggest_float("warmup_ratio", 0.15, 0.35, step=0.05)
        warmup_epochs = int(round(epochs * warmup_ratio))
        lr = trial.suggest_float("lr", 3e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

        # ===================================================================
        # 3. Reconstruction
        # ===================================================================
        nb_input_transform = trial.suggest_categorical(
            "nb_input_transform", ["log1p", "pearson_residuals"]
        )
        masking_rate = trial.suggest_float("masking_rate", 0.10, 0.40, step=0.05)
        masked_recon_weight = trial.suggest_float(
            "masked_recon_weight", 0.50, 0.90, step=0.05
        )
        masking_apply_weighted = trial.suggest_categorical(
            "masking_apply_weighted", [False, True]
        )
        nb_theta = trial.suggest_float("nb_theta", 1.0, 100.0, log=True)

        # ===================================================================
        # 4. Weighting
        # ===================================================================
        weight_exponent = trial.suggest_float("weight_exponent", 0.1, 1.0, step=0.1)
        cluster_density_alpha = trial.suggest_float(
            "cluster_density_alpha", 0.3, 0.9, step=0.1
        )
        density_knn_k = trial.suggest_categorical("density_knn_k", [10, 15, 20, 30])
        dynamic_weight_momentum = trial.suggest_float(
            "dynamic_weight_momentum", 0.5, 0.9, step=0.05
        )
        min_cell_weight = trial.suggest_float(
            "min_cell_weight", 0.1, 0.5, step=0.05
        )
        weight_fusion_mode = trial.suggest_categorical(
            "weight_fusion_mode", ["additive", "multiplicative"]
        )
        dynamic_weight_update_interval = trial.suggest_categorical(
            "dynamic_weight_update_interval", [5, 10, 15, 20]
        )

        # ===================================================================
        # 5. Rare / Triplet loss
        # ===================================================================
        rare_triplet_weight = trial.suggest_float(
            "rare_triplet_weight", 0.01, 0.30, log=True
        )
        # Start epoch relative to warmup
        triplet_start_offset = trial.suggest_int("triplet_start_offset", 0, 20, step=5)
        rare_triplet_start_epoch = warmup_epochs + triplet_start_offset
        rare_triplet_margin = trial.suggest_float(
            "rare_triplet_margin", 0.2, 1.0, step=0.1
        )
        rare_triplet_min_weight = trial.suggest_float(
            "rare_triplet_min_weight", 0.8, 2.0, step=0.2
        )
        max_triplet_anchors = trial.suggest_categorical(
            "max_triplet_anchors_per_batch", [32, 64, 128]
        )

        # ===================================================================
        # 6. DANN / Batch conditioning (conditional)
        # ===================================================================
        use_dann = trial.suggest_categorical("use_batch_conditioning", [False, True])

        adv_weight = 0.0
        adv_lambda = 1.0
        adv_start_epoch = 0
        adv_ramp_epochs = 0
        if use_dann:
            adv_weight = trial.suggest_float(
                "adversarial_batch_weight", 0.01, 0.50, log=True
            )
            adv_lambda = trial.suggest_float(
                "adversarial_lambda", 0.5, 3.0, step=0.25
            )
            adv_start_epoch = trial.suggest_int(
                "adversarial_start_epoch", 0, 30, step=5
            )
            adv_ramp_epochs = trial.suggest_int(
                "adversarial_ramp_epochs", 0, 30, step=5
            )

        # ===================================================================
        # 7. Final clustering method
        # ===================================================================
        final_clustering = trial.suggest_categorical(
            "final_clustering", ["hdbscan", "leiden"]
        )

        # HDBSCAN params (conditional on hdbscan)
        hdbscan_min_cluster_size = 4
        hdbscan_min_samples = 2
        hdbscan_selection = "eom"
        leiden_n_neighbors = 15
        if final_clustering == "hdbscan":
            hdbscan_min_cluster_size = trial.suggest_int(
                "hdbscan_min_cluster_size", 2, 15, step=1
            )
            hdbscan_min_samples = trial.suggest_int(
                "hdbscan_min_samples", 1, 8, step=1
            )
            # Ensure min_samples <= min_cluster_size
            hdbscan_min_samples = min(hdbscan_min_samples, hdbscan_min_cluster_size)
            hdbscan_selection = trial.suggest_categorical(
                "hdbscan_cluster_selection_method", ["eom", "leaf"]
            )
        elif final_clustering == "leiden":
            leiden_n_neighbors = trial.suggest_categorical(
                "leiden_n_neighbors", [10, 15, 20, 30]
            )

        # ===================================================================
        # Build param dict
        # ===================================================================
        param_overrides = {
            "hidden_layers": architecture,
            "z_dim": z_dim,
            "dropout": dropout,
            "epochs": epochs,
            "warmup_epochs": warmup_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "nb_input_transform": nb_input_transform,
            "masking_rate": masking_rate,
            "masked_recon_weight": masked_recon_weight,
            "masking_apply_weighted": masking_apply_weighted,
            "nb_theta": nb_theta,
            "weight_exponent": weight_exponent,
            "cluster_density_alpha": cluster_density_alpha,
            "density_knn_k": density_knn_k,
            "dynamic_weight_momentum": dynamic_weight_momentum,
            "min_cell_weight": min_cell_weight,
            "weight_fusion_mode": weight_fusion_mode,
            "dynamic_weight_update_interval": dynamic_weight_update_interval,
            "rare_triplet_weight": rare_triplet_weight,
            "rare_triplet_start_epoch": rare_triplet_start_epoch,
            "rare_triplet_margin": rare_triplet_margin,
            "rare_triplet_min_weight": rare_triplet_min_weight,
            "max_triplet_anchors_per_batch": max_triplet_anchors,
            "use_batch_conditioning": use_dann,
            "adversarial_batch_weight": adv_weight,
            "adversarial_lambda": adv_lambda,
            "adversarial_start_epoch": adv_start_epoch,
            "adversarial_ramp_epochs": adv_ramp_epochs,
            "hdbscan_min_cluster_size": hdbscan_min_cluster_size,
            "hdbscan_min_samples": hdbscan_min_samples,
            "hdbscan_cluster_selection_method": hdbscan_selection,
            "capture_embedding_snapshots": False,
        }

        trial_name = f"trial_{trial.number:04d}"
        trial_dir = output_root / trial_name

        # === Dry-run mode: just log config and return a dummy score ===
        if dry_run:
            trial_dir.mkdir(parents=True, exist_ok=True)
            with open(trial_dir / "config.json", "w") as f:
                json.dump(
                    {
                        "trial_number": trial.number,
                        "params": param_overrides,
                        "final_clustering": final_clustering,
                        "leiden_n_neighbors": leiden_n_neighbors,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            logger.info(
                "[DRY-RUN] Trial %d: %s clustering, DANN=%s",
                trial.number,
                final_clustering,
                use_dann,
            )
            return 0.0

        # === Real run ===
        scores = []
        for seed_offset in range(n_seeds):
            trial_seed = base_seed + seed_offset
            seed_dir = trial_dir / f"seed_{trial_seed}" if n_seeds > 1 else trial_dir

            try:
                metrics = _run_single_trial(
                    preset_name=preset_name,
                    data_path=data_path,
                    output_dir=seed_dir,
                    device=device,
                    seed=trial_seed,
                    param_overrides=param_overrides,
                    metrics_only=metrics_only,
                    final_clustering=final_clustering,
                    leiden_n_neighbors=leiden_n_neighbors,
                )
                score = metrics["composite_score"]
                scores.append(score)

                logger.info(
                    "Trial %d (seed %d): score=%.4f | ARI=%.4f NMI=%.4f F1=%.4f "
                    "BalACC=%.4f RareACC=%.4f | %s | DANN=%s | %.0fs",
                    trial.number,
                    trial_seed,
                    score,
                    metrics.get("ARI", 0),
                    metrics.get("NMI", 0),
                    metrics.get("F1_Macro", 0),
                    metrics.get("BalancedACC", 0),
                    metrics.get("RareACC", 0),
                    final_clustering,
                    use_dann,
                    metrics.get("elapsed_seconds", 0),
                )

            except Exception as exc:
                logger.error("Trial %d seed %d FAILED: %s", trial.number, trial_seed, exc)
                scores.append(0.0)
                # Save error info
                seed_dir.mkdir(parents=True, exist_ok=True)
                with open(seed_dir / "error.txt", "w") as f:
                    f.write(f"Error: {exc}\n")

        avg_score = float(np.mean(scores)) if scores else 0.0

        # Append to global CSV
        _append_to_global_csv(
            output_root / "all_trials.csv",
            trial_number=trial.number,
            score=avg_score,
            params=param_overrides,
            final_clustering=final_clustering,
            leiden_n_neighbors=leiden_n_neighbors if final_clustering == "leiden" else None,
        )

        return avg_score

    return objective


def _append_to_global_csv(
    csv_path: Path,
    trial_number: int,
    score: float,
    params: Dict[str, Any],
    final_clustering: str,
    leiden_n_neighbors: Optional[int],
) -> None:
    """Append one row to the global results CSV."""
    row = {"trial": trial_number, "composite_score": round(score, 6)}
    row["final_clustering"] = final_clustering
    if leiden_n_neighbors is not None:
        row["leiden_n_neighbors"] = leiden_n_neighbors
    # Flatten params (skip internal keys)
    for k, v in sorted(params.items()):
        if not k.startswith("_"):
            row[k] = v

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def _generate_summary_report(study, output_root: Path) -> None:
    """Generate a human-readable summary of the Optuna study."""
    report_path = output_root / "search_summary.json"
    best = study.best_trial

    summary = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "best_trial_number": best.number,
        "best_composite_score": round(best.value, 6),
        "best_params": best.params,
        "score_weights": SCORE_WEIGHTS,
        "timestamp": datetime.now().isoformat(),
    }

    # Top-10 trials
    top_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
        reverse=True,
    )[:10]
    summary["top_10_trials"] = [
        {"trial": t.number, "score": round(t.value, 6), "params": t.params}
        for t in top_trials
    ]

    # Param importance (if enough trials)
    try:
        import optuna
        if len(study.trials) >= 10:
            importances = optuna.importance.get_param_importances(study)
            summary["param_importances"] = {
                k: round(v, 4) for k, v in importances.items()
            }
    except Exception:
        pass

    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary report saved to %s", report_path)

    # Also print a nice table
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH RESULTS")
    print("=" * 80)
    print(f"Total trials: {len(study.trials)}")
    print(f"Best trial:   #{best.number} (score={best.value:.4f})")
    print("\nBest parameters:")
    for k, v in sorted(best.params.items()):
        print(f"  {k:40s} = {v}")
    print("\nTop 10 trials:")
    for t in top_trials:
        print(f"  #{t.number:4d}  score={t.value:.4f}")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ultra-comprehensive Optuna hyperparameter search for scRAW + DANN"
    )
    p.add_argument("--preset", required=True, help="Preset name (e.g. baron_best)")
    p.add_argument("--data", required=True, help="Path to .h5ad data file")
    p.add_argument("--output-root", required=True, help="Output directory")
    p.add_argument("--device", default="auto", help="Device: auto|cuda|cpu|mps")
    p.add_argument("--seed", type=int, default=42, help="Base random seed")
    p.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Number of seeds per trial for robustness (1-3)",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=300,
        help="Number of Optuna trials",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Maximum total search time in seconds (optional)",
    )
    p.add_argument("--dry-run", action="store_true", help="Log configs without training")
    p.add_argument(
        "--study-name",
        default=None,
        help="Optuna study name (defaults to auto-generated)",
    )
    p.add_argument("--metrics-only", action="store_true", default=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        return 1

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Configure numba/matplotlib cache for cluster environments
    cache_dir = output_root / ".cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir / "numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "mpl"))

    # Import optuna
    try:
        import optuna
    except ImportError:
        logger.error(
            "optuna is not installed. Run: pip install optuna"
        )
        return 1

    # Optuna logging level
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Study name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = args.study_name or f"scraw_{args.preset}_{timestamp}"

    # SQLite storage for persistence
    db_path = output_root / "optuna_study.db"
    storage = f"sqlite:///{db_path}"

    # Create or load study
    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        multivariate=True,
        n_startup_trials=20,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )

    logger.info("=" * 70)
    logger.info("scRAW Hyperparameter Search")
    logger.info("=" * 70)
    logger.info("Study name:   %s", study_name)
    logger.info("Preset:       %s", args.preset)
    logger.info("Data:         %s", data_path)
    logger.info("Output root:  %s", output_root)
    logger.info("Device:       %s", args.device)
    logger.info("N trials:     %d", args.n_trials)
    logger.info("N seeds/trial:%d", args.n_seeds)
    logger.info("Dry run:      %s", args.dry_run)
    logger.info("DB:           %s", db_path)
    logger.info("Score:        %s", SCORE_WEIGHTS)
    logger.info("=" * 70)

    # Save search config
    search_config = {
        "study_name": study_name,
        "preset": args.preset,
        "data": str(data_path),
        "device": args.device,
        "seed": args.seed,
        "n_seeds": args.n_seeds,
        "n_trials": args.n_trials,
        "timeout": args.timeout,
        "dry_run": args.dry_run,
        "score_weights": SCORE_WEIGHTS,
        "timestamp": timestamp,
    }
    with open(output_root / "search_config.json", "w") as f:
        json.dump(search_config, f, indent=2)

    # Build and run objective
    objective = _create_objective(
        preset_name=args.preset,
        data_path=data_path,
        output_root=output_root,
        device=args.device,
        base_seed=args.seed,
        n_seeds=args.n_seeds,
        metrics_only=args.metrics_only,
        dry_run=args.dry_run,
    )

    n_completed = len([t for t in study.trials if t.state.name == "COMPLETE"])
    n_remaining = max(0, args.n_trials - n_completed)

    if n_completed > 0:
        logger.info(
            "Resuming: %d completed, %d remaining", n_completed, n_remaining
        )

    if n_remaining > 0:
        study.optimize(
            objective,
            n_trials=n_remaining,
            timeout=args.timeout,
            show_progress_bar=True,
        )

    # Generate summary
    if not args.dry_run and len(study.trials) > 0:
        _generate_summary_report(study, output_root)

    logger.info("Search complete. Results in %s", output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
