#!/usr/bin/env python3
"""Single-run CLI for the standalone scRAW project."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import align_labels, compute_metrics, marker_overlap_annotation
from .presets import PRESETS, get_preset
from .preprocessing import preprocess_adata

logger = logging.getLogger("scraw_dedicated")


def _setup_logging(verbose: bool) -> None:
    """Configure le niveau et le format des logs pour le terminal.
    
    
    Args:
        verbose: Paramètre d'entrée `verbose` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def _configure_runtime_cache(output_dir: Path) -> None:
    """Configure writable cache directories for numba/matplotlib."""
    cache_root = output_dir / "tmp_cache"
    numba_dir = cache_root / "numba"
    mpl_dir = cache_root / "mpl"
    xdg_dir = cache_root / "xdg_cache"

    for p in (cache_root, numba_dir, mpl_dir, xdg_dir):
        p.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_dir))
    # Parity with reference SCRBenchmark scripts (Deep2 baseline).
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _detect_label_key(obs_columns: Sequence[str]) -> Optional[str]:
    """Détecte automatiquement la colonne de labels biologiques dans `adata.obs`.
    
    
    Args:
        obs_columns: Paramètre d'entrée `obs_columns` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    # Keep same priority as SCRBenchmark DataHandler.LABEL_COLUMNS.
    candidates = [
        "Group",
        "label",
        "cell_type",
        "cluster",
        "Groupe",
        "Y",
        "celltype",
        "CellType",
        "cell_types",
        "labels",
        "clusters",
        "assigned_cluster",
    ]
    for c in candidates:
        if c in obs_columns:
            return c
    return None


def _detect_batch_key(obs_columns: Sequence[str], preferred: Optional[str] = None) -> Optional[str]:
    """Détecte automatiquement la colonne batch dans `adata.obs`.
    
    
    Args:
        obs_columns: Paramètre d'entrée `obs_columns` utilisé dans cette étape du pipeline.
        preferred: Paramètre d'entrée `preferred` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    if preferred and preferred in obs_columns:
        return preferred
    for c in ["batch", "Batch", "study", "dataset", "donor", "sample", "patient", "tech"]:
        if c in obs_columns:
            return c
    return None


def _detect_batch_key_in_file(data_path: Path, preferred: Optional[str]) -> Optional[str]:
    """Lit un fichier h5ad en mode léger pour détecter la colonne batch sans tout charger.
    
    
    Args:
        data_path: Paramètre d'entrée `data_path` utilisé dans cette étape du pipeline.
        preferred: Paramètre d'entrée `preferred` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    import anndata as ad

    adata = ad.read_h5ad(data_path, backed="r")
    try:
        cols = list(adata.obs.columns)
        return _detect_batch_key(cols, preferred=preferred)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


def _as_jsonable(value: Any) -> Any:
    """Helper interne: as jsonable.
    
    
    Args:
        value: Paramètre d'entrée `value` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_jsonable(v) for v in value]
    return value


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Helper interne: save json.
    
    
    Args:
        path: Paramètre d'entrée `path` utilisé dans cette étape du pipeline.
        payload: Paramètre d'entrée `payload` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    path.write_text(json.dumps(_as_jsonable(payload), indent=2), encoding="utf-8")


def _save_csv(path: Path, row: Dict[str, Any]) -> None:
    """Helper interne: save csv.
    
    
    Args:
        path: Paramètre d'entrée `path` utilisé dans cette étape du pipeline.
        row: Paramètre d'entrée `row` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow({k: _as_jsonable(v) for k, v in row.items()})


def _safe_numpy(x: Any) -> np.ndarray:
    """Helper interne: safe numpy.
    
    
    Args:
        x: Paramètre d'entrée `x` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    return np.asarray(x)


def _parse_scalar(raw: str) -> Any:
    """Convertit une valeur texte CLI en booléen, entier, flottant, None ou chaîne.
    
    
    Args:
        raw: Paramètre d'entrée `raw` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    text = raw.strip()
    low = text.lower()

    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None

    if "," in text:
        return text

    if re.fullmatch(r"[-+]?\d+", text):
        try:
            return int(text)
        except Exception:
            return text

    if re.fullmatch(r"[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?", text):
        try:
            return float(text)
        except Exception:
            return text

    return text


def _parse_kv_overrides(items: Sequence[str]) -> Dict[str, Any]:
    """Convertit une liste `KEY=VALUE` en dictionnaire Python prêt à l'emploi.
    
    
    Args:
        items: Paramètre d'entrée `items` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    out: Dict[str, Any] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid override '{raw}'. Expected KEY=VALUE.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{raw}'. Empty key.")
        # Compatibility with legacy SCRBenchmark syntax: "scraw:param=value"
        # (and similar namespace prefixes for preprocess overrides).
        if ":" in key:
            ns, bare_key = key.split(":", 1)
            ns = ns.strip().lower()
            if ns in {"scraw", "algorithm", "algo", "preprocess"}:
                key = bare_key.strip()
                if not key:
                    raise ValueError(f"Invalid override '{raw}'. Empty key after namespace.")
        out[key] = _parse_scalar(value)
    return out


def _build_scraw_params(
    preset_name: str,
    seed: int,
    device: str,
    dann_mode: str,
    batch_key_override: Optional[str],
    capture_snapshots: str,
    snapshot_interval: Optional[int],
    param_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Construit la configuration finale scRAW à partir du preset et des overrides utilisateur.
    
    
    Args:
        preset_name: Paramètre d'entrée `preset_name` utilisé dans cette étape du pipeline.
        seed: Paramètre d'entrée `seed` utilisé dans cette étape du pipeline.
        device: Paramètre d'entrée `device` utilisé dans cette étape du pipeline.
        dann_mode: Paramètre d'entrée `dann_mode` utilisé dans cette étape du pipeline.
        batch_key_override: Paramètre d'entrée `batch_key_override` utilisé dans cette étape du pipeline.
        capture_snapshots: Paramètre d'entrée `capture_snapshots` utilisé dans cette étape du pipeline.
        snapshot_interval: Paramètre d'entrée `snapshot_interval` utilisé dans cette étape du pipeline.
        param_overrides: Paramètre d'entrée `param_overrides` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    preset = get_preset(preset_name)
    params = dict(preset.algorithm_params)

    params["random_state"] = int(seed)
    params["seed"] = int(seed)
    params["device"] = str(device)

    if capture_snapshots == "on":
        params["capture_embedding_snapshots"] = True
    elif capture_snapshots == "off":
        params["capture_embedding_snapshots"] = False

    if snapshot_interval is not None:
        params["snapshot_interval_epochs"] = int(snapshot_interval)

    if dann_mode == "off":
        params["use_batch_conditioning"] = False
        params["adversarial_batch_weight"] = 0.0
        params["mmd_batch_weight"] = 0.0
        params["batch_correction_key"] = str(params.get("batch_correction_key", "auto") or "auto")
    elif dann_mode == "on":
        params["use_batch_conditioning"] = True

    if batch_key_override:
        params["batch_correction_key"] = batch_key_override

    params.update(param_overrides)
    return params


def _label_encoding(labels: Sequence[Any]) -> Tuple[np.ndarray, Dict[str, str]]:
    """Helper interne: label encoding.
    
    
    Args:
        labels: Paramètre d'entrée `labels` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    labels = np.asarray([str(x) for x in labels], dtype=object)
    uniq = sorted(np.unique(labels).tolist())
    to_idx = {lab: i for i, lab in enumerate(uniq)}
    encoded = np.asarray([to_idx[l] for l in labels], dtype=np.int64)
    idx_to_label = {str(i): lab for lab, i in to_idx.items()}
    return encoded, idx_to_label


def _extract_final_cell_weights(snapshots: List[Dict[str, Any]], n_cells: int) -> Optional[np.ndarray]:
    """Helper interne: extract final cell weights.
    
    
    Args:
        snapshots: Paramètre d'entrée `snapshots` utilisé dans cette étape du pipeline.
        n_cells: Paramètre d'entrée `n_cells` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    for snap in reversed(snapshots):
        w = snap.get("cell_weights")
        if w is None:
            continue
        w = np.asarray(w, dtype=np.float32)
        if len(w) == n_cells:
            return w
    return None


def _hyperparams_declared() -> List[Dict[str, Any]]:
    """Expose la liste des hyperparamètres déclarés par l'algorithme scRAW.
    
    
    Args:
        Aucun argument explicite en dehors du contexte objet.
    
    Returns:
        Valeur calculée par la fonction.
    """
    from .algorithms.scraw_algorithm import ScRAWAlgorithm

    out: List[Dict[str, Any]] = []
    for hp in ScRAWAlgorithm.get_hyperparameters():
        out.append(
            {
                "name": hp.name,
                "display_name": hp.display_name,
                "type": hp.param_type.value,
                "default": hp.default,
                "description": hp.description,
                "min_value": hp.min_value,
                "max_value": hp.max_value,
                "choices": hp.choices,
                "category": hp.category,
                "advanced": hp.advanced,
            }
        )
    return out


def run_once(args: argparse.Namespace) -> int:
    """Exécute un run scRAW complet: préparation, entraînement, métriques et exports.
    
    
    Args:
        args: Paramètre d'entrée `args` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    preset = get_preset(args.preset)
    data_path = Path(args.data).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    preprocess_overrides = _parse_kv_overrides(args.preprocess or [])
    param_overrides = _parse_kv_overrides(args.param or [])

    preprocess_cfg = dict(preset.preprocessing)
    preprocess_cfg.update(preprocess_overrides)

    scraw_params = _build_scraw_params(
        preset_name=args.preset,
        seed=args.seed,
        device=args.device,
        dann_mode=args.dann,
        batch_key_override=args.batch_key,
        capture_snapshots=args.capture_snapshots,
        snapshot_interval=args.snapshot_interval,
        param_overrides=param_overrides,
    )

    output.mkdir(parents=True, exist_ok=True)
    config_dir = output / "config"
    data_dir = output / "data"
    figures_dir = output / "figures"
    results_dir = output / "results"
    labels_dir = results_dir / "labels"
    loss_dir = results_dir / "loss_history"
    save_processed_data = str(getattr(args, "save_processed_data", "off")).lower() == "on"

    for p in (config_dir, figures_dir, results_dir, labels_dir, loss_dir):
        p.mkdir(parents=True, exist_ok=True)
    if save_processed_data:
        data_dir.mkdir(parents=True, exist_ok=True)

    _configure_runtime_cache(output)

    # Strict DANN validation with auto-batch-key fallback.
    use_batch = bool(scraw_params.get("use_batch_conditioning", False)) or float(
        scraw_params.get("adversarial_batch_weight", 0.0) or 0.0
    ) > 0.0
    if use_batch:
        bkey = str(scraw_params.get("batch_correction_key", "")).strip() or None
        if bkey is None:
            bkey = _detect_batch_key_in_file(data_path, preferred=args.batch_key)
            if bkey is None:
                raise ValueError(
                    "DANN/batch conditioning is enabled but no batch key was found in adata.obs. "
                    "Pass --batch-key or --param batch_correction_key=..."
                )
            scraw_params["batch_correction_key"] = bkey

    import anndata as ad
    import matplotlib.pyplot as plt

    from .algorithms.scraw_algorithm import ScRAWAlgorithm
    from .visualization import (
        plot_loss_curves,
        plot_marker_overlap_heatmap,
        plot_umap_batch,
        plot_umap_comparison,
        plot_umap_evolution,
        plot_umap_weighted,
        plot_umap_weighted_gradient,
    )

    logger.info("Loading dataset: %s", data_path)
    adata = ad.read_h5ad(data_path)

    logger.info("Applying preprocessing...")
    adata_proc = preprocess_adata(adata, preprocess_cfg)
    if save_processed_data:
        adata_proc.write(data_dir / "processed.h5ad")
    else:
        logger.info("Skipping export of processed.h5ad (save_processed_data=off).")

    label_key = _detect_label_key(list(adata_proc.obs.columns))
    batch_key = _detect_batch_key(
        list(adata_proc.obs.columns),
        preferred=str(scraw_params.get("batch_correction_key", "")).strip() or None,
    )

    true_labels_raw: Optional[np.ndarray] = None
    fit_labels: Optional[np.ndarray] = None
    label_map: Dict[str, str] = {}

    if label_key is not None:
        true_labels_raw = np.asarray(adata_proc.obs[label_key].astype(str).to_numpy(), dtype=object)
        fit_labels, label_map = _label_encoding(true_labels_raw)

    if args.unsupervised:
        fit_labels = None

    algo = ScRAWAlgorithm(params=scraw_params)
    t0 = time.time()
    algo.fit(adata_proc, labels=fit_labels)
    pred_labels = _safe_numpy(algo.predict())
    runtime = float(time.time() - t0)

    embeddings = algo.get_embeddings()
    if embeddings is not None:
        embeddings = _safe_numpy(embeddings)

    loss_history = algo.get_loss_history() or []
    snapshots = algo.get_embedding_snapshots() or []
    effective_params = algo.get_effective_params()
    num_params = algo.get_num_parameters()

    metrics = compute_metrics(true_labels_raw, pred_labels, embeddings=embeddings)

    result_row: Dict[str, Any] = {
        "algorithm": "scraw",
        "run_id": 0,
        "runtime": runtime,
        "num_parameters": num_params,
        "NMI": metrics.get("NMI"),
        "ARI": metrics.get("ARI"),
        "ACC": metrics.get("ACC"),
        "UCA": metrics.get("UCA"),
        "F1_Macro": metrics.get("F1_Macro"),
        "BalancedACC": metrics.get("BalancedACC"),
        "RareACC": metrics.get("RareACC"),
        "KNN_Purity": metrics.get("KNN_Purity"),
        "ClassWise": metrics.get("ClassWise"),
        "Silhouette": metrics.get("Silhouette"),
        "n_clusters_found": metrics.get("n_clusters_found"),
        "n_samples_evaluated": metrics.get("n_samples_evaluated"),
    }
    for k, v in sorted(effective_params.items()):
        result_row[f"param_{k}"] = v

    _save_csv(results_dir / "results.csv", result_row)
    _save_csv(results_dir / "analysis_results.csv", result_row)

    label_payload: Dict[str, Any] = {
        "predicted_label": [str(x) for x in pred_labels],
    }
    if true_labels_raw is not None and len(true_labels_raw) == len(pred_labels):
        label_payload["true_label"] = [str(x) for x in true_labels_raw]
        try:
            aligned = align_labels(true_labels_raw, pred_labels)
            label_payload["aligned_predicted_label"] = [str(x) for x in aligned]
        except Exception:
            pass
    if batch_key is not None and batch_key in adata_proc.obs.columns:
        bvals = adata_proc.obs[batch_key].astype(str).to_numpy()
        if len(bvals) == len(pred_labels):
            label_payload["batch"] = [str(x) for x in bvals]

    with (labels_dir / "labels_scraw_run0.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(label_payload.keys()))
        writer.writeheader()
        rows = zip(*[label_payload[k] for k in label_payload.keys()])
        for vals in rows:
            writer.writerow({k: v for k, v in zip(label_payload.keys(), vals)})

    if label_map:
        _save_json(labels_dir / "label_map.json", label_map)

    _save_json(
        loss_dir / "loss_scraw_run0.json",
        {
            "algorithm": "scraw",
            "run_id": 0,
            "phases": loss_history,
        },
    )

    _save_json(
        results_dir / "results.json",
        {
            "results": [
                {
                    "algorithm_name": "scraw",
                    "run_id": 0,
                    "runtime": runtime,
                    "metrics": metrics,
                    "params": effective_params,
                    "num_parameters": num_params,
                    "embeddings_shape": list(embeddings.shape) if embeddings is not None else None,
                }
            ],
            "summary": {
                "scraw": {
                    "NMI_mean": metrics.get("NMI"),
                    "ARI_mean": metrics.get("ARI"),
                    "ACC_mean": metrics.get("ACC"),
                    "F1_Macro_mean": metrics.get("F1_Macro"),
                    "BalancedACC_mean": metrics.get("BalancedACC"),
                    "RareACC_mean": metrics.get("RareACC"),
                    "Silhouette_mean": metrics.get("Silhouette"),
                    "runtime_mean": runtime,
                    "n_clusters_found_mean": metrics.get("n_clusters_found"),
                }
            },
            "timestamp": datetime.now().isoformat(),
        },
    )

    config_used = {
        "data": {"file": str(data_path)},
        "preprocessing": preprocess_cfg,
        "algorithms": ["scraw"],
        "algorithm_params": {"scraw": scraw_params},
        "algorithm_effective_params_by_algorithm": {"scraw": effective_params},
        "execution": {
            "device": args.device,
            "n_repeats": 1,
            "random_seed": args.seed,
            "compute_scib_metrics": False,
            "save_processed_data": save_processed_data,
        },
        "output": {"directory": str(output)},
        "context": {
            "preset": preset.name,
            "description": preset.description,
            "timestamp": datetime.now().isoformat(),
            "label_key": label_key,
            "batch_key_detected": batch_key,
            "unsupervised": bool(args.unsupervised),
        },
    }
    _save_json(config_dir / "config_used.json", config_used)

    _save_json(
        config_dir / "algorithm_hyperparams_used.json",
        {
            "timestamp": datetime.now().isoformat(),
            "algorithms": ["scraw"],
            "declared_hyperparameters": {"scraw": _hyperparams_declared()},
            "defaults_by_algorithm": {
                "scraw": {hp["name"]: hp["default"] for hp in _hyperparams_declared()}
            },
            "overrides_by_algorithm": {"scraw": scraw_params},
            "effective_params_by_algorithm": {"scraw": effective_params},
            "execution": config_used["execution"],
            "context": {
                "mode": "standard",
                "data_file": str(data_path),
                "output_dir": str(output),
                "status": "completed",
                "preset": preset.name,
            },
            "per_run_params": [
                {
                    "algorithm": "scraw",
                    "run_id": 0,
                    "params_used": effective_params,
                }
            ],
        },
    )

    if not args.metrics_only and embeddings is not None and len(embeddings) == len(pred_labels):
        logger.info("Generating figures...")

        reverse_label_map: Optional[Dict[int, str]] = None
        if label_map:
            reverse_label_map = {}
            for k, v in label_map.items():
                try:
                    reverse_label_map[int(k)] = str(v)
                except Exception:
                    continue

        params_info = {
            "normalization": effective_params.get("nb_input_transform", "log1p"),
            "DANN_weight": effective_params.get("adversarial_batch_weight", 0),
            "MMD_weight": effective_params.get("mmd_batch_weight", 0),
            "clustering": effective_params.get("clustering_method", "hdbscan"),
            "HVG_flavor": effective_params.get("internal_hvg_flavor", "seurat"),
            "epochs": effective_params.get("epochs", "?"),
            "z_dim": effective_params.get("z_dim", "?"),
        }
        dataset_info = f"{data_path.stem} | Full data"

        if true_labels_raw is not None and len(true_labels_raw) == len(pred_labels):
            fig = plot_umap_comparison(
                embeddings=embeddings,
                true_labels=true_labels_raw,
                predicted_labels=pred_labels,
                algorithm_name="scraw",
                label_names=reverse_label_map,
                params_info=params_info,
                dataset_info=dataset_info,
            )
            fig.savefig(figures_dir / "umap_comparison_scraw.png", bbox_inches="tight", dpi=150)
            plt.close(fig)

            try:
                overlap_result = marker_overlap_annotation(
                    adata=adata_proc,
                    labels_true=true_labels_raw,
                    labels_pred=pred_labels,
                    n_top_genes=100,
                    method="wilcoxon",
                )
                fig_hm = plot_marker_overlap_heatmap(
                    overlap_matrix=overlap_result["overlap_matrix"],
                    algorithm_name="scraw",
                )
                if fig_hm is not None:
                    fig_hm.savefig(figures_dir / "marker_overlap_heatmap_scraw.png", bbox_inches="tight", dpi=150)
                    plt.close(fig_hm)

                import pandas as pd

                annot_df = pd.DataFrame(
                    {
                        "true_label": np.asarray(true_labels_raw, dtype=str),
                        "predicted_cluster": np.asarray(pred_labels, dtype=str),
                        "hungarian_annotation": np.asarray(overlap_result["hungarian_labels"], dtype=str),
                        "marker_overlap_annotation": np.asarray(overlap_result["marker_labels"], dtype=str),
                    }
                )
                annot_df.to_csv(results_dir / "annotation_comparison_scraw.csv", index=False)
                overlap_result["overlap_matrix"].to_csv(results_dir / "marker_overlap_matrix_scraw.csv")
            except Exception as exc:
                logger.warning("Marker-overlap annotation failed for scraw: %s", exc)

        if batch_key is not None and batch_key in adata_proc.obs.columns:
            batch_labels = adata_proc.obs[batch_key].astype(str).to_numpy()
            if len(batch_labels) == len(pred_labels):
                fig_b = plot_umap_batch(
                    embeddings=embeddings,
                    batch_labels=batch_labels,
                    title=f"scraw (Batch: {batch_key})",
                    params_info=params_info,
                    dataset_info=dataset_info,
                )
                fig_b.savefig(figures_dir / "umap_batch_scraw.png", bbox_inches="tight", dpi=150)
                plt.close(fig_b)

        weights = _extract_final_cell_weights(snapshots, n_cells=len(pred_labels))
        if weights is not None:
            labels_for_weight_plot = true_labels_raw if true_labels_raw is not None else pred_labels
            fig_w = plot_umap_weighted(
                embeddings=embeddings,
                labels=labels_for_weight_plot,
                cell_weights=weights,
                title="scraw (Cell Weights)",
                label_names=reverse_label_map,
                params_info=params_info,
                dataset_info=dataset_info,
            )
            fig_w.savefig(figures_dir / "umap_scraw_weighted.png", bbox_inches="tight", dpi=150)
            plt.close(fig_w)

            fig_wg = plot_umap_weighted_gradient(
                embeddings=embeddings,
                cell_weights=weights,
                title="scraw (Cell Weights Gradient)",
                params_info=params_info,
                dataset_info=dataset_info,
            )
            fig_wg.savefig(figures_dir / "umap_scraw_weighted_gradient.png", bbox_inches="tight", dpi=150)
            plt.close(fig_wg)

        if snapshots:
            labels_for_evo = true_labels_raw if true_labels_raw is not None else pred_labels
            fig_evo = plot_umap_evolution(
                embedding_snapshots=snapshots,
                labels=labels_for_evo,
                algorithm_name="scraw",
                params_info=params_info,
                dataset_info=dataset_info,
            )
            if fig_evo is not None:
                fig_evo.savefig(figures_dir / "umap_evolution_scraw_run0.png", bbox_inches="tight", dpi=150)
                plt.close(fig_evo)

        fig_loss = plot_loss_curves(loss_history, algorithm_name="scraw")
        if fig_loss is not None:
            fig_loss.savefig(figures_dir / "loss_curves_scraw_run0.png", bbox_inches="tight", dpi=150)
            plt.close(fig_loss)

    logger.info("Run completed. Output: %s", output)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Déclare toutes les options de ligne de commande supportées par le runner.
    
    
    Args:
        Aucun argument explicite en dehors du contexte objet.
    
    Returns:
        Valeur calculée par la fonction.
    """
    p = argparse.ArgumentParser(description="Standalone strict scRAW runner")
    p.add_argument("--preset", required=True, choices=sorted(PRESETS.keys()))
    p.add_argument("--data", required=True, help="Input .h5ad file")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto|cuda|cpu|mps")
    p.add_argument(
        "--save-processed-data",
        choices=["on", "off"],
        default="off",
        help="Save data/processed.h5ad in output directory (default: off)",
    )
    p.add_argument("--metrics-only", action="store_true", help="Skip figure generation")
    p.add_argument("--unsupervised", action="store_true", help="Hide labels during training")
    p.add_argument("--dann", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--batch-key", default=None, help="Batch key override when DANN is enabled")
    p.add_argument("--capture-snapshots", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--snapshot-interval", type=int, default=None)
    p.add_argument(
        "--param",
        action="append",
        default=[],
        help="Override algorithm param: KEY=VALUE (repeatable)",
    )
    p.add_argument(
        "--preprocess",
        action="append",
        default=[],
        help="Override preprocessing param: KEY=VALUE (repeatable)",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Point d'entrée principal appelé lors de l'exécution du script.
    
    
    Args:
        argv: Paramètre d'entrée `argv` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return run_once(args)
    except Exception as exc:
        logger.exception("scRAW dedicated run failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
