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


def _extract_final_weight_component(
    snapshots: List[Dict[str, Any]],
    n_cells: int,
    key: str,
) -> Optional[np.ndarray]:
    """Return last valid per-cell weight component from snapshots."""
    for snap in reversed(snapshots):
        vals = snap.get(key)
        if vals is None:
            continue
        arr = np.asarray(vals, dtype=np.float32)
        if len(arr) == n_cells:
            return arr
    return None


def _snapshot_epoch(snap: Dict[str, Any]) -> Optional[int]:
    """Parse snapshot epoch as int."""
    try:
        return int(snap.get("epoch"))
    except Exception:
        return None


def _select_snapshots_for_requested_epochs(
    snapshots: List[Dict[str, Any]],
    warmup_epochs: int,
    step: int = 10,
) -> List[Dict[str, Any]]:
    """Select snapshots: epoch 0 pre-backward, epoch warmup-1, then every `step` epochs."""
    valid = [s for s in snapshots if s.get("embeddings") is not None]
    if not valid:
        return []

    pre0 = next((s for s in valid if str(s.get("snapshot_type", "")) == "pre_backward"), None)
    by_epoch: Dict[int, Dict[str, Any]] = {}
    for s in valid:
        e = _snapshot_epoch(s)
        if e is None:
            continue
        if e not in by_epoch:
            by_epoch[e] = s

    max_epoch = max(by_epoch.keys()) if by_epoch else 0
    anchor = int(max(0, warmup_epochs - 1))
    anchor = min(anchor, max_epoch)
    target_epochs = [anchor]
    e = anchor + int(max(1, step))
    while e <= max_epoch:
        target_epochs.append(e)
        e += int(max(1, step))
    if max_epoch not in target_epochs:
        target_epochs.append(max_epoch)

    out: List[Dict[str, Any]] = []
    if pre0 is not None:
        out.append(pre0)
    for te in target_epochs:
        s = by_epoch.get(int(te))
        if s is not None and s not in out:
            out.append(s)
    if not out:
        out = [by_epoch[e] for e in sorted(by_epoch.keys())]
    return out


def _snapshot_component_vector(snapshot: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    """Read one component vector from a snapshot."""
    vals = snapshot.get(key)
    if vals is None:
        return None
    arr = np.asarray(vals, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] == 0:
        return None
    return arr


def _lagged_component_vectors(
    snapshots: List[Dict[str, Any]],
    key: str,
    lag: int,
    phase2_start_epoch: int,
) -> List[Optional[np.ndarray]]:
    """Build per-snapshot lagged vectors (epoch n-`lag`) for epoch n projections."""
    by_epoch: Dict[int, Dict[str, Any]] = {}
    for s in snapshots:
        e = _snapshot_epoch(s)
        if e is None:
            continue
        by_epoch[e] = s

    out: List[Optional[np.ndarray]] = []
    for s in snapshots:
        e = _snapshot_epoch(s)
        if e is None or e < int(phase2_start_epoch):
            out.append(None)
            continue
        prev = by_epoch.get(int(e - lag))
        if prev is None:
            out.append(None)
            continue
        out.append(_snapshot_component_vector(prev, key))
    return out


def _leiden_optimized_for_target_clusters(
    embeddings: np.ndarray,
    seed: int,
    target_clusters: int = 14,
    labels_true: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run Leiden resolution search targeting `target_clusters` and maximizing ARI when available."""
    import anndata as ad
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score

    emb = np.asarray(embeddings, dtype=np.float32)
    n_cells = int(emb.shape[0])
    if n_cells < 3:
        labels = np.zeros(n_cells, dtype=np.int64)
        return labels, {"resolution": 0.0, "n_clusters": 1, "target_clusters": int(target_clusters)}

    adata = ad.AnnData(X=emb)
    n_neighbors = max(2, min(15, n_cells - 1))
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep="X",
        method="gauss",
        transformer="sklearn",
        random_state=int(seed),
    )

    true_arr = None if labels_true is None else np.asarray(labels_true, dtype=object)
    has_truth = true_arr is not None and len(true_arr) == n_cells

    candidates: List[Dict[str, Any]] = []
    for res in np.arange(0.05, 3.01, 0.05):
        sc.tl.leiden(adata, resolution=float(res), random_state=int(seed), key_added="_leiden_tmp")
        labels = adata.obs["_leiden_tmp"].astype(int).to_numpy(dtype=np.int64, copy=False)
        n_found = int(len(np.unique(labels)))
        ari = float("nan")
        if has_truth:
            try:
                ari = float(adjusted_rand_score(true_arr, labels))
            except Exception:
                ari = float("nan")
        candidates.append(
            {
                "resolution": float(res),
                "labels": labels.copy(),
                "n_clusters": n_found,
                "diff": abs(n_found - int(target_clusters)),
                "ari": ari,
            }
        )

    if not candidates:
        labels = np.zeros(n_cells, dtype=np.int64)
        return labels, {"resolution": 0.0, "n_clusters": 1, "target_clusters": int(target_clusters)}

    exact = [c for c in candidates if c["n_clusters"] == int(target_clusters)]
    if has_truth and exact:
        finite_exact = [c for c in exact if np.isfinite(float(c["ari"]))]
        if finite_exact:
            best = sorted(finite_exact, key=lambda c: (-float(c["ari"]), float(c["resolution"])))[0]
        else:
            best = sorted(exact, key=lambda c: float(c["resolution"]))[0]
    else:
        pool = exact if exact else candidates
        if has_truth:
            best = sorted(
                pool,
                key=lambda c: (int(c["diff"]), -float(c["ari"]) if np.isfinite(c["ari"]) else 1e9, float(c["resolution"])),
            )[0]
        else:
            best = sorted(pool, key=lambda c: (int(c["diff"]), float(c["resolution"])))[0]

    info = {
        "resolution": float(best["resolution"]),
        "n_clusters": int(best["n_clusters"]),
        "target_clusters": int(target_clusters),
        "ARI_proxy": float(best["ari"]) if np.isfinite(float(best["ari"])) else None,
    }
    return np.asarray(best["labels"], dtype=np.int64), info


def _metric_row_from_bundle(
    epoch: int,
    method: str,
    metrics: Dict[str, Any],
    n_clusters: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize one metric row for CSV/plots."""
    row: Dict[str, Any] = {
        "epoch": int(epoch),
        "method": str(method),
        "NMI": metrics.get("NMI"),
        "ARI": metrics.get("ARI"),
        "ACC": metrics.get("ACC"),
        "BalancedACC": metrics.get("BalancedACC"),
        "F1_Macro": metrics.get("F1_Macro"),
        "RareACC": metrics.get("RareACC"),
        "n_clusters_found": int(n_clusters),
    }
    if extra:
        row.update(extra)
    return row


def _write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dict rows to CSV."""
    if not rows:
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(str(k))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _as_jsonable(v) for k, v in row.items()})


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
    if not args.metrics_only and args.capture_snapshots == "auto":
        # Figures epoch-wise require latent snapshots (enabled by default when plotting).
        scraw_params["capture_embedding_snapshots"] = True
        scraw_params.setdefault("snapshot_interval_epochs", 10)

    output.mkdir(parents=True, exist_ok=True)
    config_dir = output / "config"
    data_dir = output / "data"
    figures_dir = output / "figures"
    fig_umap_dir = figures_dir / "umaps"
    fig_umap_overview_dir = fig_umap_dir / "overview"
    fig_umap_labels_dir = fig_umap_dir / "labels"
    fig_umap_batch_dir = fig_umap_dir / "batch"
    fig_umap_weights_dir = fig_umap_dir / "weights"
    fig_umap_weights_cluster_dir = fig_umap_weights_dir / "cluster_component"
    fig_umap_weights_density_dir = fig_umap_weights_dir / "density_component"
    fig_umap_weights_fused_dir = fig_umap_weights_dir / "fused_weight"
    fig_loss_dir = figures_dir / "loss"
    fig_metrics_dir = figures_dir / "metrics"
    results_dir = output / "results"
    clustering_dir = results_dir / "clustering_final"
    epoch_metrics_dir = results_dir / "epoch_metrics"
    labels_dir = results_dir / "labels"
    loss_dir = results_dir / "loss_history"
    save_processed_data = str(getattr(args, "save_processed_data", "off")).lower() == "on"

    for p in (
        config_dir,
        figures_dir,
        fig_umap_dir,
        fig_umap_overview_dir,
        fig_umap_labels_dir,
        fig_umap_batch_dir,
        fig_umap_weights_dir,
        fig_umap_weights_cluster_dir,
        fig_umap_weights_density_dir,
        fig_umap_weights_fused_dir,
        fig_loss_dir,
        fig_metrics_dir,
        results_dir,
        clustering_dir,
        epoch_metrics_dir,
        labels_dir,
        loss_dir,
    ):
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
        _compute_shared_umap_sequence,
        plot_loss_curves,
        plot_loss_curves_timeline,
        plot_marker_overlap_heatmap,
        plot_umap_batch,
        plot_umap_comparison,
        plot_umap_evolution,
        plot_umap_snapshots_categorical_panels,
        plot_umap_snapshots_gradient_panels,
        plot_metric_evolution_curves,
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

    warmup_epochs_eff = int(
        effective_params.get(
            "warmup_epochs",
            scraw_params.get("warmup_epochs", 30),
        )
        or 30
    )
    epochs_eff = int(effective_params.get("epochs", scraw_params.get("epochs", 120)) or 120)
    final_epoch = max(0, epochs_eff - 1)

    selected_snapshots = _select_snapshots_for_requested_epochs(
        snapshots=snapshots,
        warmup_epochs=warmup_epochs_eff,
        step=10,
    )

    epoch_metric_rows: List[Dict[str, Any]] = []
    if selected_snapshots and true_labels_raw is not None:
        for snap in selected_snapshots:
            epoch_idx = _snapshot_epoch(snap)
            emb_snap = snap.get("embeddings")
            if epoch_idx is None or emb_snap is None:
                continue
            emb_arr = np.asarray(emb_snap, dtype=np.float32)
            if emb_arr.ndim != 2 or emb_arr.shape[0] != len(true_labels_raw):
                continue
            try:
                labels_h = algo._hdbscan_clustering(emb_arr)
                m_h = compute_metrics(true_labels_raw, labels_h, embeddings=emb_arr)
                epoch_metric_rows.append(
                    _metric_row_from_bundle(
                        epoch=epoch_idx,
                        method="hdbscan",
                        metrics=m_h,
                        n_clusters=int(len(np.unique(labels_h))),
                    )
                )
            except Exception as exc:
                logger.warning("Epoch metric (HDBSCAN) failed at epoch %s: %s", epoch_idx, exc)

            try:
                labels_l, l_info = _leiden_optimized_for_target_clusters(
                    embeddings=emb_arr,
                    seed=args.seed,
                    target_clusters=14,
                    labels_true=true_labels_raw,
                )
                m_l = compute_metrics(true_labels_raw, labels_l, embeddings=emb_arr)
                epoch_metric_rows.append(
                    _metric_row_from_bundle(
                        epoch=epoch_idx,
                        method="leiden_target14",
                        metrics=m_l,
                        n_clusters=int(l_info.get("n_clusters", len(np.unique(labels_l)))),
                        extra={"resolution": l_info.get("resolution")},
                    )
                )
            except Exception as exc:
                logger.warning("Epoch metric (Leiden target14) failed at epoch %s: %s", epoch_idx, exc)

    if epoch_metric_rows:
        _write_rows_csv(epoch_metrics_dir / "metrics_by_epoch.csv", epoch_metric_rows)
        _save_json(
            epoch_metrics_dir / "metrics_by_epoch.json",
            {
                "n_rows": len(epoch_metric_rows),
                "methods": sorted({str(r.get("method")) for r in epoch_metric_rows}),
                "rows": epoch_metric_rows,
            },
        )

    final_clustering_rows: List[Dict[str, Any]] = []
    final_clustering_rows.append(
        _metric_row_from_bundle(
            epoch=final_epoch,
            method="hdbscan_final",
            metrics=metrics,
            n_clusters=int(len(np.unique(pred_labels))),
            extra={"resolution": None},
        )
    )

    leiden_final_labels: Optional[np.ndarray] = None
    leiden_final_metrics: Optional[Dict[str, Any]] = None
    leiden_final_info: Dict[str, Any] = {}
    if embeddings is not None and len(embeddings) == len(pred_labels):
        try:
            leiden_final_labels, leiden_final_info = _leiden_optimized_for_target_clusters(
                embeddings=embeddings,
                seed=args.seed,
                target_clusters=14,
                labels_true=true_labels_raw,
            )
            leiden_final_metrics = compute_metrics(true_labels_raw, leiden_final_labels, embeddings=embeddings)
            final_clustering_rows.append(
                _metric_row_from_bundle(
                    epoch=final_epoch,
                    method="leiden_target14_final",
                    metrics=leiden_final_metrics,
                    n_clusters=int(leiden_final_info.get("n_clusters", len(np.unique(leiden_final_labels)))),
                    extra={"resolution": leiden_final_info.get("resolution")},
                )
            )
        except Exception as exc:
            logger.warning("Final Leiden target14 clustering failed: %s", exc)

    _write_rows_csv(clustering_dir / "final_clustering_comparison.csv", final_clustering_rows)
    _save_json(
        clustering_dir / "final_clustering_comparison.json",
        {
            "final_epoch": final_epoch,
            "rows": final_clustering_rows,
            "leiden_target14_info": leiden_final_info,
        },
    )

    # Save labels for reproducible post-hoc diagnostics.
    labels_compare_payload: Dict[str, List[str]] = {
        "hdbscan_final": [str(x) for x in np.asarray(pred_labels)],
    }
    if leiden_final_labels is not None:
        labels_compare_payload["leiden_target14_final"] = [
            str(x) for x in np.asarray(leiden_final_labels)
        ]
    if true_labels_raw is not None and len(true_labels_raw) == len(pred_labels):
        labels_compare_payload["true_label"] = [str(x) for x in np.asarray(true_labels_raw)]
    with (clustering_dir / "final_clustering_labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(labels_compare_payload.keys()))
        writer.writeheader()
        rows = zip(*[labels_compare_payload[k] for k in labels_compare_payload.keys()])
        for vals in rows:
            writer.writerow({k: v for k, v in zip(labels_compare_payload.keys(), vals)})

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
                    "final_clustering_comparison": final_clustering_rows,
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
        batch_labels: Optional[np.ndarray] = None
        if batch_key is not None and batch_key in adata_proc.obs.columns:
            tmp_batch = adata_proc.obs[batch_key].astype(str).to_numpy()
            if len(tmp_batch) == len(pred_labels):
                batch_labels = np.asarray(tmp_batch, dtype=object)

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
            fig.savefig(
                fig_umap_overview_dir / "umap_comparison_scraw.png",
                bbox_inches="tight",
                dpi=150,
            )
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
                    fig_hm.savefig(
                        fig_umap_overview_dir / "marker_overlap_heatmap_scraw.png",
                        bbox_inches="tight",
                        dpi=150,
                    )
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

        if batch_labels is not None:
            fig_b = plot_umap_batch(
                embeddings=embeddings,
                batch_labels=batch_labels,
                title=f"scraw (Batch: {batch_key})",
                params_info=params_info,
                dataset_info=dataset_info,
            )
            fig_b.savefig(
                fig_umap_overview_dir / "umap_batch_scraw.png",
                bbox_inches="tight",
                dpi=150,
            )
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
            fig_w.savefig(
                fig_umap_weights_fused_dir / "umap_scraw_weighted_alpha.png",
                bbox_inches="tight",
                dpi=150,
            )
            plt.close(fig_w)

            fig_wg = plot_umap_weighted_gradient(
                embeddings=embeddings,
                cell_weights=weights,
                title="scraw (Cell Weights Gradient)",
                params_info=params_info,
                dataset_info=dataset_info,
            )
            fig_wg.savefig(
                fig_umap_weights_fused_dir / "umap_scraw_weighted_gradient_final.png",
                bbox_inches="tight",
                dpi=150,
            )
            plt.close(fig_wg)

        if selected_snapshots:
            shared_snapshot_proj: Optional[List[np.ndarray]] = None
            try:
                shared_snapshot_proj, _ = _compute_shared_umap_sequence(
                    [np.asarray(s["embeddings"], dtype=np.float32) for s in selected_snapshots],
                    random_state=args.seed,
                )
            except Exception as exc:
                logger.warning("Shared UMAP projection for snapshot panels failed: %s", exc)
                shared_snapshot_proj = None

            labels_for_panels = true_labels_raw if true_labels_raw is not None else pred_labels
            fig_labels_panel = plot_umap_snapshots_categorical_panels(
                embedding_snapshots=selected_snapshots,
                labels=np.asarray(labels_for_panels),
                title="UMAP snapshots (labels) - epoch 0 pre-backward, epoch 29, then every 10 epochs",
                point_size=3,
                random_state=args.seed,
                projection_2d_per_snapshot=shared_snapshot_proj,
                params_info=params_info,
                dataset_info=dataset_info,
            )
            if fig_labels_panel is not None:
                fig_labels_panel.savefig(
                    fig_umap_labels_dir / "umap_labels_snapshots_panels.png",
                    bbox_inches="tight",
                    dpi=150,
                )
                plt.close(fig_labels_panel)

            if batch_labels is not None:
                fig_batch_panel = plot_umap_snapshots_categorical_panels(
                    embedding_snapshots=selected_snapshots,
                    labels=np.asarray(batch_labels),
                    title=f"UMAP snapshots (batch={batch_key}) - epoch 0 pre-backward, epoch 29, then every 10 epochs",
                    point_size=3,
                    random_state=args.seed,
                    projection_2d_per_snapshot=shared_snapshot_proj,
                    params_info=params_info,
                    dataset_info=dataset_info,
                )
                if fig_batch_panel is not None:
                    fig_batch_panel.savefig(
                        fig_umap_batch_dir / "umap_batch_snapshots_panels.png",
                        bbox_inches="tight",
                        dpi=150,
                    )
                    plt.close(fig_batch_panel)

            component_specs = [
                ("cluster_component_weights", "Cluster Component", fig_umap_weights_cluster_dir),
                ("density_component_weights", "Density Component", fig_umap_weights_density_dir),
                ("cell_weights", "Fused Reconstruction Weight (Cluster + Density)", fig_umap_weights_fused_dir),
            ]
            for comp_key, comp_name, comp_dir in component_specs:
                current_vectors = [
                    _snapshot_component_vector(s, comp_key) for s in selected_snapshots
                ]
                lag_vectors = _lagged_component_vectors(
                    snapshots=selected_snapshots,
                    key=comp_key,
                    lag=10,
                    phase2_start_epoch=warmup_epochs_eff,
                )
                fig_comp = plot_umap_snapshots_gradient_panels(
                    embedding_snapshots=selected_snapshots,
                    current_weights=current_vectors,
                    lagged_weights=lag_vectors,
                    title=f"UMAP snapshots ({comp_name})",
                    point_size=3,
                    random_state=args.seed,
                    projection_2d_per_snapshot=shared_snapshot_proj,
                    current_row_label="Current epoch n weights",
                    lagged_row_label="Lagged epoch n-10 weights on epoch n latent",
                    params_info=params_info,
                    dataset_info=dataset_info,
                )
                if fig_comp is not None:
                    out_name = comp_key.replace("_weights", "").replace("_", "-")
                    fig_comp.savefig(
                        comp_dir / f"umap_gradient_panels_{out_name}.png",
                        bbox_inches="tight",
                        dpi=150,
                    )
                    plt.close(fig_comp)

            labels_for_evo = true_labels_raw if true_labels_raw is not None else pred_labels
            fig_evo = plot_umap_evolution(
                embedding_snapshots=selected_snapshots,
                labels=labels_for_evo,
                algorithm_name="scraw",
                params_info=params_info,
                dataset_info=dataset_info,
            )
            if fig_evo is not None:
                fig_evo.savefig(
                    fig_umap_overview_dir / "umap_evolution_scraw_run0.png",
                    bbox_inches="tight",
                    dpi=150,
                )
                plt.close(fig_evo)

        fig_loss = plot_loss_curves(loss_history, algorithm_name="scraw")
        if fig_loss is not None:
            fig_loss.savefig(
                fig_loss_dir / "loss_curves_by_phase_scraw_run0.png",
                bbox_inches="tight",
                dpi=150,
            )
            plt.close(fig_loss)

        fig_loss_timeline = plot_loss_curves_timeline(loss_history, algorithm_name="scraw")
        if fig_loss_timeline is not None:
            fig_loss_timeline.savefig(
                fig_loss_dir / "loss_curves_timeline_scraw_run0.png",
                bbox_inches="tight",
                dpi=150,
            )
            plt.close(fig_loss_timeline)

        fig_metrics = plot_metric_evolution_curves(
            epoch_metric_rows,
            title="Epoch-wise Metrics (HDBSCAN vs Leiden target=14)",
        )
        if fig_metrics is not None:
            fig_metrics.savefig(
                fig_metrics_dir / "metrics_evolution_by_epoch_scraw_run0.png",
                bbox_inches="tight",
                dpi=150,
            )
            plt.close(fig_metrics)

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
