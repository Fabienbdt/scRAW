"""Preprocessing utilities for the scRAW pipeline."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict

import logging
import numpy as np
from scipy import sparse


logger = logging.getLogger(__name__)


def _as_dict(params: Any) -> Dict[str, Any]:
    """Convert a dataclass or mapping-like config into a plain dictionary."""
    if is_dataclass(params):
        return asdict(params)
    return dict(params)


def _has_negative_values(matrix: Any) -> bool:
    """Check whether a dense or sparse expression matrix contains negative values."""
    if sparse.issparse(matrix):
        data = np.asarray(matrix.data)
        return bool(data.size and np.nanmin(data) < 0)

    arr = np.asarray(matrix)
    return bool(arr.size and np.nanmin(arr) < 0)


def _has_nonfinite_values(matrix: Any) -> bool:
    """Return whether a dense or sparse expression matrix contains NaN/Inf."""
    values = np.asarray(matrix.data) if sparse.issparse(matrix) else np.asarray(matrix)
    return bool(values.size and not np.all(np.isfinite(values)))


def _to_dense_float32(matrix: Any) -> np.ndarray:
    """Convert one matrix to a dense float32 NumPy array."""
    if sparse.issparse(matrix):
        return matrix.toarray().astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32)


def preprocess_adata(adata: Any, params: Any) -> Any:
    """Apply filtering and the configured expression preprocessing path."""
    import scanpy as sc

    cfg = _as_dict(params)
    adata = adata.copy()

    input_mode = str(cfg.get("input_mode", "auto")).strip().lower()
    if input_mode not in {"auto", "raw", "preprocessed"}:
        raise ValueError("input_mode must be one of: auto, raw, preprocessed.")
    if _has_nonfinite_values(adata.X):
        raise ValueError("Input expression matrix contains NaN or infinite values.")

    if "original_X" not in adata.layers:
        X_orig = adata.X
        if hasattr(X_orig, "copy"):
            X_orig = X_orig.copy()
        adata.layers["original_X"] = X_orig

    min_genes = int(cfg.get("min_genes_per_cell", 0) or 0)
    if min_genes > 0:
        sc.pp.filter_cells(adata, min_genes=min_genes)

    max_genes = cfg.get("max_genes_per_cell")
    if max_genes is not None:
        sc.pp.calculate_qc_metrics(adata, inplace=True)
        if "n_genes_by_counts" in adata.obs.columns:
            adata = adata[adata.obs["n_genes_by_counts"] <= int(max_genes)].copy()

    min_cells = int(cfg.get("min_cells_per_gene", 0) or 0)
    if min_cells > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells)

    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("Preprocessing removed all cells or genes.")

    contains_negative_values = _has_negative_values(adata.X)
    if input_mode == "raw" and contains_negative_values:
        raise ValueError(
            "input_mode='raw' is incompatible with negative expression values. "
            "Use input_mode='preprocessed' for scaled data."
        )

    looks_processed = input_mode == "preprocessed" or (
        input_mode == "auto" and contains_negative_values
    )
    if looks_processed:
        if input_mode == "auto":
            logger.warning(
                "Input matrix contains negative values; assuming it is already preprocessed."
            )
    else:
        sc.pp.normalize_total(adata, target_sum=float(cfg.get("target_sum", 20000.0)))
        sc.pp.log1p(adata)

        n_top_genes = int(cfg.get("n_top_genes", 2000) or 0)
        if n_top_genes > 0 and adata.n_vars > 1:
            sc.pp.highly_variable_genes(
                adata,
                flavor=str(cfg.get("hvg_flavor", "seurat")),
                n_top_genes=min(n_top_genes, int(adata.n_vars)),
                subset=True,
            )

    X = _to_dense_float32(adata.X)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0.0] = 1.0
    X = (X - mean) / std
    scale_max_value = float(cfg.get("scale_max_value", 10.0))
    np.clip(X, -scale_max_value, scale_max_value, out=X)
    adata.X = X
    return adata
