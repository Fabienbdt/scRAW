"""Preprocessing utilities for the scRAW pipeline."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict

import logging
import numpy as np


logger = logging.getLogger(__name__)


def _as_dict(params: Any) -> Dict[str, Any]:
    """Convert a dataclass or mapping-like config into a plain dictionary."""
    if is_dataclass(params):
        return asdict(params)
    return dict(params)


def preprocess_adata(adata: Any, params: Any) -> Any:
    """Apply the default scRAW preprocessing path on a raw-count AnnData object."""
    import scanpy as sc

    cfg = _as_dict(params)
    adata = adata.copy()

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

    X_probe = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    looks_processed = bool(X_probe.size and np.nanmin(X_probe) < 0)
    if looks_processed:
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

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = np.asarray(X, dtype=np.float32)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0.0] = 1.0
    X = (X - mean) / std
    X = np.clip(X, -float(cfg.get("scale_max_value", 10.0)), float(cfg.get("scale_max_value", 10.0)))
    adata.X = np.asarray(X, dtype=np.float32)
    return adata
