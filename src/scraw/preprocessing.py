"""Preprocessing utilities for the scRAW pipeline."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict


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

    sc.pp.filter_cells(adata, min_genes=int(cfg.get("min_genes_per_cell", 100)))

    max_genes = cfg.get("max_genes_per_cell")
    if max_genes is not None:
        sc.pp.calculate_qc_metrics(adata, inplace=True)
        adata = adata[adata.obs["n_genes_by_counts"] <= int(max_genes)].copy()

    sc.pp.filter_genes(adata, min_cells=int(cfg.get("min_cells_per_gene", 3)))

    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("Preprocessing removed all cells or genes.")

    sc.pp.normalize_total(adata, target_sum=float(cfg.get("target_sum", 20000.0)))
    sc.pp.log1p(adata)

    n_top_genes = int(cfg.get("n_top_genes", 2000))
    if n_top_genes > 0 and adata.n_vars > 1:
        sc.pp.highly_variable_genes(
            adata,
            flavor=str(cfg.get("hvg_flavor", "seurat")),
            n_top_genes=min(n_top_genes, int(adata.n_vars)),
            subset=True,
        )

    sc.pp.scale(adata, max_value=float(cfg.get("scale_max_value", 10.0)))
    return adata
