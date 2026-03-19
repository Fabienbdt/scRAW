"""Pseudo-label and final clustering helpers for the scRAW pipeline."""

from __future__ import annotations

from typing import Optional
import logging

import numpy as np

from .config import ClusteringConfig, RuntimeConfig


logger = logging.getLogger(__name__)


def remap_contiguous_labels(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary cluster ids to contiguous integers starting at zero."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return labels.astype(np.int64)
    unique_labels = np.unique(labels)
    mapping = {label: idx for idx, label in enumerate(unique_labels.tolist())}
    return np.asarray([mapping[label] for label in labels], dtype=np.int64)


def sanitize_embeddings(values: np.ndarray) -> np.ndarray:
    """Ensure embeddings contain only finite clipped values."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e4, neginf=-1e4)
    return np.clip(arr, -1e4, 1e4).astype(np.float32, copy=False)


def _clip_k_to_data(k: int, n_cells: int) -> int:
    """Clip K to a valid range for the dataset size."""
    n_cells = int(max(1, n_cells))
    if n_cells == 1:
        return 1
    return int(max(2, min(int(k), max(2, n_cells - 1))))


def estimate_pseudo_k(n_cells: int, config: ClusteringConfig) -> int:
    """Estimate pseudo-label K with a simple bounded heuristic."""
    if int(config.pseudo_k) > 1:
        return _clip_k_to_data(int(config.pseudo_k), n_cells)

    k_min = int(max(2, config.pseudo_k_min))
    k_max = int(max(k_min, config.pseudo_k_max))
    heuristic = int(round(float(np.sqrt(max(1, n_cells) / 40.0))))
    heuristic = int(np.clip(heuristic, k_min, k_max))
    return _clip_k_to_data(heuristic, n_cells)


def kmeans_labels(embeddings: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Compute KMeans labels."""
    from sklearn.cluster import KMeans

    emb = sanitize_embeddings(embeddings)
    k = _clip_k_to_data(k, emb.shape[0])
    labels = KMeans(n_clusters=k, random_state=int(seed), n_init=10).fit_predict(emb)
    return remap_contiguous_labels(labels)


def leiden_labels(embeddings: np.ndarray, target_k: int, seed: int) -> np.ndarray:
    """Compute Leiden labels by scanning the resolution toward `target_k`."""
    import anndata as ad
    import scanpy as sc

    emb = sanitize_embeddings(embeddings)
    n_cells = emb.shape[0]
    if n_cells < 3:
        return np.zeros(n_cells, dtype=np.int64)

    adata = ad.AnnData(X=emb)
    sc.pp.neighbors(
        adata,
        n_neighbors=min(15, n_cells - 1),
        use_rep="X",
        method="gauss",
        transformer="sklearn",
        random_state=int(seed),
    )

    k_eff = _clip_k_to_data(target_k, n_cells)
    best_resolution = 1.0
    best_diff = n_cells
    for resolution in np.arange(0.05, 3.0, 0.05):
        sc.tl.leiden(adata, resolution=float(resolution), random_state=int(seed))
        n_found = int(np.unique(adata.obs["leiden"].astype(int).values).size)
        diff = abs(n_found - k_eff)
        if diff < best_diff:
            best_diff = diff
            best_resolution = float(resolution)
        if n_found == k_eff:
            break

    sc.tl.leiden(adata, resolution=best_resolution, random_state=int(seed))
    return remap_contiguous_labels(adata.obs["leiden"].astype(int).values)


def pseudo_labels(
    embeddings: np.ndarray,
    config: ClusteringConfig,
    runtime: RuntimeConfig,
) -> np.ndarray:
    """Compute pseudo-labels used during the weighted training phase."""
    emb = sanitize_embeddings(embeddings)
    k = estimate_pseudo_k(emb.shape[0], config)
    method = str(config.pseudo_label_method).strip().lower()

    if method == "kmeans":
        return kmeans_labels(emb, k=k, seed=runtime.seed)

    try:
        return leiden_labels(emb, target_k=k, seed=runtime.seed)
    except Exception as exc:
        logger.warning("Leiden pseudo-labels failed (%s). Falling back to KMeans.", exc)
        return kmeans_labels(emb, k=k, seed=runtime.seed)


def _reassign_noise_to_centroids(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Reassign HDBSCAN noise points to the nearest non-noise centroid."""
    labels = np.asarray(labels, dtype=np.int64).copy()
    keep = labels >= 0
    if not np.any(keep):
        return kmeans_labels(embeddings, k=max(2, min(embeddings.shape[0] - 1, 8)), seed=42)

    unique_clusters = sorted(np.unique(labels[keep]).tolist())
    centroids = np.asarray(
        [np.mean(embeddings[labels == cluster_id], axis=0) for cluster_id in unique_clusters],
        dtype=np.float32,
    )

    for noise_index in np.where(labels < 0)[0]:
        distances = np.sum((centroids - embeddings[noise_index]) ** 2, axis=1)
        labels[noise_index] = int(unique_clusters[int(np.argmin(distances))])
    return labels


def final_clustering(
    embeddings: np.ndarray,
    config: ClusteringConfig,
    runtime: RuntimeConfig,
) -> np.ndarray:
    """Run final HDBSCAN clustering with robust Leiden/KMeans fallbacks."""
    import hdbscan

    emb = sanitize_embeddings(embeddings)
    fallback_k = estimate_pseudo_k(emb.shape[0], config)
    cluster_selection_method = str(
        getattr(config, "hdbscan_cluster_selection_method", "eom") or "eom"
    ).strip().lower()
    if cluster_selection_method not in {"eom", "leaf"}:
        cluster_selection_method = "eom"

    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(2, int(config.hdbscan_min_cluster_size)),
            min_samples=max(1, int(config.hdbscan_min_samples)),
            cluster_selection_method=cluster_selection_method,
            metric="euclidean",
            core_dist_n_jobs=1,
        )
        labels = np.asarray(clusterer.fit_predict(emb), dtype=np.int64)
    except Exception as exc:
        logger.warning("HDBSCAN failed (%s). Falling back to Leiden.", exc)
        try:
            return leiden_labels(emb, target_k=fallback_k, seed=runtime.seed)
        except Exception:
            return kmeans_labels(emb, k=fallback_k, seed=runtime.seed)

    n_clusters_found = int(np.sum(np.unique(labels) >= 0))
    if n_clusters_found <= 1:
        logger.warning(
            "HDBSCAN returned %d usable cluster(s). Falling back to Leiden.",
            n_clusters_found,
        )
        try:
            return leiden_labels(emb, target_k=fallback_k, seed=runtime.seed)
        except Exception:
            return kmeans_labels(emb, k=fallback_k, seed=runtime.seed)

    if bool(config.hdbscan_reassign_noise) and np.any(labels < 0):
        labels = _reassign_noise_to_centroids(emb, labels)

    if np.any(labels < 0):
        labels = labels.copy()
        labels[labels < 0] = int(labels[labels >= 0].max()) + 1 if np.any(labels >= 0) else 0

    return remap_contiguous_labels(labels)
