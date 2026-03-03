"""Clustering and pseudo-label helpers used by scRAW."""

from __future__ import annotations

from typing import Any, Optional
import logging

import numpy as np


logger = logging.getLogger(__name__)


def remap_contiguous_labels(labels: np.ndarray) -> np.ndarray:
    """Réalise l'opération `remap contiguous labels` du module `scraw_clustering`.
    
    
    Args:
        labels: Paramètre d'entrée `labels` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    labels = np.asarray(labels)
    uniq = sorted(np.unique(labels).tolist())
    mapping = {int(v): i for i, v in enumerate(uniq)}
    return np.asarray([mapping[int(v)] for v in labels], dtype=np.int64)


class ScrawClusteringMixin:
    """Mixin with pseudo-label and final clustering routines."""

    _pseudo_fallback_method: Optional[str]
    _leiden_warning_emitted: bool

    def _param(self, key: str, default: Any) -> Any:  # pragma: no cover - provided by parent class
        """Helper interne: param.
        
        
        Args:
            key: Paramètre d'entrée `key` utilisé dans cette étape du pipeline.
            default: Paramètre d'entrée `default` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        raise NotImplementedError

    def _estimate_k(self, n_cells: int) -> int:
        """Helper interne: estimate k.
        
        
        Args:
            n_cells: Paramètre d'entrée `n_cells` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        k_user = int(self._param("n_clusters", 0) or 0)
        if k_user > 1:
            return min(max(2, k_user), max(2, n_cells - 1))
        k = int(np.sqrt(max(n_cells, 2)) / 2.0)
        k = max(2, min(40, k))
        return min(k, max(2, n_cells - 1))

    def _kmeans_pseudo_labels(self, embeddings: np.ndarray) -> np.ndarray:
        """Calcule des pseudo-labels KMeans lorsque Leiden n'est pas disponible.
        
        
        Args:
            embeddings: Paramètre d'entrée `embeddings` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        from sklearn.cluster import KMeans

        n_cells = embeddings.shape[0]
        k = self._estimate_k(n_cells)
        km = KMeans(n_clusters=k, random_state=int(self._param("seed", 42)), n_init=10)
        labels = km.fit_predict(embeddings)
        return remap_contiguous_labels(labels)

    def _leiden_pseudo_labels(self, embeddings: np.ndarray) -> np.ndarray:
        """Calcule des pseudo-labels Leiden sur les embeddings latents.
        
        
        Args:
            embeddings: Paramètre d'entrée `embeddings` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        import anndata as ad
        import scanpy as sc

        n_cells = embeddings.shape[0]
        if n_cells < 3:
            return np.zeros(n_cells, dtype=np.int64)

        adata = ad.AnnData(X=np.asarray(embeddings, dtype=np.float32))
        n_neighbors = max(5, min(30, n_cells - 1))
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X")

        resolution = float(self._param("pseudo_leiden_resolution", 1.0))
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added="_leiden",
            random_state=int(self._param("seed", 42)),
        )
        raw = adata.obs["_leiden"].astype(str).to_numpy()
        uniq = sorted(np.unique(raw).tolist())
        mapping = {v: i for i, v in enumerate(uniq)}
        return np.asarray([mapping[v] for v in raw], dtype=np.int64)

    def _pseudo_labels(self, embeddings: np.ndarray) -> np.ndarray:
        """Calcule les pseudo-labels utilisés pendant l'entraînement pondéré.
        
        
        Args:
            embeddings: Paramètre d'entrée `embeddings` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        method = str(self._param("pseudo_label_method", "leiden")).strip().lower()
        if self._pseudo_fallback_method is not None:
            method = self._pseudo_fallback_method

        if method == "kmeans":
            return self._kmeans_pseudo_labels(embeddings)

        try:
            return self._leiden_pseudo_labels(embeddings)
        except Exception as exc:
            if not self._leiden_warning_emitted:
                logger.warning(
                    "pseudo_label_method=leiden failed once (%s); falling back to KMeans for the rest of the run.",
                    exc,
                )
                self._leiden_warning_emitted = True
            self._pseudo_fallback_method = "kmeans"
            return self._kmeans_pseudo_labels(embeddings)

    def _hdbscan_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Applique HDBSCAN sur l'espace latent final pour obtenir les clusters prédits.
        
        
        Args:
            embeddings: Paramètre d'entrée `embeddings` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        min_cluster_size = int(self._param("hdbscan_min_cluster_size", 4) or 4)
        min_samples = int(self._param("hdbscan_min_samples", 2) or 2)
        method = str(self._param("hdbscan_cluster_selection_method", "eom"))
        reassign_noise = bool(self._param("hdbscan_reassign_noise", True))

        try:
            import hdbscan

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(2, min_cluster_size),
                min_samples=max(1, min_samples),
                cluster_selection_method=method,
            )
            labels = np.asarray(clusterer.fit_predict(embeddings), dtype=np.int64)
        except Exception as exc:
            logger.warning("HDBSCAN unavailable/failed (%s); fallback to KMeans.", exc)
            labels = self._kmeans_pseudo_labels(embeddings)

        if reassign_noise and np.any(labels < 0):
            labels = self._reassign_noise_to_centroids(embeddings, labels)

        if np.any(labels < 0):
            labels = labels.copy()
            labels[labels < 0] = int(labels[labels >= 0].max()) + 1 if np.any(labels >= 0) else 0

        return remap_contiguous_labels(labels)

    def _reassign_noise_to_centroids(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Helper interne: reassign noise to centroids.
        
        
        Args:
            embeddings: Paramètre d'entrée `embeddings` utilisé dans cette étape du pipeline.
            labels: Paramètre d'entrée `labels` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        labels = np.asarray(labels, dtype=np.int64).copy()
        keep = labels >= 0
        if not np.any(keep):
            return self._kmeans_pseudo_labels(embeddings)

        uniq = sorted(np.unique(labels[keep]).tolist())
        centroids = []
        for c in uniq:
            centroids.append(np.mean(embeddings[labels == c], axis=0))
        centroids = np.asarray(centroids, dtype=np.float32)

        noise_idx = np.where(labels < 0)[0]
        if noise_idx.size == 0:
            return labels

        for i in noise_idx:
            d = np.sum((centroids - embeddings[i]) ** 2, axis=1)
            labels[i] = int(uniq[int(np.argmin(d))])
        return labels
