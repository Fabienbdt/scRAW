"""Clustering and pseudo-label helpers used by scRAW (deep2-focused)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

import numpy as np


logger = logging.getLogger(__name__)


def remap_contiguous_labels(labels: np.ndarray) -> np.ndarray:
    """Remap arbitrary cluster ids to contiguous ids 0..K-1."""
    labels = np.asarray(labels)
    uniq = sorted(np.unique(labels).tolist())
    mapping = {int(v): i for i, v in enumerate(uniq)}
    return np.asarray([mapping[int(v)] for v in labels], dtype=np.int64)


class ScrawClusteringMixin:
    """Pseudo-label + final clustering routines used by the deep2 workflow."""

    _pseudo_fallback_method: Optional[str]
    _leiden_warning_emitted: bool

    def _param(self, key: str, default: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def _clip_k_to_data(self, k: int, n_cells: int) -> int:
        """Clip k to valid bounds for the dataset size."""
        n_cells = int(max(1, n_cells))
        if n_cells == 1:
            return 1
        upper = 2 if n_cells == 2 else max(2, n_cells - 1)
        return int(max(2, min(int(k), upper)))

    def _estimate_unsupervised_k_heuristic(self, n_cells: int) -> int:
        """Legacy size-based K heuristic used in SCRBenchmark."""
        n_cells = int(max(1, n_cells))
        k_min = int(max(2, self._param("unsupervised_k_min", 8)))
        k_max = int(max(2, self._param("unsupervised_k_max", 30)))
        if k_max < k_min:
            k_min, k_max = k_max, k_min

        k_est = int(round(float(np.sqrt(n_cells / 40.0))))
        k_est = int(np.clip(k_est, k_min, k_max))
        return self._clip_k_to_data(k_est, n_cells)

    def _build_unsupervised_k_candidates(self, n_cells: int) -> List[int]:
        """Build bounded candidate grid for stability-consensus K selection."""
        n_cells = int(max(1, n_cells))
        k_min = int(max(2, self._param("unsupervised_k_min", 8)))
        k_max = int(max(2, self._param("unsupervised_k_max", 30)))
        if k_max < k_min:
            k_min, k_max = k_max, k_min

        k_max = int(min(k_max, max(2, n_cells - 1)))
        k_min = int(min(k_min, k_max))
        if k_max <= 2:
            return [2]

        max_candidates = int(max(3, self._param("unsupervised_k_num_candidates", 12)))
        full_grid = list(range(k_min, k_max + 1))
        if len(full_grid) <= max_candidates:
            return full_grid

        sampled = np.linspace(k_min, k_max, num=max_candidates)
        candidates = sorted(set(int(round(v)) for v in sampled))
        candidates = [k for k in candidates if 2 <= k <= k_max]
        if k_min not in candidates:
            candidates.insert(0, k_min)
        if k_max not in candidates:
            candidates.append(k_max)
        return sorted(set(candidates))

    def _prepare_unsupervised_k_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Prepare embeddings for K selection (optional PCA reduction)."""
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[0] <= 2:
            return emb

        pca_dim = int(self._param("unsupervised_k_pca_dim", 32))
        max_components = int(min(emb.shape[1], max(1, emb.shape[0] - 1)))
        if pca_dim <= 1 or max_components <= 2 or emb.shape[1] <= pca_dim:
            return emb

        from sklearn.decomposition import PCA

        n_components = int(max(2, min(pca_dim, max_components)))
        try:
            pca = PCA(
                n_components=n_components,
                svd_solver="randomized",
                random_state=int(self._param("random_state", self._param("seed", 42))),
            )
            return pca.fit_transform(emb).astype(np.float32, copy=False)
        except Exception:
            return emb

    def _fit_kmeans_for_k_selection(
        self,
        X: np.ndarray,
        n_clusters: int,
        random_state: int,
        n_init: int,
    ) -> np.ndarray:
        """Fit KMeans (or MiniBatchKMeans) for K-selection scoring."""
        from sklearn.cluster import KMeans, MiniBatchKMeans

        Xv = np.asarray(X, dtype=np.float32)
        n_samples = int(Xv.shape[0])
        n_clusters = int(max(2, min(int(n_clusters), n_samples)))
        n_init = int(max(1, n_init))

        if n_samples > 15000:
            batch_size = int(min(2048, max(256, n_samples // 20)))
            model = MiniBatchKMeans(
                n_clusters=n_clusters,
                n_init=n_init,
                random_state=int(random_state),
                batch_size=batch_size,
                max_iter=200,
                reassignment_ratio=0.01,
            )
        else:
            model = KMeans(
                n_clusters=n_clusters,
                n_init=n_init,
                random_state=int(random_state),
            )
        labels = model.fit_predict(Xv)
        return np.asarray(labels, dtype=np.int64)

    def _rank_scores(self, metric_by_k: Dict[int, float], higher_is_better: bool) -> Dict[int, float]:
        """Convert metric values to [0,1] rank scores."""
        finite_items = [
            (int(k), float(v))
            for k, v in metric_by_k.items()
            if np.isfinite(float(v))
        ]
        if not finite_items:
            return {}

        ordered = sorted(finite_items, key=lambda kv: kv[1], reverse=bool(higher_is_better))
        n = len(ordered)
        if n == 1:
            return {ordered[0][0]: 1.0}
        return {
            int(k): float(1.0 - (idx / float(n - 1)))
            for idx, (k, _) in enumerate(ordered)
        }

    def _estimate_unsupervised_k_stability_consensus(
        self,
        embeddings: np.ndarray,
        n_cells: int,
    ) -> int:
        """Select K with weighted consensus of stability + internal CVIs."""
        from sklearn.metrics import (
            adjusted_rand_score,
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )

        emb = self._prepare_unsupervised_k_embeddings(embeddings)
        n_samples = int(emb.shape[0])
        if n_samples <= 3:
            return self._clip_k_to_data(2, n_cells)

        candidates = self._build_unsupervised_k_candidates(n_samples)
        if not candidates:
            return self._estimate_unsupervised_k_heuristic(n_cells)

        random_state = int(self._param("random_state", self._param("seed", 42)))
        rng = np.random.RandomState(random_state)

        eval_sample_size = int(max(200, self._param("unsupervised_k_eval_sample_size", 3000)))
        stability_sample_size = int(
            max(300, self._param("unsupervised_k_stability_sample_size", 4000))
        )
        stability_runs = int(max(2, self._param("unsupervised_k_stability_runs", 5)))

        eval_n = int(min(n_samples, eval_sample_size))
        stab_n = int(min(n_samples, stability_sample_size))
        eval_idx = (
            rng.choice(n_samples, size=eval_n, replace=False)
            if eval_n < n_samples
            else np.arange(n_samples)
        )
        stab_idx = (
            rng.choice(n_samples, size=stab_n, replace=False)
            if stab_n < n_samples
            else np.arange(n_samples)
        )

        emb_eval = emb[eval_idx]
        emb_stab = emb[stab_idx]

        metric_sil: Dict[int, float] = {}
        metric_ch: Dict[int, float] = {}
        metric_db: Dict[int, float] = {}
        metric_stability: Dict[int, float] = {}
        metric_tiny_frac: Dict[int, float] = {}

        tiny_cluster_fraction = float(self._param("unsupervised_k_min_cluster_fraction", 0.005))
        tiny_cluster_fraction = float(np.clip(tiny_cluster_fraction, 0.0, 0.2))

        for k in candidates:
            if k >= n_samples:
                continue

            try:
                labels_eval = self._fit_kmeans_for_k_selection(
                    X=emb_eval,
                    n_clusters=k,
                    random_state=random_state + (11 * int(k)),
                    n_init=10,
                )
                n_found = int(np.unique(labels_eval).size)
                if n_found <= 1 or n_found >= emb_eval.shape[0]:
                    continue
                counts = np.bincount(labels_eval.astype(np.int64, copy=False))
                tiny_abs = int(max(2, round(tiny_cluster_fraction * emb_eval.shape[0])))
                metric_tiny_frac[k] = float(np.mean(counts < tiny_abs))
                metric_sil[k] = float(silhouette_score(emb_eval, labels_eval, metric="euclidean"))
                metric_ch[k] = float(calinski_harabasz_score(emb_eval, labels_eval))
                metric_db[k] = float(davies_bouldin_score(emb_eval, labels_eval))
            except Exception:
                continue

            run_labels: List[np.ndarray] = []
            for run_id in range(stability_runs):
                try:
                    labels_run = self._fit_kmeans_for_k_selection(
                        X=emb_stab,
                        n_clusters=k,
                        random_state=random_state + 1009 + (run_id * 1291) + (17 * int(k)),
                        n_init=1,
                    )
                    if int(np.unique(labels_run).size) > 1:
                        run_labels.append(labels_run)
                except Exception:
                    continue

            if len(run_labels) >= 2:
                pairwise_ari: List[float] = []
                for i in range(len(run_labels)):
                    for j in range(i + 1, len(run_labels)):
                        pairwise_ari.append(float(adjusted_rand_score(run_labels[i], run_labels[j])))
                if pairwise_ari:
                    metric_stability[k] = float(np.mean(pairwise_ari))

        if not metric_sil:
            return self._estimate_unsupervised_k_heuristic(n_cells)

        score_stability = self._rank_scores(metric_stability, higher_is_better=True)
        score_sil = self._rank_scores(metric_sil, higher_is_better=True)
        score_ch = self._rank_scores(metric_ch, higher_is_better=True)
        score_db = self._rank_scores(metric_db, higher_is_better=False)
        score_tiny = self._rank_scores(metric_tiny_frac, higher_is_better=False)

        w_stability = max(0.0, float(self._param("unsupervised_k_weight_stability", 0.45)))
        w_sil = max(0.0, float(self._param("unsupervised_k_weight_silhouette", 0.25)))
        w_ch = max(0.0, float(self._param("unsupervised_k_weight_ch", 0.20)))
        w_db = max(0.0, float(self._param("unsupervised_k_weight_db", 0.10)))
        w_tiny = max(0.0, float(self._param("unsupervised_k_weight_tiny_clusters", 0.20)))
        overseg_penalty = max(0.0, float(self._param("unsupervised_k_overseg_penalty", 0.25)))
        underseg_penalty = max(0.0, float(self._param("unsupervised_k_underseg_penalty", 0.05)))
        k_anchor = self._estimate_unsupervised_k_heuristic(n_cells=n_cells)

        scored_k: Dict[int, float] = {}
        valid_ks = sorted(set(metric_sil.keys()) | set(metric_stability.keys()) | set(metric_ch.keys()) | set(metric_db.keys()))
        for k in valid_ks:
            weighted_sum = 0.0
            weight_sum = 0.0
            if k in score_stability:
                weighted_sum += w_stability * score_stability[k]
                weight_sum += w_stability
            if k in score_sil:
                weighted_sum += w_sil * score_sil[k]
                weight_sum += w_sil
            if k in score_ch:
                weighted_sum += w_ch * score_ch[k]
                weight_sum += w_ch
            if k in score_db:
                weighted_sum += w_db * score_db[k]
                weight_sum += w_db
            if k in score_tiny:
                weighted_sum += w_tiny * score_tiny[k]
                weight_sum += w_tiny

            if weight_sum <= 0:
                continue

            raw_score = float(weighted_sum / weight_sum)
            over_ratio = max(0.0, (float(k) - float(k_anchor)) / max(1.0, float(k_anchor)))
            under_ratio = max(0.0, (float(k_anchor) - float(k)) / max(1.0, float(k_anchor)))
            penalty = float(overseg_penalty * over_ratio + underseg_penalty * under_ratio)
            scored_k[k] = float(raw_score - penalty)

        if not scored_k:
            return self._estimate_unsupervised_k_heuristic(n_cells)

        best_score = max(scored_k.values())
        tied = [k for k, s in scored_k.items() if abs(s - best_score) <= 1e-12]
        best_k = int(min(tied))
        return self._clip_k_to_data(best_k, n_cells)

    def _estimate_k(self, n_cells: int, embeddings: Optional[np.ndarray] = None) -> int:
        """Estimate pseudo-label K with SCRBenchmark-compatible defaults."""
        k_effective = int(self._param("_pseudo_n_clusters", 0) or 0)
        if k_effective > 1:
            return self._clip_k_to_data(k_effective, n_cells)

        k_user = int(self._param("n_clusters", 0) or 0)
        if k_user > 1:
            return self._clip_k_to_data(k_user, n_cells)

        manual_k = int(self._param("unsupervised_k_fallback", 0) or 0)
        if manual_k > 1:
            k = self._clip_k_to_data(manual_k, n_cells)
            self.params["unsupervised_k_selected"] = int(k)
            self.params["unsupervised_k_selection_mode"] = "manual_fallback"
            return k

        mode = str(self._param("unsupervised_k_selection", "stability_consensus")).strip().lower()
        if mode not in {"stability_consensus", "heuristic"}:
            mode = "stability_consensus"

        if mode == "heuristic" or embeddings is None:
            k = self._estimate_unsupervised_k_heuristic(n_cells)
            self.params["unsupervised_k_selected"] = int(k)
            self.params["unsupervised_k_selection_mode"] = (
                "heuristic_no_embeddings" if embeddings is None and mode != "heuristic" else "heuristic"
            )
            return k

        try:
            k = self._estimate_unsupervised_k_stability_consensus(embeddings=embeddings, n_cells=n_cells)
            self.params["unsupervised_k_selected"] = int(k)
            self.params["unsupervised_k_selection_mode"] = "stability_consensus"
            return self._clip_k_to_data(k, n_cells)
        except Exception:
            k = self._estimate_unsupervised_k_heuristic(n_cells)
            self.params["unsupervised_k_selected"] = int(k)
            self.params["unsupervised_k_selection_mode"] = "heuristic_after_failure"
            return k

    def _kmeans_pseudo_labels(self, embeddings: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """Compute KMeans pseudo-labels."""
        from sklearn.cluster import KMeans

        n_cells = embeddings.shape[0]
        k_eff = int(k) if k is not None else self._estimate_k(n_cells, embeddings=embeddings)
        k_eff = self._clip_k_to_data(k_eff, n_cells)

        km = KMeans(n_clusters=k_eff, random_state=int(self._param("seed", 42)), n_init=10)
        labels = km.fit_predict(embeddings)
        return remap_contiguous_labels(labels)

    def _leiden_pseudo_labels(self, embeddings: np.ndarray, target_k: int) -> np.ndarray:
        """Compute Leiden pseudo-labels with resolution search toward target_k."""
        import anndata as ad
        import scanpy as sc

        n_cells = embeddings.shape[0]
        if n_cells < 3:
            return np.zeros(n_cells, dtype=np.int64)

        emb = np.asarray(embeddings, dtype=np.float32)
        adata = ad.AnnData(X=emb)
        rs = int(self._param("seed", 42))

        n_neighbors = min(15, n_cells - 1)
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            use_rep="X",
            method="gauss",
            transformer="sklearn",
            random_state=rs,
        )

        k_eff = max(2, min(int(target_k), n_cells - 1))
        best_res, best_diff = 1.0, n_cells
        for res in np.arange(0.05, 3.0, 0.05):
            sc.tl.leiden(adata, resolution=float(res), random_state=rs)
            n_found = len(np.unique(adata.obs["leiden"].astype(int).values))
            diff = abs(n_found - k_eff)
            if diff < best_diff:
                best_diff = diff
                best_res = float(res)
            if n_found == k_eff:
                break

        sc.tl.leiden(adata, resolution=best_res, random_state=rs)
        return adata.obs["leiden"].astype(int).values.astype(np.int64)

    def _pseudo_labels(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pseudo-labels used for dynamic weighting and triplet loss."""
        embeddings = self._sanitize_embeddings(embeddings)
        method = str(self._param("pseudo_label_method", "leiden")).strip().lower()
        if self._pseudo_fallback_method is not None:
            method = self._pseudo_fallback_method
        if method not in {"leiden", "kmeans"}:
            method = "kmeans"

        k = self._estimate_k(embeddings.shape[0], embeddings=embeddings)
        if method == "kmeans":
            return self._kmeans_pseudo_labels(embeddings, k=k)

        try:
            return self._leiden_pseudo_labels(embeddings, target_k=k)
        except Exception as exc:
            if not self._leiden_warning_emitted:
                logger.warning(
                    "pseudo_label_method=leiden failed once (%s); falling back to KMeans for the rest of the run.",
                    exc,
                )
                self._leiden_warning_emitted = True
            self._pseudo_fallback_method = "kmeans"
            return self._kmeans_pseudo_labels(embeddings, k=k)

    @staticmethod
    def _sanitize_embeddings(values: np.ndarray) -> np.ndarray:
        """Ensure finite values before distance or clustering ops."""
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return arr
        non_finite = ~np.isfinite(arr)
        if np.any(non_finite):
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e4, neginf=-1e4)
        arr = np.clip(arr, -1e4, 1e4).astype(np.float32, copy=False)
        return arr

    def _hdbscan_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply final HDBSCAN clustering on latent embeddings."""
        min_cluster_size = max(2, int(self._param("hdbscan_min_cluster_size", 4) or 4))
        min_samples = max(1, int(self._param("hdbscan_min_samples", 2) or 2))
        if min_samples > min_cluster_size:
            logger.warning(
                "HDBSCAN: min_samples (%d) > min_cluster_size (%d). Clamping to %d.",
                min_samples,
                min_cluster_size,
                min_cluster_size,
            )
            min_samples = min_cluster_size

        method = str(self._param("hdbscan_cluster_selection_method", "eom")).strip().lower()
        if method not in {"eom", "leaf"}:
            logger.warning(
                "HDBSCAN: invalid cluster_selection_method='%s'. Falling back to 'eom'.",
                method,
            )
            method = "eom"
        reassign_noise = bool(self._param("hdbscan_reassign_noise", True))

        emb = self._sanitize_embeddings(embeddings)
        k_fallback = self._estimate_k(emb.shape[0], embeddings=emb)

        def _fallback_to_leiden(reason: str) -> np.ndarray:
            logger.warning("HDBSCAN fallback to Leiden (%s).", reason)
            try:
                labels_fb = self._leiden_pseudo_labels(emb, target_k=k_fallback)
                return remap_contiguous_labels(labels_fb)
            except Exception as exc:
                logger.warning("Leiden fallback failed (%s); fallback to KMeans.", exc)
                return self._kmeans_pseudo_labels(emb, k=k_fallback)

        try:
            import hdbscan as hdbscan_lib

            clusterer = hdbscan_lib.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method=method,
                metric="euclidean",
                core_dist_n_jobs=1,
            )
            labels = np.asarray(clusterer.fit_predict(emb), dtype=np.int64)
        except Exception as exc:
            return _fallback_to_leiden(str(exc))

        n_found = int(np.sum(np.unique(labels) >= 0))
        if n_found <= 1:
            return _fallback_to_leiden(f"degenerate result (n_clusters={n_found})")

        if reassign_noise and np.any(labels < 0):
            labels = self._reassign_noise_to_centroids(emb, labels)

        if np.any(labels < 0):
            labels = labels.copy()
            labels[labels < 0] = int(labels[labels >= 0].max()) + 1 if np.any(labels >= 0) else 0

        labels = remap_contiguous_labels(labels)
        labels = self._merge_close_hdbscan_siblings(emb, labels)
        return remap_contiguous_labels(labels)

    def _reassign_noise_to_centroids(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Reassign HDBSCAN noise points to nearest non-noise centroid."""
        labels = np.asarray(labels, dtype=np.int64).copy()
        keep = labels >= 0
        if not np.any(keep):
            return self._kmeans_pseudo_labels(embeddings)

        uniq = sorted(np.unique(labels[keep]).tolist())
        centroids = np.asarray(
            [np.mean(embeddings[labels == c], axis=0) for c in uniq], dtype=np.float32
        )

        noise_idx = np.where(labels < 0)[0]
        for i in noise_idx:
            d = np.sum((centroids - embeddings[i]) ** 2, axis=1)
            labels[i] = int(uniq[int(np.argmin(d))])
        return labels

    def _merge_close_hdbscan_siblings(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Merge one obviously over-segmented close cluster pair (if any)."""
        # Disabled by default for strict parity with SCRBenchmark scRAW baseline.
        enable = bool(self._param("hdbscan_merge_close_clusters", False))
        if not enable:
            return np.asarray(labels, dtype=np.int64)

        labs = np.asarray(labels, dtype=np.int64).copy()
        uniq = sorted(np.unique(labs).tolist())
        if len(uniq) < 2:
            return labs

        min_size = int(self._param("hdbscan_merge_min_cluster_size", 200) or 200)
        max_ratio = float(self._param("hdbscan_merge_max_centroid_ratio", 1.15) or 1.15)
        if max_ratio <= 0.0:
            return labs

        sizes = {int(c): int(np.sum(labs == c)) for c in uniq}
        eligible = [int(c) for c in uniq if sizes[int(c)] >= max(1, min_size)]
        if len(eligible) < 2:
            return labs

        centroids: Dict[int, np.ndarray] = {}
        radii: Dict[int, float] = {}
        for c in eligible:
            pts = embeddings[labs == c]
            if pts.shape[0] == 0:
                continue
            cent = np.mean(pts, axis=0).astype(np.float32, copy=False)
            d = np.sqrt(np.sum((pts - cent) ** 2, axis=1))
            centroids[c] = cent
            radii[c] = float(np.median(d)) + 1e-8

        keys = sorted(centroids.keys())
        if len(keys) < 2:
            return labs

        best_pair: Optional[tuple[int, int]] = None
        best_ratio = np.inf
        for i, ca in enumerate(keys):
            for cb in keys[i + 1 :]:
                d_ab = float(np.sqrt(np.sum((centroids[ca] - centroids[cb]) ** 2)))
                ratio = d_ab / (radii[ca] + radii[cb])
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_pair = (int(ca), int(cb))

        if best_pair is None or not np.isfinite(best_ratio) or best_ratio > max_ratio:
            return labs

        a, b = best_pair
        logger.info(
            "HDBSCAN sibling merge applied: merge %d <- %d (ratio=%.4f, min_size=%d).",
            a,
            b,
            best_ratio,
            min_size,
        )
        labs[labs == b] = a
        return labs
