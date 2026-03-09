"""Reconstruction, weighting and rare-loss helpers used by scRAW.

This mixin centralizes the "math blocks" used by training:
- reconstruction targets / transforms (NB or MSE path),
- per-cell dynamic weighting (cluster frequency + latent density),
- optional rare-cell regularization (semi-hard triplet).

The training loop in `scraw_algorithm.py` decides *when* each helper is called;
this file focuses on *how* each quantity is computed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import torch

from ..defaults import DEFAULT_PARAM_OVERRIDES


logger = logging.getLogger(__name__)


def _to_numpy(x: Any, dtype: np.dtype | None = None) -> np.ndarray:
    """Convert sparse/dense inputs to numpy arrays."""
    # `AnnData` peut contenir des matrices sparse; on les densifie ici
    # pour simplifier les opérations numériques dans le reste du pipeline.
    if hasattr(x, "toarray"):
        x = x.toarray()
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


class ScrawLossWeightMixin:
    """Mixin with reconstruction losses and dynamic cell weights."""

    def _param(self, key: str, default: Any) -> Any:  # pragma: no cover - provided by parent class
        """Read one algorithm parameter, delegating to the concrete parent class."""
        raise NotImplementedError

    def _transform_nb_input(
        self,
        counts: np.ndarray,
        transform: str,
        theta: float,
        residual_clip: float,
    ) -> np.ndarray:
        """Transform raw counts for encoder input in NB mode."""
        # Cette fonction prépare l'entrée de l'encodeur (et non la loss).
        mode = str(transform).strip().lower()
        X_counts = np.asarray(counts, dtype=np.float32)
        X_counts = np.nan_to_num(X_counts, nan=0.0, posinf=0.0, neginf=0.0)
        X_counts = np.clip(X_counts, a_min=0.0, a_max=None)

        if mode in {"identity", "none"}:
            # Aucun changement de l'échelle des comptes.
            return X_counts.copy()
        if mode == "log1p":
            # Transformation standard en single-cell pour compresser les grandes valeurs.
            return np.log1p(X_counts).astype(np.float32, copy=False)
        if mode != "pearson_residuals":
            logger.warning("Unknown nb_input_transform='%s'. Falling back to log1p.", mode)
            return np.log1p(X_counts).astype(np.float32, copy=False)

        # Pearson residuals:
        # - mu: espérance sous un modèle simple de marges,
        # - var: variance NB approchée,
        # - residuals: (x - mu) / sqrt(var).
        row_sum = X_counts.sum(axis=1, keepdims=True, dtype=np.float64)
        col_sum = X_counts.sum(axis=0, keepdims=True, dtype=np.float64)
        total = float(max(row_sum.sum(), 1.0))
        mu = (row_sum @ col_sum) / total
        theta_v = max(1e-4, float(theta))
        var = mu + (mu * mu) / theta_v
        residuals = (X_counts.astype(np.float64) - mu) / np.sqrt(var + 1e-8)
        clip_v = max(1.0, float(residual_clip))
        residuals = np.clip(residuals, -clip_v, clip_v)
        gene_std = residuals.std(axis=0, keepdims=True)
        residuals = residuals / np.maximum(gene_std, 1e-3)
        return residuals.astype(np.float32, copy=False)

    def _prepare_reconstruction_target(self, data: Any, X_proc: np.ndarray) -> Tuple[np.ndarray, str]:
        """Prepare reconstruction target matrix for NB or MSE training.

        Returns:
        - target matrix with shape `(n_cells, n_genes)`,
        - mode string in `{"nb", "mse"}`.

        Note:
        - In NB mode, this function keeps legacy compatibility behavior by
          applying `nb_input_transform` to the target.
        - In the current training pipeline, `ScRAWAlgorithm.fit()` may replace
          this target with raw counts from `_prepare_nb_inputs()` when
          `adata.layers["original_X"]` is available. This keeps the final NB
          loss aligned with counts-based likelihood while preserving backward
          compatibility for setups where only `adata.X` exists.
        """
        dist = str(
            self._param(
                "reconstruction_distribution",
                DEFAULT_PARAM_OVERRIDES["reconstruction_distribution"],
            )
        ).strip().lower()
        transform = str(
            self._param("nb_input_transform", DEFAULT_PARAM_OVERRIDES["nb_input_transform"])
        ).strip().lower()
        theta = float(self._param("nb_theta", DEFAULT_PARAM_OVERRIDES["nb_theta"]))
        residual_clip = float(self._param("pearson_residual_clip", 10.0))

        if dist != "nb":
            # En MSE, la cible est directement la matrice prétraitée.
            return np.asarray(X_proc, dtype=np.float32), "mse"

        if hasattr(data, "layers") and "original_X" in data.layers:
            # Chemin NB recommandé: cible issue des comptes bruts.
            target = _to_numpy(data.layers["original_X"], dtype=np.float32)
        else:
            # Fallback de compatibilité si original_X n'existe pas.
            target = np.asarray(X_proc, dtype=np.float32)
            logger.warning("adata.layers['original_X'] missing; NB target falls back to adata.X.")

        target = np.clip(target, 0.0, None)
        if transform in {"none", "identity"}:
            pass
        elif transform == "log1p":
            target = np.log1p(target)
        elif transform == "pearson_residuals":
            target = self._transform_nb_input(
                counts=target,
                transform="pearson_residuals",
                theta=theta,
                residual_clip=residual_clip,
            )
            # Keep targets non-negative for the legacy dedicated behavior.
            t_min = float(np.nanmin(target)) if target.size else 0.0
            if np.isfinite(t_min) and t_min < 0.0:
                target = target - t_min
            target = np.clip(target, 0.0, None)
        else:
            logger.warning("Unknown nb_input_transform='%s'. Falling back to log1p.", transform)
            target = np.log1p(target)

        return np.asarray(target, dtype=np.float32), "nb"

    def _prepare_nb_inputs(
        self,
        data: Any,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Prepare NB-mode model input, target counts and size factors.

        Returns `(X_model, counts_target, size_factors)` where:
        - `X_model` is encoder input transformed by `nb_input_transform`,
        - `counts_target` are non-negative raw counts used in NB loss,
        - `size_factors` are library-size factors:
          `lib_size_i / median(lib_size_j > 0)`, clipped to `[1e-3, 1e3]`.

        Returns `None` if `adata.layers["original_X"]` is unavailable.
        """
        if not (hasattr(data, "layers") and "original_X" in data.layers):
            return None

        # Cible de loss NB = comptes bruts non négatifs.
        counts = _to_numpy(data.layers["original_X"], dtype=np.float32)
        counts = np.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
        counts = np.clip(counts, a_min=0.0, a_max=None)

        # Entrée encodeur transformée selon `nb_input_transform`.
        X_model = self._transform_nb_input(
            counts=counts,
            transform=str(
                self._param("nb_input_transform", DEFAULT_PARAM_OVERRIDES["nb_input_transform"])
            ),
            theta=float(self._param("nb_theta", DEFAULT_PARAM_OVERRIDES["nb_theta"])),
            residual_clip=float(self._param("pearson_residual_clip", 10.0)),
        )

        # Facteurs de taille cellule:
        # s_i = lib_size_i / median(lib_size_j > 0), puis clipping robuste.
        raw_library_size = counts.sum(axis=1, dtype=np.float32)
        positive_lib = raw_library_size[raw_library_size > 0]
        lib_median = float(np.median(positive_lib)) if positive_lib.size > 0 else 1.0
        size_factors = np.clip(
            raw_library_size / max(lib_median, 1e-6), 1e-3, 1e3
        ).astype(np.float32)

        return (
            np.asarray(X_model, dtype=np.float32),
            np.asarray(counts, dtype=np.float32),
            np.asarray(size_factors, dtype=np.float32),
        )

    def _cluster_frequency_weights(self, pseudo_labels: np.ndarray) -> np.ndarray:
        """Compute inverse-frequency pseudo-cluster weights.

        If `count(c)` is cluster size and `exp = weight_exponent`:
        `w_i = (1 / count(cluster_i))**exp`, then normalized to mean 1.
        """
        labels = np.asarray(pseudo_labels, dtype=np.int64)
        n = labels.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float32)

        exp = max(
            0.0,
            float(self._param("weight_exponent", DEFAULT_PARAM_OVERRIDES["weight_exponent"])),
        )
        if exp == 0.0:
            # Exposant nul => toutes les cellules ont le même poids.
            return np.ones(n, dtype=np.float32)

        uniq, cnt = np.unique(labels, return_counts=True)
        count_map = {int(u): int(c) for u, c in zip(uniq, cnt)}
        freq = np.asarray([count_map[int(x)] for x in labels], dtype=np.float32)
        freq = np.maximum(freq, 1.0)

        w = np.power(1.0 / freq, exp).astype(np.float32)
        mean = float(np.nanmean(w))
        if np.isfinite(mean) and mean > 0.0:
            w = w / mean
        else:
            w = np.ones_like(w)
        return np.asarray(w, dtype=np.float32)

    def _density_weights(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute density-aware weights from latent kNN distances.

        Uses the distance to the k-th nearest neighbor as inverse density proxy.
        Cells in sparse regions (larger kNN radius) get larger weights.
        Final weights are clipped then normalized to mean 1.
        """
        from sklearn.neighbors import NearestNeighbors

        n = embeddings.shape[0]
        if n <= 2:
            return np.ones(max(0, n), dtype=np.float32)

        k = int(self._param("density_knn_k", 15) or 15)
        k = max(2, min(k, n - 1))
        exp = max(0.0, float(self._param("density_weight_exponent", 1.0)))
        density_clip = float(
            self._param("density_weight_clip", DEFAULT_PARAM_OVERRIDES["density_weight_clip"])
        )

        # Distance au k-ième voisin: proxy simple de la densité locale.
        nn_model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn_model.fit(embeddings)
        dists, _ = nn_model.kneighbors(embeddings)
        kth = np.asarray(dists[:, -1], dtype=np.float32)

        med = float(np.nanmedian(kth))
        if not np.isfinite(med) or med <= 0.0:
            return np.ones(n, dtype=np.float32)

        w = np.power(np.clip(kth / med, 1e-8, None), exp)
        w = np.clip(w, 0.05, max(0.05, density_clip)).astype(np.float32)
        mean = float(np.nanmean(w))
        if np.isfinite(mean) and mean > 0.0:
            w = w / mean
        else:
            w = np.ones_like(w)
        return np.asarray(w, dtype=np.float32)

    def _combined_cell_weights(self, embeddings: np.ndarray, pseudo_labels: np.ndarray) -> np.ndarray:
        """Fuse cluster-frequency and density-derived weights."""
        comp = self._combined_cell_weights_components(
            embeddings=embeddings,
            pseudo_labels=pseudo_labels,
        )
        return np.asarray(comp["fused_weight"], dtype=np.float32)

    def _combined_cell_weights_components(
        self,
        embeddings: np.ndarray,
        pseudo_labels: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Return cluster, density and fused reconstruction weights.

        Output keys:
        - `cluster_component`: normalized cluster-frequency component,
        - `density_component`: normalized density component,
        - `fused_weight_unclipped`: fused score before min/max clipping,
        - `fused_weight`: final clipped reconstruction weight.

        All non-clipped components are scaled to have mean ~1 for stability.
        """
        w_cluster = self._cluster_frequency_weights(pseudo_labels)
        w_density = self._density_weights(embeddings)

        fusion_mode = str(self._param("weight_fusion_mode", "additive")).strip().lower()
        if fusion_mode not in {"additive", "multiplicative"}:
            fusion_mode = "additive"
        cluster_power = float(self._param("cluster_weight_power", 1.0))
        density_power = float(self._param("density_weight_power", 1.0))
        w_min = float(self._param("min_cell_weight", DEFAULT_PARAM_OVERRIDES["min_cell_weight"]))
        w_max = float(self._param("max_cell_weight", DEFAULT_PARAM_OVERRIDES["max_cell_weight"]))
        if w_max < w_min:
            w_max = w_min

        cw = np.power(np.maximum(w_cluster, 1e-8), max(0.0, cluster_power)).astype(np.float32, copy=False)
        dw = np.power(np.maximum(w_density, 1e-8), max(0.0, density_power)).astype(np.float32, copy=False)

        if fusion_mode == "multiplicative":
            # Renforce surtout les cellules qui sont à la fois rares et isolées.
            cluster_component_raw = cw
            density_component_raw = dw
            fused_raw = (cw * dw).astype(np.float32, copy=False)
        else:
            # Additive mode exposes explicit trade-off alpha:
            # fused = alpha * cluster_component + (1-alpha) * density_component.
            alpha = float(
                np.clip(
                    float(
                        self._param(
                            "cluster_density_alpha",
                            DEFAULT_PARAM_OVERRIDES["cluster_density_alpha"],
                        )
                    ),
                    0.0,
                    1.0,
                )
            )
            cluster_component_raw = (alpha * cw).astype(np.float32, copy=False)
            density_component_raw = ((1.0 - alpha) * dw).astype(np.float32, copy=False)
            fused_raw = (cluster_component_raw + density_component_raw).astype(np.float32, copy=False)

        mean_fused = float(np.nanmean(fused_raw))
        if not np.isfinite(mean_fused) or mean_fused <= 0.0:
            mean_fused = 1.0
            fused_norm = np.ones_like(fused_raw, dtype=np.float32)
        else:
            # Normaliser à moyenne 1 pour stabiliser l'échelle de loss entre epochs.
            fused_norm = (fused_raw / mean_fused).astype(np.float32, copy=False)

        cluster_component = (cluster_component_raw / mean_fused).astype(np.float32, copy=False)
        density_component = (density_component_raw / mean_fused).astype(np.float32, copy=False)
        # Clipping final: évite qu'une cellule domine (ou disparaisse) dans la loss.
        fused_weight = np.clip(fused_norm, w_min, w_max).astype(np.float32, copy=False)

        return {
            "cluster_component": cluster_component,
            "density_component": density_component,
            "fused_weight_unclipped": fused_norm.astype(np.float32, copy=False),
            "fused_weight": fused_weight,
        }

    def _apply_random_mask(
        self,
        x: torch.Tensor,
        rate: float,
        masking_value: float = 0.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply random feature masking and return the boolean mask."""
        if rate <= 0.0:
            return x, None
        rate = min(float(rate), 0.95)
        # Masque booléen indépendant par cellule et par feature.
        mask = torch.rand_like(x) < rate
        if not bool(torch.any(mask)):
            return x, None
        out = x.clone()
        out[mask] = float(masking_value)
        return out, mask

    def _reduce_per_sample_loss(
        self,
        per_elem_loss: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        masked_recon_weight: float = 0.75,
    ) -> torch.Tensor:
        """Reduce per-feature losses to per-cell losses, with optional mask weighting."""
        if mask is None:
            return per_elem_loss.mean(dim=1)

        weight = float(np.clip(masked_recon_weight, 0.0, 1.0))
        mask_f = mask.float()

        if weight >= 1.0:
            # Cas extrême: on n'optimise que les positions masquées.
            denom = mask_f.sum(dim=1).clamp(min=1.0)
            return (per_elem_loss * mask_f).sum(dim=1) / denom
        if weight <= 0.0:
            # Cas extrême: on ignore la distinction masqué / non-masqué.
            return per_elem_loss.mean(dim=1)

        # Cas intermédiaire: combinaison pondérée des deux zones.
        weights = mask_f * weight + (1.0 - mask_f) * (1.0 - weight)
        denom = weights.sum(dim=1).clamp(min=1e-6)
        return (per_elem_loss * weights).sum(dim=1) / denom

    def _negative_binomial_loss_per_sample(
        self,
        target: torch.Tensor,
        recon_raw: torch.Tensor,
        size_factors: Optional[torch.Tensor],
        theta: float,
        mu_clip_max: float,
        mask: Optional[torch.Tensor] = None,
        masked_recon_weight: float = 0.75,
    ) -> torch.Tensor:
        """Per-cell NB negative log-likelihood with size factors.

        Implementation notes (important for report/code alignment):
        - numerical guards are explicit (`eps`, theta clamp, mu clamp),
        - with `size_factors`, `mu = exp(clamp(recon_raw)) * size_factor`,
        - without `size_factors`, a safe fallback is used:
          `mu = softplus(recon_raw) + 1e-4`.

        Returns one scalar loss per cell after feature reduction.
        """
        eps = 1e-8
        theta_t = torch.as_tensor(float(theta), device=recon_raw.device, dtype=recon_raw.dtype)
        theta_t = torch.clamp(theta_t, min=1e-4)
        if size_factors is None:
            # Fallback branch used when no library-size factors are available.
            mu = torch.nn.functional.softplus(recon_raw) + 1e-4
        else:
            # Main branch for counts-based NB:
            # decoder output predicts a log-scale rate, then multiplied by size factor.
            sf = torch.clamp(size_factors.view(-1, 1), min=1e-4)
            mu = torch.exp(torch.clamp(recon_raw, min=-12.0, max=12.0)) * sf
            mu = torch.clamp(mu, min=1e-8, max=float(mu_clip_max))
        x = torch.clamp(target, min=0.0)

        # Formule NB (log-vraisemblance négative) calculée élément par élément.
        log_theta_mu = torch.log(theta_t + mu + eps)
        per_elem_nll = -(
            torch.lgamma(x + theta_t)
            - torch.lgamma(theta_t)
            - torch.lgamma(x + 1.0)
            + theta_t * (torch.log(theta_t + eps) - log_theta_mu)
            + x * (torch.log(mu + eps) - log_theta_mu)
        )
        # Garde-fou: remplace NaN/Inf pour ne pas casser le backward.
        per_elem_nll = torch.nan_to_num(per_elem_nll, nan=1e3, posinf=1e3, neginf=1e3)
        return self._reduce_per_sample_loss(
            per_elem_loss=per_elem_nll,
            mask=mask,
            masked_recon_weight=masked_recon_weight,
        )

    def _reconstruction_loss_per_sample(
        self,
        target: torch.Tensor,
        recon_raw: torch.Tensor,
        mode: str,
        size_factors: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        masked_recon_weight: float = 0.75,
    ) -> torch.Tensor:
        """Compute per-cell reconstruction loss (NB or MSE)."""
        if mode == "nb":
            if size_factors is None:
                # Branche de secours si aucun facteur de taille n'est fourni.
                size_factors = torch.ones(target.shape[0], device=target.device, dtype=target.dtype)
            return self._negative_binomial_loss_per_sample(
                target=target,
                recon_raw=recon_raw,
                size_factors=size_factors,
                theta=float(self._param("nb_theta", DEFAULT_PARAM_OVERRIDES["nb_theta"])),
                mu_clip_max=float(self._param("nb_mu_clip_max", 1e6)),
                mask=mask,
                masked_recon_weight=masked_recon_weight,
            )

        per_elem_mse = (recon_raw - target) ** 2
        return self._reduce_per_sample_loss(
            per_elem_loss=per_elem_mse,
            mask=mask,
            masked_recon_weight=masked_recon_weight,
        )

    def _rare_triplet_loss(
        self,
        z: torch.Tensor,
        pseudo_labels_batch: np.ndarray,
        weights_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Semi-hard triplet loss focused on high-weight (rare) anchors.

        Anchor selection is restricted to `weights_batch >= rare_triplet_min_weight`.
        For each anchor:
        - positive: farthest sample in same pseudo-cluster (hard positive),
        - negative: closest sample with distance `> d_pos` when available
          (semi-hard), otherwise closest negative overall (fallback).
        """
        if z.shape[0] < 3:
            return torch.tensor(0.0, device=z.device)

        margin = float(self._param("rare_triplet_margin", 0.4))
        min_w = float(
            self._param("rare_triplet_min_weight", DEFAULT_PARAM_OVERRIDES["rare_triplet_min_weight"])
        )
        max_anchors = int(self._param("max_triplet_anchors_per_batch", 64) or 64)

        labels = torch.as_tensor(
            np.asarray(pseudo_labels_batch, dtype=np.int64),
            device=z.device,
            dtype=torch.long,
        )
        candidate = torch.nonzero(weights_batch >= min_w, as_tuple=False).flatten()
        if candidate.numel() == 0:
            return torch.tensor(0.0, device=z.device)

        if max_anchors > 0 and candidate.numel() > max_anchors:
            perm = torch.randperm(candidate.numel(), device=candidate.device)
            candidate = candidate[perm[:max_anchors]]

        dists = torch.cdist(z, z, p=2)

        losses: List[torch.Tensor] = []
        for a in candidate:
            same = labels == labels[a]
            same[a] = False
            diff = ~same
            diff[a] = False
            if not bool(torch.any(same)) or not bool(torch.any(diff)):
                continue

            # Hardest positive + semi-hard negative (FaceNet-style fallback).
            d_pos = dists[a, same].max()
            neg_dists = dists[a, diff]
            semi_hard_mask = neg_dists > d_pos
            if bool(torch.any(semi_hard_mask)):
                d_neg = neg_dists[semi_hard_mask].min()
            else:
                # Fallback explicite: s'il n'y a pas de semi-hard, prendre le plus proche.
                d_neg = neg_dists.min()

            losses.append(torch.relu(d_pos - d_neg + margin))

        if not losses:
            return torch.tensor(0.0, device=z.device)
        return torch.stack(losses).mean()
