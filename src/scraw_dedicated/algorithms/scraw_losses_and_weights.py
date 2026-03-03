"""Reconstruction, weighting and rare-loss helpers used by scRAW."""

from __future__ import annotations

from typing import Any, List, Tuple
import logging

import numpy as np
import torch


logger = logging.getLogger(__name__)


def _to_numpy(x: Any, dtype: np.dtype | None = None) -> np.ndarray:
    """Helper interne: to numpy.
    
    
    Args:
        x: Paramètre d'entrée `x` utilisé dans cette étape du pipeline.
        dtype: Paramètre d'entrée `dtype` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    # Uniformise les entrées sparse/dense vers un tableau numpy dense.
    if hasattr(x, "toarray"):
        x = x.toarray()
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


class ScrawLossWeightMixin:
    """Mixin with reconstruction losses and dynamic cell weights."""

    def _param(self, key: str, default: Any) -> Any:  # pragma: no cover - provided by parent class
        """Helper interne: param.
        
        
        Args:
            key: Paramètre d'entrée `key` utilisé dans cette étape du pipeline.
            default: Paramètre d'entrée `default` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        raise NotImplementedError

    def _prepare_reconstruction_target(self, data: Any, X_proc: np.ndarray) -> Tuple[np.ndarray, str]:
        """Prépare la matrice cible de reconstruction selon le mode NB/MSE demandé.
        
        
        Args:
            data: Paramètre d'entrée `data` utilisé dans cette étape du pipeline.
            X_proc: Paramètre d'entrée `X_proc` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        # Cette fonction construit la "vérité terrain" de reconstruction.
        dist = str(self._param("reconstruction_distribution", "nb")).strip().lower()
        transform = str(self._param("nb_input_transform", "log1p")).strip().lower()
        theta = float(self._param("nb_theta", 10.0))

        if dist != "nb":
            # Mode simple: on reconstruit directement la matrice prétraitée (MSE).
            return np.asarray(X_proc, dtype=np.float32), "mse"

        # Mode NB: on privilégie les comptes bruts si disponibles.
        if hasattr(data, "layers") and "original_X" in data.layers:
            target = _to_numpy(data.layers["original_X"], dtype=np.float32)
        else:
            target = np.asarray(X_proc, dtype=np.float32)
            logger.warning("adata.layers['original_X'] missing; NB target falls back to adata.X.")

        target = np.clip(target, 0.0, None)

        # Optionnel: transformation de la cible NB (none / log1p / résidus de Pearson).
        if transform == "none":
            pass
        elif transform == "log1p":
            target = np.log1p(target)
        elif transform == "pearson_residuals":
            target = self._pearson_residual_transform(target, theta=theta)
            t_min = float(np.nanmin(target)) if target.size else 0.0
            if np.isfinite(t_min) and t_min < 0.0:
                target = target - t_min
            target = np.clip(target, 0.0, None)
        else:
            logger.warning("Unknown nb_input_transform='%s'. Using log1p.", transform)
            target = np.log1p(target)

        return target.astype(np.float32), "nb"

    def _pearson_residual_transform(self, counts: np.ndarray, theta: float) -> np.ndarray:
        """Calcule les résidus de Pearson stabilisés à partir des comptes bruts.
        
        
        Args:
            counts: Paramètre d'entrée `counts` utilisé dans cette étape du pipeline.
            theta: Paramètre d'entrée `theta` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        # Résidus de Pearson standardisés pour stabiliser la variance gène par gène.
        clip = float(self._param("pearson_residual_clip", 10.0))
        X = np.clip(np.asarray(counts, dtype=np.float64), 0.0, None)
        if X.size == 0:
            return np.asarray(X, dtype=np.float32)

        row_sum = X.sum(axis=1, keepdims=True)
        col_sum = X.sum(axis=0, keepdims=True)
        total = float(X.sum())
        if total <= 0:
            return np.zeros_like(X, dtype=np.float32)

        mu = (row_sum @ col_sum) / total
        denom = np.sqrt(mu + (mu**2) / max(theta, 1e-6))
        denom = np.where(denom <= 1e-12, 1.0, denom)
        resid = (X - mu) / denom
        resid = np.clip(resid, -clip, clip)
        return resid.astype(np.float32)

    def _cluster_frequency_weights(self, pseudo_labels: np.ndarray) -> np.ndarray:
        """Helper interne: cluster frequency weights.
        
        
        Args:
            pseudo_labels: Paramètre d'entrée `pseudo_labels` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        # Pondération globale: plus un cluster est rare, plus son poids augmente.
        labels = np.asarray(pseudo_labels, dtype=np.int64)
        n = labels.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float32)

        exp = max(0.0, float(self._param("weight_exponent", 0.2)))
        if exp == 0.0:
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
        """Helper interne: density weights.
        
        
        Args:
            embeddings: Paramètre d'entrée `embeddings` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        # Pondération locale: cellules isolées en latent => poids plus fort.
        from sklearn.neighbors import NearestNeighbors

        n = embeddings.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        if n <= 2:
            return np.ones(n, dtype=np.float32)

        k = int(self._param("density_knn_k", 15) or 15)
        k = max(2, min(k, n - 1))
        exp = max(0.0, float(self._param("density_weight_exponent", 1.0)))

        nn_model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn_model.fit(embeddings)
        dists, _ = nn_model.kneighbors(embeddings)
        kth = np.asarray(dists[:, -1], dtype=np.float32)

        med = float(np.nanmedian(kth))
        if not np.isfinite(med) or med <= 0.0:
            return np.ones(n, dtype=np.float32)

        w = np.power(np.clip(kth / med, 1e-6, None), exp).astype(np.float32)
        mean = float(np.nanmean(w))
        if np.isfinite(mean) and mean > 0.0:
            w = w / mean
        else:
            w = np.ones_like(w)
        return np.asarray(w, dtype=np.float32)

    def _combined_cell_weights(self, embeddings: np.ndarray, pseudo_labels: np.ndarray) -> np.ndarray:
        """Fusionne les poids de rareté globale et de densité locale par cellule.
        
        
        Args:
            embeddings: Paramètre d'entrée `embeddings` utilisé dans cette étape du pipeline.
            pseudo_labels: Paramètre d'entrée `pseudo_labels` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        # Fusion pondération globale (fréquence) + locale (densité).
        w_cluster = self._cluster_frequency_weights(pseudo_labels)
        w_density = self._density_weights(embeddings)

        alpha = float(self._param("cluster_density_alpha", 0.6))
        alpha = float(np.clip(alpha, 0.0, 1.0))
        w = (1.0 - alpha) * w_cluster + alpha * w_density

        # Clamp final pour contrôler la stabilité numérique à l'entraînement.
        w_min = float(self._param("min_cell_weight", 1.0))
        w_max = float(self._param("max_cell_weight", 10.0))
        if w_max < w_min:
            w_max = w_min
        w = np.clip(w, w_min, w_max)
        return np.asarray(w, dtype=np.float32)

    def _apply_random_mask(self, x: torch.Tensor, rate: float) -> torch.Tensor:
        """Helper interne: apply random mask.
        
        
        Args:
            x: Paramètre d'entrée `x` utilisé dans cette étape du pipeline.
            rate: Paramètre d'entrée `rate` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        # Masquage type denoising autoencoder.
        if rate <= 0.0:
            return x
        keep = torch.rand_like(x) >= rate
        return x * keep

    def _negative_binomial_loss_per_sample(
        self,
        target: torch.Tensor,
        recon_raw: torch.Tensor,
        theta: float,
    ) -> torch.Tensor:
        """Calcule la perte négative log-vraisemblance binomiale négative par cellule.
        
        
        Args:
            target: Paramètre d'entrée `target` utilisé dans cette étape du pipeline.
            recon_raw: Paramètre d'entrée `recon_raw` utilisé dans cette étape du pipeline.
            theta: Paramètre d'entrée `theta` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        # Perte NB calculée par cellule puis moyennée sur les gènes.
        target = torch.clamp(target, min=0.0)
        mu = torch.nn.functional.softplus(recon_raw) + 1e-4
        theta_t = torch.tensor(float(theta), device=mu.device, dtype=mu.dtype)

        ll = (
            torch.lgamma(target + theta_t)
            - torch.lgamma(theta_t)
            - torch.lgamma(target + 1.0)
            + theta_t * (torch.log(theta_t + 1e-8) - torch.log(theta_t + mu + 1e-8))
            + target * (torch.log(mu + 1e-8) - torch.log(theta_t + mu + 1e-8))
        )
        nll = -ll
        return nll.mean(dim=1)

    def _reconstruction_loss_per_sample(
        self,
        target: torch.Tensor,
        recon_raw: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        """Calcule la loss de reconstruction par cellule (NB ou MSE).
        
        
        Args:
            target: Paramètre d'entrée `target` utilisé dans cette étape du pipeline.
            recon_raw: Paramètre d'entrée `recon_raw` utilisé dans cette étape du pipeline.
            mode: Paramètre d'entrée `mode` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        # Point d'aiguillage unique entre NB et MSE.
        if mode == "nb":
            return self._negative_binomial_loss_per_sample(
                target=target,
                recon_raw=recon_raw,
                theta=float(self._param("nb_theta", 10.0)),
            )
        return torch.mean((recon_raw - target) ** 2, dim=1)

    def _rare_triplet_loss(
        self,
        z: torch.Tensor,
        pseudo_labels_batch: np.ndarray,
        weights_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Calcule la régularisation triplet ciblant les cellules pondérées comme rares.
        
        
        Args:
            z: Paramètre d'entrée `z` utilisé dans cette étape du pipeline.
            pseudo_labels_batch: Paramètre d'entrée `pseudo_labels_batch` utilisé dans cette étape du pipeline.
            weights_batch: Paramètre d'entrée `weights_batch` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        # Régularisation rare: rapproche les cellules d'un même pseudo-cluster rare
        # et éloigne les cellules d'autres clusters.
        if z.shape[0] < 3:
            return torch.tensor(0.0, device=z.device)

        margin = float(self._param("rare_triplet_margin", 0.4))
        min_w = float(self._param("rare_triplet_min_weight", 1.2))
        max_anchors = int(self._param("max_triplet_anchors_per_batch", 64) or 64)

        labels = np.asarray(pseudo_labels_batch, dtype=np.int64)
        rare_mask = weights_batch.detach().cpu().numpy() >= min_w
        # On ne retient que les ancres suffisamment pondérées comme "rares".
        candidate = np.where(rare_mask)[0]
        if candidate.size == 0:
            return torch.tensor(0.0, device=z.device)

        rng = np.random.default_rng(int(self._param("seed", 42)))
        rng.shuffle(candidate)
        candidate = candidate[:max_anchors]

        losses: List[torch.Tensor] = []
        for a in candidate:
            same = np.where(labels == labels[a])[0]
            diff = np.where(labels != labels[a])[0]
            same = same[same != a]
            if same.size == 0 or diff.size == 0:
                continue

            p = int(rng.choice(same))
            n = int(rng.choice(diff))

            da = torch.norm(z[a] - z[p], p=2)
            dn = torch.norm(z[a] - z[n], p=2)
            losses.append(torch.relu(da - dn + margin))

        if not losses:
            return torch.tensor(0.0, device=z.device)
        return torch.stack(losses).mean()
