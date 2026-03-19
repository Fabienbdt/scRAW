"""Training loop for the scRAW pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .clustering import final_clustering, pseudo_labels
from .config import ScRAWConfig
from .model import MLPAutoencoder, encode_in_batches, gradient_reversal, resolve_device


logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    model: MLPAutoencoder
    device: str
    embeddings: np.ndarray
    labels: np.ndarray
    pseudo_labels: np.ndarray
    cell_weights: np.ndarray
    cluster_component: np.ndarray
    density_component: np.ndarray
    loss_history: List[Dict[str, Any]]


def _apply_random_mask(
    x: torch.Tensor,
    rate: float,
    masking_value: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply feature masking for denoising-style reconstruction."""
    if rate <= 0.0:
        return x, None
    rate = min(float(rate), 0.95)
    mask = torch.rand_like(x) < rate
    if not bool(torch.any(mask)):
        return x, None
    out = x.clone()
    out[mask] = float(masking_value)
    return out, mask


def _reduce_mse_per_sample(
    target: torch.Tensor,
    recon: torch.Tensor,
    mask: Optional[torch.Tensor],
    masked_recon_weight: float,
) -> torch.Tensor:
    """Reduce element-wise MSE to one value per cell."""
    per_elem_loss = (recon - target) ** 2
    if mask is None:
        return per_elem_loss.mean(dim=1)

    weight = float(np.clip(masked_recon_weight, 0.0, 1.0))
    mask_float = mask.float()

    if weight >= 1.0:
        denom = mask_float.sum(dim=1).clamp(min=1.0)
        return (per_elem_loss * mask_float).sum(dim=1) / denom
    if weight <= 0.0:
        return per_elem_loss.mean(dim=1)

    combined_weights = mask_float * weight + (1.0 - mask_float) * (1.0 - weight)
    denom = combined_weights.sum(dim=1).clamp(min=1e-6)
    return (per_elem_loss * combined_weights).sum(dim=1) / denom


def _cluster_frequency_weights(labels: np.ndarray, exponent: float) -> np.ndarray:
    """Compute inverse-frequency cluster weights normalized to mean 1."""
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return np.zeros(0, dtype=np.float32)

    exponent = max(0.0, float(exponent))
    if exponent == 0.0:
        return np.ones(labels.shape[0], dtype=np.float32)

    unique_labels, counts = np.unique(labels, return_counts=True)
    count_map = {int(label): int(count) for label, count in zip(unique_labels, counts)}
    frequencies = np.asarray([count_map[int(label)] for label in labels], dtype=np.float32)
    weights = np.power(1.0 / np.maximum(frequencies, 1.0), exponent).astype(np.float32)
    mean_value = float(np.nanmean(weights))
    if np.isfinite(mean_value) and mean_value > 0.0:
        weights = weights / mean_value
    return np.asarray(weights, dtype=np.float32)


def _density_weights(
    embeddings: np.ndarray,
    density_knn_k: int,
    density_weight_exponent: float,
    density_weight_clip: float,
) -> np.ndarray:
    """Compute density-derived weights from latent-space kNN distances."""
    from sklearn.neighbors import NearestNeighbors

    emb = np.asarray(embeddings, dtype=np.float32)
    n_cells = emb.shape[0]
    if n_cells <= 2:
        return np.ones(max(0, n_cells), dtype=np.float32)

    k = max(2, min(int(density_knn_k), n_cells - 1))
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn_model.fit(emb)
    distances, _ = nn_model.kneighbors(emb)
    kth_distances = np.asarray(distances[:, -1], dtype=np.float32)

    median_distance = float(np.nanmedian(kth_distances))
    if not np.isfinite(median_distance) or median_distance <= 0.0:
        return np.ones(n_cells, dtype=np.float32)

    weights = np.power(
        np.clip(kth_distances / median_distance, 1e-8, None),
        max(0.0, float(density_weight_exponent)),
    )
    weights = np.clip(weights, 0.05, max(0.05, float(density_weight_clip))).astype(np.float32)
    mean_value = float(np.nanmean(weights))
    if np.isfinite(mean_value) and mean_value > 0.0:
        weights = weights / mean_value
    return np.asarray(weights, dtype=np.float32)


def _combined_cell_weights(
    embeddings: np.ndarray,
    pseudo_label_values: np.ndarray,
    config: ScRAWConfig,
) -> Dict[str, np.ndarray]:
    """Fuse cluster and density signals into one reconstruction weight vector."""
    cluster_weights = _cluster_frequency_weights(
        pseudo_label_values,
        exponent=config.weighting.weight_exponent,
    )
    density_weights = _density_weights(
        embeddings,
        density_knn_k=config.weighting.density_knn_k,
        density_weight_exponent=config.weighting.density_weight_exponent,
        density_weight_clip=config.weighting.density_weight_clip,
    )

    alpha = float(np.clip(config.weighting.cluster_density_alpha, 0.0, 1.0))
    cluster_component_raw = (alpha * cluster_weights).astype(np.float32, copy=False)
    density_component_raw = ((1.0 - alpha) * density_weights).astype(np.float32, copy=False)
    fused_raw = (cluster_component_raw + density_component_raw).astype(np.float32, copy=False)

    mean_fused = float(np.nanmean(fused_raw))
    if not np.isfinite(mean_fused) or mean_fused <= 0.0:
        mean_fused = 1.0
        fused_normalized = np.ones_like(fused_raw, dtype=np.float32)
    else:
        fused_normalized = (fused_raw / mean_fused).astype(np.float32, copy=False)

    fused_weight = np.clip(
        fused_normalized,
        float(config.weighting.min_cell_weight),
        float(config.weighting.max_cell_weight),
    ).astype(np.float32, copy=False)

    return {
        "cluster_component": (cluster_component_raw / mean_fused).astype(np.float32, copy=False),
        "density_component": (density_component_raw / mean_fused).astype(np.float32, copy=False),
        "cell_weights": fused_weight,
    }


def _rare_triplet_loss(
    z: torch.Tensor,
    pseudo_labels_batch: np.ndarray,
    weights_batch: torch.Tensor,
    config: ScRAWConfig,
) -> torch.Tensor:
    """Semi-hard triplet loss focused on the highest-weight cells."""
    if z.shape[0] < 3:
        return torch.tensor(0.0, device=z.device)

    labels = torch.as_tensor(
        np.asarray(pseudo_labels_batch, dtype=np.int64),
        dtype=torch.long,
        device=z.device,
    )
    candidate = torch.nonzero(
        weights_batch >= float(config.triplet.min_anchor_weight),
        as_tuple=False,
    ).flatten()
    if candidate.numel() == 0:
        return torch.tensor(0.0, device=z.device)

    max_anchors = int(config.triplet.max_anchors_per_batch)
    if max_anchors > 0 and candidate.numel() > max_anchors:
        perm = torch.randperm(candidate.numel(), device=candidate.device)
        candidate = candidate[perm[:max_anchors]]

    distances = torch.cdist(z, z, p=2)
    losses: List[torch.Tensor] = []
    for anchor in candidate:
        same_cluster = labels == labels[anchor]
        same_cluster[anchor] = False
        different_cluster = ~same_cluster
        different_cluster[anchor] = False
        if not bool(torch.any(same_cluster)) or not bool(torch.any(different_cluster)):
            continue

        positive_distance = distances[anchor, same_cluster].max()
        negative_distances = distances[anchor, different_cluster]
        semi_hard = negative_distances > positive_distance
        if bool(torch.any(semi_hard)):
            negative_distance = negative_distances[semi_hard].min()
        else:
            negative_distance = negative_distances.min()

        losses.append(
            torch.relu(positive_distance - negative_distance + float(config.triplet.margin))
        )

    if not losses:
        return torch.tensor(0.0, device=z.device)
    return torch.stack(losses).mean()


def _record_epoch(
    history: List[Dict[str, Any]],
    epoch: int,
    phase: str,
    total_loss: float,
    reconstruction_loss: float,
    triplet_loss: float,
    batch_adv_loss: float = 0.0,
) -> None:
    """Append one epoch summary to the loss history."""
    history.append(
        {
            "epoch": int(epoch),
            "phase": str(phase),
            "total_loss": float(total_loss),
            "reconstruction_loss": float(reconstruction_loss),
            "triplet_loss": float(triplet_loss),
            "batch_adv_loss": float(batch_adv_loss),
        }
    )


class ScRAWTrainer:
    """Train the scRAW model on a preprocessed matrix."""

    def __init__(self, config: ScRAWConfig) -> None:
        self.config = config
        self.device = resolve_device(config.runtime.device)

    def fit(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray] = None,
        batch_ids: Optional[np.ndarray] = None,
    ) -> TrainingResult:
        """Train the autoencoder, update dynamic weights, and cluster embeddings."""
        torch.manual_seed(int(self.config.runtime.seed))
        np.random.seed(int(self.config.runtime.seed))

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Input matrix must be a non-empty 2D array.")

        n_cells, n_features = X.shape
        resolved_pseudo_k = 0
        if int(self.config.clustering.pseudo_k) > 1:
            resolved_pseudo_k = int(self.config.clustering.pseudo_k)
        elif labels is not None:
            resolved_pseudo_k = int(len(np.unique(np.asarray(labels))))
        clustering_config = (
            replace(self.config.clustering, pseudo_k=resolved_pseudo_k)
            if resolved_pseudo_k > 1
            else self.config.clustering
        )

        batch_enabled = bool(self.config.batch_correction.enabled)
        adv_weight = float(self.config.batch_correction.adversarial_weight)
        batch_index = None if batch_ids is None else np.asarray(batch_ids, dtype=object)
        batch_index_encoded: Optional[np.ndarray] = None
        n_batches = 1
        if batch_enabled and adv_weight > 0.0:
            if batch_index is None:
                raise ValueError("Batch correction is enabled but no batch ids were provided.")
            if len(batch_index) != n_cells:
                raise ValueError("Batch id length does not match the number of cells.")
            unique_batches = sorted(np.unique(batch_index).tolist())
            n_batches = len(unique_batches)
            if n_batches >= 2:
                mapping = {value: idx for idx, value in enumerate(unique_batches)}
                batch_index_encoded = np.asarray(
                    [mapping[value] for value in batch_index],
                    dtype=np.int64,
                )
            else:
                logger.warning(
                    "Batch correction requested but only one batch was found; disabling the adversarial branch."
                )

        model = MLPAutoencoder(input_dim=n_features, config=self.config.model).to(self.device)
        params: List[torch.nn.Parameter] = list(model.parameters())
        batch_head: Optional[nn.Module] = None
        if batch_enabled and adv_weight > 0.0 and batch_index_encoded is not None:
            hidden_dim = max(8, int(self.config.model.latent_dim) // 2)
            batch_head = nn.Sequential(
                nn.Linear(int(self.config.model.latent_dim), hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, n_batches),
            ).to(self.device)
            params += list(batch_head.parameters())

        optimizer = torch.optim.Adam(params, lr=float(self.config.training.learning_rate))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(self.config.training.epochs)),
            eta_min=float(self.config.training.learning_rate) * 0.01,
        )

        from torch.utils.data import DataLoader, TensorDataset

        x_tensor_cpu = torch.from_numpy(X).float()
        index_tensor_cpu = torch.arange(n_cells, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(x_tensor_cpu, index_tensor_cpu),
            batch_size=int(self.config.training.batch_size),
            shuffle=True,
        )

        current_weights = np.ones(n_cells, dtype=np.float32)
        current_cluster_component = np.ones(n_cells, dtype=np.float32)
        current_density_component = np.ones(n_cells, dtype=np.float32)
        current_pseudo = np.zeros(n_cells, dtype=np.int64)
        loss_history: List[Dict[str, Any]] = []
        adv_start_epoch = int(self.config.batch_correction.start_epoch)
        adv_ramp_epochs = int(self.config.batch_correction.ramp_epochs)
        adv_lambda = float(self.config.batch_correction.adversarial_lambda)

        for epoch in range(int(self.config.training.epochs)):
            weighted_phase = epoch >= int(self.config.training.warmup_epochs)
            refresh_interval = int(self.config.weighting.dynamic_weight_update_interval)
            should_refresh_weights = weighted_phase and (
                epoch == int(self.config.training.warmup_epochs)
                or (refresh_interval > 0 and ((epoch - int(self.config.training.warmup_epochs)) % refresh_interval == 0))
            )

            if should_refresh_weights:
                embeddings_for_weights = encode_in_batches(
                    model,
                    X,
                    device=self.device,
                    batch_size=int(self.config.training.batch_size),
                )
                pseudo_for_weights = pseudo_labels(
                    embeddings_for_weights,
                    config=clustering_config,
                    runtime=self.config.runtime,
                )
                weight_components = _combined_cell_weights(
                    embeddings_for_weights,
                    pseudo_for_weights,
                    config=self.config,
                )
                new_weights = np.asarray(weight_components["cell_weights"], dtype=np.float32)
                new_cluster_component = np.asarray(
                    weight_components["cluster_component"],
                    dtype=np.float32,
                )
                new_density_component = np.asarray(
                    weight_components["density_component"],
                    dtype=np.float32,
                )

                if epoch == int(self.config.training.warmup_epochs):
                    current_weights = new_weights
                    current_cluster_component = new_cluster_component
                    current_density_component = new_density_component
                else:
                    momentum = float(np.clip(self.config.weighting.dynamic_weight_momentum, 0.0, 1.0))
                    current_weights = (
                        momentum * current_weights + (1.0 - momentum) * new_weights
                    ).astype(np.float32, copy=False)
                    current_cluster_component = (
                        momentum * current_cluster_component + (1.0 - momentum) * new_cluster_component
                    ).astype(np.float32, copy=False)
                    current_density_component = (
                        momentum * current_density_component + (1.0 - momentum) * new_density_component
                    ).astype(np.float32, copy=False)

                    mean_weight = float(np.mean(current_weights))
                    if np.isfinite(mean_weight) and mean_weight > 0.0:
                        current_weights = current_weights / mean_weight
                    current_weights = np.clip(
                        current_weights,
                        float(self.config.weighting.min_cell_weight),
                        float(self.config.weighting.max_cell_weight),
                    ).astype(np.float32, copy=False)

                current_pseudo = pseudo_for_weights

            model.train()
            total_sum = 0.0
            reconstruction_sum = 0.0
            triplet_sum = 0.0
            batch_adv_sum = 0.0
            n_batches_seen = 0

            for xb, idx_tensor in loader:
                xb = xb.to(self.device)
                idx = idx_tensor.detach().cpu().numpy()

                if float(self.config.training.masking_rate) > 0.0 and (
                    (not weighted_phase) or bool(self.config.training.masking_in_weighted_phase)
                ):
                    x_in, mask = _apply_random_mask(
                        xb,
                        rate=float(self.config.training.masking_rate),
                        masking_value=float(self.config.training.masking_value),
                    )
                else:
                    x_in, mask = xb, None

                z, recon = model(x_in)
                loss_per_sample = _reduce_mse_per_sample(
                    target=xb,
                    recon=recon,
                    mask=mask,
                    masked_recon_weight=float(self.config.training.masked_recon_weight),
                )

                if weighted_phase:
                    weight_tensor = torch.tensor(
                        current_weights[idx],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    reconstruction_loss = torch.mean(loss_per_sample * weight_tensor)
                else:
                    weight_tensor = torch.ones(idx_tensor.shape[0], dtype=torch.float32, device=self.device)
                    reconstruction_loss = torch.mean(loss_per_sample)

                triplet_loss = torch.tensor(0.0, device=self.device)
                triplet_active = (
                    bool(self.config.triplet.enabled)
                    and float(self.config.triplet.weight) > 0.0
                    and weighted_phase
                    and epoch >= int(self.config.triplet.start_epoch)
                )
                if triplet_active:
                    triplet_loss = _rare_triplet_loss(
                        z,
                        pseudo_labels_batch=current_pseudo[idx],
                        weights_batch=weight_tensor,
                        config=self.config,
                    )
                    ramp_epochs = max(1, min(20, int(self.config.training.epochs) - int(self.config.triplet.start_epoch)))
                    triplet_ramp = min(1.0, (epoch - int(self.config.triplet.start_epoch)) / ramp_epochs)
                else:
                    triplet_ramp = 0.0

                adv_loss = torch.tensor(0.0, device=self.device)
                if batch_head is not None and batch_index_encoded is not None and epoch >= adv_start_epoch:
                    if adv_ramp_epochs > 0:
                        frac = min(1.0, (epoch - adv_start_epoch + 1) / float(adv_ramp_epochs))
                    else:
                        frac = 1.0
                    logits = batch_head(gradient_reversal(z, lambda_=adv_lambda * frac))
                    yb = torch.tensor(batch_index_encoded[idx], dtype=torch.long, device=self.device)
                    adv_loss = nn.functional.cross_entropy(logits, yb)

                total_loss = (
                    reconstruction_loss
                    + (triplet_ramp * float(self.config.triplet.weight) * triplet_loss)
                    + (adv_weight * adv_loss)
                )

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    params,
                    max_norm=float(self.config.training.gradient_clip),
                )
                optimizer.step()

                total_sum += float(total_loss.detach().cpu().item())
                reconstruction_sum += float(reconstruction_loss.detach().cpu().item())
                triplet_sum += float(triplet_loss.detach().cpu().item())
                batch_adv_sum += float(adv_loss.detach().cpu().item())
                n_batches_seen += 1

            if n_batches_seen == 0:
                raise RuntimeError("No mini-batch was processed during training.")

            scheduler.step()
            phase = "weighted" if weighted_phase else "warmup"
            _record_epoch(
                history=loss_history,
                epoch=epoch,
                phase=phase,
                total_loss=total_sum / n_batches_seen,
                reconstruction_loss=reconstruction_sum / n_batches_seen,
                triplet_loss=triplet_sum / n_batches_seen,
                batch_adv_loss=batch_adv_sum / n_batches_seen,
            )

        final_embeddings = encode_in_batches(
            model,
            X,
            device=self.device,
            batch_size=int(self.config.training.batch_size),
        )
        final_labels = final_clustering(
            final_embeddings,
            config=clustering_config,
            runtime=self.config.runtime,
        )

        return TrainingResult(
            model=model,
            device=str(self.device),
            embeddings=final_embeddings,
            labels=final_labels,
            pseudo_labels=current_pseudo,
            cell_weights=current_weights,
            cluster_component=current_cluster_component,
            density_component=current_density_component,
            loss_history=loss_history,
        )
