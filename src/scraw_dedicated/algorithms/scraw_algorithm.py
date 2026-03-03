"""Main scRAW algorithm orchestration.

This file keeps the training orchestration readable by delegating:
- clustering/pseudo-label logic to `scraw_clustering.py`
- reconstruction/weighting/triplet logic to `scraw_losses_and_weights.py`
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn

from ..core.algorithm_registry import AlgorithmInfo, AlgorithmRegistry
from ..core.config import HyperparameterConfig, ParamType
from .base_autoencoder import BaseAutoencoderAlgorithm, gradient_reversal
from .scraw_clustering import ScrawClusteringMixin
from .scraw_losses_and_weights import ScrawLossWeightMixin


logger = logging.getLogger(__name__)


@AlgorithmRegistry.register
class ScRAWAlgorithm(BaseAutoencoderAlgorithm, ScrawLossWeightMixin, ScrawClusteringMixin):
    """Lean scRAW implementation focused on reproducible ablation runs."""

    @classmethod
    def get_info(cls) -> AlgorithmInfo:
        """Réalise l'opération `get info` du module `scraw_algorithm`.
        
        
        Args:
            Aucun argument explicite en dehors du contexte objet.
        
        Returns:
            Valeur calculée par la fonction.
        """
        return AlgorithmInfo(
            name="scraw",
            display_name="scRAW (dedicated)",
            description=(
                "Simplified scRAW with weighted reconstruction, rare triplet loss, "
                "optional DANN, and HDBSCAN final clustering."
            ),
            category="deep_learning",
            requires_gpu=False,
            supports_labels=True,
            preprocessing_notes=(
                "Use preprocessed adata.X; NB reconstruction uses adata.layers['original_X'] "
                "when available."
            ),
            has_internal_preprocessing=False,
            recommended_data="preprocessed",
        )

    @classmethod
    def get_hyperparameters(cls) -> List[HyperparameterConfig]:
        """Réalise l'opération `get hyperparameters` du module `scraw_algorithm`.
        
        
        Args:
            Aucun argument explicite en dehors du contexte objet.
        
        Returns:
            Valeur calculée par la fonction.
        """
        hp = list(BaseAutoencoderAlgorithm.get_hyperparameters())
        hp.extend(
            [
                HyperparameterConfig(
                    name="clustering_method",
                    display_name="Final Clustering",
                    param_type=ParamType.CHOICE,
                    default="hdbscan",
                    choices=["hdbscan"],
                    description="Final clustering method on latent embeddings.",
                    category="Clustering",
                ),
                HyperparameterConfig(
                    name="hdbscan_min_cluster_size",
                    display_name="HDBSCAN Min Cluster Size",
                    param_type=ParamType.INTEGER,
                    default=4,
                    min_value=2,
                    max_value=200,
                    description="Minimum cluster size for HDBSCAN.",
                    category="Clustering",
                ),
                HyperparameterConfig(
                    name="hdbscan_min_samples",
                    display_name="HDBSCAN Min Samples",
                    param_type=ParamType.INTEGER,
                    default=2,
                    min_value=1,
                    max_value=200,
                    description="Min samples for HDBSCAN core points.",
                    category="Clustering",
                ),
                HyperparameterConfig(
                    name="hdbscan_cluster_selection_method",
                    display_name="HDBSCAN Selection",
                    param_type=ParamType.CHOICE,
                    default="eom",
                    choices=["eom", "leaf"],
                    description="Cluster selection method.",
                    category="Clustering",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="hdbscan_reassign_noise",
                    display_name="Reassign Noise",
                    param_type=ParamType.BOOLEAN,
                    default=True,
                    description="Reassign HDBSCAN noise points to nearest centroid.",
                    category="Clustering",
                ),
                HyperparameterConfig(
                    name="pseudo_label_method",
                    display_name="Pseudo-label Method",
                    param_type=ParamType.CHOICE,
                    default="leiden",
                    choices=["leiden", "kmeans"],
                    description="Pseudo-clustering method used for weighting and triplet loss.",
                    category="Pseudo Labels",
                ),
                HyperparameterConfig(
                    name="n_clusters",
                    display_name="Pseudo K",
                    param_type=ParamType.INTEGER,
                    default=0,
                    min_value=0,
                    max_value=300,
                    description="K for KMeans pseudo-labels (0 = heuristic).",
                    category="Pseudo Labels",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="pseudo_leiden_resolution",
                    display_name="Leiden Resolution",
                    param_type=ParamType.FLOAT,
                    default=1.0,
                    min_value=0.01,
                    max_value=10.0,
                    description="Resolution used for Leiden pseudo-labels.",
                    category="Pseudo Labels",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="weight_exponent",
                    display_name="Cluster Weight Exponent",
                    param_type=ParamType.FLOAT,
                    default=0.2,
                    min_value=0.0,
                    max_value=4.0,
                    description="Exponent for inverse-frequency cluster weighting.",
                    category="Weighting",
                ),
                HyperparameterConfig(
                    name="cluster_density_alpha",
                    display_name="Density Mix Alpha",
                    param_type=ParamType.FLOAT,
                    default=0.6,
                    min_value=0.0,
                    max_value=1.0,
                    description="Blend ratio between cluster and density weights.",
                    category="Weighting",
                ),
                HyperparameterConfig(
                    name="density_knn_k",
                    display_name="Density kNN",
                    param_type=ParamType.INTEGER,
                    default=15,
                    min_value=2,
                    max_value=200,
                    description="k used for latent density estimation.",
                    category="Weighting",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="density_weight_exponent",
                    display_name="Density Exponent",
                    param_type=ParamType.FLOAT,
                    default=1.0,
                    min_value=0.0,
                    max_value=4.0,
                    description="Exponent applied to density-derived weights.",
                    category="Weighting",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="weight_fusion_mode",
                    display_name="Weight Fusion",
                    param_type=ParamType.CHOICE,
                    default="additive",
                    choices=["additive"],
                    description="Fusion mode for cluster+density weights.",
                    category="Weighting",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="min_cell_weight",
                    display_name="Min Cell Weight",
                    param_type=ParamType.FLOAT,
                    default=1.0,
                    min_value=0.0,
                    max_value=100.0,
                    description="Lower bound for per-cell reconstruction weights.",
                    category="Weighting",
                ),
                HyperparameterConfig(
                    name="max_cell_weight",
                    display_name="Max Cell Weight",
                    param_type=ParamType.FLOAT,
                    default=10.0,
                    min_value=0.1,
                    max_value=100.0,
                    description="Upper bound for per-cell reconstruction weights.",
                    category="Weighting",
                ),
                HyperparameterConfig(
                    name="dynamic_weight_update_interval",
                    display_name="Weight Update Interval",
                    param_type=ParamType.INTEGER,
                    default=10,
                    min_value=0,
                    max_value=200,
                    description="Recompute global cell weights every N epochs in weighted phase.",
                    category="Weighting",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="dynamic_weight_momentum",
                    display_name="Weight Momentum",
                    param_type=ParamType.FLOAT,
                    default=0.7,
                    min_value=0.0,
                    max_value=1.0,
                    description="EMA momentum for weight updates across epochs.",
                    category="Weighting",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="reconstruction_distribution",
                    display_name="Reconstruction Loss",
                    param_type=ParamType.CHOICE,
                    default="nb",
                    choices=["nb", "mse"],
                    description="Reconstruction objective.",
                    category="Reconstruction",
                ),
                HyperparameterConfig(
                    name="nb_theta",
                    display_name="NB Theta",
                    param_type=ParamType.FLOAT,
                    default=10.0,
                    min_value=1e-3,
                    max_value=1e4,
                    description="Dispersion parameter for NB loss.",
                    category="Reconstruction",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="nb_input_transform",
                    display_name="NB Target Transform",
                    param_type=ParamType.CHOICE,
                    default="log1p",
                    choices=["none", "log1p", "pearson_residuals"],
                    description="Transform applied to NB reconstruction targets.",
                    category="Reconstruction",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="pearson_residual_clip",
                    display_name="Pearson Clip",
                    param_type=ParamType.FLOAT,
                    default=10.0,
                    min_value=0.5,
                    max_value=100.0,
                    description="Clip value for Pearson residual transform.",
                    category="Reconstruction",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="masking_rate",
                    display_name="Masking Rate",
                    param_type=ParamType.FLOAT,
                    default=0.2,
                    min_value=0.0,
                    max_value=0.95,
                    description="Fraction of features masked at input during training.",
                    category="Reconstruction",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="masking_apply_weighted",
                    display_name="Masking In Weighted Phase",
                    param_type=ParamType.BOOLEAN,
                    default=False,
                    description="If false, masking is only applied in warm-up phase.",
                    category="Reconstruction",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="rare_loss_type",
                    display_name="Rare Loss Type",
                    param_type=ParamType.CHOICE,
                    default="triplet",
                    choices=["triplet"],
                    description="Rare-cell regularization type.",
                    category="Rare",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="rare_triplet_weight",
                    display_name="Triplet Weight",
                    param_type=ParamType.FLOAT,
                    default=0.1,
                    min_value=0.0,
                    max_value=100.0,
                    description="Triplet regularization strength.",
                    category="Rare",
                ),
                HyperparameterConfig(
                    name="rare_triplet_start_epoch",
                    display_name="Triplet Start Epoch",
                    param_type=ParamType.INTEGER,
                    default=35,
                    min_value=0,
                    max_value=2000,
                    description="Epoch from which triplet loss is enabled.",
                    category="Rare",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="rare_triplet_margin",
                    display_name="Triplet Margin",
                    param_type=ParamType.FLOAT,
                    default=0.4,
                    min_value=0.0,
                    max_value=10.0,
                    description="Triplet margin.",
                    category="Rare",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="rare_triplet_min_weight",
                    display_name="Triplet Min Weight",
                    param_type=ParamType.FLOAT,
                    default=1.2,
                    min_value=0.0,
                    max_value=100.0,
                    description="Minimum cell weight to be considered as rare anchor.",
                    category="Rare",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="max_triplet_anchors_per_batch",
                    display_name="Max Triplet Anchors",
                    param_type=ParamType.INTEGER,
                    default=64,
                    min_value=1,
                    max_value=10000,
                    description="Max number of anchor cells sampled per mini-batch.",
                    category="Rare",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="use_batch_conditioning",
                    display_name="Use DANN",
                    param_type=ParamType.BOOLEAN,
                    default=False,
                    description="Enable adversarial batch conditioning.",
                    category="Batch",
                ),
                HyperparameterConfig(
                    name="batch_correction_key",
                    display_name="Batch Key",
                    param_type=ParamType.STRING,
                    default="",
                    description="obs key containing batch labels.",
                    category="Batch",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="adversarial_batch_weight",
                    display_name="DANN Weight",
                    param_type=ParamType.FLOAT,
                    default=0.0,
                    min_value=0.0,
                    max_value=100.0,
                    description="Weight of adversarial batch classification loss.",
                    category="Batch",
                ),
                HyperparameterConfig(
                    name="adversarial_lambda",
                    display_name="GRL Lambda",
                    param_type=ParamType.FLOAT,
                    default=1.0,
                    min_value=0.0,
                    max_value=10.0,
                    description="Gradient reversal scale.",
                    category="Batch",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="adversarial_start_epoch",
                    display_name="DANN Start Epoch",
                    param_type=ParamType.INTEGER,
                    default=10,
                    min_value=0,
                    max_value=2000,
                    description="Epoch where adversarial branch starts.",
                    category="Batch",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="adversarial_ramp_epochs",
                    display_name="DANN Ramp Epochs",
                    param_type=ParamType.INTEGER,
                    default=20,
                    min_value=0,
                    max_value=2000,
                    description="Linear ramp length for DANN weight/lambda.",
                    category="Batch",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="mmd_batch_weight",
                    display_name="MMD Weight",
                    param_type=ParamType.FLOAT,
                    default=0.0,
                    min_value=0.0,
                    max_value=100.0,
                    description="Kept for compatibility; currently ignored in simplified version.",
                    category="Batch",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="capture_embedding_snapshots",
                    display_name="Capture Snapshots",
                    param_type=ParamType.BOOLEAN,
                    default=False,
                    description="Capture periodic latent snapshots for UMAP evolution figures.",
                    category="Monitoring",
                ),
                HyperparameterConfig(
                    name="snapshot_interval_epochs",
                    display_name="Snapshot Interval",
                    param_type=ParamType.INTEGER,
                    default=10,
                    min_value=1,
                    max_value=1000,
                    description="Epoch interval between periodic snapshots.",
                    category="Monitoring",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="seed",
                    display_name="Seed",
                    param_type=ParamType.INTEGER,
                    default=42,
                    description="Random seed.",
                    category="General",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="random_state",
                    display_name="Random State",
                    param_type=ParamType.INTEGER,
                    default=42,
                    description="Alias for seed.",
                    category="General",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="input_type",
                    display_name="Input Type",
                    param_type=ParamType.STRING,
                    default="processed",
                    description="Kept for compatibility.",
                    category="General",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="cluster_preprocess_mode",
                    display_name="Cluster Preprocess Mode",
                    param_type=ParamType.STRING,
                    default="none",
                    description="Kept for compatibility (not used in simplified mode).",
                    category="General",
                    advanced=True,
                ),
            ]
        )
        return hp

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Helper interne: init.
        
        
        Args:
            params: Paramètre d'entrée `params` utilisé dans cette étape du pipeline.
        
        Returns:
            `None` ou une valeur interne selon le flux d'exécution.
        """
        super().__init__(params=params)
        self._batch_info: Tuple[Optional[str], int] = (None, 0)
        self._pseudo_fallback_method: Optional[str] = None
        self._leiden_warning_emitted = False

    def _param(self, key: str, default: Any) -> Any:
        """Helper interne: param.
        
        
        Args:
            key: Paramètre d'entrée `key` utilisé dans cette étape du pipeline.
            default: Paramètre d'entrée `default` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        return self.params.get(key, default)

    def _infer_batch_ids(self, data: Any, n_cells: int) -> Tuple[Optional[np.ndarray], int, Optional[str]]:
        """Helper interne: infer batch ids.
        
        
        Args:
            data: Paramètre d'entrée `data` utilisé dans cette étape du pipeline.
            n_cells: Paramètre d'entrée `n_cells` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        # Prépare les IDs batch nécessaires à la branche adversariale (DANN).
        use_batch = bool(self._param("use_batch_conditioning", False))
        adv_w = float(self._param("adversarial_batch_weight", 0.0) or 0.0)
        if not use_batch and adv_w <= 0.0:
            return None, 0, None

        key = str(self._param("batch_correction_key", "")).strip()
        if not key:
            raise ValueError("Batch conditioning enabled but 'batch_correction_key' is empty.")
        if not hasattr(data, "obs") or key not in data.obs.columns:
            raise ValueError(f"Batch key '{key}' not found in adata.obs.")

        raw = np.asarray(data.obs[key].astype(str).to_numpy(), dtype=object)
        if len(raw) != n_cells:
            raise ValueError("Batch label length does not match number of cells.")

        uniq = sorted(np.unique(raw).tolist())
        mapping = {v: i for i, v in enumerate(uniq)}
        ids = np.asarray([mapping[v] for v in raw], dtype=np.int64)
        return ids, len(uniq), key

    def _snapshot(
        self,
        epoch: int,
        phase: str,
        embeddings: np.ndarray,
        cell_weights: np.ndarray,
        snapshot_type: str = "periodic",
    ) -> None:
        """Helper interne: snapshot.
        
        
        Args:
            epoch: Paramètre d'entrée `epoch` utilisé dans cette étape du pipeline.
            phase: Paramètre d'entrée `phase` utilisé dans cette étape du pipeline.
            embeddings: Paramètre d'entrée `embeddings` utilisé dans cette étape du pipeline.
            cell_weights: Paramètre d'entrée `cell_weights` utilisé dans cette étape du pipeline.
            snapshot_type: Paramètre d'entrée `snapshot_type` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        # Chaque snapshot contient embeddings + poids cellule au même epoch.
        self._embedding_snapshots.append(
            {
                "epoch": int(epoch),
                "phase": str(phase),
                "snapshot_type": str(snapshot_type),
                "embeddings": np.asarray(embeddings, dtype=np.float32),
                "cell_weights": np.asarray(cell_weights, dtype=np.float32),
            }
        )

    def _encode_full(self, X: np.ndarray) -> np.ndarray:
        """Helper interne: encode full.
        
        
        Args:
            X: Paramètre d'entrée `X` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        # Encodage complet du dataset, utilisé pour pseudo-labels et export final.
        return self._encode_numpy(X, batch_size=max(512, int(self._param("batch_size", 256))))

    def fit(self, data: Any, labels: Optional[Any] = None) -> "ScRAWAlgorithm":
        """Entraîne le modèle sur les données fournies.
        
        
        Args:
            data: Paramètre d'entrée `data` utilisé dans cette étape du pipeline.
            labels: Paramètre d'entrée `labels` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        # 1) Initialisation déterministe pour rendre le run reproductible.
        seed = int(self._param("seed", self._param("random_state", 42)))
        self._set_seed(seed)

        # 2) Chargement des matrices d'entrée et de reconstruction.
        X = self._as_numpy_matrix(data)
        n_cells, n_features = X.shape

        recon_target, recon_mode = self._prepare_reconstruction_target(data, X)
        if recon_target.shape != X.shape:
            raise ValueError(
                f"Reconstruction target shape {recon_target.shape} does not match input shape {X.shape}."
            )

        # 3) Optionnel: extraction du batch pour DANN.
        batch_ids_np, n_batches, batch_key = self._infer_batch_ids(data, n_cells)
        self._batch_info = (batch_key, int(n_batches))

        # 4) Construction du modèle autoencodeur principal.
        model = self._build_model(input_dim=n_features)
        device = torch.device(self.get_device())
        model.to(device)

        use_batch = bool(self._param("use_batch_conditioning", False))
        adv_weight = float(self._param("adversarial_batch_weight", 0.0) or 0.0)

        batch_head: Optional[nn.Module] = None
        if use_batch and adv_weight > 0.0 and n_batches >= 2:
            # Tête de classification batch utilisée avec gradient reversal.
            z_dim = int(self._param("z_dim", 128))
            hidden = max(8, z_dim // 2)
            batch_head = nn.Sequential(
                nn.Linear(z_dim, hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, n_batches),
            ).to(device)
        elif use_batch and adv_weight > 0.0:
            logger.warning("DANN requested but only one batch found; adversarial branch disabled.")

        params = list(model.parameters())
        if batch_head is not None:
            params += list(batch_head.parameters())

        # 5) Optimiseur commun (autoencodeur + tête DANN si active).
        optimizer = torch.optim.Adam(params, lr=float(self._param("lr", 1e-3)))

        epochs = int(self._param("epochs", 120))
        warmup = int(self._param("warmup_epochs", 30))
        batch_size = int(self._param("batch_size", 256))
        mask_rate = float(self._param("masking_rate", 0.2))
        mask_in_weighted = bool(self._param("masking_apply_weighted", False))

        update_interval = int(self._param("dynamic_weight_update_interval", 10))
        momentum = float(self._param("dynamic_weight_momentum", 0.7))
        momentum = float(np.clip(momentum, 0.0, 1.0))

        triplet_weight = float(self._param("rare_triplet_weight", 0.1))
        triplet_start = int(self._param("rare_triplet_start_epoch", 35))

        capture = bool(self._param("capture_embedding_snapshots", False))
        snap_interval = int(self._param("snapshot_interval_epochs", 10) or 10)
        snap_interval = max(1, snap_interval)

        # Poids de reconstruction par cellule (mis à jour dans la phase weighted).
        current_weights = np.ones(n_cells, dtype=np.float32)
        current_pseudo = np.zeros(n_cells, dtype=np.int64)

        warm_hist = {
            "name": "warm-up",
            "epochs": [],
            "train_loss": [],
            "components": {"reconstruction": [], "triplet": [], "batch_adv": []},
        }
        weighted_hist = {
            "name": "weighted",
            "epochs": [],
            "train_loss": [],
            "components": {"reconstruction": [], "triplet": [], "batch_adv": []},
        }

        # 6) Boucle principale d'entraînement.
        for epoch in range(epochs):
            weighted_phase = epoch >= warmup
            if weighted_phase and (
                epoch == warmup
                or (update_interval > 0 and ((epoch - warmup) % update_interval == 0))
            ):
                # Recalcule pseudo-labels + poids globaux à intervalle régulier.
                emb_for_weights = self._encode_full(X)
                pseudo_new = self._pseudo_labels(emb_for_weights)
                weights_new = self._combined_cell_weights(emb_for_weights, pseudo_new)

                if epoch == warmup:
                    current_weights = weights_new
                else:
                    current_weights = momentum * current_weights + (1.0 - momentum) * weights_new
                current_pseudo = pseudo_new

            perm = np.random.permutation(n_cells)
            total_loss_sum = 0.0
            rec_sum = 0.0
            triplet_sum = 0.0
            adv_sum = 0.0
            n_batches_seen = 0

            model.train()
            if batch_head is not None:
                batch_head.train()

            # 7) Entraînement mini-batch.
            for start in range(0, n_cells, batch_size):
                idx = perm[start : start + batch_size]
                xb_np = X[idx]
                tb_np = recon_target[idx]

                xb = torch.tensor(xb_np, dtype=torch.float32, device=device)
                tb = torch.tensor(tb_np, dtype=torch.float32, device=device)

                if mask_rate > 0.0 and (not weighted_phase or mask_in_weighted):
                    x_in = self._apply_random_mask(xb, mask_rate)
                else:
                    x_in = xb

                # Forward AE: embeddings latents + reconstruction.
                z, recon_raw = model(x_in)
                loss_per_sample = self._reconstruction_loss_per_sample(
                    target=tb,
                    recon_raw=recon_raw,
                    mode=recon_mode,
                )

                if weighted_phase:
                    # En phase weighted, chaque cellule contribue selon son poids dynamique.
                    w_t = torch.tensor(current_weights[idx], dtype=torch.float32, device=device)
                    reconstruction_loss = torch.mean(loss_per_sample * w_t)
                else:
                    w_t = torch.ones(len(idx), dtype=torch.float32, device=device)
                    reconstruction_loss = torch.mean(loss_per_sample)

                triplet_loss = torch.tensor(0.0, device=device)
                if weighted_phase and triplet_weight > 0.0 and epoch >= triplet_start:
                    # Régularisation rare activée seulement après un certain epoch.
                    triplet_loss = self._rare_triplet_loss(
                        z=z,
                        pseudo_labels_batch=current_pseudo[idx],
                        weights_batch=w_t,
                    )

                adv_loss = torch.tensor(0.0, device=device)
                if batch_head is not None and batch_ids_np is not None and adv_weight > 0.0:
                    start_epoch = int(self._param("adversarial_start_epoch", 10))
                    if epoch >= start_epoch:
                        # Ramp-up progressif de la force adversariale pour stabiliser le training.
                        ramp_epochs = int(self._param("adversarial_ramp_epochs", 20) or 0)
                        if ramp_epochs > 0:
                            frac = min(1.0, (epoch - start_epoch + 1) / float(ramp_epochs))
                        else:
                            frac = 1.0
                        lam = float(self._param("adversarial_lambda", 1.0)) * frac
                        logits = batch_head(gradient_reversal(z, lambda_=lam))
                        yb = torch.tensor(batch_ids_np[idx], dtype=torch.long, device=device)
                        adv_loss = nn.functional.cross_entropy(logits, yb)

                # Loss totale = reconstruction + triplet + batch adversarial.
                total_loss = reconstruction_loss + triplet_weight * triplet_loss + adv_weight * adv_loss

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

                total_loss_sum += float(total_loss.detach().cpu().item())
                rec_sum += float(reconstruction_loss.detach().cpu().item())
                triplet_sum += float(triplet_loss.detach().cpu().item())
                adv_sum += float(adv_loss.detach().cpu().item())
                n_batches_seen += 1

            if n_batches_seen == 0:
                raise RuntimeError("No mini-batch processed during training.")

            avg_total = total_loss_sum / n_batches_seen
            avg_rec = rec_sum / n_batches_seen
            avg_triplet = triplet_sum / n_batches_seen
            avg_adv = adv_sum / n_batches_seen

            hist = weighted_hist if weighted_phase else warm_hist
            hist["epochs"].append(int(epoch))
            hist["train_loss"].append(float(avg_total))
            hist["components"]["reconstruction"].append(float(avg_rec))
            hist["components"]["triplet"].append(float(avg_triplet))
            hist["components"]["batch_adv"].append(float(avg_adv))

            if capture and (epoch == 0 or epoch == epochs - 1 or (epoch % snap_interval == 0)):
                # Snapshot périodique pour la figure d'évolution UMAP.
                emb_snap = self._encode_full(X)
                phase = "weighted" if weighted_phase else "warm-up"
                self._snapshot(
                    epoch=epoch,
                    phase=phase,
                    embeddings=emb_snap,
                    cell_weights=current_weights,
                    snapshot_type="periodic",
                )

        # 8) Fin d'entraînement: embeddings finaux + clustering final.
        self._embeddings = self._encode_full(X)
        self._labels = self._hdbscan_clustering(self._embeddings)
        self._fitted = True

        # 9) Historique de loss exportable.
        history = []
        if warm_hist["epochs"]:
            history.append(warm_hist)
        if weighted_hist["epochs"]:
            history.append(weighted_hist)
        self._loss_history = history

        # 10) Trace des paramètres effectivement utilisés pendant ce run.
        resolved = {
            "seed": seed,
            "random_state": seed,
            "reconstruction_distribution": recon_mode,
            "pseudo_label_method_effective": self._pseudo_fallback_method
            or str(self._param("pseudo_label_method", "leiden")),
            "batch_correction_key_effective": batch_key,
            "n_batches_effective": int(n_batches),
            "mmd_batch_weight_effective": 0.0,
        }
        self.set_effective_params(resolved)

        return self

    def get_batch_info(self) -> tuple:
        """Réalise l'opération `get batch info` du module `scraw_algorithm`.
        
        Args:
            Aucun argument explicite en dehors du contexte objet.
        
        Returns:
            Valeur calculée par la fonction.
        """
        return self._batch_info

    def predict(self, data: Any = None) -> Any:
        """Retourne les clusters prédits après entraînement.
        
        Args:
            data: Paramètre d'entrée `data` utilisé dans cette étape du pipeline.
        
        Returns:
            Valeur calculée par la fonction.
        """
        if not self._fitted or self._labels is None:
            raise RuntimeError("Algorithm not fitted.")
        return self._labels
