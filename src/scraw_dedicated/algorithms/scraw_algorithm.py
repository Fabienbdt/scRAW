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
                    name="weight_exponent",
                    display_name="Cluster Weight Exponent",
                    param_type=ParamType.FLOAT,
                    default=0.4,
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
                    name="density_weight_clip",
                    display_name="Density Weight Clip",
                    param_type=ParamType.FLOAT,
                    default=5.0,
                    min_value=0.1,
                    max_value=100.0,
                    description="Maximum density-derived weight before normalization.",
                    category="Weighting",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="weight_fusion_mode",
                    display_name="Weight Fusion Mode",
                    param_type=ParamType.CHOICE,
                    default="additive",
                    choices=["additive", "multiplicative"],
                    description="How cluster and density weights are fused.",
                    category="Weighting",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="cluster_weight_power",
                    display_name="Cluster Weight Power",
                    param_type=ParamType.FLOAT,
                    default=1.0,
                    min_value=0.0,
                    max_value=4.0,
                    description="Power applied to cluster-frequency weights before fusion.",
                    category="Weighting",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="density_weight_power",
                    display_name="Density Weight Power",
                    param_type=ParamType.FLOAT,
                    default=1.0,
                    min_value=0.0,
                    max_value=4.0,
                    description="Power applied to density weights before fusion.",
                    category="Weighting",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="min_cell_weight",
                    display_name="Min Cell Weight",
                    param_type=ParamType.FLOAT,
                    default=0.25,
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
                    name="nb_mu_clip_max",
                    display_name="NB Mu Clip Max",
                    param_type=ParamType.FLOAT,
                    default=1e6,
                    min_value=1.0,
                    max_value=1e9,
                    description="Upper clip value for NB mean parameter.",
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
                    name="masked_recon_weight",
                    display_name="Masked Recon Weight",
                    param_type=ParamType.FLOAT,
                    default=0.75,
                    min_value=0.0,
                    max_value=1.0,
                    description="Relative weight of masked positions in reconstruction loss.",
                    category="Reconstruction",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="masking_value",
                    display_name="Masking Value",
                    param_type=ParamType.FLOAT,
                    default=0.0,
                    min_value=-100.0,
                    max_value=100.0,
                    description="Value injected at masked positions.",
                    category="Reconstruction",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="rare_loss_type",
                    display_name="Rare Loss Type",
                    param_type=ParamType.CHOICE,
                    default="triplet",
                    choices=["triplet"],
                    description="Rare-cell regularization objective.",
                    category="Rare",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="rare_triplet_weight",
                    display_name="Triplet Weight",
                    param_type=ParamType.FLOAT,
                    default=0.10,
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
                    min_value=-1,
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
                    default="auto",
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
                    default=0,
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
                    default=0,
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
                    description="Compatibility placeholder (ignored in dedicated version).",
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
                    description="Compatibility parameter (kept for parity with SCRBenchmark).",
                    category="General",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="cluster_preprocess_mode",
                    display_name="Cluster Preprocess Mode",
                    param_type=ParamType.STRING,
                    default="none",
                    description="Compatibility parameter (not used in dedicated version).",
                    category="General",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="unsupervised_k_fallback",
                    display_name="Unsupervised K Fallback",
                    param_type=ParamType.INTEGER,
                    default=0,
                    min_value=0,
                    max_value=1000,
                    description="Manual K override used when n_clusters=0 and labels are unavailable.",
                    category="Pseudo Labels",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="unsupervised_k_selection",
                    display_name="Unsupervised K Selection",
                    param_type=ParamType.CHOICE,
                    default="stability_consensus",
                    choices=["stability_consensus", "heuristic"],
                    description="Automatic K selection strategy when n_clusters=0 and labels are unavailable.",
                    category="Pseudo Labels",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="unsupervised_k_min",
                    display_name="Unsupervised K Min",
                    param_type=ParamType.INTEGER,
                    default=8,
                    min_value=2,
                    max_value=1000,
                    description="Minimum candidate K in unsupervised selection.",
                    category="Pseudo Labels",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="unsupervised_k_max",
                    display_name="Unsupervised K Max",
                    param_type=ParamType.INTEGER,
                    default=30,
                    min_value=2,
                    max_value=1000,
                    description="Maximum candidate K in unsupervised selection.",
                    category="Pseudo Labels",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="unsupervised_k_num_candidates",
                    display_name="Unsupervised K Candidates",
                    param_type=ParamType.INTEGER,
                    default=12,
                    min_value=3,
                    max_value=200,
                    description="Maximum number of K candidates evaluated.",
                    category="Pseudo Labels",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="unsupervised_k_pca_dim",
                    display_name="Unsupervised K PCA Dim",
                    param_type=ParamType.INTEGER,
                    default=32,
                    min_value=2,
                    max_value=512,
                    description="PCA dimension used before unsupervised K scoring.",
                    category="Pseudo Labels",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="unsupervised_k_eval_sample_size",
                    display_name="Unsupervised K Eval Size",
                    param_type=ParamType.INTEGER,
                    default=3000,
                    min_value=200,
                    max_value=100000,
                    description="Sample size used for CVI scoring during K selection.",
                    category="Pseudo Labels",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="unsupervised_k_stability_runs",
                    display_name="Unsupervised K Stability Runs",
                    param_type=ParamType.INTEGER,
                    default=5,
                    min_value=2,
                    max_value=100,
                    description="Number of repeated KMeans runs for stability scoring.",
                    category="Pseudo Labels",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="unsupervised_k_stability_sample_size",
                    display_name="Unsupervised K Stability Size",
                    param_type=ParamType.INTEGER,
                    default=4000,
                    min_value=300,
                    max_value=100000,
                    description="Sample size used for stability scoring.",
                    category="Pseudo Labels",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="unsupervised_k_weight_stability",
                    display_name="K Weight Stability",
                    param_type=ParamType.FLOAT,
                    default=0.45,
                    min_value=0.0,
                    max_value=5.0,
                    description="Weight of stability score in K consensus.",
                    category="Pseudo Labels",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="unsupervised_k_weight_silhouette",
                    display_name="K Weight Silhouette",
                    param_type=ParamType.FLOAT,
                    default=0.25,
                    min_value=0.0,
                    max_value=5.0,
                    description="Weight of silhouette score in K consensus.",
                    category="Pseudo Labels",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="unsupervised_k_weight_ch",
                    display_name="K Weight CH",
                    param_type=ParamType.FLOAT,
                    default=0.20,
                    min_value=0.0,
                    max_value=5.0,
                    description="Weight of Calinski-Harabasz score in K consensus.",
                    category="Pseudo Labels",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="unsupervised_k_weight_db",
                    display_name="K Weight DB",
                    param_type=ParamType.FLOAT,
                    default=0.10,
                    min_value=0.0,
                    max_value=5.0,
                    description="Weight of Davies-Bouldin score in K consensus.",
                    category="Pseudo Labels",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="unsupervised_k_weight_tiny_clusters",
                    display_name="K Weight Tiny Clusters",
                    param_type=ParamType.FLOAT,
                    default=0.20,
                    min_value=0.0,
                    max_value=5.0,
                    description="Penalty weight for tiny clusters in K consensus.",
                    category="Pseudo Labels",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="unsupervised_k_min_cluster_fraction",
                    display_name="K Tiny Cluster Fraction",
                    param_type=ParamType.FLOAT,
                    default=0.005,
                    min_value=0.0,
                    max_value=0.2,
                    description="Threshold defining tiny clusters during K consensus.",
                    category="Pseudo Labels",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="unsupervised_k_overseg_penalty",
                    display_name="K Overseg Penalty",
                    param_type=ParamType.FLOAT,
                    default=0.25,
                    min_value=0.0,
                    max_value=5.0,
                    description="Penalty applied when K is above anchor heuristic.",
                    category="Pseudo Labels",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="unsupervised_k_underseg_penalty",
                    display_name="K Underseg Penalty",
                    param_type=ParamType.FLOAT,
                    default=0.05,
                    min_value=0.0,
                    max_value=5.0,
                    description="Penalty applied when K is below anchor heuristic.",
                    category="Pseudo Labels",
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
        """Resolve batch ids from adata.obs for reporting and optional DANN."""
        use_batch = bool(self._param("use_batch_conditioning", False))
        adv_w = float(self._param("adversarial_batch_weight", 0.0) or 0.0)
        mmd_w = float(self._param("mmd_batch_weight", 0.0) or 0.0)
        want_batch = use_batch or adv_w > 0.0 or mmd_w > 0.0

        # Match SCRBenchmark behavior: do not auto-detect/use batch ids when
        # batch conditioning losses are disabled.
        if not want_batch:
            return None, 1, None

        if not hasattr(data, "obs"):
            if want_batch:
                raise ValueError("Batch conditioning requested but input has no obs table.")
            return None, 1, None

        obs_cols = list(data.obs.columns)
        requested = str(self._param("batch_correction_key", "auto")).strip()
        if not requested:
            requested = "auto"

        key: Optional[str] = None
        if requested.lower() != "auto":
            if requested in obs_cols:
                key = requested
            elif want_batch:
                raise ValueError(f"Batch key '{requested}' not found in adata.obs.")
        else:
            auto_keys = (
                "batch",
                "tech",
                "study",
                "batch_key",
                "_scvi_batch",
                "system",
                "patient",
                "sample",
                "donor",
            )
            lower_map = {str(c).lower(): str(c) for c in obs_cols}
            for k in auto_keys:
                if k in lower_map:
                    key = lower_map[k]
                    break

        if key is None:
            if want_batch:
                raise ValueError(
                    "Batch conditioning requested but no batch key was found in adata.obs."
                )
            return None, 0, None

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
        # Align with SCRBenchmark: single-pass full-dataset encoding to minimize
        # tiny chunking differences that can change HDBSCAN boundaries.
        if self.model is None:
            return self._encode_numpy(X, batch_size=max(512, int(self._param("batch_size", 256))))

        device = torch.device(self.get_device())
        self.model.to(device)
        was_training = bool(self.model.training)
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32, device=device)
            emb = self.model.encoder(xb).detach().cpu().numpy().astype(np.float32, copy=False)
        if was_training:
            self.model.train()
        return emb

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
        X_proc = self._as_numpy_matrix(data)
        n_cells, _ = X_proc.shape

        configured_n_clusters = int(self._param("n_clusters", 0) or 0)
        if configured_n_clusters == 0 and labels is not None:
            try:
                pseudo_k = int(len(np.unique(np.asarray(labels))))
            except Exception:
                pseudo_k = 0
        else:
            pseudo_k = configured_n_clusters
        self.params["_pseudo_n_clusters"] = int(max(0, pseudo_k))

        X = np.asarray(X_proc, dtype=np.float32)
        recon_target, recon_mode = self._prepare_reconstruction_target(data, X_proc)
        size_factors_np: Optional[np.ndarray] = None

        # En mode NB, utiliser _prepare_nb_inputs pour obtenir :
        #  - X_model : log1p(raw_counts) comme entrée du modèle (au lieu de adata.X preprocessed)
        #  - raw counts comme cible NB (au lieu de log1p(counts))
        #  - size factors réels (au lieu de 1.0)
        if recon_mode == "nb":
            nb_result = self._prepare_nb_inputs(data)
            if nb_result is not None:
                X_model_nb, counts_nb, sf_nb = nb_result
                X = X_model_nb
                recon_target = counts_nb
                size_factors_np = sf_nb

        n_cells, n_features = X.shape
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
        lr = float(self._param("lr", 1e-3))
        optimizer = torch.optim.Adam(params, lr=lr)

        epochs = int(self._param("epochs", 120))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=lr * 0.01
        )
        warmup = int(self._param("warmup_epochs", 30))
        batch_size = int(self._param("batch_size", 256))
        mask_rate = float(self._param("masking_rate", 0.2))
        mask_in_weighted = bool(self._param("masking_apply_weighted", False))
        masked_recon_weight = float(np.clip(self._param("masked_recon_weight", 0.75), 0.0, 1.0))
        masking_value = float(self._param("masking_value", 0.0))

        update_interval = int(self._param("dynamic_weight_update_interval", 10))
        momentum = float(self._param("dynamic_weight_momentum", 0.7))
        momentum = float(np.clip(momentum, 0.0, 1.0))

        triplet_weight = float(self._param("rare_triplet_weight", 0.10))
        raw_triplet_start = int(self._param("rare_triplet_start_epoch", 35))
        triplet_start = raw_triplet_start if raw_triplet_start >= 0 else warmup

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

        # DataLoader aligné avec SCRBenchmark pour reproduire l'ordre mini-batch PyTorch.
        from torch.utils.data import DataLoader, TensorDataset

        X_tensor_cpu = torch.from_numpy(X).float()
        target_tensor_cpu = torch.from_numpy(recon_target).float()
        index_tensor_cpu = torch.arange(n_cells, dtype=torch.long)
        if size_factors_np is not None:
            sf_tensor_cpu = torch.from_numpy(size_factors_np).float()
            dataset = TensorDataset(
                X_tensor_cpu, target_tensor_cpu, sf_tensor_cpu, index_tensor_cpu
            )
        else:
            dataset = TensorDataset(X_tensor_cpu, target_tensor_cpu, index_tensor_cpu)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
                    mixed = momentum * current_weights + (1.0 - momentum) * weights_new
                    mean_w = float(np.mean(mixed))
                    if np.isfinite(mean_w) and mean_w > 0.0:
                        mixed = mixed / mean_w
                    w_min = float(self._param("min_cell_weight", 0.25))
                    w_max = float(self._param("max_cell_weight", 10.0))
                    if w_max < w_min:
                        w_max = w_min
                    current_weights = np.clip(mixed, w_min, w_max).astype(np.float32)
                current_pseudo = pseudo_new

            total_loss_sum = 0.0
            rec_sum = 0.0
            triplet_sum = 0.0
            adv_sum = 0.0
            n_batches_seen = 0

            model.train()
            if batch_head is not None:
                batch_head.train()

            # 7) Entraînement mini-batch.
            for batch in loader:
                if size_factors_np is not None:
                    xb, tb, sf_t, idx_t = batch
                    sf_t = sf_t.to(device)
                else:
                    xb, tb, idx_t = batch
                    sf_t = None
                xb = xb.to(device)
                tb = tb.to(device)
                idx = idx_t.detach().cpu().numpy()

                if mask_rate > 0.0 and (not weighted_phase or mask_in_weighted):
                    x_in, mask = self._apply_random_mask(
                        xb, mask_rate, masking_value=masking_value
                    )
                else:
                    x_in = xb
                    mask = None

                # Forward AE: embeddings latents + reconstruction.
                z, recon_raw = model(x_in)
                loss_per_sample = self._reconstruction_loss_per_sample(
                    target=tb,
                    recon_raw=recon_raw,
                    mode=recon_mode,
                    size_factors=sf_t,
                    mask=mask,
                    masked_recon_weight=masked_recon_weight,
                )

                if weighted_phase:
                    # En phase weighted, chaque cellule contribue selon son poids dynamique.
                    w_t = torch.tensor(current_weights[idx], dtype=torch.float32, device=device)
                    reconstruction_loss = torch.mean(loss_per_sample * w_t)
                else:
                    w_t = torch.ones(idx_t.shape[0], dtype=torch.float32, device=device)
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
                    start_epoch = int(self._param("adversarial_start_epoch", 0))
                    if epoch >= start_epoch:
                        # Ramp-up progressif de la force adversariale pour stabiliser le training.
                        ramp_epochs = int(self._param("adversarial_ramp_epochs", 0) or 0)
                        if ramp_epochs > 0:
                            frac = min(1.0, (epoch - start_epoch + 1) / float(ramp_epochs))
                        else:
                            frac = 1.0
                        lam = float(self._param("adversarial_lambda", 1.0)) * frac
                        logits = batch_head(gradient_reversal(z, lambda_=lam))
                        yb = torch.tensor(batch_ids_np[idx], dtype=torch.long, device=device)
                        adv_loss = nn.functional.cross_entropy(logits, yb)

                # Ramp-up linéaire de la triplet loss sur 20 epochs pour éviter
                # une perturbation brutale de l'espace latent.
                rare_loss_ramp = 0.0
                if triplet_weight > 0.0 and epoch >= triplet_start:
                    ramp_epochs = max(1, min(20, epochs - triplet_start))
                    rare_loss_ramp = min(1.0, (epoch - triplet_start) / ramp_epochs)

                # Loss totale = reconstruction + triplet (rampée) + batch adversarial.
                total_loss = reconstruction_loss + (rare_loss_ramp * triplet_weight * triplet_loss) + adv_weight * adv_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
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

            scheduler.step()

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
        n_clusters_effective = int(
            self.params.get("_pseudo_n_clusters", 0)
            or self.params.get("unsupervised_k_selected", 0)
            or 0
        )
        resolved = {
            "seed": seed,
            "random_state": seed,
            "reconstruction_distribution": recon_mode,
            "pseudo_label_method_effective": self._pseudo_fallback_method
            or str(self._param("pseudo_label_method", "leiden")),
            "batch_correction_key_effective": batch_key,
            "n_batches_effective": int(n_batches),
            "mmd_batch_weight_effective": float(self._param("mmd_batch_weight", 0.0) or 0.0),
            "n_clusters_effective": n_clusters_effective,
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
