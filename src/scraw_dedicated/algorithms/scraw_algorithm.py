"""Main scRAW algorithm orchestration.

This file keeps the training orchestration readable by delegating:
- clustering/pseudo-label logic to `scraw_clustering.py`
- reconstruction/weighting/triplet logic to `scraw_losses_and_weights.py`

High-level fit pipeline:
1) prepare model inputs / targets (MSE or NB path),
2) warm-up reconstruction phase (uniform per-cell contribution),
3) weighted phase with pseudo-label refresh and optional regularizers,
4) final embedding + final clustering export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn

from ..core.algorithm_registry import AlgorithmInfo, AlgorithmRegistry
from ..core.config import HyperparameterConfig, ParamType
from ..defaults import DEFAULT_PARAM_OVERRIDES
from .base_autoencoder import BaseAutoencoderAlgorithm, gradient_reversal
from .scraw_clustering import ScrawClusteringMixin
from .scraw_losses_and_weights import ScrawLossWeightMixin


logger = logging.getLogger(__name__)


@AlgorithmRegistry.register
class ScRAWAlgorithm(BaseAutoencoderAlgorithm, ScrawLossWeightMixin, ScrawClusteringMixin):
    """Lean scRAW implementation focused on reproducible ablation runs.

    Design goals:
    - deterministic/restartable training,
    - explicit effective parameters for auditability,
    - robust fallbacks when pseudo-label or final clustering fails.
    """

    @classmethod
    def get_info(cls) -> AlgorithmInfo:
        """Return static metadata used by the algorithm registry/UI."""
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
        """Expose all tunable hyperparameters for training and clustering."""
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
                    default=int(DEFAULT_PARAM_OVERRIDES["hdbscan_min_cluster_size"]),
                    min_value=2,
                    max_value=200,
                    description="Minimum cluster size for HDBSCAN.",
                    category="Clustering",
                ),
                HyperparameterConfig(
                    name="hdbscan_min_samples",
                    display_name="HDBSCAN Min Samples",
                    param_type=ParamType.INTEGER,
                    default=int(DEFAULT_PARAM_OVERRIDES["hdbscan_min_samples"]),
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
                    default=float(DEFAULT_PARAM_OVERRIDES["weight_exponent"]),
                    min_value=0.0,
                    max_value=4.0,
                    description="Exponent for inverse-frequency cluster weighting.",
                    category="Weighting",
                ),
                HyperparameterConfig(
                    name="cluster_density_alpha",
                    display_name="Density Mix Alpha",
                    param_type=ParamType.FLOAT,
                    default=float(DEFAULT_PARAM_OVERRIDES["cluster_density_alpha"]),
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
                    default=float(DEFAULT_PARAM_OVERRIDES["density_weight_clip"]),
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
                    default=float(DEFAULT_PARAM_OVERRIDES["min_cell_weight"]),
                    min_value=0.0,
                    max_value=100.0,
                    description="Lower bound for per-cell reconstruction weights.",
                    category="Weighting",
                ),
                HyperparameterConfig(
                    name="max_cell_weight",
                    display_name="Max Cell Weight",
                    param_type=ParamType.FLOAT,
                    default=float(DEFAULT_PARAM_OVERRIDES["max_cell_weight"]),
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
                    default=float(DEFAULT_PARAM_OVERRIDES["dynamic_weight_momentum"]),
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
                    default=str(DEFAULT_PARAM_OVERRIDES["reconstruction_distribution"]),
                    choices=["nb", "mse"],
                    description="Reconstruction objective.",
                    category="Reconstruction",
                ),
                HyperparameterConfig(
                    name="nb_theta",
                    display_name="NB Theta",
                    param_type=ParamType.FLOAT,
                    default=float(DEFAULT_PARAM_OVERRIDES["nb_theta"]),
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
                    default=float(DEFAULT_PARAM_OVERRIDES["masking_rate"]),
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
                    default=bool(DEFAULT_PARAM_OVERRIDES["masking_apply_weighted"]),
                    description="If false, masking is only applied in warm-up phase.",
                    category="Reconstruction",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="masked_recon_weight",
                    display_name="Masked Recon Weight",
                    param_type=ParamType.FLOAT,
                    default=float(DEFAULT_PARAM_OVERRIDES["masked_recon_weight"]),
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
                    default=float(DEFAULT_PARAM_OVERRIDES["rare_triplet_weight"]),
                    min_value=0.0,
                    max_value=100.0,
                    description="Triplet regularization strength.",
                    category="Rare",
                ),
                HyperparameterConfig(
                    name="rare_triplet_start_epoch",
                    display_name="Triplet Start Epoch",
                    param_type=ParamType.INTEGER,
                    default=int(DEFAULT_PARAM_OVERRIDES["rare_triplet_start_epoch"]),
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
                    default=float(DEFAULT_PARAM_OVERRIDES["rare_triplet_min_weight"]),
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
                    default=bool(DEFAULT_PARAM_OVERRIDES["use_batch_conditioning"]),
                    description="Enable adversarial batch conditioning.",
                    category="Batch",
                ),
                HyperparameterConfig(
                    name="batch_correction_key",
                    display_name="Batch Key",
                    param_type=ParamType.STRING,
                    default=str(DEFAULT_PARAM_OVERRIDES["batch_correction_key"]),
                    description="obs key containing batch labels.",
                    category="Batch",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="adversarial_batch_weight",
                    display_name="DANN Weight",
                    param_type=ParamType.FLOAT,
                    default=float(DEFAULT_PARAM_OVERRIDES["adversarial_batch_weight"]),
                    min_value=0.0,
                    max_value=100.0,
                    description="Weight of adversarial batch classification loss.",
                    category="Batch",
                ),
                HyperparameterConfig(
                    name="adversarial_lambda",
                    display_name="GRL Lambda",
                    param_type=ParamType.FLOAT,
                    default=float(DEFAULT_PARAM_OVERRIDES["adversarial_lambda"]),
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
                    default=int(DEFAULT_PARAM_OVERRIDES["adversarial_start_epoch"]),
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
                    default=int(DEFAULT_PARAM_OVERRIDES["adversarial_ramp_epochs"]),
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
                    name="resume_checkpoint_path",
                    display_name="Resume Checkpoint",
                    param_type=ParamType.STRING,
                    default="",
                    description=(
                        "Optional training checkpoint path used to resume optimization "
                        "from a shared latent state."
                    ),
                    category="Monitoring",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="save_checkpoint_path",
                    display_name="Save Checkpoint",
                    param_type=ParamType.STRING,
                    default="",
                    description=(
                        "Optional output path where a training checkpoint is dumped "
                        "when `stop_after_epoch` is reached."
                    ),
                    category="Monitoring",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="stop_after_epoch",
                    display_name="Stop After Epoch",
                    param_type=ParamType.INTEGER,
                    default=-1,
                    min_value=-1,
                    max_value=2000,
                    description=(
                        "Early-stop training after this epoch index (inclusive); "
                        "-1 disables early stop."
                    ),
                    category="Monitoring",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="resume_load_optimizer",
                    display_name="Resume Optimizer",
                    param_type=ParamType.BOOLEAN,
                    default=True,
                    description=(
                        "When resuming from checkpoint, also restore optimizer/scheduler "
                        "states when compatible."
                    ),
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
                    description="Compatibility parameter kept for legacy config files.",
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

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialize training state and fallback flags."""
        super().__init__(params=params)
        self._batch_info: Tuple[Optional[str], int] = (None, 0)
        self._pseudo_fallback_method: Optional[str] = None
        self._leiden_warning_emitted: bool = False

    def _param(self, key: str, default: Any) -> Any:
        """Read one parameter with a default fallback."""
        return self.params.get(key, default)

    def _path_param(self, key: str) -> Optional[Path]:
        """Parse one path-like hyperparameter into a normalized Path."""
        raw = self._param(key, "")
        text = "" if raw is None else str(raw).strip()
        if not text:
            return None
        # On normalise vers un chemin absolu pour éviter les ambiguïtés CWD.
        return Path(text).expanduser().resolve()

    def _save_training_checkpoint(
        self,
        path: Path,
        *,
        next_epoch: int,
        model: nn.Module,
        batch_head: Optional[nn.Module],
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        current_weights: np.ndarray,
        current_cluster_component: np.ndarray,
        current_density_component: np.ndarray,
        current_fused_unclipped: np.ndarray,
        current_pseudo: np.ndarray,
        warm_hist: Dict[str, Any],
        weighted_hist: Dict[str, Any],
    ) -> None:
        """Save a full training checkpoint for resume-at-epoch workflows."""
        # Snapshot complet: assez d'information pour reprendre l'entraînement
        # sans perdre l'état dynamique des poids/pseudo-labels.
        payload = {
            "version": 1,
            "next_epoch": int(next_epoch),
            "model_state": model.state_dict(),
            "batch_head_state": None if batch_head is None else batch_head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "current_weights": np.asarray(current_weights, dtype=np.float32),
            "current_cluster_component": np.asarray(current_cluster_component, dtype=np.float32),
            "current_density_component": np.asarray(current_density_component, dtype=np.float32),
            "current_fused_unclipped": np.asarray(current_fused_unclipped, dtype=np.float32),
            "current_pseudo": np.asarray(current_pseudo, dtype=np.int64),
            "warm_hist": warm_hist,
            "weighted_hist": weighted_hist,
            "embedding_snapshots": self._embedding_snapshots,
            "params": dict(self.params),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(path))

    def _infer_batch_ids(self, data: Any, n_cells: int) -> Tuple[Optional[np.ndarray], int, Optional[str]]:
        """Resolve batch ids from adata.obs for reporting and optional DANN.

        Batch key resolution order:
        - explicit `batch_correction_key` when provided,
        - otherwise automatic scan over common column names.

        Returns `(batch_ids, n_batches, key_used)`.
        """
        use_batch = bool(
            self._param("use_batch_conditioning", DEFAULT_PARAM_OVERRIDES["use_batch_conditioning"])
        )
        adv_w = float(
            self._param(
                "adversarial_batch_weight",
                DEFAULT_PARAM_OVERRIDES["adversarial_batch_weight"],
            )
            or 0.0
        )
        mmd_w = float(self._param("mmd_batch_weight", 0.0) or 0.0)
        # `want_batch` active la recherche de colonne batch même si `use_batch=False`
        # dès qu'une pénalité batch (adv/mmd) est non nulle.
        want_batch = use_batch or adv_w > 0.0 or mmd_w > 0.0

        if not want_batch:
            return None, 1, None

        if not hasattr(data, "obs"):
            if want_batch:
                raise ValueError("Batch conditioning requested but input has no obs table.")
            return None, 1, None

        obs_cols = list(data.obs.columns)
        requested = str(
            self._param("batch_correction_key", DEFAULT_PARAM_OVERRIDES["batch_correction_key"])
        ).strip()
        if not requested:
            requested = "auto"

        key: Optional[str] = None
        if requested.lower() != "auto":
            if requested in obs_cols:
                key = requested
            elif want_batch:
                raise ValueError(f"Batch key '{requested}' not found in adata.obs.")
        else:
            # Recherche automatique sur une liste de noms fréquents.
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
        # Encodage string -> int pour l'entraînement de la tête adversariale.
        mapping = {v: i for i, v in enumerate(uniq)}
        ids = np.asarray([mapping[v] for v in raw], dtype=np.int64)
        return ids, len(uniq), key

    def _snapshot(
        self,
        epoch: int,
        phase: str,
        embeddings: np.ndarray,
        cell_weights: np.ndarray,
        pseudo_labels: Optional[np.ndarray] = None,
        cluster_component_weights: Optional[np.ndarray] = None,
        density_component_weights: Optional[np.ndarray] = None,
        fused_weight_unclipped: Optional[np.ndarray] = None,
        snapshot_type: str = "periodic",
    ) -> None:
        """Store one latent snapshot for post-hoc analysis/visualization.

        Snapshot payload intentionally mirrors quantities used in weighting:
        embeddings, pseudo-labels, and each weight component at the same epoch.
        """
        # Chaque snapshot contient embeddings + poids cellule au même epoch.
        self._embedding_snapshots.append(
            {
                "epoch": int(epoch),
                "phase": str(phase),
                "snapshot_type": str(snapshot_type),
                "embeddings": np.asarray(embeddings, dtype=np.float32),
                "cell_weights": np.asarray(cell_weights, dtype=np.float32),
                "pseudo_labels": None
                if pseudo_labels is None
                else np.asarray(pseudo_labels, dtype=np.int64),
                "cluster_component_weights": None
                if cluster_component_weights is None
                else np.asarray(cluster_component_weights, dtype=np.float32),
                "density_component_weights": None
                if density_component_weights is None
                else np.asarray(density_component_weights, dtype=np.float32),
                "fused_weight_unclipped": None
                if fused_weight_unclipped is None
                else np.asarray(fused_weight_unclipped, dtype=np.float32),
            }
        )

    def _encode_full(self, X: np.ndarray) -> np.ndarray:
        """Encode the full dataset in one pass when memory allows.

        Falls back to chunked numpy path when model is absent.
        """
        if self.model is None:
            return self._encode_numpy(
                X,
                batch_size=max(
                    512,
                    int(self._param("batch_size", DEFAULT_PARAM_OVERRIDES["batch_size"])),
                ),
            )

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
        """Train scRAW end-to-end and compute final clusters.

        Workflow summary:
        1) Build model and reconstruction targets.
        2) Warm-up with unweighted reconstruction.
        3) Weighted phase with pseudo-label updates and optional rare/batch losses.
        4) Final latent embedding + HDBSCAN clustering.

        Expected input conventions:
        - `data.X`: processed matrix used in MSE mode (and as fallback),
        - `data.layers["original_X"]`: raw counts used by NB path when present,
        - `data.obs`: optional metadata for batch conditioning.
        """
        # --- Étape 1: initialisation déterministe (reproductibilité) ---
        seed = int(self._param("seed", self._param("random_state", 42)))
        self._set_seed(seed)
        self._embedding_snapshots = []
        self._loss_history = []

        # --- Étape 2: préparation des matrices d'entrée / cible ---
        X_proc = self._as_numpy_matrix(data)
        n_cells, _ = X_proc.shape

        configured_n_clusters = int(self._param("n_clusters", 0) or 0)
        # Si l'utilisateur n'impose pas K et que des labels sont fournis,
        # on initialise un K pseudo-supervisé avec le nombre de classes observées.
        if configured_n_clusters == 0 and labels is not None:
            try:
                pseudo_k = int(len(np.unique(np.asarray(labels))))
            except Exception:
                pseudo_k = 0
        else:
            pseudo_k = configured_n_clusters
        # `_pseudo_n_clusters` est ensuite relu par le mixin clustering.
        self.params["_pseudo_n_clusters"] = int(max(0, pseudo_k))

        X = np.asarray(X_proc, dtype=np.float32)
        recon_target, recon_mode = self._prepare_reconstruction_target(data, X_proc)
        size_factors_np: Optional[np.ndarray] = None

        # NB path: override generic target preparation with count-aware tensors:
        # - model input transformed from raw counts (e.g., log1p),
        # - NB target on raw counts,
        # - explicit per-cell size factors.
        # This is the branch that makes the effective NB loss counts-based.
        if recon_mode == "nb":
            nb_result = self._prepare_nb_inputs(data)
            if nb_result is not None:
                X_model_nb, counts_nb, sf_nb = nb_result
                # Ce trio aligne explicitement l'entraînement NB:
                # entrée transformée + cible counts bruts + size factors.
                X = X_model_nb
                recon_target = counts_nb
                size_factors_np = sf_nb

        n_cells, n_features = X.shape
        if recon_target.shape != X.shape:
            raise ValueError(
                f"Reconstruction target shape {recon_target.shape} does not match input shape {X.shape}."
            )

        # --- Étape 3: résolution batch (si nécessaire) ---
        batch_ids_np, n_batches, batch_key = self._infer_batch_ids(data, n_cells)
        self._batch_info = (batch_key, int(n_batches))

        # --- Étape 4: construction des modules réseau ---
        model = self._build_model(input_dim=n_features)
        device = torch.device(self.get_device())
        model.to(device)

        use_batch = bool(
            self._param("use_batch_conditioning", DEFAULT_PARAM_OVERRIDES["use_batch_conditioning"])
        )
        adv_weight = float(
            self._param(
                "adversarial_batch_weight",
                DEFAULT_PARAM_OVERRIDES["adversarial_batch_weight"],
            )
            or 0.0
        )

        batch_head: Optional[nn.Module] = None
        # La tête adversariale batch n'est instanciée que si:
        # - le mode batch est demandé,
        # - son poids est strictement positif,
        # - au moins 2 batches existent (sinon classification impossible).
        if use_batch and adv_weight > 0.0 and n_batches >= 2:
            # Tête de classification batch utilisée avec gradient reversal.
            z_dim = int(self._param("z_dim", DEFAULT_PARAM_OVERRIDES["z_dim"]))
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

        # --- Étape 5: configuration optimisation / planning ---
        lr = float(self._param("lr", DEFAULT_PARAM_OVERRIDES["lr"]))
        optimizer = torch.optim.Adam(params, lr=lr)

        epochs = int(self._param("epochs", DEFAULT_PARAM_OVERRIDES["epochs"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=lr * 0.01
        )
        warmup = int(self._param("warmup_epochs", DEFAULT_PARAM_OVERRIDES["warmup_epochs"]))
        batch_size = int(self._param("batch_size", DEFAULT_PARAM_OVERRIDES["batch_size"]))
        mask_rate = float(self._param("masking_rate", DEFAULT_PARAM_OVERRIDES["masking_rate"]))
        mask_in_weighted = bool(
            self._param("masking_apply_weighted", DEFAULT_PARAM_OVERRIDES["masking_apply_weighted"])
        )
        masked_recon_weight = float(
            np.clip(
                self._param("masked_recon_weight", DEFAULT_PARAM_OVERRIDES["masked_recon_weight"]),
                0.0,
                1.0,
            )
        )
        masking_value = float(self._param("masking_value", 0.0))

        update_interval = int(self._param("dynamic_weight_update_interval", 10))
        momentum = float(
            self._param("dynamic_weight_momentum", DEFAULT_PARAM_OVERRIDES["dynamic_weight_momentum"])
        )
        momentum = float(np.clip(momentum, 0.0, 1.0))

        triplet_weight = float(
            self._param("rare_triplet_weight", DEFAULT_PARAM_OVERRIDES["rare_triplet_weight"])
        )
        raw_triplet_start = int(
            self._param("rare_triplet_start_epoch", DEFAULT_PARAM_OVERRIDES["rare_triplet_start_epoch"])
        )
        triplet_start = raw_triplet_start if raw_triplet_start >= 0 else warmup

        capture = bool(self._param("capture_embedding_snapshots", False))
        snap_interval = int(self._param("snapshot_interval_epochs", 10) or 10)
        snap_interval = max(1, snap_interval)
        snapshot_anchor = int(max(0, warmup - 1))
        if epochs > 0:
            snapshot_anchor = int(min(snapshot_anchor, epochs - 1))

        def _should_capture_epoch(epoch_idx: int) -> bool:
            """Return whether an embedding snapshot should be captured at this epoch."""
            # On capture toujours:
            # - le dernier epoch,
            # - l'epoch "anchor" (fin warm-up),
            # - puis périodiquement après cet anchor.
            if epoch_idx == epochs - 1:
                return True
            if epoch_idx == snapshot_anchor:
                return True
            if epoch_idx > snapshot_anchor:
                return bool(((epoch_idx - snapshot_anchor) % snap_interval) == 0)
            return False

        # Poids de reconstruction par cellule (mis à jour dans la phase weighted).
        current_weights = np.ones(n_cells, dtype=np.float32)
        current_cluster_component = np.ones(n_cells, dtype=np.float32)
        current_density_component = np.ones(n_cells, dtype=np.float32)
        current_fused_unclipped = np.ones(n_cells, dtype=np.float32)
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

        resume_checkpoint_path = self._path_param("resume_checkpoint_path")
        save_checkpoint_path = self._path_param("save_checkpoint_path")
        stop_after_epoch = int(self._param("stop_after_epoch", -1) or -1)
        load_optimizer_state = bool(self._param("resume_load_optimizer", True))
        start_epoch = 0

        # DataLoader standard PyTorch avec shuffle pour l'entraînement mini-batch.
        from torch.utils.data import DataLoader, TensorDataset

        X_tensor_cpu = torch.from_numpy(X).float()
        target_tensor_cpu = torch.from_numpy(recon_target).float()
        index_tensor_cpu = torch.arange(n_cells, dtype=torch.long)
        # La structure du dataset dépend de la présence des size factors NB:
        # - NB: (x, target, size_factor, index)
        # - MSE: (x, target, index)
        if size_factors_np is not None:
            sf_tensor_cpu = torch.from_numpy(size_factors_np).float()
            dataset = TensorDataset(
                X_tensor_cpu, target_tensor_cpu, sf_tensor_cpu, index_tensor_cpu
            )
        else:
            dataset = TensorDataset(X_tensor_cpu, target_tensor_cpu, index_tensor_cpu)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optional resume: restore model and, when compatible, optimizer/scheduler.
        # Dynamic vectors (weights/pseudo-labels) and history are restored too so
        # resumed runs preserve phase state and exported diagnostics.
        if resume_checkpoint_path is not None:
            if not resume_checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Resume checkpoint not found: {resume_checkpoint_path}"
                )
            logger.info("Resuming from checkpoint: %s", resume_checkpoint_path)
            try:
                try:
                    checkpoint = torch.load(
                        str(resume_checkpoint_path),
                        map_location="cpu",
                        weights_only=False,
                    )
                except TypeError:
                    checkpoint = torch.load(str(resume_checkpoint_path), map_location="cpu")
            except Exception as exc:
                raise RuntimeError(f"Failed to load checkpoint {resume_checkpoint_path}: {exc}") from exc

            if not isinstance(checkpoint, dict) or checkpoint.get("model_state") is None:
                raise RuntimeError(
                    f"Invalid checkpoint format: {resume_checkpoint_path}"
                )

            model.load_state_dict(checkpoint["model_state"], strict=True)

            ckpt_head_state = checkpoint.get("batch_head_state")
            if batch_head is not None and ckpt_head_state is not None:
                try:
                    batch_head.load_state_dict(ckpt_head_state, strict=True)
                except Exception as exc:
                    logger.warning("Batch head state restore failed: %s", exc)
            elif batch_head is not None and ckpt_head_state is None:
                logger.info("No batch head state in checkpoint; DANN head starts from fresh init.")

            optimizer_loaded = False
            if load_optimizer_state and checkpoint.get("optimizer_state") is not None:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
                    optimizer_loaded = True
                except Exception as exc:
                    # Non bloquant: on peut repartir avec un optimiseur neuf.
                    logger.warning("Optimizer state restore failed (continuing): %s", exc)

            scheduler_loaded = False
            if load_optimizer_state and checkpoint.get("scheduler_state") is not None:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler_state"])
                    scheduler_loaded = True
                except Exception as exc:
                    # Non bloquant: on recale plus bas la phase du scheduler.
                    logger.warning("Scheduler state restore failed (continuing): %s", exc)

            def _restore_vec(key: str, default: np.ndarray, dtype: Any) -> np.ndarray:
                """Restore a 1D checkpoint vector with shape guard and dtype coercion."""
                raw = checkpoint.get(key)
                if raw is None:
                    return default
                arr = np.asarray(raw, dtype=dtype).reshape(-1)
                if arr.shape[0] != n_cells:
                    return default
                return arr

            current_weights = _restore_vec("current_weights", current_weights, np.float32)
            current_cluster_component = _restore_vec(
                "current_cluster_component", current_cluster_component, np.float32
            )
            current_density_component = _restore_vec(
                "current_density_component", current_density_component, np.float32
            )
            current_fused_unclipped = _restore_vec(
                "current_fused_unclipped", current_fused_unclipped, np.float32
            )
            current_pseudo = _restore_vec("current_pseudo", current_pseudo, np.int64)

            warm_hist_ckpt = checkpoint.get("warm_hist")
            weighted_hist_ckpt = checkpoint.get("weighted_hist")
            if isinstance(warm_hist_ckpt, dict):
                warm_hist = warm_hist_ckpt
            if isinstance(weighted_hist_ckpt, dict):
                weighted_hist = weighted_hist_ckpt

            snapshots_ckpt = checkpoint.get("embedding_snapshots")
            if isinstance(snapshots_ckpt, list):
                self._embedding_snapshots = snapshots_ckpt

            start_epoch = int(checkpoint.get("next_epoch", 0) or 0)
            start_epoch = int(np.clip(start_epoch, 0, max(0, epochs)))

            if start_epoch > 0 and not scheduler_loaded:
                # Keep LR schedule phase coherent even when optimizer state is not restored.
                scheduler.last_epoch = int(start_epoch - 1)
                try:
                    lrs = scheduler._get_closed_form_lr()
                    for pg, lr_val in zip(optimizer.param_groups, lrs):
                        pg["lr"] = float(lr_val)
                    scheduler._last_lr = [float(x) for x in lrs]
                except Exception:
                    pass

            if start_epoch > 0 and not optimizer_loaded:
                logger.info(
                    "Checkpoint resumed at epoch %d with fresh optimizer state.",
                    start_epoch,
                )

        # Optional pre-backward snapshot to capture latent state before any update.
        if capture and epochs > 0 and start_epoch == 0 and not self._embedding_snapshots:
            # Ce snapshot sert de "photo initiale" pour les figures d'évolution.
            emb_init = self._encode_full(X)
            pseudo_init = self._pseudo_labels(emb_init)
            init_comp = self._combined_cell_weights_components(
                embeddings=emb_init,
                pseudo_labels=pseudo_init,
            )
            current_cluster_component = np.asarray(
                init_comp["cluster_component"], dtype=np.float32
            )
            current_density_component = np.asarray(
                init_comp["density_component"], dtype=np.float32
            )
            current_fused_unclipped = np.asarray(
                init_comp["fused_weight_unclipped"], dtype=np.float32
            )
            current_weights = np.asarray(init_comp["fused_weight"], dtype=np.float32)
            current_pseudo = np.asarray(pseudo_init, dtype=np.int64)
            self._snapshot(
                epoch=0,
                phase="pretrain",
                embeddings=emb_init,
                cell_weights=current_weights,
                pseudo_labels=current_pseudo,
                cluster_component_weights=current_cluster_component,
                density_component_weights=current_density_component,
                fused_weight_unclipped=current_fused_unclipped,
                snapshot_type="pre_backward",
            )

        # --- Étape 6: boucle principale d'entraînement ---
        for epoch in range(start_epoch, epochs):
            # warm-up: uniform reconstruction; weighted phase: dynamic cell weights.
            weighted_phase = epoch >= warmup
            # Weight refresh policy:
            # - first weighted epoch,
            # - then every `update_interval` epochs.
            if weighted_phase and (
                epoch == warmup
                or (update_interval > 0 and ((epoch - warmup) % update_interval == 0))
            ):
                # Recompute pseudo-labels + global weights on full dataset.
                emb_for_weights = self._encode_full(X)
                pseudo_new = self._pseudo_labels(emb_for_weights)
                comp_new = self._combined_cell_weights_components(
                    embeddings=emb_for_weights,
                    pseudo_labels=pseudo_new,
                )
                weights_new = np.asarray(comp_new["fused_weight"], dtype=np.float32)
                cluster_new = np.asarray(comp_new["cluster_component"], dtype=np.float32)
                density_new = np.asarray(comp_new["density_component"], dtype=np.float32)
                fused_unclipped_new = np.asarray(
                    comp_new["fused_weight_unclipped"], dtype=np.float32
                )

                if epoch == warmup:
                    # First weighted epoch: direct assignment.
                    current_weights = weights_new
                    current_cluster_component = cluster_new
                    current_density_component = density_new
                    current_fused_unclipped = fused_unclipped_new
                else:
                    # Subsequent updates use EMA smoothing to avoid abrupt swings.
                    mixed = momentum * current_weights + (1.0 - momentum) * weights_new
                    mean_w = float(np.mean(mixed))
                    if np.isfinite(mean_w) and mean_w > 0.0:
                        mixed = mixed / mean_w
                    w_min = float(
                        self._param("min_cell_weight", DEFAULT_PARAM_OVERRIDES["min_cell_weight"])
                    )
                    w_max = float(
                        self._param("max_cell_weight", DEFAULT_PARAM_OVERRIDES["max_cell_weight"])
                    )
                    if w_max < w_min:
                        w_max = w_min
                    current_weights = np.clip(mixed, w_min, w_max).astype(np.float32)
                    # Apply same smoothing to each diagnostic component.
                    cluster_mixed = (
                        momentum * current_cluster_component + (1.0 - momentum) * cluster_new
                    )
                    density_mixed = (
                        momentum * current_density_component + (1.0 - momentum) * density_new
                    )
                    fused_unclipped_mixed = (
                        momentum * current_fused_unclipped + (1.0 - momentum) * fused_unclipped_new
                    )
                    cluster_mean = float(np.mean(cluster_mixed))
                    density_mean = float(np.mean(density_mixed))
                    fused_mean = float(np.mean(fused_unclipped_mixed))
                    if np.isfinite(cluster_mean) and cluster_mean > 0.0:
                        cluster_mixed = cluster_mixed / cluster_mean
                    if np.isfinite(density_mean) and density_mean > 0.0:
                        density_mixed = density_mixed / density_mean
                    if np.isfinite(fused_mean) and fused_mean > 0.0:
                        fused_unclipped_mixed = fused_unclipped_mixed / fused_mean
                    current_cluster_component = np.asarray(cluster_mixed, dtype=np.float32)
                    current_density_component = np.asarray(density_mixed, dtype=np.float32)
                    current_fused_unclipped = np.asarray(
                        fused_unclipped_mixed, dtype=np.float32
                    )
                current_pseudo = pseudo_new

            total_loss_sum = 0.0
            rec_sum = 0.0
            triplet_sum = 0.0
            adv_sum = 0.0
            n_batches_seen = 0

            model.train()
            if batch_head is not None:
                batch_head.train()

            # 7) Mini-batch optimization.
            for batch in loader:
                # Batch unpacking depends on NB (size factors present) vs MSE.
                if size_factors_np is not None:
                    xb, tb, sf_t, idx_t = batch
                    sf_t = sf_t.to(device)
                else:
                    xb, tb, idx_t = batch
                    sf_t = None
                xb = xb.to(device)
                tb = tb.to(device)
                # `idx` permet de récupérer les poids globaux alignés sur ces cellules.
                idx = idx_t.detach().cpu().numpy()

                # Input masking can be warm-up only or enabled in weighted phase.
                if mask_rate > 0.0 and (not weighted_phase or mask_in_weighted):
                    x_in, mask = self._apply_random_mask(
                        xb, mask_rate, masking_value=masking_value
                    )
                else:
                    x_in = xb
                    mask = None

                # AE forward pass: latent embedding + reconstruction head output.
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
                    # Weighted phase: per-cell contribution scaled by global dynamic weight.
                    w_t = torch.tensor(current_weights[idx], dtype=torch.float32, device=device)
                    reconstruction_loss = torch.mean(loss_per_sample * w_t)
                else:
                    # Warm-up: uniform contribution.
                    w_t = torch.ones(idx_t.shape[0], dtype=torch.float32, device=device)
                    reconstruction_loss = torch.mean(loss_per_sample)

                triplet_loss = torch.tensor(0.0, device=device)
                # Triplet active only after start epoch and only in weighted phase.
                if weighted_phase and triplet_weight > 0.0 and epoch >= triplet_start:
                    triplet_loss = self._rare_triplet_loss(
                        z=z,
                        pseudo_labels_batch=current_pseudo[idx],
                        weights_batch=w_t,
                    )

                adv_loss = torch.tensor(0.0, device=device)
                if batch_head is not None and batch_ids_np is not None and adv_weight > 0.0:
                    adv_start_epoch = int(
                        self._param(
                            "adversarial_start_epoch",
                            DEFAULT_PARAM_OVERRIDES["adversarial_start_epoch"],
                        )
                    )
                    # Adversarial branch can start later than reconstruction.
                    if epoch >= adv_start_epoch:
                        # Linear ramp stabilizes adversarial signal at startup.
                        ramp_epochs = int(
                            self._param(
                                "adversarial_ramp_epochs",
                                DEFAULT_PARAM_OVERRIDES["adversarial_ramp_epochs"],
                            )
                            or 0
                        )
                        if ramp_epochs > 0:
                            frac = min(1.0, (epoch - adv_start_epoch + 1) / float(ramp_epochs))
                        else:
                            frac = 1.0
                        lam = float(
                            self._param("adversarial_lambda", DEFAULT_PARAM_OVERRIDES["adversarial_lambda"])
                        ) * frac
                        logits = batch_head(gradient_reversal(z, lambda_=lam))
                        # Cible batch locale correspondant aux indices de mini-batch.
                        yb = torch.tensor(batch_ids_np[idx], dtype=torch.long, device=device)
                        adv_loss = nn.functional.cross_entropy(logits, yb)

                # Triplet ramp-up (bounded by 20 epochs) avoids abrupt latent distortion.
                rare_loss_ramp = 0.0
                if triplet_weight > 0.0 and epoch >= triplet_start:
                    ramp_epochs = max(1, min(20, epochs - triplet_start))
                    rare_loss_ramp = min(1.0, (epoch - triplet_start) / ramp_epochs)

                # Total loss = reconstruction + ramped triplet + adversarial batch.
                total_loss = reconstruction_loss + (rare_loss_ramp * triplet_weight * triplet_loss) + adv_weight * adv_loss

                optimizer.zero_grad()
                total_loss.backward()
                # Global grad clipping limits unstable updates.
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                optimizer.step()

                total_loss_sum += float(total_loss.detach().cpu().item())
                rec_sum += float(reconstruction_loss.detach().cpu().item())
                triplet_sum += float(triplet_loss.detach().cpu().item())
                adv_sum += float(adv_loss.detach().cpu().item())
                n_batches_seen += 1

            if n_batches_seen == 0:
                raise RuntimeError("No mini-batch processed during training.")

            # Epoch-level averages used by result export and plotting.
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

            # Scheduler stepped une fois par epoch.
            scheduler.step()

            if capture and _should_capture_epoch(epoch):
                # Periodic snapshot for post-hoc embedding evolution plots.
                emb_snap = self._encode_full(X)
                phase = "weighted" if weighted_phase else "warm-up"
                self._snapshot(
                    epoch=epoch,
                    phase=phase,
                    embeddings=emb_snap,
                    cell_weights=current_weights,
                    pseudo_labels=current_pseudo,
                    cluster_component_weights=current_cluster_component,
                    density_component_weights=current_density_component,
                    fused_weight_unclipped=current_fused_unclipped,
                    snapshot_type="periodic",
                )

            # Optional stop-and-save checkpoint for staged/long runs.
            if stop_after_epoch >= 0 and epoch >= stop_after_epoch:
                if save_checkpoint_path is not None:
                    self._save_training_checkpoint(
                        save_checkpoint_path,
                        next_epoch=epoch + 1,
                        model=model,
                        batch_head=batch_head,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        current_weights=current_weights,
                        current_cluster_component=current_cluster_component,
                        current_density_component=current_density_component,
                        current_fused_unclipped=current_fused_unclipped,
                        current_pseudo=current_pseudo,
                        warm_hist=warm_hist,
                        weighted_hist=weighted_hist,
                    )
                    logger.info(
                        "Saved training checkpoint at epoch %d -> %s",
                        epoch,
                        save_checkpoint_path,
                    )
                logger.info("Early stop requested at epoch %d", epoch)
                break

        # --- Étape 8: embedding final + clustering final ---
        self._embeddings = self._encode_full(X)
        self._labels = self._hdbscan_clustering(self._embeddings)
        self._fitted = True

        # --- Étape 9: historisation des losses ---
        history = []
        if warm_hist["epochs"]:
            history.append(warm_hist)
        if weighted_hist["epochs"]:
            history.append(weighted_hist)
        self._loss_history = history

        # --- Étape 10: traçabilité des paramètres effectifs ---
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
            "resume_checkpoint_path_effective": None
            if resume_checkpoint_path is None
            else str(resume_checkpoint_path),
            "start_epoch_effective": int(start_epoch),
            "stop_after_epoch_effective": int(stop_after_epoch),
        }
        self.set_effective_params(resolved)

        return self

    def get_batch_info(self) -> Tuple[Optional[str], int]:
        """Return `(batch_key_used, n_batches)` resolved during fit."""
        return self._batch_info

    def predict(self, data: Any = None) -> Any:
        """Return final cluster labels computed at the end of training."""
        if not self._fitted or self._labels is None:
            raise RuntimeError("Algorithm not fitted.")
        return self._labels
