"""Central default configuration for standalone scRAW."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


DEFAULT_PRESET_NAME = "default"

DEFAULT_TRIAL_CONFIGURATION: Dict[str, Any] = {
    "trial_number": 206,
    "final_clustering_requested": "hdbscan",
    "dann_enabled": True,
    "param_overrides": {
        "hidden_layers": "512,256,128",
        "z_dim": 192,
        "dropout": 0.3,
        "epochs": 210,
        "warmup_epochs": 74,
        "lr": 0.00233670337683859,
        "batch_size": 192,
        "reconstruction_distribution": "mse",
        "nb_input_transform": "log1p",
        "nb_theta": 2.5815883941220323,
        "masking_rate": 0.15000000000000002,
        "masked_recon_weight": 0.8,
        "masking_apply_weighted": True,
        "weight_exponent": 0.7000000000000001,
        "cluster_density_alpha": 0.30000000000000004,
        "density_knn_k": 15,
        "density_weight_clip": 8.0,
        "dynamic_weight_momentum": 0.8500000000000001,
        "dynamic_weight_update_interval": 10,
        "weight_fusion_mode": "additive",
        "min_cell_weight": 0.45000000000000007,
        "max_cell_weight": 8.0,
        "rare_triplet_weight": 0.2346243650039478,
        "rare_triplet_start_epoch": 84,
        "rare_triplet_margin": 0.4,
        "rare_triplet_min_weight": 1.8,
        "max_triplet_anchors_per_batch": 64,
        "pseudo_label_method": "leiden",
        "hdbscan_min_cluster_size": 8,
        "hdbscan_min_samples": 8,
        "hdbscan_cluster_selection_method": "eom",
        "hdbscan_reassign_noise": True,
        "use_batch_conditioning": True,
        "adversarial_batch_weight": 0.056150696336115635,
        "adversarial_lambda": 1.75,
        "adversarial_start_epoch": 55,
        "adversarial_ramp_epochs": 60,
        "mmd_batch_weight": 0.0,
        "capture_embedding_snapshots": False,
        "batch_correction_key": "batch",
    },
}

DEFAULT_PARAM_OVERRIDES: Dict[str, Any] = deepcopy(DEFAULT_TRIAL_CONFIGURATION["param_overrides"])

DEFAULT_PREPROCESSING: Dict[str, Any] = {
    "n_top_genes": 2000,
    "min_genes_per_cell": 100,
    "max_genes_per_cell": 10000,
    "min_cells_per_gene": 3,
    "target_sum": 20000,
    "scale_max_value": 10.0,
    "hvg_flavor": "seurat",
    "hvg_strategy": "train_only",
    "dropout_method": "none",
    "noise_level": 0.0,
}

DEFAULT_ALGORITHM_STATIC_PARAMS: Dict[str, Any] = {
    "input_type": "processed",
    "masking_value": 0.0,
    "nb_mu_clip_max": 1e6,
    "clustering_method": str(DEFAULT_TRIAL_CONFIGURATION["final_clustering_requested"]),
    "density_weight_exponent": 1.0,
    "cluster_weight_power": 1.0,
    "density_weight_power": 1.0,
    "n_clusters": 0,
    "unsupervised_k_fallback": 0,
    "unsupervised_k_selection": "stability_consensus",
    "unsupervised_k_min": 8,
    "unsupervised_k_max": 30,
    "unsupervised_k_num_candidates": 12,
    "unsupervised_k_pca_dim": 32,
    "unsupervised_k_eval_sample_size": 3000,
    "unsupervised_k_stability_runs": 5,
    "unsupervised_k_stability_sample_size": 4000,
    "unsupervised_k_weight_stability": 0.45,
    "unsupervised_k_weight_silhouette": 0.25,
    "unsupervised_k_weight_ch": 0.20,
    "unsupervised_k_weight_db": 0.10,
    "unsupervised_k_weight_tiny_clusters": 0.20,
    "unsupervised_k_min_cluster_fraction": 0.005,
    "unsupervised_k_overseg_penalty": 0.25,
    "unsupervised_k_underseg_penalty": 0.05,
    "rare_loss_type": "triplet",
    "snapshot_interval_epochs": 10,
    "random_state": 42,
    "seed": 42,
}

DEFAULT_ALGORITHM_PARAMS: Dict[str, Any] = {
    **DEFAULT_ALGORITHM_STATIC_PARAMS,
    **DEFAULT_PARAM_OVERRIDES,
}


def copy_default_algorithm_params() -> Dict[str, Any]:
    """Return a deep copy of the default scRAW algorithm parameters."""
    return deepcopy(DEFAULT_ALGORITHM_PARAMS)


def copy_default_preprocessing() -> Dict[str, Any]:
    """Return a deep copy of the default scRAW preprocessing configuration."""
    return deepcopy(DEFAULT_PREPROCESSING)


def copy_default_trial_configuration() -> Dict[str, Any]:
    """Return a deep copy of the Optuna-derived default configuration manifest."""
    return deepcopy(DEFAULT_TRIAL_CONFIGURATION)
