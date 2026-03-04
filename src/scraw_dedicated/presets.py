#!/usr/bin/env python3
"""Strict scRAW presets derived from reference best runs.

Only parameters used in the target Baron/Pancreas reference runs are exposed.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ScrawPreset:
    name: str
    description: str
    preprocessing: Dict[str, Any]
    algorithm_params: Dict[str, Any]
    supports_dann: bool


_COMMON_PREPROCESSING: Dict[str, Any] = {
    "n_top_genes": 2000,
    "min_cells_per_gene": 3,
    "target_sum": 20000,
    "scale_max_value": 10.0,
    "hvg_flavor": "seurat",
    "hvg_strategy": "train_only",
    "dropout_method": "none",
    "noise_level": 0.0,
}

_COMMON_ALGO: Dict[str, Any] = {
    "input_type": "processed",
    "hidden_layers": "512,256,128",
    "z_dim": 128,
    "masking_rate": 0.2,
    "masked_recon_weight": 0.75,
    "masking_value": 0.0,
    "masking_apply_weighted": False,
    "reconstruction_distribution": "nb",
    "nb_theta": 10,
    "nb_mu_clip_max": 1e6,
    "weight_exponent": 0.4,
    "weight_fusion_mode": "additive",
    "cluster_weight_power": 1.0,
    "density_weight_power": 1.0,
    "cluster_density_alpha": 0.6,
    "density_weight_clip": 5.0,
    "clustering_method": "hdbscan",
    "hdbscan_min_cluster_size": 4,
    "hdbscan_min_samples": 2,
    "hdbscan_cluster_selection_method": "eom",
    "hdbscan_reassign_noise": True,
    "pseudo_label_method": "leiden",
    "density_knn_k": 15,
    "density_weight_exponent": 1.0,
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
    "random_state": 42,
    "seed": 42,
    "rare_loss_type": "triplet",
    "rare_triplet_margin": 0.4,
    "rare_triplet_min_weight": 1.2,
    "max_triplet_anchors_per_batch": 64,
    "min_cell_weight": 0.25,
}


def _merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Fusionne deux dictionnaires de configuration sans modifier l'original.
    
    
    Args:
        base: Paramètre d'entrée `base` utilisé dans cette étape du pipeline.
        update: Paramètre d'entrée `update` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    out = deepcopy(base)
    out.update(update)
    return out


PRESETS: Dict[str, ScrawPreset] = {
    "baron_best": ScrawPreset(
        name="baron_best",
        description=(
            "Best Baron run (trip10_d10_s35) with HDBSCAN + Leiden pseudo-labels + "
            "NB(log1p) and no DANN."
        ),
        preprocessing=_merge(
            _COMMON_PREPROCESSING,
            {
                "min_genes_per_cell": 200,
            },
        ),
        algorithm_params=_merge(
            _COMMON_ALGO,
            {
                "epochs": 120,
                "warmup_epochs": 30,
                "nb_input_transform": "log1p",
                "rare_triplet_weight": 0.1,
                "rare_triplet_start_epoch": 35,
                "capture_embedding_snapshots": False,
                "use_batch_conditioning": False,
                "batch_correction_key": "auto",
                "adversarial_batch_weight": 0.0,
                "adversarial_start_epoch": 0,
                "adversarial_ramp_epochs": 0,
                "mmd_batch_weight": 0.0,
            },
        ),
        supports_dann=False,
    ),
    "pancreas_best": ScrawPreset(
        name="pancreas_best",
        description=(
            "Best Pancreas baseline with DANN on study batches, HDBSCAN + Leiden "
            "pseudo-labels and NB(Pearson residuals)."
        ),
        preprocessing=_merge(
            _COMMON_PREPROCESSING,
            {
                "min_genes_per_cell": 100,
                "max_genes_per_cell": 10000,
            },
        ),
        algorithm_params=_merge(
            _COMMON_ALGO,
            {
                "epochs": 80,
                "warmup_epochs": 20,
                "nb_input_transform": "pearson_residuals",
                "pearson_residual_clip": 10.0,
                "rare_triplet_weight": 0.05,
                "rare_triplet_start_epoch": 30,
                "weight_exponent": 0.2,
                "capture_embedding_snapshots": True,
                "use_batch_conditioning": True,
                "batch_correction_key": "study",
                "adversarial_batch_weight": 0.1,
                "adversarial_lambda": 1.0,
                "adversarial_start_epoch": 10,
                "adversarial_ramp_epochs": 0,
                "mmd_batch_weight": 0.0,
            },
        ),
        supports_dann=True,
    ),
}


def get_preset(name: str) -> ScrawPreset:
    """Retourne un preset valide à partir de son nom et lève une erreur sinon.
    
    
    Args:
        name: Paramètre d'entrée `name` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    key = name.strip().lower()
    if key not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[key]
