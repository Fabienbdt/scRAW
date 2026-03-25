from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scraw.config import ScRAWConfig


def _make_toy_adata(include_batch: bool):
    from anndata import AnnData

    rng = np.random.default_rng(0)
    n_clusters = 3
    cells_per_cluster = 6
    genes_per_cluster = 6
    n_genes = n_clusters * genes_per_cluster

    blocks = []
    labels: list[str] = []
    batches: list[str] = []
    for cluster_id in range(n_clusters):
        lam = np.full((cells_per_cluster, n_genes), 0.2, dtype=np.float32)
        gene_start = cluster_id * genes_per_cluster
        gene_end = gene_start + genes_per_cluster
        lam[:, gene_start:gene_end] = 6.0
        blocks.append(rng.poisson(lam).astype(np.float32))
        labels.extend([f"type_{cluster_id}"] * cells_per_cluster)
        batches.extend([f"batch_{cell_idx % 2}" for cell_idx in range(cells_per_cluster)])

    counts = np.vstack(blocks)
    obs_dict = {"cell_type": labels}
    if include_batch:
        obs_dict["batch"] = batches

    n_cells = counts.shape[0]
    obs = pd.DataFrame(obs_dict, index=[f"cell_{idx}" for idx in range(n_cells)])
    var = pd.DataFrame(index=[f"gene_{idx}" for idx in range(n_genes)])
    return AnnData(X=sparse.csr_matrix(counts), obs=obs, var=var)


@pytest.fixture
def toy_adata():
    return _make_toy_adata(include_batch=True)


@pytest.fixture
def toy_adata_no_batch():
    return _make_toy_adata(include_batch=False)


@pytest.fixture
def base_config() -> ScRAWConfig:
    config = ScRAWConfig()
    config.runtime.device = "cpu"
    config.runtime.strict_repro = False
    config.training.epochs = 2
    config.training.warmup_epochs = 1
    config.training.batch_size = 6
    config.training.learning_rate = 1e-3
    config.training.masking_rate = 0.0
    config.triplet.enabled = False
    config.outputs.save_figures = False
    config.outputs.save_model = False
    config.batch_correction.enabled = False
    config.preprocessing.min_genes_per_cell = 0
    config.preprocessing.min_cells_per_gene = 0
    config.preprocessing.n_top_genes = 12
    config.preprocessing.target_sum = 1e4
    config.clustering.pseudo_label_method = "kmeans"
    config.clustering.pseudo_k = 3
    config.clustering.pseudo_k_min = 2
    config.clustering.pseudo_k_max = 4
    config.clustering.hdbscan_min_cluster_size = 2
    config.clustering.hdbscan_min_samples = 1
    return config
