"""Regression tests for scientific configuration and runtime safety."""

from __future__ import annotations

import json

import anndata as ad
import hdbscan
import numpy as np
import pytest
import torch

from scraw.clustering import final_clustering
from scraw.config import ScRAWConfig
from scraw.model import resolve_device
from scraw.pipeline import (
    _detect_batch_key,
    _detect_label_key,
    run_inference_from_checkpoint,
    run_pipeline,
)
from scraw.preprocessing import preprocess_adata
from scraw.trainer import ScRAWTrainer


def _tiny_config(tmp_path, data_path) -> ScRAWConfig:
    config = ScRAWConfig()
    config.data.data_path = str(data_path)
    config.data.output_dir = str(tmp_path / "output")
    config.data.label_key = None
    config.runtime.device = "cpu"
    config.preprocessing.input_mode = "raw"
    config.preprocessing.min_genes_per_cell = 0
    config.preprocessing.min_cells_per_gene = 0
    config.preprocessing.n_top_genes = 0
    config.model.hidden_layers = [8]
    config.model.latent_dim = 4
    config.model.dropout = 0.0
    config.training.epochs = 1
    config.training.warmup_epochs = 1
    config.training.batch_size = 8
    config.training.masking_rate = 0.0
    config.clustering.pseudo_label_method = "kmeans"
    config.clustering.pseudo_k = 3
    config.clustering.pseudo_k_min = 2
    config.clustering.pseudo_k_max = 4
    config.clustering.hdbscan_min_cluster_size = 2
    config.clustering.hdbscan_min_samples = 1
    config.triplet.enabled = False
    config.batch_correction.enabled = False
    config.batch_correction.key = None
    config.outputs.save_figures = False
    config.outputs.save_model = True
    return config


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"trainng": {}}, "Unknown configuration section"),
        ({"training": {"epoch": 2}}, "Unknown field"),
        ({"weighting": {"weight_fusion_mode": "multiply"}}, "weight_fusion_mode"),
        ({"clustering": {"pseudo_label_method": "keans"}}, "pseudo_label_method"),
        (
            {"clustering": {"hdbscan_cluster_selection_method": "invalid"}},
            "cluster_selection_method",
        ),
    ],
)
def test_config_rejects_typos(payload, message) -> None:
    with pytest.raises(ValueError, match=message):
        ScRAWConfig.from_dict(payload)


def test_config_rejects_unimplemented_mmd_objective() -> None:
    with pytest.raises(NotImplementedError, match="mmd_weight is not implemented"):
        ScRAWConfig.from_dict({"batch_correction": {"mmd_weight": 0.1}})


def test_device_parser_rejects_typos() -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device("cdua")


def test_device_parser_accepts_an_indexed_cuda_device(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert resolve_device("CUDA:1") == torch.device("cuda:1")


def test_explicit_missing_observation_keys_fail() -> None:
    adata = ad.AnnData(np.ones((3, 2), dtype=np.float32))
    adata.obs["label"] = ["a", "a", "b"]
    adata.obs["batch"] = ["x", "x", "y"]

    with pytest.raises(KeyError, match="Configured label column 'celltype'"):
        _detect_label_key(adata, "celltype")
    with pytest.raises(KeyError, match="Configured batch column 'sample'"):
        _detect_batch_key(adata, "sample")

    assert _detect_label_key(adata, None) == "label"
    assert _detect_batch_key(adata, None) == "batch"


def test_preprocessing_mode_prevents_ambiguous_double_processing(monkeypatch) -> None:
    adata = ad.AnnData(np.asarray([[1.0, 2.0], [2.0, 3.0], [4.0, 8.0]], dtype=np.float32))
    config = ScRAWConfig().preprocessing
    config.input_mode = "preprocessed"
    config.min_genes_per_cell = 0
    config.min_cells_per_gene = 0

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("raw-count preprocessing was called")

    import scanpy as sc

    monkeypatch.setattr(sc.pp, "normalize_total", unexpected_call)
    monkeypatch.setattr(sc.pp, "log1p", unexpected_call)
    monkeypatch.setattr(sc.pp, "highly_variable_genes", unexpected_call)

    processed = preprocess_adata(adata, config)

    assert processed.shape == adata.shape
    assert np.all(np.isfinite(processed.X))


def test_raw_mode_and_trainer_reject_invalid_expression_values() -> None:
    raw_config = ScRAWConfig().preprocessing
    raw_config.input_mode = "raw"
    raw_config.min_genes_per_cell = 0
    raw_config.min_cells_per_gene = 0

    with pytest.raises(ValueError, match="incompatible with negative"):
        preprocess_adata(ad.AnnData(np.asarray([[1.0, -1.0], [2.0, 3.0]])), raw_config)

    trainer = ScRAWTrainer(ScRAWConfig.from_dict({"runtime": {"device": "cpu"}}))
    with pytest.raises(ValueError, match="NaN or infinite"):
        trainer._validate_input_matrix(np.asarray([[1.0, np.nan], [2.0, 3.0]]))
    with pytest.raises(ValueError, match="At least two cells"):
        trainer._validate_input_matrix(np.ones((1, 3), dtype=np.float32))


def test_loader_avoids_a_final_singleton_batch() -> None:
    config = ScRAWConfig.from_dict(
        {"runtime": {"device": "cpu"}, "training": {"batch_size": 2}}
    )
    trainer = ScRAWTrainer(config)

    batch_sizes = [
        len(batch[0])
        for batch in trainer._build_loader(np.ones((5, 3), dtype=np.float32))
    ]
    second_sizes = [
        len(batch[0])
        for batch in trainer._build_loader(np.ones((7, 3), dtype=np.float32))
    ]

    assert batch_sizes == [3, 2]
    assert second_sizes == [4, 3]


def test_reference_protocol_uses_known_class_count_for_automatic_k() -> None:
    config = ScRAWConfig.from_dict(
        {
            "runtime": {"device": "cpu"},
            "clustering": {"pseudo_k": 0},
        }
    )
    trainer = ScRAWTrainer(config)
    labels = np.asarray(["alpha", "alpha", "beta", "gamma"])

    resolved = trainer._resolve_clustering_config(labels)

    assert resolved.pseudo_k == 3


def test_reference_hdbscan_noise_handling_is_preserved(monkeypatch) -> None:
    class DummyHDBSCAN:
        def __init__(self, **_kwargs):
            pass

        def fit_predict(self, _embeddings):
            return np.asarray([-1, 0, 0, 1, 1], dtype=np.int64)

    monkeypatch.setattr(hdbscan, "HDBSCAN", DummyHDBSCAN)
    config = ScRAWConfig().clustering
    config.hdbscan_reassign_noise = False

    labels = final_clustering(
        np.arange(10, dtype=np.float32).reshape(5, 2),
        config=config,
        runtime=ScRAWConfig().runtime,
    )

    assert labels.tolist() == [2, 0, 0, 1, 1]


def test_weighted_triplet_and_batch_adversarial_training_are_finite() -> None:
    config = ScRAWConfig()
    config.runtime.device = "cpu"
    config.model.hidden_layers = [8]
    config.model.latent_dim = 4
    config.model.dropout = 0.0
    config.training.epochs = 3
    config.training.warmup_epochs = 1
    config.training.batch_size = 6
    config.training.masking_rate = 0.0
    config.clustering.pseudo_label_method = "kmeans"
    config.clustering.pseudo_k = 2
    config.clustering.pseudo_k_min = 2
    config.clustering.pseudo_k_max = 3
    config.clustering.hdbscan_min_cluster_size = 2
    config.clustering.hdbscan_min_samples = 1
    config.triplet.enabled = True
    config.triplet.start_epoch = 1
    config.triplet.min_anchor_weight = 0.0
    config.triplet.margin = 10.0
    config.batch_correction.enabled = True
    config.batch_correction.start_epoch = 0
    config.batch_correction.ramp_epochs = 0
    rng = np.random.default_rng(17)
    matrix = rng.normal(size=(12, 6)).astype(np.float32)
    true_labels = np.asarray(["a"] * 6 + ["b"] * 6)
    batch_ids = np.asarray(["x", "y"] * 6)

    result = ScRAWTrainer(config).fit(matrix, labels=true_labels, batch_ids=batch_ids)

    assert len(result.loss_history) == 3
    assert all(np.isfinite(row["total_loss"]) for row in result.loss_history)
    assert max(row["batch_adv_loss"] for row in result.loss_history) > 0.0
    assert max(row["triplet_loss"] for row in result.loss_history) > 0.0


def test_tiny_cpu_pipeline_writes_strict_json_and_data_order(tmp_path) -> None:
    rng = np.random.default_rng(4)
    adata = ad.AnnData(rng.poisson(2.0, size=(24, 10)).astype(np.float32))
    adata.obs_names = [f"cell-{index}" for index in range(adata.n_obs)]
    adata.var_names = [f"gene-{index}" for index in range(adata.n_vars)]
    data_path = tmp_path / "tiny.h5ad"
    adata.write_h5ad(data_path)
    config = _tiny_config(tmp_path, data_path)

    result = run_pipeline(config)

    results_dir = tmp_path / "output" / "results"
    payload = json.loads((results_dir / "results.json").read_text(encoding="utf-8"))
    assert payload["metrics"]["NMI"] is None
    assert payload["effective_pseudo_k"] == 3
    assert payload["provenance"]["data_sha256"]
    assert payload["provenance"]["resolved_device"] == "cpu"
    assert np.load(results_dir / "cell_ids.npy").tolist() == adata.obs_names.tolist()
    assert np.load(results_dir / "feature_ids.npy").tolist() == adata.var_names.tolist()
    assert result["embeddings"].shape == (adata.n_obs, config.model.latent_dim)

    inference = run_inference_from_checkpoint(
        config=config,
        checkpoint_path=tmp_path / "output" / "models" / "autoencoder.pt",
        output_dir=tmp_path / "inference-output",
        device="cpu",
    )
    inference_results = tmp_path / "inference-output" / "results"
    inference_payload = json.loads(
        (inference_results / "results.json").read_text(encoding="utf-8")
    )
    assert inference_payload["mode"] == "inference_only"
    assert inference_payload["provenance"]["checkpoint_sha256"]
    assert np.load(inference_results / "cell_ids.npy").tolist() == adata.obs_names.tolist()
    np.testing.assert_array_equal(inference["embeddings"], result["embeddings"])
    np.testing.assert_array_equal(inference["labels"], result["labels"])
