from __future__ import annotations

import logging
import os

import numpy as np
import pytest
import torch


def _make_trainer_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n_clusters = 3
    cells_per_cluster = 6
    genes_per_cluster = 6
    n_genes = n_clusters * genes_per_cluster

    blocks = []
    labels: list[str] = []
    batches: list[str] = []
    for cluster_id in range(n_clusters):
        mean = np.full((cells_per_cluster, n_genes), 0.2, dtype=np.float32)
        gene_start = cluster_id * genes_per_cluster
        gene_end = gene_start + genes_per_cluster
        mean[:, gene_start:gene_end] = 4.0
        blocks.append(rng.normal(loc=mean, scale=0.1).astype(np.float32))
        labels.extend([f"type_{cluster_id}"] * cells_per_cluster)
        batches.extend([f"batch_{cell_idx % 2}" for cell_idx in range(cells_per_cluster)])

    X = np.clip(np.vstack(blocks), 0.0, None).astype(np.float32)
    return X, np.asarray(labels, dtype=object), np.asarray(batches, dtype=object)


def test_trainer_disables_batch_branch_without_batch_ids(base_config, caplog) -> None:
    from scraw.trainer import ScRAWTrainer

    config = base_config
    config.batch_correction.enabled = True
    config.batch_correction.adversarial_weight = 0.2

    X, _, _ = _make_trainer_data()
    with caplog.at_level(logging.WARNING):
        result = ScRAWTrainer(config).fit(X)

    assert "no batch ids were provided" in caplog.text
    assert result.device == "cpu"
    assert result.labels.shape == (X.shape[0],)
    assert len(result.loss_history) == config.training.epochs


def test_trainer_rejects_invalid_batch_length(base_config) -> None:
    from scraw.trainer import ScRAWTrainer

    config = base_config
    config.batch_correction.enabled = True
    config.batch_correction.adversarial_weight = 0.2

    X, _, _ = _make_trainer_data()
    bad_batch_ids = np.asarray(["batch_0"] * (X.shape[0] - 1), dtype=object)

    with pytest.raises(ValueError, match="Batch id length does not match"):
        ScRAWTrainer(config).fit(X, batch_ids=bad_batch_ids)


def test_trainer_refreshes_weights_after_warmup(base_config) -> None:
    from scraw.trainer import ScRAWTrainer

    config = base_config
    config.weighting.dynamic_weight_update_interval = 1

    X, labels, _ = _make_trainer_data()
    result = ScRAWTrainer(config).fit(X, labels=labels)

    assert [row["phase"] for row in result.loss_history] == ["warmup", "weighted"]
    assert result.pseudo_labels.shape == (X.shape[0],)
    assert not np.all(result.pseudo_labels == 0)
    assert result.cell_weights.shape == (X.shape[0],)


def test_trainer_runs_with_batch_ids(base_config) -> None:
    from scraw.trainer import ScRAWTrainer

    config = base_config
    config.batch_correction.enabled = True
    config.batch_correction.adversarial_weight = 0.2
    config.batch_correction.start_epoch = 0
    config.batch_correction.ramp_epochs = 1

    X, _, batch_ids = _make_trainer_data()
    result = ScRAWTrainer(config).fit(X, batch_ids=batch_ids)

    assert len(result.loss_history) == config.training.epochs
    assert result.labels.shape == (X.shape[0],)


def test_trainer_strict_repro_configures_runtime_environment(base_config, monkeypatch) -> None:
    from scraw.trainer import ScRAWTrainer

    config = base_config
    config.runtime.strict_repro = True

    workspace_key = "CUBLAS_WORKSPACE_CONFIG"
    monkeypatch.delenv(workspace_key, raising=False)

    deterministic_calls: list[bool] = []
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda enabled: deterministic_calls.append(bool(enabled)),
    )

    old_benchmark = torch.backends.cudnn.benchmark
    old_deterministic = torch.backends.cudnn.deterministic
    try:
        trainer = ScRAWTrainer(config)
        assert trainer.device.type == "cpu"
        assert os.environ[workspace_key] == ":4096:8"
        assert deterministic_calls == [True]
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True
    finally:
        torch.backends.cudnn.benchmark = old_benchmark
        torch.backends.cudnn.deterministic = old_deterministic


def test_trainer_strict_repro_seeds_python_numpy_torch_and_cuda(base_config, monkeypatch) -> None:
    import scraw.trainer as trainer_module

    config = base_config
    config.runtime.strict_repro = True

    old_benchmark = torch.backends.cudnn.benchmark
    old_deterministic = torch.backends.cudnn.deterministic
    old_use_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        trainer = trainer_module.ScRAWTrainer(config)
        seed_calls: dict[str, list[int]] = {
            "python": [],
            "numpy": [],
            "torch": [],
            "cuda": [],
        }
        monkeypatch.setattr(
            trainer_module.random,
            "seed",
            lambda seed: seed_calls["python"].append(seed),
        )
        monkeypatch.setattr(
            trainer_module.np.random,
            "seed",
            lambda seed: seed_calls["numpy"].append(seed),
        )
        monkeypatch.setattr(
            trainer_module.torch,
            "manual_seed",
            lambda seed: seed_calls["torch"].append(seed),
        )
        monkeypatch.setattr(
            trainer_module.torch.cuda,
            "manual_seed_all",
            lambda seed: seed_calls["cuda"].append(seed),
        )

        trainer._set_random_seeds()

        assert seed_calls == {
            "python": [42],
            "numpy": [42],
            "torch": [42],
            "cuda": [42],
        }
    finally:
        torch.use_deterministic_algorithms(old_use_deterministic)
        torch.backends.cudnn.benchmark = old_benchmark
        torch.backends.cudnn.deterministic = old_deterministic
