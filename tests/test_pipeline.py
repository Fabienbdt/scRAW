from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.skip(
    reason="anndata/torch aborts under pytest in this environment due an OpenMP shared-memory issue",
)


def test_pipeline_runs_without_batch_column(base_config, toy_adata_no_batch, tmp_path) -> None:
    from scraw.pipeline import run_pipeline

    config = base_config
    config.batch_correction.enabled = True
    config.batch_correction.adversarial_weight = 0.2

    data_path = tmp_path / "toy_no_batch.h5ad"
    output_dir = tmp_path / "out_no_batch"
    toy_adata_no_batch.write_h5ad(data_path)

    config.data.data_path = str(data_path)
    config.data.output_dir = str(output_dir)
    result = run_pipeline(config)

    assert result["batch_key"] is None
    assert result["config"]["runtime"]["device"] == "cpu"
    assert result["labels"].shape == (toy_adata_no_batch.n_obs,)
    assert Path(result["output_dir"]).exists()


def test_pipeline_detects_batch_column(base_config, toy_adata, tmp_path) -> None:
    from scraw.pipeline import run_pipeline

    config = base_config
    config.batch_correction.enabled = True
    config.batch_correction.adversarial_weight = 0.2
    config.batch_correction.start_epoch = 0
    config.batch_correction.ramp_epochs = 1

    data_path = tmp_path / "toy_with_batch.h5ad"
    output_dir = tmp_path / "out_with_batch"
    toy_adata.write_h5ad(data_path)

    config.data.data_path = str(data_path)
    config.data.output_dir = str(output_dir)
    result = run_pipeline(config)

    assert result["batch_key"] == "batch"
    assert result["config"]["runtime"]["device"] == "cpu"
    assert result["labels"].shape == (toy_adata.n_obs,)
    assert (output_dir / "results" / "results.json").exists()
