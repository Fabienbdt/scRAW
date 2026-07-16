"""Focused tests for the lightweight scRAW command-line interface."""

from __future__ import annotations

import json

from scraw import cli


def test_run_applies_explicit_overrides(monkeypatch, capsys, tmp_path) -> None:
    observed = {}

    def fake_run_pipeline(config):
        observed["config"] = config
        return {
            "output_dir": config.data.output_dir,
            "label_key": config.data.label_key,
            "batch_key": None,
            "known_label_count": 14,
            "effective_pseudo_k": 14,
            "metrics": {"ARI": 0.75},
        }

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    status = cli.main(
        [
            "run",
            "--data-path",
            "input.h5ad",
            "--output-dir",
            str(tmp_path / "run-output"),
            "--label-key",
            "cell_type",
            "--batch-key",
            "sample",
            "--input-mode",
            "raw",
            "--device",
            "cpu",
            "--seed",
            "7",
            "--no-figures",
            "--no-model",
        ]
    )

    assert status == 0
    config = observed["config"]
    assert config.data.data_path == "input.h5ad"
    assert config.data.output_dir == str(tmp_path / "run-output")
    assert config.data.label_key == "cell_type"
    assert config.batch_correction.key == "sample"
    assert config.preprocessing.input_mode == "raw"
    assert config.runtime.device == "cpu"
    assert config.runtime.seed == 7
    assert config.outputs.save_figures is False
    assert config.outputs.save_model is False
    assert json.loads(capsys.readouterr().out) == {
        "batch_key": None,
        "effective_pseudo_k": 14,
        "known_label_count": 14,
        "label_key": "cell_type",
        "metrics": {"ARI": 0.75},
        "mode": "training",
        "output_dir": str(tmp_path / "run-output"),
    }


def test_infer_uses_public_checkpoint_api(monkeypatch, capsys, tmp_path) -> None:
    observed = {}

    def fake_inference(**kwargs):
        observed.update(kwargs)
        return {
            "checkpoint_path": kwargs["checkpoint_path"],
            "output_dir": kwargs["output_dir"],
            "label_key": "label",
            "batch_key": "batch",
            "metrics": {},
        }

    monkeypatch.setattr(cli, "run_inference_from_checkpoint", fake_inference)

    status = cli.main(
        [
            "infer",
            "--config",
            "config.json",
            "--checkpoint",
            "autoencoder.pt",
            "--output-dir",
            str(tmp_path / "inference-output"),
            "--data-path",
            "new.h5ad",
            "--device",
            "cpu",
        ]
    )

    assert status == 0
    assert observed == {
        "config": "config.json",
        "checkpoint_path": "autoencoder.pt",
        "output_dir": str(tmp_path / "inference-output"),
        "data_path": "new.h5ad",
        "device": "cpu",
    }
    assert json.loads(capsys.readouterr().out)["mode"] == "inference_only"


def test_show_config_is_valid_json(capsys) -> None:
    status = cli.main(["show-config"])

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) >= {"data", "runtime", "training", "outputs"}


def test_reference_doctor_controls_strict_exit_status(monkeypatch, capsys) -> None:
    report = {"reference_compatible": False, "packages": {}}
    monkeypatch.setattr(cli, "inspect_reference_environment", lambda _root: report)

    assert cli.main(["doctor"]) == 0
    assert json.loads(capsys.readouterr().out) == report
    assert cli.main(["doctor", "--reference"]) == 1
    assert json.loads(capsys.readouterr().out) == report


def test_cli_json_replaces_nonfinite_metrics_with_null(capsys) -> None:
    cli._print_json(
        {
            "finite": 0.5,
            "nan": float("nan"),
            "positive_infinity": float("inf"),
            "negative_infinity": float("-inf"),
        }
    )

    assert json.loads(capsys.readouterr().out) == {
        "finite": 0.5,
        "nan": None,
        "positive_infinity": None,
        "negative_infinity": None,
    }


def test_expected_run_error_has_nonzero_exit(monkeypatch, capsys) -> None:
    def fail(_config):
        raise ValueError("invalid scientific option")

    monkeypatch.setattr(cli, "run_pipeline", fail)

    assert cli.main(["run"]) == 1
    assert "scraw: error: invalid scientific option" in capsys.readouterr().err


def test_nonempty_output_requires_explicit_overwrite(monkeypatch, capsys, tmp_path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("user data", encoding="utf-8")
    called = False

    def fake_run_pipeline(_config):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    status = cli.main(["run", "--output-dir", str(output_dir)])

    assert status == 1
    assert called is False
    assert "pass --overwrite" in capsys.readouterr().err
    assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_overwrite_authorizes_run_without_deleting_directory(monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    existing_file = output_dir / "keep.txt"
    existing_file.write_text("user data", encoding="utf-8")

    def fake_run_pipeline(config):
        return {"output_dir": config.data.output_dir, "metrics": {}}

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    status = cli.main(
        ["run", "--output-dir", str(output_dir), "--overwrite"]
    )

    assert status == 0
    assert existing_file.read_text(encoding="utf-8") == "user data"
