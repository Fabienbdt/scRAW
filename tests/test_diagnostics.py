"""Tests for reproducibility diagnostics that do not require reference hardware."""

from __future__ import annotations

from hashlib import sha256

from scraw import diagnostics


def test_reference_hashes_match_the_bundled_files() -> None:
    repository_root = diagnostics._find_repository_root(None)

    for relative_path, expected_hash in diagnostics.REFERENCE_FILES.values():
        assert diagnostics._sha256_file(repository_root / relative_path) == expected_hash


def test_reference_file_checks_detect_a_mismatch(monkeypatch, tmp_path) -> None:
    config_dir = tmp_path / "configs"
    data_dir = tmp_path / "data"
    config_dir.mkdir()
    data_dir.mkdir()
    (tmp_path / "requirements.txt").write_text("reference", encoding="utf-8")
    (config_dir / "default_scraw.json").write_text("config", encoding="utf-8")
    (data_dir / "input.h5ad").write_bytes(b"changed")

    expected_config_hash = sha256(b"config").hexdigest()
    monkeypatch.setattr(
        diagnostics,
        "REFERENCE_FILES",
        {
            "default_config": ("configs/default_scraw.json", expected_config_hash),
            "dataset": ("data/input.h5ad", "not-the-current-hash"),
        },
    )
    monkeypatch.setattr(
        diagnostics,
        "_installed_cuda_version",
        lambda: (diagnostics.REFERENCE_CUDA, None),
    )
    monkeypatch.setattr(
        diagnostics,
        "_installed_version",
        lambda name: diagnostics.REFERENCE_PACKAGES[name],
    )
    monkeypatch.setattr(
        diagnostics.platform,
        "python_version",
        lambda: diagnostics.REFERENCE_PYTHON,
    )
    monkeypatch.setattr(
        diagnostics.platform,
        "system",
        lambda: diagnostics.REFERENCE_SYSTEM,
    )
    monkeypatch.setattr(
        diagnostics.platform,
        "machine",
        lambda: diagnostics.REFERENCE_MACHINE,
    )

    report = diagnostics.inspect_reference_environment(tmp_path)

    assert report["files"]["default_config"]["ok"] is True
    assert report["files"]["dataset"]["ok"] is False
    assert report["reference_compatible"] is False
