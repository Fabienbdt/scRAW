"""Smoke tests for installed package metadata and entry points."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
import sys

import pytest
import scraw


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_distribution_and_import_versions_match() -> None:
    try:
        installed_version = version("scraw")
    except PackageNotFoundError:
        pytest.skip("distribution metadata is available after package installation")
    assert installed_version == scraw.__version__


def test_module_entry_point_has_lightweight_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "scraw", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "{run,infer,show-config,doctor}" in completed.stdout


def test_reference_and_portable_requirements_are_distinct() -> None:
    reference = (REPOSITORY_ROOT / "requirements.txt").read_text(encoding="utf-8")
    portable = (REPOSITORY_ROOT / "requirements-portable.txt").read_text(
        encoding="utf-8"
    )
    cuda_alias = (REPOSITORY_ROOT / "requirements-cuda124.txt").read_text(
        encoding="utf-8"
    )

    assert "--extra-index-url" not in portable
    assert "+cu124" not in portable
    assert "torch==2.5.1+cu124" in reference
    assert "https://download.pytorch.org/whl/cu124" in reference
    assert "-r requirements.txt" in cuda_alias
