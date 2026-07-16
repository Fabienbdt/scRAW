"""Reference-environment diagnostics for reproducible scRAW runs."""

from __future__ import annotations

from hashlib import sha256
from importlib import metadata
from pathlib import Path
import platform
from typing import Any


REFERENCE_PYTHON = "3.12.3"
REFERENCE_SYSTEM = "Linux"
REFERENCE_MACHINE = "x86_64"
REFERENCE_CUDA = "12.4"
REFERENCE_PACKAGES = {
    "numpy": "1.26.4",
    "scipy": "1.14.1",
    "pandas": "2.2.3",
    "scikit-learn": "1.5.2",
    "matplotlib": "3.9.2",
    "anndata": "0.10.9",
    "scanpy": "1.10.3",
    "h5py": "3.12.1",
    "umap-learn": "0.5.7",
    "pynndescent": "0.5.13",
    "numba": "0.60.0",
    "hdbscan": "0.8.40",
    "igraph": "0.11.8",
    "leidenalg": "0.10.2",
    "torch": "2.5.1+cu124",
}
REFERENCE_FILES = {
    "dataset": (
        "data/baron_human_pancreas.h5ad",
        "aa784472d90c2a6cdb99cdd076525b2dbe83a7254543092eaccca25bc60e359e",
    ),
    "default_config": (
        "configs/default_scraw.json",
        "f2a6550ffae5c14f800b5cbdfdae41a91be8f0ef671d825448867155792dd767",
    ),
    "requirements_lock": (
        "requirements.txt",
        "fcb0eb645938dce438406c2a1d6ff7f212d400cce274681c7b37de26baee9d1f",
    ),
}


def _sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_repository_root(explicit_root: str | Path | None) -> Path:
    """Locate the source checkout that owns reference data and configuration."""
    if explicit_root is not None:
        return Path(explicit_root).expanduser().resolve()

    candidates = [Path.cwd(), Path(__file__).resolve().parents[2]]
    candidates.extend(Path.cwd().parents)
    for candidate in candidates:
        if (candidate / "requirements.txt").is_file() and (
            candidate / "configs/default_scraw.json"
        ).is_file():
            return candidate.resolve()
    return Path.cwd().resolve()


def _check(expected: Any, actual: Any) -> dict[str, Any]:
    """Return one serializable equality check."""
    return {"expected": expected, "actual": actual, "ok": actual == expected}


def _installed_version(distribution: str) -> str | None:
    """Read one installed distribution version without importing the package."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _installed_cuda_version() -> tuple[str | None, str | None]:
    """Read PyTorch's compiled CUDA version, retaining import errors for the report."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on the host installation
        return None, f"{type(exc).__name__}: {exc}"
    return getattr(torch.version, "cuda", None), None


def inspect_reference_environment(
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compare the current host and checkout with the exact reference setup."""
    root = _find_repository_root(repository_root)
    cuda_version, torch_import_error = _installed_cuda_version()

    environment_checks = {
        "python": _check(REFERENCE_PYTHON, platform.python_version()),
        "system": _check(REFERENCE_SYSTEM, platform.system()),
        "machine": _check(REFERENCE_MACHINE, platform.machine()),
        "cuda": _check(REFERENCE_CUDA, cuda_version),
    }
    if torch_import_error is not None:
        environment_checks["cuda"]["error"] = torch_import_error

    package_checks = {
        name: _check(expected, _installed_version(name))
        for name, expected in REFERENCE_PACKAGES.items()
    }

    file_checks: dict[str, dict[str, Any]] = {}
    for name, (relative_path, expected_hash) in REFERENCE_FILES.items():
        path = root / relative_path
        actual_hash = _sha256_file(path) if path.is_file() else None
        file_checks[name] = {
            **_check(expected_hash, actual_hash),
            "path": str(path),
        }

    all_checks = [
        *environment_checks.values(),
        *package_checks.values(),
        *file_checks.values(),
    ]
    return {
        "reference_compatible": all(bool(check["ok"]) for check in all_checks),
        "repository_root": str(root),
        "environment": environment_checks,
        "packages": package_checks,
        "files": file_checks,
    }
