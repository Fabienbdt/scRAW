"""Public API for the scRAW package."""

from .config import ScRAWConfig, load_config, save_config


def run_pipeline(*args, **kwargs):
    """Import the execution pipeline only when needed."""
    from .pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)

__all__ = [
    "ScRAWConfig",
    "load_config",
    "save_config",
    "run_pipeline",
]
