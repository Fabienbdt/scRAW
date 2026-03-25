"""Public API for the scRAW package."""

from .config import ScRAWConfig, load_config, save_config


def run_pipeline(*args, **kwargs):
    """Import the execution pipeline only when needed."""
    from .pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)


def run_inference_from_checkpoint(*args, **kwargs):
    """Import the checkpoint replay path only when needed."""
    from .pipeline import run_inference_from_checkpoint as _run_inference_from_checkpoint

    return _run_inference_from_checkpoint(*args, **kwargs)

__all__ = [
    "ScRAWConfig",
    "load_config",
    "save_config",
    "run_pipeline",
    "run_inference_from_checkpoint",
]
