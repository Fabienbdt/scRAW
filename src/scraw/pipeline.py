"""End-to-end execution for the scRAW pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from ._version import __version__
from .clustering import estimate_pseudo_k, final_clustering
from .config import ScRAWConfig, load_config
from .metrics import compute_metrics
from .model import MLPAutoencoder, encode_in_batches
from .plots import (
    plot_embedding_categories,
    plot_embedding_weights,
    plot_loss_history,
    save_figure,
)
from .preprocessing import preprocess_adata
from .trainer import ScRAWTrainer, TrainingResult


logger = logging.getLogger(__name__)


def _as_jsonable(value: Any) -> Any:
    """Convert numpy and path values into JSON-safe Python objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.ndarray):
        return _as_jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_jsonable(v) for v in value]
    return value


def _detect_label_key(adata: Any, configured_key: Optional[str]) -> Optional[str]:
    """Resolve the biological label column used for evaluation/plots."""
    explicit_key = None if configured_key is None else str(configured_key).strip()
    if explicit_key:
        if explicit_key not in adata.obs.columns:
            raise KeyError(
                f"Configured label column '{explicit_key}' was not found in adata.obs."
            )
        return explicit_key

    for candidate in [
        "Group",
        "label",
        "cell_type",
        "celltype",
        "CellType",
        "cell_types",
        "cluster",
        "labels",
    ]:
        if candidate in adata.obs.columns:
            return candidate
    return None


def _detect_batch_key(adata: Any, preferred: Optional[str]) -> Optional[str]:
    """Resolve the batch column used by the adversarial branch."""
    explicit_key = None if preferred is None else str(preferred).strip()
    if explicit_key:
        if explicit_key not in adata.obs.columns:
            raise KeyError(
                f"Configured batch column '{explicit_key}' was not found in adata.obs."
            )
        return explicit_key

    for candidate in [
        "batch",
        "Batch",
        "study",
        "dataset",
        "donor",
        "sample",
        "patient",
        "tech",
    ]:
        if candidate in adata.obs.columns:
            return candidate
    return None


def _obs_values(adata: Any, key: Optional[str], role: str) -> Optional[np.ndarray]:
    """Extract one complete observation column without hiding missing values."""
    if key is None:
        return None
    values = adata.obs[key]
    if bool(values.isna().any()):
        raise ValueError(f"The {role} column '{key}' contains missing values.")
    return np.asarray(values.astype(str).to_numpy(), dtype=object)


def _prepare_output_dirs(output_dir: Path) -> Dict[str, Path]:
    """Create the output directory tree used by the pipeline."""
    paths = {
        "root": output_dir,
        "config": output_dir / "config",
        "results": output_dir / "results",
        "figures": output_dir / "figures",
        "models": output_dir / "models",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _save_metrics_csv(metrics: Dict[str, Any], path: Path) -> None:
    """Save scalar metrics to a one-row CSV file."""
    flat_metrics = {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, (dict, list))
    }
    pd.DataFrame([flat_metrics]).to_csv(path, index=False)


def _save_arrays(result: TrainingResult, output_dir: Path) -> None:
    """Persist the main numpy outputs for later inspection."""
    np.save(output_dir / "embeddings.npy", np.asarray(result.embeddings, dtype=np.float32))
    np.save(output_dir / "final_labels.npy", np.asarray(result.labels, dtype=np.int64))
    np.save(output_dir / "pseudo_labels.npy", np.asarray(result.pseudo_labels, dtype=np.int64))
    np.save(output_dir / "cell_weights.npy", np.asarray(result.cell_weights, dtype=np.float32))


def _save_data_order(adata: Any, output_dir: Path) -> None:
    """Save the exact cell and feature order corresponding to exported arrays."""
    np.save(output_dir / "cell_ids.npy", np.asarray(adata.obs_names, dtype=str))
    np.save(output_dir / "feature_ids.npy", np.asarray(adata.var_names, dtype=str))


def _sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_version(distribution: str) -> Optional[str]:
    """Return an installed distribution version when metadata is available."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _build_provenance(
    data_path: Path,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Collect non-computational metadata needed to identify one run."""
    provenance: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "scraw_version": _installed_version("scraw") or __version__,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "scanpy_version": _installed_version("scanpy"),
        "data_path": str(data_path),
        "data_sha256": _sha256_file(data_path),
    }
    if checkpoint_path is not None:
        provenance["checkpoint_path"] = str(checkpoint_path)
        provenance["checkpoint_sha256"] = _sha256_file(checkpoint_path)
    return provenance


def _effective_pseudo_k(
    config: ScRAWConfig,
    n_cells: int,
    true_labels: Optional[np.ndarray],
) -> tuple[int, Optional[int]]:
    """Report the K selected by the reference known-class-count protocol."""
    known_label_count = (
        None if true_labels is None else int(len(np.unique(np.asarray(true_labels))))
    )
    if int(config.clustering.pseudo_k) > 1:
        target_k = int(config.clustering.pseudo_k)
    elif known_label_count is not None:
        target_k = known_label_count
    else:
        target_k = estimate_pseudo_k(n_cells, config.clustering)
    return int(target_k), known_label_count


def _save_inference_arrays(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
) -> None:
    """Persist inference-only arrays for checkpoint replay diagnostics."""
    np.save(output_dir / "embeddings.npy", np.asarray(embeddings, dtype=np.float32))
    np.save(output_dir / "final_labels.npy", np.asarray(labels, dtype=np.int64))


def _save_figures(
    result: TrainingResult,
    true_labels: Optional[np.ndarray],
    output_dir: Path,
    seed: int,
) -> None:
    """Generate a small default figure set."""
    save_figure(
        plot_loss_history(result.loss_history),
        output_dir / "loss_history.png",
    )
    save_figure(
        plot_embedding_categories(
            result.embeddings,
            result.labels,
            title="scRAW latent space colored by final clusters",
            random_state=seed,
        ),
        output_dir / "latent_clusters.png",
    )
    save_figure(
        plot_embedding_weights(
            result.embeddings,
            result.cell_weights,
            title="scRAW latent space colored by cell weights",
            random_state=seed,
        ),
        output_dir / "latent_weights.png",
    )
    if true_labels is not None:
        save_figure(
            plot_embedding_categories(
                result.embeddings,
                true_labels,
                title="scRAW latent space colored by ground-truth labels",
                random_state=seed,
            ),
            output_dir / "latent_ground_truth.png",
        )


def _save_inference_figures(
    embeddings: np.ndarray,
    labels: np.ndarray,
    true_labels: Optional[np.ndarray],
    output_dir: Path,
    seed: int,
) -> None:
    """Generate the figure subset that remains meaningful in inference-only mode."""
    save_figure(
        plot_embedding_categories(
            embeddings,
            labels,
            title="scRAW latent space colored by final clusters",
            random_state=seed,
        ),
        output_dir / "latent_clusters.png",
    )
    if true_labels is not None:
        save_figure(
            plot_embedding_categories(
                embeddings,
                true_labels,
                title="scRAW latent space colored by ground-truth labels",
                random_state=seed,
            ),
            output_dir / "latent_ground_truth.png",
        )


def _load_checkpoint_model(
    checkpoint_path: str | Path,
    input_dim: int,
    config: ScRAWConfig,
    device: torch.device,
) -> MLPAutoencoder:
    """Rebuild one autoencoder and load a saved state dict onto the target device."""
    model = MLPAutoencoder(input_dim=input_dim, config=config.model).to(device)
    state_dict = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_pipeline(config: ScRAWConfig | str | Path | None = None) -> Dict[str, Any]:
    """Run scRAW from the built-in stable default or from a config override."""
    if config is None:
        config = ScRAWConfig()
    elif not isinstance(config, ScRAWConfig):
        config = load_config(config)

    config.validate()
    output_dir = Path(config.data.output_dir).expanduser().resolve()
    output_paths = _prepare_output_dirs(output_dir)
    resolved_data_path = Path(config.data.data_path).expanduser().resolve()

    import scanpy as sc

    adata = sc.read_h5ad(resolved_data_path)
    adata_proc = preprocess_adata(adata, config.preprocessing)
    label_key = _detect_label_key(adata_proc, config.data.label_key)
    true_labels = _obs_values(adata_proc, label_key, role="label")
    batch_key = _detect_batch_key(
        adata_proc,
        preferred=str(config.batch_correction.key or "").strip() or None,
    )
    batch_ids = _obs_values(adata_proc, batch_key, role="batch")
    X_proc = np.asarray(adata_proc.X, dtype=np.float32)

    trainer = ScRAWTrainer(config)
    result = trainer.fit(X_proc, labels=true_labels, batch_ids=batch_ids)
    metrics = compute_metrics(
        labels_true=true_labels,
        labels_pred=result.labels,
        embeddings=result.embeddings,
    )

    config_used = config.to_dict()
    effective_pseudo_k, known_label_count = _effective_pseudo_k(
        config,
        n_cells=int(adata_proc.n_obs),
        true_labels=true_labels,
    )
    provenance = _build_provenance(resolved_data_path)
    provenance["resolved_device"] = result.device
    summary = {
        "label_key": label_key,
        "batch_key": batch_key,
        "known_label_count": known_label_count,
        "effective_pseudo_k": effective_pseudo_k,
        "n_cells": int(adata_proc.n_obs),
        "n_genes": int(adata_proc.n_vars),
        "device": result.device,
        "provenance": provenance,
        "metrics": metrics,
        "loss_history": result.loss_history,
    }

    (output_paths["config"] / "config_used.json").write_text(
        json.dumps(_as_jsonable(config_used), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_paths["results"] / "results.json").write_text(
        json.dumps(_as_jsonable(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _save_metrics_csv(metrics, output_paths["results"] / "analysis_results.csv")
    _save_arrays(result, output_paths["results"])
    _save_data_order(adata_proc, output_paths["results"])

    if bool(config.outputs.save_model):
        torch.save(result.model.state_dict(), output_paths["models"] / "autoencoder.pt")

    if bool(config.outputs.save_figures):
        _save_figures(
            result=result,
            true_labels=true_labels,
            output_dir=output_paths["figures"],
            seed=int(config.runtime.seed),
        )

    return {
        "config": config_used,
        "label_key": label_key,
        "batch_key": batch_key,
        "known_label_count": known_label_count,
        "effective_pseudo_k": effective_pseudo_k,
        "provenance": provenance,
        "metrics": metrics,
        "embeddings": result.embeddings,
        "labels": result.labels,
        "pseudo_labels": result.pseudo_labels,
        "cell_weights": result.cell_weights,
        "loss_history": result.loss_history,
        "output_dir": str(output_dir),
    }


def run_inference_from_checkpoint(
    config: ScRAWConfig | str | Path,
    checkpoint_path: str | Path,
    output_dir: Optional[str | Path] = None,
    data_path: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Replay preprocessing, encoding, clustering, and metrics from saved weights only."""
    if isinstance(config, ScRAWConfig):
        config = ScRAWConfig.from_dict(config.to_dict())
    else:
        config = load_config(config)

    if output_dir is not None:
        config.data.output_dir = str(output_dir)
    if data_path is not None:
        config.data.data_path = str(data_path)
    if device is not None:
        config.runtime.device = str(device)

    config.validate()
    resolved_output_dir = Path(config.data.output_dir).expanduser().resolve()
    output_paths = _prepare_output_dirs(resolved_output_dir)
    resolved_checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    resolved_data_path = Path(config.data.data_path).expanduser().resolve()

    import scanpy as sc

    adata = sc.read_h5ad(resolved_data_path)
    adata_proc = preprocess_adata(adata, config.preprocessing)
    label_key = _detect_label_key(adata_proc, config.data.label_key)
    true_labels = _obs_values(adata_proc, label_key, role="label")
    batch_key = _detect_batch_key(
        adata_proc,
        preferred=str(config.batch_correction.key or "").strip() or None,
    )
    X_proc = np.asarray(adata_proc.X, dtype=np.float32)

    trainer = ScRAWTrainer(config)
    trainer._set_random_seeds()
    model = _load_checkpoint_model(
        checkpoint_path=resolved_checkpoint_path,
        input_dim=int(X_proc.shape[1]),
        config=config,
        device=trainer.device,
    )
    embeddings = encode_in_batches(
        model,
        X_proc,
        device=trainer.device,
        batch_size=int(config.training.batch_size),
    )
    final_labels = final_clustering(
        embeddings,
        config=config.clustering,
        runtime=config.runtime,
    )
    metrics = compute_metrics(
        labels_true=true_labels,
        labels_pred=final_labels,
        embeddings=embeddings,
    )

    config_used = config.to_dict()
    effective_pseudo_k, known_label_count = _effective_pseudo_k(
        config,
        n_cells=int(adata_proc.n_obs),
        true_labels=true_labels,
    )
    provenance = _build_provenance(
        resolved_data_path,
        checkpoint_path=resolved_checkpoint_path,
    )
    provenance["resolved_device"] = str(trainer.device)
    summary = {
        "mode": "inference_only",
        "checkpoint_path": str(resolved_checkpoint_path),
        "label_key": label_key,
        "batch_key": batch_key,
        "known_label_count": known_label_count,
        "effective_pseudo_k": effective_pseudo_k,
        "n_cells": int(adata_proc.n_obs),
        "n_genes": int(adata_proc.n_vars),
        "device": str(trainer.device),
        "provenance": provenance,
        "metrics": metrics,
        "loss_history": [],
    }

    (output_paths["config"] / "config_used.json").write_text(
        json.dumps(_as_jsonable(config_used), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_paths["results"] / "results.json").write_text(
        json.dumps(_as_jsonable(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _save_metrics_csv(metrics, output_paths["results"] / "analysis_results.csv")
    _save_inference_arrays(embeddings, final_labels, output_paths["results"])
    _save_data_order(adata_proc, output_paths["results"])

    if bool(config.outputs.save_figures):
        _save_inference_figures(
            embeddings=embeddings,
            labels=final_labels,
            true_labels=true_labels,
            output_dir=output_paths["figures"],
            seed=int(config.runtime.seed),
        )

    return {
        "config": config_used,
        "checkpoint_path": str(resolved_checkpoint_path),
        "label_key": label_key,
        "batch_key": batch_key,
        "known_label_count": known_label_count,
        "effective_pseudo_k": effective_pseudo_k,
        "provenance": provenance,
        "metrics": metrics,
        "embeddings": embeddings,
        "labels": final_labels,
        "output_dir": str(resolved_output_dir),
        "mode": "inference_only",
    }
