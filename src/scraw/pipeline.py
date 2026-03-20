"""End-to-end execution for the scRAW pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from .config import ScRAWConfig, load_config
from .metrics import compute_metrics
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
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_jsonable(v) for v in value]
    return value


def _detect_label_key(adata: Any, configured_key: Optional[str]) -> Optional[str]:
    """Resolve the biological label column used for evaluation/plots."""
    if configured_key and configured_key in adata.obs.columns:
        return configured_key

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
    if preferred and preferred in adata.obs.columns:
        return preferred

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


def run_pipeline(config: ScRAWConfig | str | Path) -> Dict[str, Any]:
    """Run the default scRAW pipeline from a config object or JSON file."""
    if not isinstance(config, ScRAWConfig):
        config = load_config(config)

    output_dir = Path(config.data.output_dir).expanduser().resolve()
    output_paths = _prepare_output_dirs(output_dir)

    import scanpy as sc

    adata = sc.read_h5ad(Path(config.data.data_path).expanduser().resolve())
    adata_proc = preprocess_adata(adata, config.preprocessing)
    label_key = _detect_label_key(adata_proc, config.data.label_key)
    true_labels = (
        None
        if label_key is None
        else np.asarray(adata_proc.obs[label_key].astype(str).to_numpy(), dtype=object)
    )
    batch_key = _detect_batch_key(
        adata_proc,
        preferred=str(config.batch_correction.key or "").strip() or None,
    )
    batch_ids = (
        None
        if batch_key is None
        else np.asarray(adata_proc.obs[batch_key].astype(str).to_numpy(), dtype=object)
    )
    X_proc = np.asarray(adata_proc.X, dtype=np.float32)

    trainer = ScRAWTrainer(config)
    result = trainer.fit(X_proc, labels=true_labels, batch_ids=batch_ids)
    metrics = compute_metrics(
        labels_true=true_labels,
        labels_pred=result.labels,
        embeddings=result.embeddings,
    )

    config_used = config.to_dict()
    summary = {
        "label_key": label_key,
        "batch_key": batch_key,
        "n_cells": int(adata_proc.n_obs),
        "n_genes": int(adata_proc.n_vars),
        "device": result.device,
        "metrics": metrics,
        "loss_history": result.loss_history,
    }

    (output_paths["config"] / "config_used.json").write_text(
        json.dumps(_as_jsonable(config_used), indent=2),
        encoding="utf-8",
    )
    (output_paths["results"] / "results.json").write_text(
        json.dumps(_as_jsonable(summary), indent=2),
        encoding="utf-8",
    )
    _save_metrics_csv(metrics, output_paths["results"] / "analysis_results.csv")
    _save_arrays(result, output_paths["results"])

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
        "metrics": metrics,
        "embeddings": result.embeddings,
        "labels": result.labels,
        "pseudo_labels": result.pseudo_labels,
        "cell_weights": result.cell_weights,
        "loss_history": result.loss_history,
        "output_dir": str(output_dir),
    }
