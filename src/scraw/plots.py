"""Plotting utilities for the scRAW pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np


def _compute_2d_projection(embeddings: np.ndarray, random_state: int) -> np.ndarray:
    """Project latent embeddings to 2D with UMAP and a PCA fallback."""
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if emb.shape[1] == 1:
        return np.column_stack([emb[:, 0], np.zeros(emb.shape[0], dtype=np.float32)])
    if emb.shape[1] == 2:
        return emb.astype(np.float32, copy=False)

    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, max(2, emb.shape[0] - 1)),
            min_dist=0.3,
            random_state=int(random_state),
        )
        return reducer.fit_transform(emb).astype(np.float32, copy=False)
    except Exception:
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=int(random_state)).fit_transform(emb).astype(
            np.float32,
            copy=False,
        )


def _encode_categories(labels: Iterable[object]) -> tuple[np.ndarray, list[str]]:
    """Convert arbitrary labels to contiguous integers and readable names."""
    labels = np.asarray(list(labels), dtype=object)
    unique_labels = sorted(np.unique(labels).tolist())
    mapping = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = np.asarray([mapping[label] for label in labels], dtype=np.int64)
    return encoded, [str(label) for label in unique_labels]


def plot_embedding_categories(
    embeddings: np.ndarray,
    labels: Iterable[object],
    title: str,
    random_state: int = 42,
) -> Optional[plt.Figure]:
    """Plot a 2D projection colored by categorical labels."""
    emb_2d = _compute_2d_projection(embeddings, random_state=random_state)
    if emb_2d.shape[0] == 0:
        return None

    encoded, names = _encode_categories(labels)
    cmap = plt.get_cmap("tab20", max(1, len(names)))
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=encoded,
        cmap=cmap,
        s=8,
        alpha=0.85,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title)
    ax.set_xlabel("Projection 1")
    ax.set_ylabel("Projection 2")

    if len(names) <= 20:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=label,
                markerfacecolor=cmap(index),
                markersize=7,
            )
            for index, label in enumerate(names)
        ]
        ax.legend(handles=handles, title="Labels", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    else:
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Encoded label")

    plt.tight_layout()
    return fig


def plot_embedding_weights(
    embeddings: np.ndarray,
    weights: np.ndarray,
    title: str,
    random_state: int = 42,
) -> Optional[plt.Figure]:
    """Plot a 2D projection colored by continuous cell weights."""
    emb_2d = _compute_2d_projection(embeddings, random_state=random_state)
    if emb_2d.shape[0] == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=np.asarray(weights, dtype=np.float32),
        cmap="plasma",
        s=8,
        alpha=0.9,
        linewidths=0,
        rasterized=True,
    )
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Cell weight")
    ax.set_title(title)
    ax.set_xlabel("Projection 1")
    ax.set_ylabel("Projection 2")
    plt.tight_layout()
    return fig


def plot_loss_history(loss_history: list[dict[str, object]]) -> Optional[plt.Figure]:
    """Plot total, reconstruction, and triplet losses over training epochs."""
    if not loss_history:
        return None

    epochs = [int(row["epoch"]) for row in loss_history]
    total = [float(row["total_loss"]) for row in loss_history]
    reconstruction = [float(row["reconstruction_loss"]) for row in loss_history]
    triplet = [float(row["triplet_loss"]) for row in loss_history]
    phases = [str(row["phase"]) for row in loss_history]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(epochs, total, label="Total loss", linewidth=2.0)
    ax.plot(epochs, reconstruction, label="Reconstruction loss", linewidth=1.5)
    ax.plot(epochs, triplet, label="Triplet loss", linewidth=1.5)

    for epoch, phase in zip(epochs, phases):
        if phase == "weighted":
            ax.axvline(epoch, color="#dddddd", linewidth=0.3)
            break

    ax.set_title("Training losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    return fig


def save_figure(fig: Optional[plt.Figure], path: str | Path) -> None:
    """Persist one figure to disk and close it."""
    if fig is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
