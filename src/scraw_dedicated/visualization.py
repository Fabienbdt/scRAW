#!/usr/bin/env python3
"""Standalone figure generation for scRAW dedicated runs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _to_2d(embeddings: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Helper interne: to 2d.
    
    
    Args:
        embeddings: Paramètre d'entrée `embeddings` utilisé dans cette étape du pipeline.
        random_state: Paramètre d'entrée `random_state` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    X = np.asarray(embeddings)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if X.shape[1] == 2:
        return X.astype(np.float32)

    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=random_state)
        Z = reducer.fit_transform(X)
        return np.asarray(Z, dtype=np.float32)
    except Exception:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=random_state)
        Z = pca.fit_transform(X)
        return np.asarray(Z, dtype=np.float32)


def _map_categories(labels: Sequence[Any]) -> Tuple[np.ndarray, Dict[str, int]]:
    """Helper interne: map categories.
    
    
    Args:
        labels: Paramètre d'entrée `labels` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    labels_str = np.asarray([str(x) for x in labels], dtype=object)
    uniq = sorted(np.unique(labels_str).tolist())
    mapping = {v: i for i, v in enumerate(uniq)}
    idx = np.asarray([mapping[v] for v in labels_str], dtype=int)
    return idx, mapping


def _scatter_cat(ax: Any, coords: np.ndarray, labels: Sequence[Any], title: str, point_size: float = 4.0) -> None:
    """Helper interne: scatter cat.
    
    
    Args:
        ax: Paramètre d'entrée `ax` utilisé dans cette étape du pipeline.
        coords: Paramètre d'entrée `coords` utilisé dans cette étape du pipeline.
        labels: Paramètre d'entrée `labels` utilisé dans cette étape du pipeline.
        title: Paramètre d'entrée `title` utilisé dans cette étape du pipeline.
        point_size: Paramètre d'entrée `point_size` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    idx, mapping = _map_categories(labels)
    cmap = plt.get_cmap("tab20", max(len(mapping), 1))
    ax.scatter(coords[:, 0], coords[:, 1], c=idx, s=point_size, cmap=cmap, linewidths=0, alpha=0.9)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_umap_comparison(
    embeddings: np.ndarray,
    true_labels: Sequence[Any],
    predicted_labels: Sequence[Any],
    title: str,
) -> plt.Figure:
    """Trace un UMAP final comparant Ground Truth et prédictions.
    
    
    Args:
        embeddings: Paramètre d'entrée `embeddings` utilisé dans cette étape du pipeline.
        true_labels: Paramètre d'entrée `true_labels` utilisé dans cette étape du pipeline.
        predicted_labels: Paramètre d'entrée `predicted_labels` utilisé dans cette étape du pipeline.
        title: Paramètre d'entrée `title` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    coords = _to_2d(embeddings)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _scatter_cat(axes[0], coords, true_labels, "Ground Truth")
    _scatter_cat(axes[1], coords, predicted_labels, "Prediction")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_umap_batch(
    embeddings: np.ndarray,
    batch_labels: Sequence[Any],
    title: str,
) -> plt.Figure:
    """Trace un UMAP final coloré par batch d'origine.
    
    
    Args:
        embeddings: Paramètre d'entrée `embeddings` utilisé dans cette étape du pipeline.
        batch_labels: Paramètre d'entrée `batch_labels` utilisé dans cette étape du pipeline.
        title: Paramètre d'entrée `title` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    coords = _to_2d(embeddings)
    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter_cat(ax, coords, batch_labels, title)
    fig.tight_layout()
    return fig


def _nonlinear_alpha(weights: np.ndarray, power: float = 4.0) -> np.ndarray:
    """Helper interne: nonlinear alpha.
    
    
    Args:
        weights: Paramètre d'entrée `weights` utilisé dans cette étape du pipeline.
        power: Paramètre d'entrée `power` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    w = np.asarray(weights, dtype=np.float32)
    if w.size == 0:
        return w
    w_min = float(np.nanmin(w))
    w_max = float(np.nanmax(w))
    if not np.isfinite(w_min) or not np.isfinite(w_max) or w_max <= w_min:
        return np.full_like(w, 0.9, dtype=np.float32)
    wn = (w - w_min) / (w_max - w_min)
    return (0.02 + 0.98 * np.power(np.clip(wn, 0.0, 1.0), power)).astype(np.float32)


def plot_umap_weighted(
    embeddings: np.ndarray,
    labels: Sequence[Any],
    cell_weights: Sequence[float],
    title: str,
) -> plt.Figure:
    """Trace un UMAP où l'opacité varie selon le poids de reconstruction.
    
    
    Args:
        embeddings: Paramètre d'entrée `embeddings` utilisé dans cette étape du pipeline.
        labels: Paramètre d'entrée `labels` utilisé dans cette étape du pipeline.
        cell_weights: Paramètre d'entrée `cell_weights` utilisé dans cette étape du pipeline.
        title: Paramètre d'entrée `title` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    coords = _to_2d(embeddings)
    idx, mapping = _map_categories(labels)
    cmap = plt.get_cmap("tab20", max(len(mapping), 1))
    base_colors = cmap(idx)
    alpha = _nonlinear_alpha(np.asarray(cell_weights, dtype=np.float32), power=4.0)

    colors = base_colors.copy()
    colors[:, 3] = alpha

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=4.0, linewidths=0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_umap_weighted_gradient(
    embeddings: np.ndarray,
    cell_weights: Sequence[float],
    title: str,
) -> plt.Figure:
    """Trace un UMAP avec dégradé de couleur selon le poids de reconstruction.
    
    
    Args:
        embeddings: Paramètre d'entrée `embeddings` utilisé dans cette étape du pipeline.
        cell_weights: Paramètre d'entrée `cell_weights` utilisé dans cette étape du pipeline.
        title: Paramètre d'entrée `title` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    coords = _to_2d(embeddings)
    w = np.asarray(cell_weights, dtype=np.float32)
    if w.size:
        w = np.clip(w, np.nanpercentile(w, 1), np.nanpercentile(w, 99))

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=w, cmap="plasma", s=5.0, linewidths=0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Reconstruction weight")
    fig.tight_layout()
    return fig


def _select_periodic_snapshots_every_10(snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Helper interne: select periodic snapshots every 10.
    
    
    Args:
        snapshots: Paramètre d'entrée `snapshots` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    periodic = [
        s for s in snapshots
        if s.get("embeddings") is not None and len(s.get("embeddings")) > 0 and s.get("snapshot_type", "periodic") == "periodic"
    ]
    if not periodic:
        periodic = [s for s in snapshots if s.get("embeddings") is not None and len(s.get("embeddings")) > 0]
    if not periodic:
        return []

    selected = []
    for s in periodic:
        e = s.get("epoch")
        if isinstance(e, (int, np.integer)) and int(e) % 10 == 0:
            selected.append(s)

    if periodic[0] not in selected:
        selected.insert(0, periodic[0])
    if periodic[-1] not in selected:
        selected.append(periodic[-1])

    dedup = []
    seen = set()
    for s in selected:
        sid = id(s)
        if sid in seen:
            continue
        seen.add(sid)
        dedup.append(s)
    return dedup


def plot_umap_evolution(
    snapshots: List[Dict[str, Any]],
    labels: Sequence[Any],
    title: str,
    max_panels: int = 12,
) -> Optional[plt.Figure]:
    """Trace l'évolution de l'espace latent à plusieurs epochs.
    
    
    Args:
        snapshots: Paramètre d'entrée `snapshots` utilisé dans cette étape du pipeline.
        labels: Paramètre d'entrée `labels` utilisé dans cette étape du pipeline.
        title: Paramètre d'entrée `title` utilisé dans cette étape du pipeline.
        max_panels: Paramètre d'entrée `max_panels` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    selected = _select_periodic_snapshots_every_10(snapshots)
    if not selected:
        return None

    selected = selected[:max_panels]
    n = len(selected)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows))
    axes_arr = np.asarray(axes).reshape(-1)

    for ax in axes_arr[n:]:
        ax.axis("off")

    # Fit one shared 2D projector on the last snapshot for faster, comparable panels.
    projector = None
    last_emb = np.asarray(selected[-1].get("embeddings"))
    try:
        import umap  # type: ignore

        projector = umap.UMAP(n_components=2, random_state=42)
        projector.fit(last_emb)
    except Exception:
        try:
            from sklearn.decomposition import PCA

            projector = PCA(n_components=2, random_state=42)
            projector.fit(last_emb)
        except Exception:
            projector = None

    for i, snap in enumerate(selected):
        emb = snap.get("embeddings")
        if emb is None or len(emb) == 0:
            axes_arr[i].axis("off")
            continue
        if len(emb) != len(labels):
            axes_arr[i].axis("off")
            continue
        emb_arr = np.asarray(emb)
        if projector is not None and hasattr(projector, "transform"):
            try:
                coords = projector.transform(emb_arr)
                coords = np.asarray(coords, dtype=np.float32)
            except Exception:
                coords = _to_2d(emb_arr, random_state=42)
        else:
            coords = _to_2d(emb_arr, random_state=42)
        _scatter_cat(axes_arr[i], coords, labels, f"Epoch {snap.get('epoch', '?')}", point_size=2.5)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_loss_curves(loss_history: List[Dict[str, Any]], title: str) -> Optional[plt.Figure]:
    """Trace les courbes de loss par phase d'entraînement.
    
    
    Args:
        loss_history: Paramètre d'entrée `loss_history` utilisé dans cette étape du pipeline.
        title: Paramètre d'entrée `title` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    if not loss_history:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for phase in loss_history:
        name = str(phase.get("name", "phase"))
        epochs = phase.get("epochs", [])
        train = phase.get("train_loss", [])
        if epochs and train and len(epochs) == len(train):
            ax.plot(epochs, train, label=f"{name} train")
        val = phase.get("val_loss", [])
        if epochs and val and len(epochs) == len(val):
            ax.plot(epochs, val, linestyle="--", label=f"{name} val")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_marker_overlap_heatmap(true_labels: Sequence[Any], pred_labels: Sequence[Any], title: str) -> plt.Figure:
    """Trace la heatmap de recouvrement labels réels vs clusters prédits.
    
    
    Args:
        true_labels: Paramètre d'entrée `true_labels` utilisé dans cette étape du pipeline.
        pred_labels: Paramètre d'entrée `pred_labels` utilisé dans cette étape du pipeline.
        title: Paramètre d'entrée `title` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray([str(x) for x in true_labels], dtype=object)
    y_pred = np.asarray([str(x) for x in pred_labels], dtype=object)

    classes_true = sorted(np.unique(y_true).tolist())
    classes_pred = sorted(np.unique(y_pred).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=classes_true)
    row_sum = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(max(8, len(classes_pred) * 0.4), max(6, len(classes_true) * 0.35)))
    im = ax.imshow(cmn, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Predicted cluster")
    ax.set_ylabel("True label")
    ax.set_yticks(np.arange(len(classes_true)))
    ax.set_yticklabels(classes_true, fontsize=7)
    # Keep x ticks sparse for readability.
    if len(classes_pred) <= 40:
        ax.set_xticks(np.arange(len(classes_pred)))
        ax.set_xticklabels(classes_pred, rotation=90, fontsize=6)
    else:
        ax.set_xticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized overlap")
    fig.tight_layout()
    return fig
