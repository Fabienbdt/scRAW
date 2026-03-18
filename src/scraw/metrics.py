"""Clustering metrics used by the scRAW pipeline."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


NOISE_LABELS = {-1, "-1", "noise", "Noise", "NOISE", "unassigned", "Unassigned"}


def _to_array(values: Any) -> np.ndarray:
    """Convert any array-like object into a NumPy array."""
    return np.asarray(values)


def _filter_noise(
    labels_true: Optional[np.ndarray],
    labels_pred: np.ndarray,
    embeddings: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """Remove entries marked as noise from labels and embeddings."""
    labels_pred = _to_array(labels_pred)
    mask_pred = np.ones(len(labels_pred), dtype=bool)
    labels_pred_str = labels_pred.astype(str)
    for value in NOISE_LABELS:
        mask_pred &= labels_pred_str != str(value)

    if labels_true is None:
        filtered_embeddings = embeddings[mask_pred] if embeddings is not None else None
        return None, labels_pred[mask_pred], filtered_embeddings

    labels_true = _to_array(labels_true)
    mask_true = np.ones(len(labels_true), dtype=bool)
    labels_true_str = labels_true.astype(str)
    for value in NOISE_LABELS:
        mask_true &= labels_true_str != str(value)

    mask = mask_pred & mask_true
    filtered_embeddings = embeddings[mask] if embeddings is not None else None
    return labels_true[mask], labels_pred[mask], filtered_embeddings


def align_labels(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    """Align predicted clusters to true labels with Hungarian matching."""
    labels_true = _to_array(labels_true)
    labels_pred = _to_array(labels_pred)
    if len(labels_true) != len(labels_pred):
        raise ValueError("labels_true and labels_pred must have the same length.")

    unique_true = np.unique(labels_true)
    unique_pred = np.unique(labels_pred)
    true_to_index = {label: index for index, label in enumerate(unique_true)}
    pred_to_index = {label: index for index, label in enumerate(unique_pred)}

    weights = np.zeros((len(unique_pred), len(unique_true)), dtype=np.int64)
    for label_true, label_pred in zip(labels_true, labels_pred):
        weights[pred_to_index[label_pred], true_to_index[label_true]] += 1

    row_ind, col_ind = linear_sum_assignment(weights.max() - weights)
    mapping = {unique_pred[row]: unique_true[col] for row, col in zip(row_ind, col_ind)}
    return np.asarray([mapping.get(label, f"Unmatched_{label}") for label in labels_pred], dtype=object)


def _accuracy(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute clustering accuracy after Hungarian alignment."""
    aligned = align_labels(labels_true, labels_pred)
    return float(np.mean(aligned == labels_true))


def _balanced_metrics(labels_true: np.ndarray, labels_pred: np.ndarray) -> Dict[str, float]:
    """Compute macro F1 and balanced accuracy after alignment."""
    from sklearn.metrics import balanced_accuracy_score, f1_score

    aligned = align_labels(labels_true, labels_pred)
    return {
        "F1_Macro": float(f1_score(labels_true, aligned, average="macro", zero_division=0)),
        "BalancedACC": float(balanced_accuracy_score(labels_true, aligned)),
    }


def _rare_acc(labels_true: np.ndarray, labels_pred: np.ndarray, threshold: float = 0.05) -> Optional[float]:
    """Compute accuracy restricted to rare classes."""
    aligned = align_labels(labels_true, labels_pred)
    classes, counts = np.unique(labels_true, return_counts=True)
    frequencies = counts / max(len(labels_true), 1)
    rare_classes = classes[frequencies < threshold]
    if len(rare_classes) == 0:
        return None
    rare_mask = np.isin(labels_true, rare_classes)
    if not np.any(rare_mask):
        return None
    return float(np.mean(aligned[rare_mask] == labels_true[rare_mask]))


def _knn_purity(latent: np.ndarray, labels: np.ndarray, n_neighbors: int = 30) -> float:
    """Compute class-balanced kNN purity in latent space."""
    from sklearn.neighbors import NearestNeighbors

    latent = _to_array(latent)
    labels = _to_array(labels)
    if len(labels) < 2:
        return float("nan")

    k = min(int(n_neighbors), len(labels) - 1)
    if k <= 0:
        return float("nan")

    neighbor_index = NearestNeighbors(n_neighbors=k + 1).fit(latent)
    nearest_indices = neighbor_index.kneighbors(latent, return_distance=False)[:, 1:]
    neighbor_labels = labels[nearest_indices]
    per_cell = (neighbor_labels == labels.reshape(-1, 1)).mean(axis=1)
    per_class = [np.mean(per_cell[labels == cls]) for cls in np.unique(labels)]
    return float(np.mean(per_class))


def _silhouette(embeddings: np.ndarray, labels_pred: np.ndarray, sample_size: Optional[int] = 5000) -> float:
    """Compute silhouette score with optional subsampling."""
    from sklearn.metrics import silhouette_score

    if len(np.unique(labels_pred)) < 2:
        return 0.0

    X = embeddings
    y = labels_pred
    if sample_size is not None and len(y) > sample_size:
        indices = np.random.choice(len(y), sample_size, replace=False)
        X = X[indices]
        y = y[indices]

    try:
        return float(silhouette_score(X, y))
    except Exception:
        return 0.0


def compute_metrics(
    labels_true: Optional[np.ndarray],
    labels_pred: np.ndarray,
    embeddings: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute the metric bundle saved by the pipeline."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    labels_pred = _to_array(labels_pred)
    labels_true = _to_array(labels_true) if labels_true is not None else None
    labels_true, labels_pred, embeddings = _filter_noise(labels_true, labels_pred, embeddings)

    metrics: Dict[str, Any] = {
        "NMI": float("nan"),
        "ARI": float("nan"),
        "ACC": float("nan"),
        "F1_Macro": float("nan"),
        "BalancedACC": float("nan"),
        "RareACC": float("nan"),
        "KNN_Purity": float("nan"),
        "Silhouette": float("nan"),
        "n_clusters_found": int(len(np.unique(labels_pred))) if len(labels_pred) else 0,
        "n_samples_evaluated": int(len(labels_pred)),
    }

    if labels_true is not None and len(labels_true) > 0:
        metrics["NMI"] = float(normalized_mutual_info_score(labels_true, labels_pred))
        metrics["ARI"] = float(adjusted_rand_score(labels_true, labels_pred))
        metrics["ACC"] = _accuracy(labels_true, labels_pred)
        metrics.update(_balanced_metrics(labels_true, labels_pred))

        rare_accuracy = _rare_acc(labels_true, labels_pred)
        if rare_accuracy is not None:
            metrics["RareACC"] = float(rare_accuracy)

        if embeddings is not None and len(embeddings) == len(labels_true):
            metrics["KNN_Purity"] = _knn_purity(embeddings, labels_true)

    if embeddings is not None and len(embeddings) == len(labels_pred):
        metrics["Silhouette"] = _silhouette(embeddings, labels_pred)

    return metrics
