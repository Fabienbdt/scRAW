#!/usr/bin/env python3
"""Lightweight metric computation for standalone scRAW runs."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

NOISE_LABELS = {-1, "-1", "noise", "Noise", "NOISE", "unassigned", "Unassigned"}
logger = logging.getLogger(__name__)


def _to_array(values: Any) -> np.ndarray:
    """Convert any array-like input to a NumPy array."""
    return np.asarray(values)


def _filter_noise(
    labels_true: Optional[np.ndarray], labels_pred: np.ndarray, embeddings: Optional[np.ndarray]
) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """Remove rows marked as noise in predicted/true labels (and embeddings)."""
    mask_pred = np.ones(len(labels_pred), dtype=bool)
    labels_pred_str = labels_pred.astype(str)
    for v in NOISE_LABELS:
        mask_pred &= labels_pred_str != str(v)

    if labels_true is None:
        if embeddings is not None:
            embeddings = embeddings[mask_pred]
        return None, labels_pred[mask_pred], embeddings

    labels_true_str = labels_true.astype(str)
    mask_true = np.ones(len(labels_true), dtype=bool)
    for v in NOISE_LABELS:
        mask_true &= labels_true_str != str(v)

    mask = mask_pred & mask_true
    emb_f = embeddings[mask] if embeddings is not None else None
    return labels_true[mask], labels_pred[mask], emb_f


def align_labels(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    """Align predicted clusters to true labels with Hungarian matching."""
    labels_true = _to_array(labels_true)
    labels_pred = _to_array(labels_pred)
    if len(labels_true) != len(labels_pred):
        raise ValueError("labels_true and labels_pred must have same length")

    true_u = np.unique(labels_true)
    pred_u = np.unique(labels_pred)
    true_map = {lab: i for i, lab in enumerate(true_u)}
    pred_map = {lab: i for i, lab in enumerate(pred_u)}

    w = np.zeros((len(pred_u), len(true_u)), dtype=np.int64)
    for t, p in zip(labels_true, labels_pred):
        w[pred_map[p], true_map[t]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    mapping = {pred_u[r]: true_u[c] for r, c in zip(row_ind, col_ind)}

    out = np.array([mapping.get(p, f"Unmatched_{p}") for p in labels_pred], dtype=object)
    return out


def marker_overlap_annotation(
    adata: Any,
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    n_top_genes: int = 100,
    method: str = "wilcoxon",
) -> Dict[str, Any]:
    """
    Annotate predicted clusters using marker-gene overlap with gold labels.

    Returns a dict with marker labels, overlap matrix, DEG lists, and Hungarian labels.
    """
    import pandas as pd
    import scanpy as sc

    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    if len(labels_true) != adata.n_obs or len(labels_pred) != adata.n_obs:
        raise ValueError(
            f"Label lengths ({len(labels_true)}, {len(labels_pred)}) "
            f"must match adata.n_obs ({adata.n_obs})."
        )

    def _compute_degs_per_group(adata_work: Any, labels: np.ndarray, group_name: str) -> Dict[str, list]:
        adata_copy = adata_work.copy()
        adata_copy.obs[group_name] = labels.astype(str)

        unique_groups = sorted(adata_copy.obs[group_name].unique())
        degs_dict: Dict[str, list] = {}

        try:
            sc.tl.rank_genes_groups(
                adata_copy,
                groupby=group_name,
                method=method,
                n_genes=n_top_genes,
                use_raw=False,
            )
        except Exception:
            sc.tl.rank_genes_groups(
                adata_copy,
                groupby=group_name,
                method="t-test",
                n_genes=n_top_genes,
                use_raw=False,
            )

        for grp in unique_groups:
            try:
                df = sc.get.rank_genes_groups_df(adata_copy, group=grp)
                degs_dict[grp] = df["names"].head(n_top_genes).tolist()
            except Exception as exc:
                logger.warning("Could not get DEGs for group %s: %s", grp, exc)
                degs_dict[grp] = []

        return degs_dict

    logger.info("Computing gold-standard DEGs from ground-truth labels...")
    gold_degs = _compute_degs_per_group(adata, labels_true, "_gold")

    logger.info("Computing DEGs from predicted clusters...")
    pred_degs = _compute_degs_per_group(adata, labels_pred, "_pred")

    gold_types = sorted(gold_degs.keys())
    pred_clusters = sorted(pred_degs.keys())
    overlap_data = np.zeros((len(pred_clusters), len(gold_types)))

    for i, pred_cluster in enumerate(pred_clusters):
        pred_genes = set(pred_degs.get(pred_cluster, []))
        for j, gold_type in enumerate(gold_types):
            gold_genes = set(gold_degs.get(gold_type, []))
            if pred_genes and gold_genes:
                overlap_data[i, j] = len(pred_genes & gold_genes) / float(n_top_genes)
            else:
                overlap_data[i, j] = 0.0

    overlap_df = pd.DataFrame(
        overlap_data,
        index=pred_clusters,
        columns=gold_types,
    )
    overlap_df.index.name = "Predicted Cluster"
    overlap_df.columns.name = "Gold Standard Type"

    cluster_to_type: Dict[str, str] = {}
    for i, pred_cluster in enumerate(pred_clusters):
        best_idx = int(np.argmax(overlap_data[i]))
        best_score = overlap_data[i, best_idx]
        best_type = gold_types[best_idx]
        cluster_to_type[pred_cluster] = best_type if best_score > 0 else f"Unknown_{pred_cluster}"

    marker_labels = np.array(
        [cluster_to_type.get(str(lbl), f"Unknown_{lbl}") for lbl in labels_pred],
        dtype=object,
    )
    hungarian_labels = align_labels(labels_true, labels_pred)

    return {
        "marker_labels": marker_labels,
        "overlap_matrix": overlap_df,
        "gold_degs": gold_degs,
        "pred_degs": pred_degs,
        "hungarian_labels": hungarian_labels,
        "cluster_to_type": cluster_to_type,
    }


def _accuracy(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute accuracy after Hungarian alignment."""
    labels_true = _to_array(labels_true)
    labels_pred = _to_array(labels_pred)
    aligned = align_labels(labels_true, labels_pred)
    return float(np.mean(aligned == labels_true))


def _balanced_metrics(labels_true: np.ndarray, labels_pred: np.ndarray) -> Dict[str, float]:
    """Compute macro F1 and balanced accuracy after label alignment."""
    from sklearn.metrics import balanced_accuracy_score, f1_score

    aligned = align_labels(labels_true, labels_pred)
    return {
        "F1_Macro": float(f1_score(labels_true, aligned, average="macro", zero_division=0)),
        "BalancedACC": float(balanced_accuracy_score(labels_true, aligned)),
    }


def _rare_acc(labels_true: np.ndarray, labels_pred: np.ndarray, threshold: float = 0.05) -> Optional[float]:
    """Compute accuracy restricted to rare classes below `threshold` frequency."""
    labels_true = _to_array(labels_true)
    aligned = align_labels(labels_true, labels_pred)
    classes, counts = np.unique(labels_true, return_counts=True)
    freq = counts / max(len(labels_true), 1)
    rare = classes[freq < threshold]
    if len(rare) == 0:
        return None
    mask = np.isin(labels_true, rare)
    if not np.any(mask):
        return None
    return float(np.mean(aligned[mask] == labels_true[mask]))


def _classwise(labels_true: np.ndarray, labels_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Return per-class precision/recall/F1/support after alignment."""
    from sklearn.metrics import precision_recall_fscore_support

    labels_true = _to_array(labels_true)
    aligned = align_labels(labels_true, labels_pred)
    classes = np.unique(labels_true)
    p, r, f, s = precision_recall_fscore_support(
        labels_true, aligned, labels=classes, zero_division=0
    )

    out: Dict[str, Dict[str, float]] = {}
    for i, cls in enumerate(classes):
        out[str(cls)] = {
            "Precision": float(p[i]),
            "Recall": float(r[i]),
            "F1": float(f[i]),
            "Support": int(s[i]),
        }
    return out


def _silhouette(embeddings: np.ndarray, labels_pred: np.ndarray, sample_size: Optional[int] = 5000) -> float:
    """Compute silhouette score with optional random subsampling."""
    from sklearn.metrics import silhouette_score

    if len(np.unique(labels_pred)) < 2:
        return 0.0

    X = embeddings
    y = labels_pred
    if sample_size is not None and len(y) > sample_size:
        idx = np.random.choice(len(y), sample_size, replace=False)
        X = X[idx]
        y = y[idx]

    try:
        return float(silhouette_score(X, y))
    except Exception:
        return 0.0


def _knn_purity(latent: np.ndarray, labels: np.ndarray, n_neighbors: int = 30) -> float:
    """Compute class-balanced kNN purity in latent space."""
    from sklearn.neighbors import NearestNeighbors

    latent = _to_array(latent)
    labels = _to_array(labels)
    if len(labels) < 2:
        return float("nan")

    k = min(n_neighbors, len(labels) - 1)
    if k <= 0:
        return float("nan")

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(latent)
    idx = nbrs.kneighbors(latent, return_distance=False)[:, 1:]
    neigh_labels = labels[idx]
    per_cell = (neigh_labels == labels.reshape(-1, 1)).mean(axis=1)
    per_class = [np.mean(per_cell[labels == c]) for c in np.unique(labels)]
    return float(np.mean(per_class))


def compute_metrics(
    labels_true: Optional[np.ndarray],
    labels_pred: np.ndarray,
    embeddings: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute the full clustering metric bundle used by scRAW outputs."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    labels_pred = _to_array(labels_pred)
    labels_true = _to_array(labels_true) if labels_true is not None else None
    labels_true, labels_pred, embeddings = _filter_noise(labels_true, labels_pred, embeddings)

    out: Dict[str, Any] = {
        "NMI": float("nan"),
        "ARI": float("nan"),
        "ACC": float("nan"),
        "UCA": float("nan"),
        "F1_Macro": float("nan"),
        "BalancedACC": float("nan"),
        "RareACC": float("nan"),
        "KNN_Purity": float("nan"),
        "ClassWise": {},
        "Silhouette": float("nan"),
        "n_clusters_found": int(len(np.unique(labels_pred))) if len(labels_pred) else 0,
        "n_samples_evaluated": int(len(labels_pred)),
    }

    if labels_true is not None and len(labels_true) > 0:
        out["NMI"] = float(normalized_mutual_info_score(labels_true, labels_pred))
        out["ARI"] = float(adjusted_rand_score(labels_true, labels_pred))
        out["ACC"] = _accuracy(labels_true, labels_pred)
        out["UCA"] = out["ACC"]

        bm = _balanced_metrics(labels_true, labels_pred)
        out.update(bm)

        rare = _rare_acc(labels_true, labels_pred)
        if rare is not None:
            out["RareACC"] = float(rare)

        out["ClassWise"] = _classwise(labels_true, labels_pred)

        if embeddings is not None and len(embeddings) == len(labels_true):
            out["KNN_Purity"] = _knn_purity(embeddings, labels_true)

    if embeddings is not None and len(embeddings) == len(labels_pred):
        out["Silhouette"] = _silhouette(embeddings, labels_pred)

    return out
