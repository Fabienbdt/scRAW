#!/usr/bin/env python3
"""Visualization utilities for standalone scRAW runs (SCRBenchmark parity)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .metrics import align_labels


def _compute_umap_or_2d(embeddings: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, bool]:
    """Return a 2D projection and whether we had to fallback to first 2 dims."""
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2 or embeddings.shape[1] < 2:
        raise ValueError("Embeddings must be a 2D array with at least 2 columns.")

    if embeddings.shape[1] <= 2:
        return embeddings, False

    if embeddings.shape[1] > 50:
        from sklearn.decomposition import PCA

        n_comps = min(50, embeddings.shape[0] - 1, embeddings.shape[1])
        pca = PCA(n_components=n_comps, random_state=random_state)
        embeddings = pca.fit_transform(embeddings)

    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=random_state)
        return reducer.fit_transform(embeddings), False
    except Exception:
        return embeddings[:, :2], True


def _select_snapshot_indices_for_gallery(snapshots: List[Dict[str, Any]], max_snapshots: int) -> List[int]:
    """Select representative snapshot indices for gallery-like UMAP evolution."""
    n = len(snapshots)
    if n == 0:
        return []
    if max_snapshots <= 0 or n <= max_snapshots:
        return list(range(n))

    must_keep: set[int] = {0, n - 1}
    prev_phase = str(snapshots[0].get("phase", ""))
    warmup_indices: List[int] = []
    for i, snap in enumerate(snapshots):
        phase = str(snap.get("phase", ""))
        if phase.lower().startswith("warm"):
            warmup_indices.append(i)
        if i > 0 and phase != prev_phase:
            must_keep.add(i - 1)
            must_keep.add(i)
        prev_phase = phase

    if warmup_indices:
        must_keep.add(max(warmup_indices))

    refresh_indices = [
        i for i, snap in enumerate(snapshots) if snap.get("snapshot_type") == "weight_refresh"
    ]
    if 0 < len(refresh_indices) <= 3:
        must_keep.update(refresh_indices)
    elif len(refresh_indices) > 3:
        must_keep.update(
            [
                refresh_indices[0],
                refresh_indices[len(refresh_indices) // 2],
                refresh_indices[-1],
            ]
        )

    selected = sorted(i for i in must_keep if 0 <= i < n)
    if len(selected) >= max_snapshots:
        key_positions = np.linspace(0, len(selected) - 1, num=max_snapshots)
        picked = sorted({selected[int(round(pos))] for pos in key_positions})
        return picked[:max_snapshots]

    selected_set = set(selected)
    fill_candidates = [
        int(round(pos))
        for pos in np.linspace(0, n - 1, num=max(max_snapshots * 3, n))
    ]
    for idx in fill_candidates:
        if idx not in selected_set:
            selected.append(idx)
            selected_set.add(idx)
            if len(selected) >= max_snapshots:
                break

    return sorted(selected)


def _compute_shared_umap_sequence(
    embeddings_per_snapshot: List[np.ndarray],
    random_state: int = 42,
) -> Tuple[List[np.ndarray], bool]:
    """Compute one shared projector and transform each snapshot with it."""
    if not embeddings_per_snapshot:
        return [], True

    arrays = [np.asarray(e) for e in embeddings_per_snapshot]
    if len(arrays) == 1:
        one, used_fb = _compute_umap_or_2d(arrays[0], random_state=random_state)
        return [one], used_fb

    first_dim = arrays[0].shape[1] if arrays[0].ndim == 2 else -1
    if first_dim < 2 or any(a.ndim != 2 or a.shape[1] != first_dim for a in arrays):
        out = []
        used_any_fallback = False
        for emb in arrays:
            emb2d, used_fb = _compute_umap_or_2d(emb, random_state=random_state)
            out.append(emb2d)
            used_any_fallback = used_any_fallback or used_fb
        return out, used_any_fallback

    common = np.vstack(arrays)
    transformed_arrays = arrays
    if common.shape[1] > 50:
        from sklearn.decomposition import PCA

        n_comps = min(50, common.shape[0] - 1, common.shape[1])
        pca_model = PCA(n_components=n_comps, random_state=random_state)
        common = pca_model.fit_transform(common)
        transformed_arrays = [pca_model.transform(emb) for emb in arrays]

    if common.shape[1] <= 2:
        return [emb[:, :2] for emb in transformed_arrays], False

    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=random_state)
        reducer.fit(common)
        out = [reducer.transform(emb) for emb in transformed_arrays]
        return out, False
    except Exception:
        return [emb[:, :2] for emb in transformed_arrays], True


def _encode_labels(labels: np.ndarray) -> Tuple[np.ndarray, List[Any], Dict[Any, int]]:
    """Encode arbitrary labels to stable integer IDs for plotting."""
    labels_arr = np.asarray(labels, dtype=object)
    sentinel = "__MISSING__"
    normalized = np.array([sentinel if pd.isna(x) else x for x in labels_arr], dtype=object)

    if isinstance(labels, pd.Series) and isinstance(labels.dtype, pd.CategoricalDtype):
        unique_labels = [sentinel if pd.isna(x) else x for x in labels.cat.categories]
        extra = [x for x in pd.unique(normalized) if x not in unique_labels]
        unique_labels.extend(extra)
    else:
        unique_labels = list(pd.unique(normalized))

    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    encoded = np.array([label_map[x] for x in normalized])
    return encoded, unique_labels, label_map


def _decode_label_name(label: Any, label_names: Optional[Dict[int, str]]) -> str:
    """Decode numeric labels to readable names when a map is provided."""
    if label == "__MISSING__":
        return "NA"
    label_text = str(label)
    if label_text.startswith("Unmatched_Cluster_"):
        return label_text.replace("Unmatched_Cluster_", "Unmatched cluster ")
    if label_names is None:
        return label_text

    try:
        key = int(float(label))
        if key in label_names:
            return str(label_names[key])
    except (ValueError, TypeError):
        pass
    return label_text


def _tag_unmatched_predicted_labels(
    predicted_labels: np.ndarray,
    label_names: Optional[Dict[int, str]],
) -> np.ndarray:
    """Mark predicted labels that are outside known label IDs as unmatched."""
    if not label_names:
        return np.asarray(predicted_labels, dtype=object)

    known_ids = set()
    for k in label_names.keys():
        try:
            known_ids.add(int(k))
        except (TypeError, ValueError):
            continue

    out = []
    for raw in np.asarray(predicted_labels, dtype=object):
        txt = str(raw)
        if txt.startswith("Unmatched_Cluster_"):
            out.append(txt)
            continue
        try:
            idx = int(float(txt))
        except (TypeError, ValueError):
            out.append(raw)
            continue
        if idx in known_ids:
            out.append(raw)
        else:
            out.append(f"Unmatched_Cluster_{idx}")
    return np.asarray(out, dtype=object)


def _draw_cluster_overlays(
    ax: plt.Axes,
    points_2d: np.ndarray,
    labels: np.ndarray,
    outline_mode: str = "ellipse",
    n_std: float = 1.8,
    show_centroids: bool = False,
    centroid_label_names: Optional[Dict[int, str]] = None,
) -> None:
    """Draw optional cluster outlines and centroid labels."""
    from matplotlib.patches import Ellipse, Polygon

    labels_arr = np.asarray(labels)
    _, unique_labels, _ = _encode_labels(labels_arr)
    n_labels = len(unique_labels)
    cmap = plt.cm.tab20 if n_labels <= 20 else plt.cm.gist_ncar
    norm = plt.Normalize(vmin=0, vmax=max(n_labels - 1, 1))

    mode = (outline_mode or "ellipse").lower()
    if mode in ("hull", "convex"):
        mode = "convex_hull"
    if mode not in {"none", "ellipse", "convex_hull", "density"}:
        mode = "ellipse"

    top_for_density = set(unique_labels)
    if mode == "density" and n_labels > 16:
        counts = pd.Series(labels_arr).value_counts()
        top_for_density = set(counts.index[:16].tolist())

    for idx, group_label in enumerate(unique_labels):
        mask = labels_arr == group_label
        points = points_2d[mask]
        if len(points) < 3:
            continue

        color = cmap(norm(idx))
        try:
            if mode == "ellipse":
                mean = np.mean(points, axis=0)
                cov = np.cov(points.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                order = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[order]
                eigenvectors = eigenvectors[:, order]
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                width = 2 * n_std * np.sqrt(max(eigenvalues[0], 1e-3))
                height = 2 * n_std * np.sqrt(max(eigenvalues[1], 1e-3))

                ax.add_patch(
                    Ellipse(
                        xy=mean,
                        width=width,
                        height=height,
                        angle=angle,
                        facecolor=color,
                        edgecolor="none",
                        alpha=0.10,
                    )
                )
                ax.add_patch(
                    Ellipse(
                        xy=mean,
                        width=width,
                        height=height,
                        angle=angle,
                        facecolor="none",
                        edgecolor=color,
                        linewidth=1.8,
                        alpha=0.70,
                    )
                )
            elif mode == "convex_hull":
                from scipy.spatial import ConvexHull

                hull = ConvexHull(points)
                hull_pts = points[hull.vertices]
                ax.add_patch(
                    Polygon(
                        hull_pts,
                        closed=True,
                        facecolor=color,
                        edgecolor="none",
                        alpha=0.10,
                    )
                )
                ax.add_patch(
                    Polygon(
                        hull_pts,
                        closed=True,
                        facecolor="none",
                        edgecolor=color,
                        linewidth=1.8,
                        alpha=0.75,
                    )
                )
            elif mode == "density":
                if group_label in top_for_density and len(points) >= 20:
                    sns.kdeplot(
                        x=points[:, 0],
                        y=points[:, 1],
                        ax=ax,
                        levels=1,
                        color=color,
                        linewidths=1.5,
                        fill=False,
                        thresh=0.20,
                    )
        except Exception:
            continue

        if show_centroids:
            center = points.mean(axis=0)
            label_text = _decode_label_name(group_label, centroid_label_names)
            ax.text(
                center[0],
                center[1],
                label_text,
                fontsize=7,
                ha="center",
                va="center",
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    alpha=0.75,
                    edgecolor="none",
                ),
            )


def _add_param_annotation(
    fig: plt.Figure,
    params_info: Optional[Dict[str, Any]] = None,
    dataset_info: Optional[str] = None,
) -> None:
    """Draw a small annotation box with run context."""
    lines: List[str] = []
    if dataset_info:
        lines.append(dataset_info)
    if params_info:
        for key, val in params_info.items():
            lines.append(f"{key}: {val}")
    if not lines:
        return
    text = "\n".join(lines)
    fig.text(
        0.01,
        0.01,
        text,
        fontsize=7,
        fontfamily="monospace",
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray"),
        transform=fig.transFigure,
    )


def plot_umap_comparison(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    algorithm_name: str = "Algorithm",
    save_path: Optional[str] = None,
    n_std: float = 1.8,
    label_names: Optional[Dict[int, str]] = None,
    outline_mode: str = "none",
    show_cluster_centroids: bool = True,
    projection_name: str = "UMAP",
    params_info: Optional[Dict[str, Any]] = None,
    dataset_info: Optional[str] = None,
) -> plt.Figure:
    """Plot side-by-side UMAP comparison: Ground Truth vs Predicted clusters."""
    embeddings = np.asarray(embeddings)
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)

    try:
        predicted_labels = align_labels(true_labels, predicted_labels)
    except Exception:
        pass
    predicted_labels = _tag_unmatched_predicted_labels(predicted_labels, label_names)

    embeddings_2d, _ = _compute_umap_or_2d(embeddings)

    unmatched_prefix = "Unmatched_Cluster_"
    unmatched_color = "#888888"

    true_str = np.asarray(true_labels, dtype=object).astype(str)
    pred_str = np.asarray(predicted_labels, dtype=object).astype(str)

    true_unique = list(pd.unique(true_str))
    pred_unmatched = set(x for x in pd.unique(pred_str) if str(x).startswith(unmatched_prefix))
    pred_matched_extra = [
        x for x in pd.unique(pred_str)
        if not str(x).startswith(unmatched_prefix) and x not in true_unique
    ]
    real_unique: List[Any] = list(true_unique) + list(pred_matched_extra)
    real_map: Dict[Any, int] = {lbl: i for i, lbl in enumerate(real_unique)}
    n_real = len(real_unique)
    shared_cmap = plt.cm.tab20 if n_real <= 20 else plt.cm.gist_ncar
    shared_norm = plt.Normalize(vmin=0, vmax=max(n_real - 1, 1))

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    def _draw_panel(
        ax: plt.Axes,
        labels: np.ndarray,
        panel_title: str,
        use_label_names: bool = False,
        custom_label_names: Optional[Dict[int, str]] = None,
    ) -> None:
        labels_arr = np.asarray(labels, dtype=object).astype(str)
        target_map = (
            custom_label_names
            if custom_label_names is not None
            else (label_names if use_label_names else None)
        )

        is_unmatched = np.array([str(x) in pred_unmatched for x in labels_arr], dtype=bool)
        matched_mask = ~is_unmatched

        if np.any(matched_mask):
            encoded = np.array([real_map.get(str(x), 0) for x in labels_arr[matched_mask]])
            ax.scatter(
                embeddings_2d[matched_mask, 0],
                embeddings_2d[matched_mask, 1],
                c=encoded,
                cmap=shared_cmap,
                norm=shared_norm,
                s=3,
                alpha=0.8,
            )
        if np.any(is_unmatched):
            ax.scatter(
                embeddings_2d[is_unmatched, 0],
                embeddings_2d[is_unmatched, 1],
                c=unmatched_color,
                s=5,
                alpha=0.9,
                marker="x",
                linewidths=0.5,
            )

        if outline_mode != "none" or show_cluster_centroids:
            _draw_cluster_overlays(
                ax=ax,
                points_2d=embeddings_2d,
                labels=labels_arr,
                outline_mode=outline_mode,
                n_std=n_std,
                show_centroids=show_cluster_centroids,
                centroid_label_names=target_map,
            )

        ax.set_title(panel_title, fontsize=13, fontweight="bold")
        ax.set_xlabel(f"{projection_name} 1", fontsize=10)
        ax.set_ylabel(f"{projection_name} 2", fontsize=10)

        present = set(labels_arr)
        all_labels_ordered = list(real_unique) + sorted(pred_unmatched)
        n_present = sum(1 for lbl in all_labels_ordered if lbl in present)
        if n_present <= 30:
            handles = []
            for lbl in all_labels_ordered:
                if lbl not in present:
                    continue
                if str(lbl) in pred_unmatched:
                    handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            marker="x",
                            color=unmatched_color,
                            markerfacecolor=unmatched_color,
                            markersize=7,
                            linestyle="None",
                            label=_decode_label_name(lbl, target_map),
                        )
                    )
                else:
                    i = real_map.get(str(lbl), 0)
                    handles.append(
                        plt.Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor=shared_cmap(shared_norm(i)),
                            markersize=7,
                            label=_decode_label_name(lbl, target_map),
                        )
                    )
            ax.legend(
                handles=handles,
                fontsize=7,
                markerscale=1.2,
                bbox_to_anchor=(1.01, 1),
                loc="upper left",
                borderaxespad=0,
                framealpha=0.9,
            )
        else:
            ax.text(
                0.02,
                0.02,
                f"({n_present} clusters - legend hidden)",
                transform=ax.transAxes,
                fontsize=8,
                alpha=0.7,
            )

    _draw_panel(axes[0], true_str, "Ground Truth (Cell Types)", use_label_names=True)
    _draw_panel(axes[1], predicted_labels, "Predicted Clusters (IDs)", use_label_names=False)
    _draw_panel(
        axes[2],
        pred_str,
        "Predicted (Aligned & Named)",
        use_label_names=True,
        custom_label_names=label_names,
    )

    fig.suptitle(f"{projection_name} Comparison: {algorithm_name}", fontsize=16, fontweight="bold", y=1.02)

    if params_info or dataset_info:
        _add_param_annotation(fig, params_info, dataset_info)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_loss_curves(
    loss_history: List[Dict[str, Any]],
    algorithm_name: str,
    show_components: bool = True,
) -> Optional[plt.Figure]:
    """Plot training/validation loss curves from phase-wise loss history."""
    if not loss_history:
        return None

    n_phases = len(loss_history)
    fig, axes = plt.subplots(1, n_phases, figsize=(6 * n_phases, 4), squeeze=False)
    component_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]

    for idx, phase in enumerate(loss_history):
        ax = axes[0, idx]
        train_loss = phase.get("train_loss", [])
        val_loss = phase.get("val_loss", [])
        default_len = max(len(train_loss), len(val_loss))
        epochs = phase.get("epochs", list(range(default_len)))
        phase_name = phase.get("name", f"Phase {idx + 1}")

        if not train_loss and not val_loss:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(phase_name)
            continue

        if train_loss:
            train_epochs = epochs[:len(train_loss)] if len(epochs) >= len(train_loss) else list(range(len(train_loss)))
            ax.plot(train_epochs, train_loss, color="black", linewidth=2, label="Train loss")
        if val_loss:
            val_epochs = epochs[:len(val_loss)] if len(epochs) >= len(val_loss) else list(range(len(val_loss)))
            ax.plot(val_epochs, val_loss, color="#1f77b4", linewidth=2, label="Val loss")

        if show_components and "components" in phase:
            for c_idx, (comp_name, comp_values) in enumerate(phase["components"].items()):
                if comp_values == train_loss or comp_values == val_loss:
                    continue
                comp_epochs = epochs[:len(comp_values)] if len(epochs) >= len(comp_values) else list(range(len(comp_values)))
                color = component_colors[c_idx % len(component_colors)]
                ax.plot(
                    comp_epochs,
                    comp_values,
                    color=color,
                    linewidth=1,
                    linestyle="--",
                    alpha=0.7,
                    label=comp_name,
                )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(phase_name, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    has_val = any(phase.get("val_loss") for phase in loss_history)
    title_suffix = "Training/Validation Loss" if has_val else "Training Loss"
    fig.suptitle(f"{algorithm_name} - {title_suffix}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_loss_curves_timeline(
    loss_history: List[Dict[str, Any]],
    algorithm_name: str,
) -> Optional[plt.Figure]:
    """Plot one clear timeline view of losses across all phases."""
    if not loss_history:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    palette = {
        "train_loss": "#111111",
        "reconstruction": "#e74c3c",
        "triplet": "#1f77b4",
        "batch_adv": "#2ca02c",
    }

    phase_starts: List[Tuple[int, str]] = []
    first_train = True
    seen_component_labels: set[str] = set()
    used_any = False
    for phase in loss_history:
        phase_name = str(phase.get("name", "phase"))
        train_loss = phase.get("train_loss", []) or []
        epochs = phase.get("epochs", list(range(len(train_loss)))) or []
        if len(epochs) != len(train_loss):
            epochs = list(range(len(train_loss)))
        if not epochs:
            continue
        phase_starts.append((int(epochs[0]), phase_name))

        if train_loss:
            ax.plot(
                epochs,
                train_loss,
                color=palette["train_loss"],
                linewidth=2.2,
                label="train_loss" if first_train else None,
            )
            first_train = False
            used_any = True

        components = phase.get("components", {}) or {}
        for cname in ("reconstruction", "triplet", "batch_adv"):
            vals = components.get(cname, []) if isinstance(components, dict) else []
            if not vals:
                continue
            comp_epochs = epochs[: len(vals)] if len(epochs) >= len(vals) else list(range(len(vals)))
            ax.plot(
                comp_epochs,
                vals,
                color=palette.get(cname, None),
                linewidth=1.5,
                alpha=0.9,
                linestyle="--",
                label=cname if cname not in seen_component_labels else None,
            )
            seen_component_labels.add(cname)
            used_any = True

    if not used_any:
        plt.close(fig)
        return None

    for idx, (start_epoch, phase_name) in enumerate(phase_starts):
        if idx == 0:
            continue
        ax.axvline(start_epoch, color="#777777", linewidth=1.0, alpha=0.5)
        ax.text(
            start_epoch,
            ax.get_ylim()[1],
            f" {phase_name}",
            rotation=90,
            va="top",
            ha="left",
            fontsize=8,
            alpha=0.8,
        )

    handles, labels = ax.get_legend_handles_labels()
    uniq_h: List[Any] = []
    uniq_l: List[str] = []
    for h, l in zip(handles, labels):
        if not l or l in uniq_l:
            continue
        uniq_h.append(h)
        uniq_l.append(l)
    if uniq_h:
        ax.legend(uniq_h, uniq_l, loc="upper right", fontsize=9)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{algorithm_name} - Loss Timeline", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig


def _snapshot_title(snapshot: Dict[str, Any]) -> str:
    """Compact snapshot title used in panel figures."""
    epoch = snapshot.get("epoch", "?")
    phase = str(snapshot.get("phase", ""))
    stype = str(snapshot.get("snapshot_type", ""))
    if stype == "pre_backward":
        return f"Epoch {epoch} (pre-backward)"
    if phase:
        return f"Epoch {epoch} ({phase})"
    return f"Epoch {epoch}"


def plot_umap_snapshots_categorical_panels(
    embedding_snapshots: List[Dict[str, Any]],
    labels: np.ndarray,
    title: str,
    point_size: int = 3,
    random_state: int = 42,
    max_cols: int = 4,
    params_info: Optional[Dict[str, Any]] = None,
    dataset_info: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plot multi-panel UMAP snapshots with one categorical color mapping."""
    from math import ceil

    valid = [
        s
        for s in embedding_snapshots
        if s.get("embeddings") is not None and len(np.asarray(s.get("embeddings"))) > 0
    ]
    if not valid:
        return None

    first_emb = np.asarray(valid[0]["embeddings"])
    if first_emb.ndim != 2 or first_emb.shape[0] == 0:
        return None
    n_cells = int(first_emb.shape[0])
    labels_arr = np.asarray(labels)
    if len(labels_arr) != n_cells:
        return None

    umap_list, _ = _compute_shared_umap_sequence(
        [np.asarray(s["embeddings"]) for s in valid],
        random_state=random_state,
    )
    if not umap_list:
        return None

    encoded, unique_labels, _ = _encode_labels(labels_arr)
    n_unique = len(unique_labels)
    if n_unique <= 10:
        cmap = plt.cm.tab10
    elif n_unique <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.nipy_spectral
    colors = [cmap(i / max(n_unique - 1, 1)) for i in range(n_unique)]

    n_snap = len(valid)
    n_cols = max(1, min(int(max_cols), n_snap))
    n_rows = ceil(n_snap / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 4.0 * n_rows),
        squeeze=False,
    )

    legend_unique = None
    legend_colors = None
    for idx, (snap, emb2d) in enumerate(zip(valid, umap_list)):
        ax = axes[idx // n_cols][idx % n_cols]
        for lab_idx in range(n_unique):
            mask = encoded == lab_idx
            if not np.any(mask):
                continue
            ax.scatter(
                emb2d[mask, 0],
                emb2d[mask, 1],
                c=[colors[lab_idx]],
                s=float(point_size),
                alpha=0.8,
                linewidths=0,
                rasterized=True,
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(_snapshot_title(snap), fontsize=9, fontweight="bold")
        if legend_unique is None and n_unique <= 20:
            legend_unique = unique_labels
            legend_colors = colors

    for idx in range(n_snap, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    if legend_unique is not None and legend_colors is not None:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=legend_colors[i],
                markersize=6,
                label=str(legend_unique[i]),
            )
            for i in range(len(legend_unique))
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(8, len(handles)),
            fontsize=7,
            bbox_to_anchor=(0.5, -0.02),
            frameon=True,
        )

    fig.suptitle(title, fontsize=13, fontweight="bold")
    if params_info or dataset_info:
        _add_param_annotation(fig, params_info, dataset_info)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def _normalize_weights_global(weight_arrays: List[np.ndarray]) -> Tuple[float, float]:
    """Return robust global percentile bounds for weight color normalization."""
    vals = []
    for arr in weight_arrays:
        if arr is None:
            continue
        v = np.asarray(arr, dtype=np.float64)
        if v.size == 0:
            continue
        vals.append(v[np.isfinite(v)])
    if not vals:
        return 0.0, 1.0
    flat = np.concatenate(vals)
    if flat.size == 0:
        return 0.0, 1.0
    lo = float(np.nanpercentile(flat, 5))
    hi = float(np.nanpercentile(flat, 95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(flat))
        hi = float(np.nanmax(flat))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return 0.0, 1.0
    return lo, hi


def _normalize_weights_with_bounds(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Normalize raw weights with fixed bounds to [0, 1]."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return np.asarray([], dtype=np.float64)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full(v.shape[0], 0.5, dtype=np.float64)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def plot_umap_snapshots_gradient_panels(
    embedding_snapshots: List[Dict[str, Any]],
    current_weights: List[Optional[np.ndarray]],
    title: str,
    lagged_weights: Optional[List[Optional[np.ndarray]]] = None,
    point_size: int = 3,
    random_state: int = 42,
    max_cols: int = 4,
    current_row_label: str = "Epoch n",
    lagged_row_label: str = "Epoch n-10 projected on epoch n latent",
    params_info: Optional[Dict[str, Any]] = None,
    dataset_info: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plot gradient UMAP panels for snapshot weights, with optional lagged row."""
    from math import ceil
    from matplotlib.cm import ScalarMappable

    if not embedding_snapshots:
        return None
    n_snap = len(embedding_snapshots)
    if len(current_weights) != n_snap:
        return None
    if lagged_weights is not None and len(lagged_weights) != n_snap:
        return None

    arrays = [np.asarray(s.get("embeddings")) for s in embedding_snapshots]
    if any(a.ndim != 2 or a.shape[0] == 0 for a in arrays):
        return None
    umap_list, _ = _compute_shared_umap_sequence(arrays, random_state=random_state)
    if not umap_list:
        return None

    rows = 2 if lagged_weights is not None else 1
    n_cols = max(1, min(int(max_cols), n_snap))
    n_rows_panels = ceil(n_snap / n_cols)
    fig, axes = plt.subplots(
        rows * n_rows_panels,
        n_cols,
        figsize=(4.2 * n_cols, 3.8 * rows * n_rows_panels),
        squeeze=False,
    )

    bounds_lo, bounds_hi = _normalize_weights_global(
        [w for w in current_weights if w is not None]
        + ([w for w in lagged_weights if w is not None] if lagged_weights is not None else [])
    )

    def _plot_one(
        ax: plt.Axes,
        emb2d: np.ndarray,
        w_raw: Optional[np.ndarray],
        header: str,
        snap: Dict[str, Any],
    ) -> None:
        if w_raw is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(header, fontsize=8)
            return
        w = np.asarray(w_raw, dtype=np.float64)
        if w.shape[0] != emb2d.shape[0]:
            ax.text(0.5, 0.5, "Weight size mismatch", ha="center", va="center", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(header, fontsize=8)
            return
        w_norm = _normalize_weights_with_bounds(w, bounds_lo, bounds_hi)
        order = np.argsort(w_norm)
        sc = ax.scatter(
            emb2d[order, 0],
            emb2d[order, 1],
            c=w_norm[order],
            cmap="plasma",
            vmin=0.0,
            vmax=1.0,
            s=float(point_size),
            alpha=0.95,
            linewidths=0,
            rasterized=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{header}\n{_snapshot_title(snap)}", fontsize=8, fontweight="bold")
        return sc

    for idx, snap in enumerate(embedding_snapshots):
        row0 = idx // n_cols
        col = idx % n_cols
        sc_obj = _plot_one(
            axes[row0][col],
            umap_list[idx],
            current_weights[idx],
            current_row_label,
            snap,
        )
        if lagged_weights is not None:
            _plot_one(
                axes[row0 + n_rows_panels][col],
                umap_list[idx],
                lagged_weights[idx],
                lagged_row_label,
                snap,
            )

    total_rows = rows * n_rows_panels
    for idx in range(n_snap, n_rows_panels * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r][c].axis("off")
        if lagged_weights is not None:
            axes[r + n_rows_panels][c].axis("off")

    sm = ScalarMappable(cmap="plasma")
    sm.set_clim(0.0, 1.0)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    cbar.set_label("Normalized weight (global robust scale)", rotation=90, labelpad=10)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    if params_info or dataset_info:
        _add_param_annotation(fig, params_info, dataset_info)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    return fig


def plot_metric_evolution_curves(
    metric_rows: List[Dict[str, Any]],
    title: str = "Metrics Evolution Across Epochs",
) -> Optional[plt.Figure]:
    """Plot epoch-wise clustering metrics (multiple methods supported)."""
    if not metric_rows:
        return None
    df = pd.DataFrame(metric_rows)
    if df.empty or "epoch" not in df.columns or "method" not in df.columns:
        return None

    metric_specs = [
        ("ARI", "ARI"),
        ("NMI", "NMI"),
        ("ACC", "ACC"),
        ("BalancedACC", "Balanced ACC"),
        ("F1_Macro", "F1 Macro"),
        ("RareACC", "Rare ACC"),
    ]
    methods = sorted(df["method"].astype(str).unique().tolist())
    if not methods:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    colors = ["#111111", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    cmap = {m: colors[i % len(colors)] for i, m in enumerate(methods)}

    for idx, (col_name, pretty_name) in enumerate(metric_specs):
        ax = axes[idx // 3][idx % 3]
        plotted_any = False
        for method in methods:
            sub = df[df["method"].astype(str) == method].copy()
            if col_name not in sub.columns:
                continue
            sub = sub.sort_values("epoch")
            ys = pd.to_numeric(sub[col_name], errors="coerce")
            xs = pd.to_numeric(sub["epoch"], errors="coerce")
            mask = np.isfinite(xs.to_numpy()) & np.isfinite(ys.to_numpy())
            if not np.any(mask):
                continue
            ax.plot(
                xs.to_numpy()[mask],
                ys.to_numpy()[mask],
                marker="o",
                markersize=3,
                linewidth=1.8,
                color=cmap[method],
                label=method,
            )
            plotted_any = True
        if not plotted_any:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(pretty_name, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(pretty_name)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(-0.02, 1.02)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=9)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    return fig


def plot_umap_batch(
    embeddings: np.ndarray,
    batch_labels: np.ndarray,
    title: str = "UMAP - Batch",
    point_size: int = 2,
    random_state: int = 42,
    params_info: Optional[Dict[str, Any]] = None,
    dataset_info: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plot UMAP colored by batch labels."""
    embeddings = np.asarray(embeddings)
    if embeddings.shape[0] == 0:
        return None

    embeddings_2d, _ = _compute_umap_or_2d(embeddings, random_state=random_state)
    encoded, unique_labels, _ = _encode_labels(np.asarray(batch_labels))
    n_labels = len(unique_labels)
    cmap = plt.cm.tab10 if n_labels <= 10 else plt.cm.tab20
    norm = plt.Normalize(vmin=0, vmax=max(n_labels - 1, 1))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=encoded,
        cmap=cmap,
        norm=norm,
        s=point_size,
        alpha=0.7,
        rasterized=True,
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    if n_labels <= 30:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cmap(norm(i)),
                markersize=6,
                label=str(lbl),
            )
            for i, lbl in enumerate(unique_labels)
        ]
        ax.legend(
            handles=handles,
            fontsize=8,
            markerscale=1.5,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            framealpha=0.9,
        )

    if params_info or dataset_info:
        _add_param_annotation(fig, params_info, dataset_info)

    plt.tight_layout()
    return fig


def _normalize_cell_weights_robust(cell_weights: np.ndarray) -> np.ndarray:
    """Robustly normalize weights to [0, 1] using 5th/95th percentiles."""
    w = np.asarray(cell_weights, dtype=np.float64)
    if len(w) == 0:
        return np.asarray([], dtype=np.float64)

    w_lo = float(np.nanpercentile(w, 5))
    w_hi = float(np.nanpercentile(w, 95))
    if not np.isfinite(w_lo) or not np.isfinite(w_hi) or w_hi <= w_lo:
        w_lo = float(np.nanmin(w))
        w_hi = float(np.nanmax(w))
    if np.isfinite(w_hi) and w_hi > w_lo:
        return np.clip((w - w_lo) / (w_hi - w_lo), 0.0, 1.0)
    return np.full(len(w), 0.5, dtype=np.float64)


def _weights_to_alpha_exponential(
    w_norm: np.ndarray,
    strength: float = 14.0,
    min_alpha: float = 0.002,
) -> np.ndarray:
    """Convert normalized weights to alpha with strong exponential contrast."""
    w = np.asarray(w_norm, dtype=np.float64)
    return np.clip(np.exp((w - 1.0) * strength), min_alpha, 1.0)


def plot_umap_weighted(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cell_weights: np.ndarray,
    title: str = "UMAP - Cell Weights (Opacity ∝ loss weight)",
    point_size: int = 3,
    random_state: int = 42,
    label_names: Optional[Dict[int, str]] = None,
    params_info: Optional[Dict[str, Any]] = None,
    dataset_info: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plot UMAP with label colors and opacity proportional to reconstruction weight."""
    embeddings = np.asarray(embeddings)
    cell_weights = np.asarray(cell_weights, dtype=np.float64)
    if embeddings.shape[0] == 0:
        return None

    embeddings_2d, _ = _compute_umap_or_2d(embeddings, random_state=random_state)
    encoded, unique_labels, _ = _encode_labels(np.asarray(labels))
    n_labels = len(unique_labels)
    cmap = plt.cm.tab20 if n_labels <= 20 else plt.cm.gist_ncar
    norm_c = plt.Normalize(vmin=0, vmax=max(n_labels - 1, 1))

    w_norm = _normalize_cell_weights_robust(cell_weights)
    alpha_arr = _weights_to_alpha_exponential(w_norm, strength=14.0, min_alpha=0.002)

    sort_order = np.argsort(w_norm)
    e2d_sorted = embeddings_2d[sort_order]
    enc_sorted = encoded[sort_order]
    alpha_sorted = alpha_arr[sort_order]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors_per_point = np.array(
        [list(cmap(norm_c(i))) for i in enc_sorted], dtype=np.float64
    )
    colors_per_point[:, 3] = alpha_sorted

    ax.scatter(
        e2d_sorted[:, 0],
        e2d_sorted[:, 1],
        c=colors_per_point,
        s=float(point_size),
        rasterized=True,
        linewidths=0,
    )

    ax.set_title(
        title + "\n"
        "(opacite ∝ poids de reconstruction - rare opaque, commun transparent)",
        fontsize=12,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    if n_labels <= 30:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cmap(norm_c(i)),
                markersize=7,
                label=_decode_label_name(unique_labels[i], label_names),
            )
            for i in range(n_labels)
        ]
        ax.legend(
            handles=handles,
            fontsize=8,
            markerscale=1.0,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            framealpha=0.9,
            title="Cell type",
            title_fontsize=8,
        )

    if params_info or dataset_info:
        _add_param_annotation(fig, params_info, dataset_info)

    plt.tight_layout()
    return fig


def plot_umap_weighted_gradient(
    embeddings: np.ndarray,
    cell_weights: np.ndarray,
    title: str = "UMAP - Cell Weights (Gradient)",
    point_size: int = 3,
    random_state: int = 42,
    params_info: Optional[Dict[str, Any]] = None,
    dataset_info: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plot UMAP with continuous plasma gradient for reconstruction weights."""
    embeddings = np.asarray(embeddings)
    cell_weights = np.asarray(cell_weights, dtype=np.float64)
    if embeddings.shape[0] == 0:
        return None

    embeddings_2d, _ = _compute_umap_or_2d(embeddings, random_state=random_state)
    w_norm = _normalize_cell_weights_robust(cell_weights)

    sort_order = np.argsort(w_norm)
    e2d_s = embeddings_2d[sort_order]
    w_s = w_norm[sort_order]

    fig, ax = plt.subplots(figsize=(11, 8))
    sc = ax.scatter(
        e2d_s[:, 0],
        e2d_s[:, 1],
        c=w_s,
        cmap="plasma",
        vmin=0.0,
        vmax=1.0,
        s=float(point_size),
        alpha=0.95,
        rasterized=True,
        linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        "Poids de reconstruction (normalise)\nFaible -> Eleve",
        rotation=90,
        labelpad=12,
        fontsize=10,
    )
    w_raw = cell_weights
    cbar.ax.text(
        0.5,
        -0.02,
        f"min={float(np.nanmin(w_raw)):.2f}",
        transform=cbar.ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#444",
    )
    cbar.ax.text(
        0.5,
        1.02,
        f"max={float(np.nanmax(w_raw)):.2f}",
        transform=cbar.ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444",
    )

    ax.set_title(
        title + "\n"
        "(degrade plasma: violet fonce = poids faible, jaune vif = poids eleve)",
        fontsize=12,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    if params_info or dataset_info:
        _add_param_annotation(fig, params_info, dataset_info)

    plt.tight_layout()
    return fig


def plot_umap_evolution(
    embedding_snapshots: List[Dict[str, Any]],
    labels: np.ndarray,
    algorithm_name: str = "",
    max_cols: int = 4,
    point_size: int = 3,
    random_state: int = 42,
    max_points: int = 5000,
    max_snapshots: int = 18,
    projection_mode: str = "shared",
    color_mode: str = "ground_truth",
    cell_weights_per_snapshot: Optional[List[np.ndarray]] = None,
    batch_labels: Optional[np.ndarray] = None,
    labels_per_snapshot: Optional[List[np.ndarray]] = None,
    params_info: Optional[Dict[str, Any]] = None,
    dataset_info: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plot gallery of UMAP projections across training snapshots."""
    from math import ceil

    fixed_label_mode = color_mode in {"ground_truth", "ground_truth_weighted", "batch"}
    plot_labels = batch_labels if color_mode == "batch" and batch_labels is not None else labels

    valid_snapshots_all = [
        {"snapshot": s, "orig_idx": i}
        for i, s in enumerate(embedding_snapshots)
        if s.get("embeddings") is not None and len(s["embeddings"]) > 0
    ]
    selected_indices = _select_snapshot_indices_for_gallery(
        [x["snapshot"] for x in valid_snapshots_all],
        max_snapshots=max_snapshots,
    )
    valid_snapshots = [valid_snapshots_all[i] for i in selected_indices]
    if not valid_snapshots:
        return None

    if projection_mode not in {"shared", "per_snapshot"}:
        projection_mode = "shared"

    n_snapshots = len(valid_snapshots)
    n_cols = min(n_snapshots, max_cols)
    n_rows = ceil(n_snapshots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    encoded_fixed = None
    unique_labels_fixed = None
    colors_fixed = None
    if fixed_label_mode:
        encoded_fixed, unique_labels_fixed, _ = _encode_labels(np.asarray(plot_labels))
        n_unique_fixed = len(unique_labels_fixed)
        if n_unique_fixed <= 10:
            cmap = plt.cm.tab10
        elif n_unique_fixed <= 20:
            cmap = plt.cm.tab20
        else:
            cmap = plt.cm.nipy_spectral
        colors_fixed = [cmap(i / max(n_unique_fixed - 1, 1)) for i in range(n_unique_fixed)]

    first_emb = np.asarray(valid_snapshots[0]["snapshot"]["embeddings"])
    n_cells = first_emb.shape[0]
    use_subsample = int(max_points) > 0 and n_cells > int(max_points)
    subsample_idx = None
    if use_subsample:
        rng = np.random.default_rng(int(random_state))
        subsample_idx = np.sort(rng.choice(n_cells, size=int(max_points), replace=False))
        if encoded_fixed is not None:
            encoded_fixed = encoded_fixed[subsample_idx]

    legend_unique_labels = None
    legend_colors = None

    shared_umap_fallback = False
    shared_umap_2d_by_snapshot: Optional[List[np.ndarray]] = None
    if projection_mode == "shared":
        shared_input = []
        for entry in valid_snapshots:
            emb = np.asarray(entry["snapshot"]["embeddings"])
            if use_subsample and subsample_idx is not None:
                emb = emb[subsample_idx]
            shared_input.append(emb)
        shared_umap_2d_by_snapshot, shared_umap_fallback = _compute_shared_umap_sequence(
            shared_input,
            random_state=random_state,
        )

    for idx, entry in enumerate(valid_snapshots):
        snapshot = entry["snapshot"]
        orig_idx = int(entry["orig_idx"])
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row][col]

        emb = snapshot["embeddings"]
        if use_subsample and subsample_idx is not None:
            emb = np.asarray(emb)[subsample_idx]
        epoch = snapshot.get("epoch", "?")
        phase = snapshot.get("phase", "")

        if shared_umap_2d_by_snapshot is not None and idx < len(shared_umap_2d_by_snapshot):
            umap_2d = shared_umap_2d_by_snapshot[idx]
        else:
            umap_2d, _ = _compute_umap_or_2d(emb, random_state=random_state)

        use_weighted_alpha = (
            color_mode == "ground_truth_weighted"
            and cell_weights_per_snapshot is not None
            and orig_idx < len(cell_weights_per_snapshot)
            and cell_weights_per_snapshot[orig_idx] is not None
        )
        alpha_arr = None
        if use_weighted_alpha:
            w = np.asarray(cell_weights_per_snapshot[orig_idx], dtype=np.float64)
            if use_subsample and subsample_idx is not None:
                w = w[subsample_idx]
            w_norm = _normalize_cell_weights_robust(w)
            alpha_arr = _weights_to_alpha_exponential(w_norm)

        encoded_curr = None
        unique_curr = None
        colors_curr = None
        if fixed_label_mode:
            encoded_curr = encoded_fixed
            unique_curr = unique_labels_fixed
            colors_curr = colors_fixed
        else:
            snap_labels = None
            if labels_per_snapshot is not None and orig_idx < len(labels_per_snapshot):
                snap_labels = labels_per_snapshot[orig_idx]
            if snap_labels is None:
                snap_labels = snapshot.get("pseudo_labels")
            if snap_labels is None:
                ax.text(0.5, 0.5, "No pseudo-labels", ha="center", va="center", fontsize=9)
                ax.set_title(f"Epoch {epoch} ({phase})", fontsize=9, fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            snap_labels = np.asarray(snap_labels)
            if len(snap_labels) != len(snapshot["embeddings"]):
                ax.text(0.5, 0.5, "Pseudo-label size mismatch", ha="center", va="center", fontsize=9)
                ax.set_title(f"Epoch {epoch} ({phase})", fontsize=9, fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            if use_subsample and subsample_idx is not None:
                snap_labels = snap_labels[subsample_idx]

            encoded_curr, unique_curr, _ = _encode_labels(snap_labels)
            n_unique_curr = len(unique_curr)
            if n_unique_curr <= 10:
                cmap_curr = plt.cm.tab10
            elif n_unique_curr <= 20:
                cmap_curr = plt.cm.tab20
            else:
                cmap_curr = plt.cm.nipy_spectral
            colors_curr = [cmap_curr(i / max(n_unique_curr - 1, 1)) for i in range(n_unique_curr)]

        if encoded_curr is None or unique_curr is None or colors_curr is None:
            ax.text(0.5, 0.5, "No labels", ha="center", va="center", fontsize=9)
            ax.set_title(f"Epoch {epoch} ({phase})", fontsize=9, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        if legend_unique_labels is None and len(unique_curr) <= 20:
            legend_unique_labels = unique_curr
            legend_colors = colors_curr

        for label_idx, label_name in enumerate(unique_curr):
            mask = encoded_curr == label_idx
            if mask.sum() == 0:
                continue

            if alpha_arr is not None:
                base_rgba = np.array(colors_curr[label_idx], dtype=np.float64)
                rgba = np.tile(base_rgba, (int(mask.sum()), 1))
                rgba[:, 3] = alpha_arr[mask]
                ax.scatter(
                    umap_2d[mask, 0],
                    umap_2d[mask, 1],
                    c=rgba,
                    s=float(point_size),
                    label=str(label_name) if idx == 0 else None,
                    rasterized=True,
                )
            else:
                ax.scatter(
                    umap_2d[mask, 0],
                    umap_2d[mask, 1],
                    c=[colors_curr[label_idx]],
                    s=point_size,
                    alpha=0.75,
                    label=str(label_name) if idx == 0 else None,
                    rasterized=True,
                )

        title = f"Epoch {epoch} ({phase})"
        if snapshot.get("snapshot_type") == "weight_refresh":
            wr_idx = snapshot.get("weight_refresh_index")
            if wr_idx is not None:
                title += f" | refresh#{wr_idx}"
        if color_mode == "pseudo_cluster":
            ax.set_title(
                f"{title} | k={len(unique_curr)}",
                fontsize=9,
                fontweight="bold",
            )
        else:
            ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n_snapshots, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row][col].axis("off")

    if legend_unique_labels is not None and legend_colors is not None and len(legend_unique_labels) <= 20:
        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=legend_colors[i],
                markersize=6,
                label=str(legend_unique_labels[i]),
            )
            for i in range(len(legend_unique_labels))
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(legend_unique_labels), 8),
            fontsize=7,
            bbox_to_anchor=(0.5, -0.02),
            frameon=True,
        )

    mode_suffix = {
        "ground_truth": "Ground Truth",
        "ground_truth_weighted": "Ground Truth + Cell Weights (Alpha)",
        "batch": "Batch",
        "pseudo_cluster": "Pseudo-Clusters",
    }.get(color_mode, "Ground Truth")
    proj_suffix = (
        "shared-UMAP"
        if projection_mode == "shared" and not shared_umap_fallback
        else ("shared-fallback-2D" if projection_mode == "shared" else "per-snapshot")
    )
    coverage_suffix = f"{n_snapshots}/{len(valid_snapshots_all)} snapshots"
    fig.suptitle(
        f"UMAP Evolution - {algorithm_name} ({mode_suffix} | {proj_suffix} | {coverage_suffix})",
        fontsize=13,
        fontweight="bold",
    )

    if params_info or dataset_info:
        _add_param_annotation(fig, params_info, dataset_info)

    legend_size = len(legend_unique_labels) if legend_unique_labels is not None else 0
    plt.tight_layout(rect=[0, 0.03 if legend_size <= 20 and legend_size > 0 else 0, 1, 0.96])
    return fig


def plot_marker_overlap_heatmap(
    overlap_matrix: pd.DataFrame,
    algorithm_name: str = "Algorithm",
    figsize: tuple = None,
) -> Optional[plt.Figure]:
    """Plot marker-overlap matrix (predicted clusters x gold cell types)."""
    if overlap_matrix is None or overlap_matrix.empty:
        return None

    n_rows, n_cols = overlap_matrix.shape
    if figsize is None:
        figsize = (max(8, n_cols * 0.9), max(5, n_rows * 0.5))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        overlap_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="lightgray",
        cbar_kws={"label": "Overlap Score (∩ / 100)"},
        ax=ax,
        vmin=0,
        vmax=1,
    )

    ax.set_title(
        f"Marker Gene Overlap - {algorithm_name}\n"
        "(Predicted Clusters x Gold-Standard Types)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Gold-Standard Cell Type", fontsize=11)
    ax.set_ylabel("Predicted Cluster", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    return fig
