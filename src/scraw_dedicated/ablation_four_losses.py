#!/usr/bin/env python3
"""Light ablation study with 4 loss combinations from shared epoch-30 latent."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .defaults import DEFAULT_PRESET_NAME
from .hyperparam_search import (
    DANN_BASE_OVERRIDES,
    _as_cli_value,
    _load_overrides_json,
    _read_metrics_from_run,
    _score_from_row,
    _subprocess_env,
)
from .presets import get_preset


@dataclass(frozen=True)
class LossVariant:
    name: str
    triplet_on: bool
    dann_on: bool


VARIANTS: List[LossVariant] = [
    LossVariant(name="01_nb_only", triplet_on=False, dann_on=False),
    LossVariant(name="02_triplet_nb", triplet_on=True, dann_on=False),
    LossVariant(name="03_dann_nb", triplet_on=False, dann_on=True),
    LossVariant(name="04_dann_nb_triplet", triplet_on=True, dann_on=True),
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description=(
            "Run 4-loss ablations (NB, Triplet+NB, DANN+NB, DANN+NB+Triplet) "
            "from a shared latent checkpoint at epoch 30."
        )
    )
    p.add_argument("--search-root", required=True, help="Source search root with ranked summary.")
    p.add_argument(
        "--output-root",
        default=None,
        help="Target output root (default: <search-root>/ablation_loss_impact).",
    )
    p.add_argument("--preset", default=None, help="Override preset (default from search manifest).")
    p.add_argument("--data", default=None, help="Override dataset path (default from search manifest).")
    p.add_argument("--seed", type=int, default=None, help="Override seed (default from search manifest).")
    p.add_argument("--device", default=None, help="Override device (default from search manifest).")
    p.add_argument("--python-bin", default=sys.executable, help="Python executable for subprocess calls.")
    p.add_argument(
        "--resume-epoch",
        type=int,
        default=30,
        help="Epoch used to build shared latent checkpoint (default: 30).",
    )
    p.add_argument(
        "--total-epochs",
        type=int,
        default=0,
        help="Total epochs for resumed runs (0 = from best config).",
    )
    p.add_argument("--snapshot-interval", type=int, default=10, help="Snapshot interval for figures.")
    p.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite output root before run (default: true).",
    )
    p.add_argument("--skip-existing", action="store_true", help="Skip completed runs.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    return p.parse_args(argv)


def _as_float(value: Any) -> float:
    """Convert value to float with NaN fallback."""
    try:
        return float(value)
    except Exception:
        return float("nan")


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    """Read CSV rows, empty when unavailable."""
    if not path.exists():
        return []
    try:
        with path.open("r", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _load_search_manifest(search_root: Path) -> Dict[str, Any]:
    """Load source search manifest."""
    path = search_root / "meta" / "search_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing search manifest: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid search manifest: {path}")
    return payload


def _read_ranked_rows(search_root: Path) -> List[Dict[str, Any]]:
    """Load ranked rows from search summaries."""
    ranked_path = search_root / "summaries" / "ranked_by_score.csv"
    ranked = _read_csv_rows(ranked_path)
    if ranked:
        return ranked
    all_path = search_root / "summaries" / "all_runs_metrics.csv"
    rows = _read_csv_rows(all_path)
    rows.sort(
        key=lambda r: (
            -_as_float(r.get("score")) if np.isfinite(_as_float(r.get("score"))) else float("inf"),
            str(r.get("group", "")),
            str(r.get("name", "")),
        )
    )
    return rows


def _pick_reference_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick best finite-score row."""
    for row in rows:
        if np.isfinite(_as_float(row.get("score"))):
            return row
    raise RuntimeError("No finite-score reference row found.")


def _build_cmd(
    *,
    python_bin: str,
    preset: str,
    data_path: Path,
    output_dir: Path,
    seed: int,
    device: str,
    metrics_only: bool,
    capture_snapshots: str,
    snapshot_interval: int,
    params: Dict[str, Any],
) -> List[str]:
    """Build one CLI command with explicit param overrides."""
    cmd = [
        str(python_bin),
        "-m",
        "scraw_dedicated.cli",
        "--preset",
        str(preset),
        "--data",
        str(data_path),
        "--output",
        str(output_dir),
        "--seed",
        str(int(seed)),
        "--device",
        str(device),
        "--capture-snapshots",
        str(capture_snapshots),
        "--snapshot-interval",
        str(max(1, int(snapshot_interval))),
    ]
    if metrics_only:
        cmd.append("--metrics-only")
    for key, value in sorted(params.items()):
        cmd.extend(["--param", f"{key}={_as_cli_value(value)}"])
    return cmd


def _read_final_clustering_rows(run_dir: Path) -> List[Dict[str, Any]]:
    """Read final clustering comparison for one run."""
    path = run_dir / "results" / "clustering_final" / "final_clustering_comparison.csv"
    rows = _read_csv_rows(path)
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "method": str(row.get("method", "")),
                "NMI": _as_float(row.get("NMI")),
                "ARI": _as_float(row.get("ARI")),
                "ACC": _as_float(row.get("ACC")),
                "BalancedACC": _as_float(row.get("BalancedACC")),
                "F1_Macro": _as_float(row.get("F1_Macro")),
                "RareACC": _as_float(row.get("RareACC")),
                "n_clusters_found": _as_float(row.get("n_clusters_found")),
                "resolution": row.get("resolution"),
            }
        )
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """Write CSV rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _has_completed_variant(run_dir: Path) -> bool:
    """Check essential outputs for one variant."""
    must_have = [
        run_dir / "results" / "analysis_results.csv",
        run_dir / "results" / "clustering_final" / "final_clustering_comparison.csv",
        run_dir / "figures" / "umaps" / "labels" / "umap_labels_snapshots_panels.png",
        run_dir / "figures" / "umaps" / "batch" / "umap_batch_snapshots_panels.png",
        run_dir / "figures" / "umaps" / "weights" / "cluster_component" / "umap_gradient_panels_cluster-component.png",
        run_dir / "figures" / "umaps" / "weights" / "density_component" / "umap_gradient_panels_density-component.png",
        run_dir / "figures" / "umaps" / "weights" / "fused_weight" / "umap_gradient_panels_cell.png",
        run_dir / "figures" / "loss" / "loss_curves_by_phase_scraw_run0.png",
        run_dir / "figures" / "metrics" / "metrics_evolution_by_epoch_scraw_run0.png",
    ]
    return all(p.exists() for p in must_have)


def _base_triplet_weight(base_params: Dict[str, Any]) -> float:
    """Resolve triplet-on reference weight."""
    w = float(base_params.get("rare_triplet_weight", 0.1) or 0.0)
    return w if w > 0.0 else 0.1


def _dann_on_params(base_params: Dict[str, Any], resume_epoch: int) -> Dict[str, Any]:
    """Resolve DANN-on params for ablation variants."""
    out: Dict[str, Any] = {}
    out["use_batch_conditioning"] = True
    out["batch_correction_key"] = str(base_params.get("batch_correction_key", DANN_BASE_OVERRIDES["batch_correction_key"]))
    adv_w = float(base_params.get("adversarial_batch_weight", 0.0) or 0.0)
    out["adversarial_batch_weight"] = adv_w if adv_w > 0.0 else float(DANN_BASE_OVERRIDES["adversarial_batch_weight"])
    out["adversarial_lambda"] = float(base_params.get("adversarial_lambda", DANN_BASE_OVERRIDES["adversarial_lambda"]) or DANN_BASE_OVERRIDES["adversarial_lambda"])
    out["adversarial_start_epoch"] = int(resume_epoch)
    out["adversarial_ramp_epochs"] = int(
        base_params.get("adversarial_ramp_epochs", DANN_BASE_OVERRIDES["adversarial_ramp_epochs"])
        or DANN_BASE_OVERRIDES["adversarial_ramp_epochs"]
    )
    out["mmd_batch_weight"] = 0.0
    return out


def _best_method_rows(method_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pick best final clustering method by ARI for each run."""
    by_run: Dict[str, List[Dict[str, Any]]] = {}
    for row in method_rows:
        key = str(row.get("name", ""))
        by_run.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for name, rows in sorted(by_run.items()):
        best = None
        for row in rows:
            ari = _as_float(row.get("ARI"))
            if not np.isfinite(ari):
                continue
            if best is None or ari > _as_float(best.get("ARI")):
                best = row
        hdb = next((r for r in rows if str(r.get("method")) == "hdbscan_final"), None)
        lei = next((r for r in rows if str(r.get("method")) == "leiden_target14_final"), None)
        out.append(
            {
                "name": name,
                "best_method_by_ARI": "" if best is None else str(best.get("method")),
                "best_ARI": np.nan if best is None else _as_float(best.get("ARI")),
                "best_NMI": np.nan if best is None else _as_float(best.get("NMI")),
                "best_ACC": np.nan if best is None else _as_float(best.get("ACC")),
                "best_n_clusters_found": np.nan if best is None else _as_float(best.get("n_clusters_found")),
                "hdbscan_ARI": np.nan if hdb is None else _as_float(hdb.get("ARI")),
                "leiden_target14_ARI": np.nan if lei is None else _as_float(lei.get("ARI")),
                "delta_leiden_minus_hdbscan_ARI": (
                    np.nan
                    if hdb is None or lei is None
                    else (_as_float(lei.get("ARI")) - _as_float(hdb.get("ARI")))
                ),
                "run_dir": "" if best is None else str(best.get("run_dir", "")),
            }
        )
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    search_root = Path(args.search_root).expanduser().resolve()
    if not search_root.exists():
        raise FileNotFoundError(f"Search root not found: {search_root}")

    manifest_src = _load_search_manifest(search_root)
    ranked_rows = _read_ranked_rows(search_root)
    ref_row = _pick_reference_row(ranked_rows)

    preset = str(args.preset or manifest_src.get("preset", DEFAULT_PRESET_NAME))
    data_path = Path(args.data or manifest_src.get("data", "")).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    seed = int(args.seed if args.seed is not None else manifest_src.get("seed", 42))
    device = str(args.device or manifest_src.get("device", "auto") or "auto")
    resume_epoch = int(max(1, args.resume_epoch))

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else (search_root / "ablation_loss_impact").resolve()
    )
    common_root = output_root / "common_phase1"
    runs_root = output_root / "runs"
    logs_root = output_root / "logs"
    summaries_root = output_root / "summaries"
    meta_root = output_root / "meta"

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    for d in (output_root, common_root, runs_root, logs_root, summaries_root, meta_root):
        d.mkdir(parents=True, exist_ok=True)

    preset_params = dict(get_preset(preset).algorithm_params)
    ref_overrides = _load_overrides_json(ref_row.get("overrides_json", "{}"))
    base_params = dict(preset_params)
    base_params.update(ref_overrides)

    total_epochs = int(args.total_epochs if args.total_epochs > 0 else base_params.get("epochs", 120))
    if total_epochs <= resume_epoch:
        raise ValueError(
            f"total_epochs ({total_epochs}) must be > resume_epoch ({resume_epoch})."
        )

    shared_ckpt_dir = Path(
        os.environ.get(
            "SCRAW_ABLATION_SHARED_CKPT_DIR",
            "/tmp/scraw_ablation_shared_checkpoints",
        )
    ).expanduser().resolve()
    shared_ckpt_dir.mkdir(parents=True, exist_ok=True)
    safe_name = output_root.name.replace(" ", "_")
    checkpoint_path = shared_ckpt_dir / f"{safe_name}_epoch{resume_epoch}.pt"
    checkpoint_artifact = common_root / f"shared_state_epoch{resume_epoch}.pt"

    trunk_params = dict(base_params)
    trunk_params.update(
        {
            "reconstruction_distribution": "nb",
            "epochs": total_epochs,
            "warmup_epochs": resume_epoch,
            "rare_triplet_weight": 0.0,
            "rare_triplet_start_epoch": resume_epoch,
            "use_batch_conditioning": False,
            "adversarial_batch_weight": 0.0,
            "mmd_batch_weight": 0.0,
            "capture_embedding_snapshots": True,
            "snapshot_interval_epochs": int(max(1, args.snapshot_interval)),
            "stop_after_epoch": int(resume_epoch - 1),
            "save_checkpoint_path": str(checkpoint_path),
            "resume_checkpoint_path": "",
            "resume_load_optimizer": True,
        }
    )

    manifest = {
        "source": "ablation_four_losses",
        "search_root": str(search_root),
        "preset": preset,
        "data": str(data_path),
        "seed": seed,
        "device": device,
        "resume_epoch": resume_epoch,
        "total_epochs": total_epochs,
        "snapshot_interval_epochs": int(max(1, args.snapshot_interval)),
        "reference_search_run": {
            "group": ref_row.get("group"),
            "name": ref_row.get("name"),
            "score": _as_float(ref_row.get("score")),
            "run_dir": ref_row.get("run_dir"),
            "overrides": ref_overrides,
        },
        "shared_checkpoint": str(checkpoint_path),
        "shared_checkpoint_artifact": str(checkpoint_artifact),
        "variants": [
            {"name": v.name, "triplet_on": v.triplet_on, "dann_on": v.dann_on}
            for v in VARIANTS
        ],
    }
    manifest_path = meta_root / "ablation_four_losses_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    env = _subprocess_env()
    env.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl_scraw")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_scraw")

    # Step 1: build shared latent/checkpoint at epoch 30.
    common_run_dir = common_root / "run"
    common_log = logs_root / "00_common_phase1.log"
    common_run_dir.mkdir(parents=True, exist_ok=True)
    common_log.parent.mkdir(parents=True, exist_ok=True)
    common_cmd = _build_cmd(
        python_bin=args.python_bin,
        preset=preset,
        data_path=data_path,
        output_dir=common_run_dir,
        seed=seed,
        device=device,
        metrics_only=True,
        capture_snapshots="on",
        snapshot_interval=args.snapshot_interval,
        params=trunk_params,
    )
    print("[COMMON] Shared phase-1 checkpoint run")
    print("  " + " ".join(shlex.quote(c) for c in common_cmd))
    if args.dry_run:
        pass
    else:
        with common_log.open("w") as fh:
            proc = subprocess.run(common_cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        if int(proc.returncode) != 0:
            raise RuntimeError(f"Shared checkpoint run failed (rc={proc.returncode}). See {common_log}")
        if not checkpoint_path.exists():
            raise RuntimeError(f"Shared checkpoint missing: {checkpoint_path}")
        checkpoint_artifact.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_path, checkpoint_artifact)

    # Step 2: run 4 variants from shared state.
    variant_rows: List[Dict[str, Any]] = []
    method_rows: List[Dict[str, Any]] = []
    failures = 0

    triplet_on_weight = _base_triplet_weight(base_params)
    dann_on_defaults = _dann_on_params(base_params, resume_epoch=resume_epoch)

    for idx, variant in enumerate(VARIANTS, start=1):
        run_dir = runs_root / variant.name
        log_file = logs_root / f"{variant.name}.log"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        params = dict(base_params)
        params.update(
            {
                "reconstruction_distribution": "nb",
                "epochs": total_epochs,
                "warmup_epochs": resume_epoch,
                "capture_embedding_snapshots": True,
                "snapshot_interval_epochs": int(max(1, args.snapshot_interval)),
                "stop_after_epoch": -1,
                "save_checkpoint_path": "",
                "resume_checkpoint_path": str(checkpoint_path),
                "rare_triplet_start_epoch": int(resume_epoch),
            }
        )

        if variant.triplet_on:
            params["rare_triplet_weight"] = float(triplet_on_weight)
        else:
            params["rare_triplet_weight"] = 0.0

        if variant.dann_on:
            params.update(dann_on_defaults)
            params["resume_load_optimizer"] = False
        else:
            params.update(
                {
                    "use_batch_conditioning": False,
                    "adversarial_batch_weight": 0.0,
                    "mmd_batch_weight": 0.0,
                }
            )
            params["resume_load_optimizer"] = True

        cmd = _build_cmd(
            python_bin=args.python_bin,
            preset=preset,
            data_path=data_path,
            output_dir=run_dir,
            seed=seed,
            device=device,
            metrics_only=False,
            capture_snapshots="on",
            snapshot_interval=args.snapshot_interval,
            params=params,
        )
        print(f"[{idx:02d}/04] {variant.name}")
        print("  " + " ".join(shlex.quote(c) for c in cmd))

        if args.skip_existing and _has_completed_variant(run_dir):
            rc = 0
            status = "existing"
        elif args.dry_run:
            rc = 0
            status = "dry_run"
        else:
            if not checkpoint_path.exists():
                if checkpoint_artifact.exists():
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(checkpoint_artifact, checkpoint_path)
                else:
                    raise FileNotFoundError(
                        f"Shared checkpoint vanished and no artifact found: {checkpoint_path}"
                    )
            with log_file.open("w") as fh:
                proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
            rc = int(proc.returncode)
            status = "ok" if rc == 0 else f"failed_{rc}"

        metrics = _read_metrics_from_run(run_dir)
        if rc != 0:
            failures += 1

        row = {
            "name": variant.name,
            "triplet_on": variant.triplet_on,
            "dann_on": variant.dann_on,
            "status": status if status != "ok" else metrics.get("status", "ok"),
            "run_dir": str(run_dir),
            "log_file": str(log_file),
            "NMI": metrics.get("NMI"),
            "ARI": metrics.get("ARI"),
            "ACC": metrics.get("ACC"),
            "F1_Macro": metrics.get("F1_Macro"),
            "BalancedACC": metrics.get("BalancedACC"),
            "RareACC": metrics.get("RareACC"),
            "Silhouette": metrics.get("Silhouette"),
            "n_clusters_found": metrics.get("n_clusters_found"),
            "runtime": metrics.get("runtime"),
            "figures_complete": _has_completed_variant(run_dir),
        }
        row["score"] = _score_from_row(row)
        variant_rows.append(row)

        final_rows = _read_final_clustering_rows(run_dir)
        for fr in final_rows:
            method_rows.append(
                {
                    "name": variant.name,
                    "triplet_on": variant.triplet_on,
                    "dann_on": variant.dann_on,
                    "status": row["status"],
                    "run_dir": str(run_dir),
                    **fr,
                }
            )

    metrics_fields = [
        "name",
        "triplet_on",
        "dann_on",
        "status",
        "run_dir",
        "log_file",
        "NMI",
        "ARI",
        "ACC",
        "F1_Macro",
        "BalancedACC",
        "RareACC",
        "Silhouette",
        "n_clusters_found",
        "runtime",
        "score",
        "figures_complete",
    ]
    metrics_csv = summaries_root / "ablation_four_losses_metrics.csv"
    _write_csv(metrics_csv, variant_rows, metrics_fields)

    ranked = sorted(
        variant_rows,
        key=lambda r: (
            -_as_float(r.get("score")) if np.isfinite(_as_float(r.get("score"))) else float("inf"),
            str(r.get("name", "")),
        ),
    )
    ranked_csv = summaries_root / "ablation_four_losses_ranked_by_score.csv"
    _write_csv(ranked_csv, ranked, metrics_fields)

    method_fields = [
        "name",
        "triplet_on",
        "dann_on",
        "status",
        "method",
        "NMI",
        "ARI",
        "ACC",
        "BalancedACC",
        "F1_Macro",
        "RareACC",
        "n_clusters_found",
        "resolution",
        "run_dir",
    ]
    methods_csv = summaries_root / "final_clustering_all_methods.csv"
    _write_csv(methods_csv, method_rows, method_fields)

    best_rows = _best_method_rows(method_rows)
    best_fields = [
        "name",
        "best_method_by_ARI",
        "best_ARI",
        "best_NMI",
        "best_ACC",
        "best_n_clusters_found",
        "hdbscan_ARI",
        "leiden_target14_ARI",
        "delta_leiden_minus_hdbscan_ARI",
        "run_dir",
    ]
    best_csv = summaries_root / "final_clustering_best_method_by_run.csv"
    _write_csv(best_csv, best_rows, best_fields)

    summary = {
        "status": "ok" if failures == 0 else "partial_failures",
        "n_variants": len(VARIANTS),
        "n_failures": failures,
        "output_root": str(output_root),
        "manifest_json": str(manifest_path),
        "metrics_csv": str(metrics_csv),
        "ranked_csv": str(ranked_csv),
        "final_clustering_methods_csv": str(methods_csv),
        "final_clustering_best_csv": str(best_csv),
        "common_checkpoint": str(checkpoint_path),
    }
    meta_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_path = meta_root / "ablation_four_losses_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nCompleted.")
    print(f" - {metrics_csv}")
    print(f" - {ranked_csv}")
    print(f" - {methods_csv}")
    print(f" - {best_csv}")
    print(f" - {summary_path}")

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
