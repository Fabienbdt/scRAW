#!/usr/bin/env python3
"""Regenerate ablation_loss_impact with full figures and final clustering summaries.

This script reuses the best run from an existing hyperparameter search result
directory, rebuilds the loss-ablation plan, and reruns each ablation in
figure-rich mode (snapshots enabled).
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .hyperparam_search import (
    RunSpec,
    _as_cli_value,
    _build_loss_ablation_plan,
    _load_overrides_json,
    _read_metrics_from_run,
    _score_from_row,
    _subprocess_env,
)


EXPECTED_FIGURES_REL = [
    Path("figures/umaps/labels/umap_labels_snapshots_panels.png"),
    Path("figures/umaps/batch/umap_batch_snapshots_panels.png"),
    Path("figures/umaps/weights/cluster_component/umap_gradient_panels_cluster-component.png"),
    Path("figures/umaps/weights/density_component/umap_gradient_panels_density-component.png"),
    Path("figures/umaps/weights/fused_weight/umap_gradient_panels_cell.png"),
    Path("figures/loss/loss_curves_by_phase_scraw_run0.png"),
    Path("figures/loss/loss_curves_timeline_scraw_run0.png"),
    Path("figures/metrics/metrics_evolution_by_epoch_scraw_run0.png"),
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description=(
            "Regenerate loss-ablation runs with figures + final clustering "
            "comparisons (HDBSCAN vs Leiden target=14)."
        )
    )
    p.add_argument(
        "--search-root",
        required=True,
        help="Existing hparam search root (contains meta/search_manifest.json).",
    )
    p.add_argument(
        "--output-root",
        default=None,
        help="Target ablation folder (default: <search-root>/ablation_loss_impact).",
    )
    p.add_argument("--preset", default=None, help="Override preset (default: from search manifest).")
    p.add_argument("--data", default=None, help="Override dataset path (default: from search manifest).")
    p.add_argument("--seed", type=int, default=None, help="Override seed (default: from search manifest).")
    p.add_argument(
        "--device",
        default=None,
        help="Override device (default: from search manifest, fallback 'auto').",
    )
    p.add_argument("--python-bin", default=sys.executable, help="Python executable for subprocess runs.")
    p.add_argument("--snapshot-interval", type=int, default=10, help="Snapshot interval in epochs.")
    p.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite output root before starting (default: true).",
    )
    p.add_argument("--skip-existing", action="store_true", help="Skip runs with complete artifacts.")
    p.add_argument("--max-runs", type=int, default=0, help="Limit number of runs (0 = all).")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing runs.")
    return p.parse_args(argv)


def _as_float(value: Any) -> float:
    """Convert value to float with NaN fallback."""
    try:
        return float(value)
    except Exception:
        return float("nan")


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    """Read CSV rows from path, empty list if missing/invalid."""
    if not path.exists():
        return []
    try:
        with path.open("r", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _load_search_manifest(search_root: Path) -> Dict[str, Any]:
    """Load search manifest JSON."""
    manifest_path = search_root / "meta" / "search_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing search manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid search manifest content: {manifest_path}")
    return payload


def _read_ranked_rows(search_root: Path) -> List[Dict[str, Any]]:
    """Load ranked rows from original search summaries."""
    ranked_csv = search_root / "summaries" / "ranked_by_score.csv"
    rows = _read_csv_rows(ranked_csv)
    if rows:
        return rows

    all_csv = search_root / "summaries" / "all_runs_metrics.csv"
    rows = _read_csv_rows(all_csv)
    for row in rows:
        score = _as_float(row.get("score"))
        if not np.isfinite(score):
            row["score"] = _score_from_row(
                {
                    "NMI": _as_float(row.get("NMI")),
                    "ARI": _as_float(row.get("ARI")),
                    "ACC": _as_float(row.get("ACC")),
                    "F1_Macro": _as_float(row.get("F1_Macro")),
                    "BalancedACC": _as_float(row.get("BalancedACC")),
                }
            )
    rows.sort(
        key=lambda r: (
            -_as_float(r.get("score")) if np.isfinite(_as_float(r.get("score"))) else float("inf"),
            str(r.get("group", "")),
            str(r.get("name", "")),
        )
    )
    return rows


def _select_reference_row(ranked_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick best finite-score row from ranked rows."""
    for row in ranked_rows:
        score = _as_float(row.get("score"))
        if np.isfinite(score):
            return row
    raise RuntimeError("No finite-score run found in ranked rows.")


def _build_cli_cmd(
    python_bin: str,
    preset: str,
    data_path: Path,
    run_dir: Path,
    seed: int,
    device: str,
    snapshot_interval: int,
    spec: RunSpec,
) -> List[str]:
    """Build one figure-rich scRAW CLI command."""
    cmd = [
        str(python_bin),
        "-m",
        "scraw_dedicated.cli",
        "--preset",
        str(preset),
        "--data",
        str(data_path),
        "--output",
        str(run_dir),
        "--seed",
        str(int(seed)),
        "--device",
        str(device),
        "--capture-snapshots",
        "on",
        "--snapshot-interval",
        str(max(1, int(snapshot_interval))),
    ]
    for key, value in sorted(spec.overrides.items()):
        cmd.extend(["--param", f"{key}={_as_cli_value(value)}"])
    return cmd


def _has_complete_artifacts(run_dir: Path) -> bool:
    """Check if run already has metrics + final clustering + required figures."""
    must_have = [
        run_dir / "results" / "analysis_results.csv",
        run_dir / "results" / "clustering_final" / "final_clustering_comparison.csv",
    ]
    must_have.extend(run_dir / rel for rel in EXPECTED_FIGURES_REL)
    return all(p.exists() for p in must_have)


def _read_final_clustering_rows(run_dir: Path) -> List[Dict[str, Any]]:
    """Read final clustering comparison rows from one run."""
    path = run_dir / "results" / "clustering_final" / "final_clustering_comparison.csv"
    rows = _read_csv_rows(path)
    if not rows:
        return []
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
    """Write rows to CSV with stable field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _method_summary_rows(method_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate final clustering metrics per method across runs."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in method_rows:
        m = str(row.get("method", ""))
        groups.setdefault(m, []).append(row)

    out: List[Dict[str, Any]] = []
    for method, rows in sorted(groups.items()):
        ari_vals = [float(r.get("ARI", np.nan)) for r in rows if np.isfinite(_as_float(r.get("ARI")))]
        nmi_vals = [float(r.get("NMI", np.nan)) for r in rows if np.isfinite(_as_float(r.get("NMI")))]
        acc_vals = [float(r.get("ACC", np.nan)) for r in rows if np.isfinite(_as_float(r.get("ACC")))]
        out.append(
            {
                "method": method,
                "n_runs": len(rows),
                "ARI_mean": float(np.mean(ari_vals)) if ari_vals else np.nan,
                "ARI_std": float(np.std(ari_vals)) if ari_vals else np.nan,
                "NMI_mean": float(np.mean(nmi_vals)) if nmi_vals else np.nan,
                "ACC_mean": float(np.mean(acc_vals)) if acc_vals else np.nan,
            }
        )
    return out


def _best_method_rows(method_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pick best final clustering method (by ARI) for each run."""
    by_run: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in method_rows:
        key = (str(row.get("group", "")), str(row.get("name", "")))
        by_run.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for (group, name), rows in sorted(by_run.items()):
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
                "group": group,
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

    search_manifest = _load_search_manifest(search_root)
    ranked_rows = _read_ranked_rows(search_root)
    if not ranked_rows:
        raise RuntimeError("No ranked rows found in search summaries.")
    ref_row = _select_reference_row(ranked_rows)
    ref_overrides = _load_overrides_json(ref_row.get("overrides_json", "{}"))
    plan = _build_loss_ablation_plan(ref_overrides)
    if args.max_runs and args.max_runs > 0:
        plan = plan[: int(args.max_runs)]
    if not plan:
        raise RuntimeError("Ablation plan is empty.")

    preset = str(args.preset or search_manifest.get("preset", "baron_best"))
    data_path = Path(args.data or search_manifest.get("data", "")).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    seed = int(args.seed if args.seed is not None else search_manifest.get("seed", 42))
    device = str(args.device or search_manifest.get("device", "auto") or "auto")

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else (search_root / "ablation_loss_impact").resolve()
    )
    runs_root = output_root / "runs"
    logs_root = output_root / "logs"
    summaries_root = output_root / "summaries"
    meta_root = output_root / "meta"

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    for d in (output_root, runs_root, logs_root, summaries_root, meta_root):
        d.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": "regenerate_ablation_figures",
        "search_root": str(search_root),
        "search_manifest": str(search_root / "meta" / "search_manifest.json"),
        "preset": preset,
        "data": str(data_path),
        "seed": seed,
        "device": device,
        "snapshot_interval_epochs": int(max(1, args.snapshot_interval)),
        "overwrite": bool(args.overwrite),
        "skip_existing": bool(args.skip_existing),
        "reference_search_run": {
            "group": ref_row.get("group"),
            "name": ref_row.get("name"),
            "score": _as_float(ref_row.get("score")),
            "run_dir": ref_row.get("run_dir"),
            "overrides": ref_overrides,
        },
        "expected_figures_rel": [str(p) for p in EXPECTED_FIGURES_REL],
        "n_planned_runs": len(plan),
        "runs": [{"group": s.group, "name": s.name, "overrides": s.overrides} for s in plan],
    }
    manifest_path = meta_root / "ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    env = _subprocess_env()
    env.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl_scraw")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_scraw")

    run_rows: List[Dict[str, Any]] = []
    method_rows: List[Dict[str, Any]] = []
    failures = 0

    print(f"Search root: {search_root}")
    print(f"Output root: {output_root}")
    print(f"Reference run: {ref_row.get('group')}/{ref_row.get('name')} (score={ref_row.get('score')})")
    print(f"Planned runs: {len(plan)}")

    for idx, spec in enumerate(plan, start=1):
        run_dir = runs_root / spec.group / spec.name
        log_file = logs_root / spec.group / f"{spec.name}.log"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = _build_cli_cmd(
            python_bin=args.python_bin,
            preset=preset,
            data_path=data_path,
            run_dir=run_dir,
            seed=seed,
            device=device,
            snapshot_interval=args.snapshot_interval,
            spec=spec,
        )
        cmd_text = " ".join(shlex.quote(c) for c in cmd)
        print(f"[{idx:03d}/{len(plan):03d}] {spec.group}/{spec.name}")
        print(f"  {cmd_text}")

        if args.skip_existing and _has_complete_artifacts(run_dir):
            rc = 0
            status = "existing"
        elif args.dry_run:
            rc = 0
            status = "dry_run"
        else:
            with log_file.open("w") as fh:
                proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
            rc = int(proc.returncode)
            status = "ok" if rc == 0 else f"failed_{rc}"

        metrics = _read_metrics_from_run(run_dir)
        if rc != 0:
            failures += 1

        row: Dict[str, Any] = {
            "group": spec.group,
            "name": spec.name,
            "status": status if status != "ok" else metrics.get("status", "ok"),
            "run_dir": str(run_dir),
            "log_file": str(log_file),
            "overrides_json": json.dumps(spec.overrides, sort_keys=True),
            "NMI": metrics.get("NMI"),
            "ARI": metrics.get("ARI"),
            "ACC": metrics.get("ACC"),
            "F1_Macro": metrics.get("F1_Macro"),
            "BalancedACC": metrics.get("BalancedACC"),
            "RareACC": metrics.get("RareACC"),
            "Silhouette": metrics.get("Silhouette"),
            "n_clusters_found": metrics.get("n_clusters_found"),
            "runtime": metrics.get("runtime"),
            "figures_complete": _has_complete_artifacts(run_dir),
        }
        row["score"] = _score_from_row(row)
        run_rows.append(row)

        final_rows = _read_final_clustering_rows(run_dir)
        for frow in final_rows:
            frow_out = {
                "group": spec.group,
                "name": spec.name,
                "status": row["status"],
                "run_dir": str(run_dir),
                **frow,
            }
            method_rows.append(frow_out)

    run_fields = [
        "group",
        "name",
        "status",
        "run_dir",
        "log_file",
        "overrides_json",
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
    all_runs_csv = summaries_root / "ablation_all_runs.csv"
    _write_csv(all_runs_csv, run_rows, run_fields)

    ranked = sorted(
        run_rows,
        key=lambda r: (
            -_as_float(r.get("score")) if np.isfinite(_as_float(r.get("score"))) else float("inf"),
            str(r.get("group", "")),
            str(r.get("name", "")),
        ),
    )
    ranked_csv = summaries_root / "ablation_ranked_by_score.csv"
    _write_csv(ranked_csv, ranked, run_fields)

    method_fields = [
        "group",
        "name",
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
        "group",
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

    method_summary = _method_summary_rows(method_rows)
    method_summary_fields = ["method", "n_runs", "ARI_mean", "ARI_std", "NMI_mean", "ACC_mean"]
    method_summary_csv = summaries_root / "final_clustering_method_summary.csv"
    _write_csv(method_summary_csv, method_summary, method_summary_fields)

    summary = {
        "status": "ok" if failures == 0 else "partial_failures",
        "n_runs": len(run_rows),
        "n_failures": failures,
        "output_root": str(output_root),
        "manifest_json": str(manifest_path),
        "ablation_all_runs_csv": str(all_runs_csv),
        "ablation_ranked_csv": str(ranked_csv),
        "final_clustering_all_methods_csv": str(methods_csv),
        "final_clustering_best_method_by_run_csv": str(best_csv),
        "final_clustering_method_summary_csv": str(method_summary_csv),
    }
    summary_path = meta_root / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nCompleted.")
    print(f" - {all_runs_csv}")
    print(f" - {ranked_csv}")
    print(f" - {methods_csv}")
    print(f" - {best_csv}")
    print(f" - {method_summary_csv}")
    print(f" - {summary_path}")

    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
