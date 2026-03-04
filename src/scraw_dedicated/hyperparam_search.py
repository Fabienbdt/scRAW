#!/usr/bin/env python3
"""Comprehensive hyperparameter search runner for standalone scRAW (metrics only)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .presets import get_preset


@dataclass(frozen=True)
class RunSpec:
    """One concrete run in the hyperparameter search plan."""

    group: str
    name: str
    overrides: Dict[str, Any]


SINGLE_PARAM_SWEEP: Dict[str, List[Any]] = {
    "lr": [5e-4, 1e-3, 2e-3],
    "batch_size": [128, 256, 512],
    "dropout": [0.05, 0.1, 0.2],
    "z_dim": [64, 128, 256],
    "weight_exponent": [0.1, 0.2, 0.4],
    "cluster_density_alpha": [0.4, 0.6, 0.8],
    "rare_triplet_weight": [0.05, 0.1, 0.2],
    "rare_triplet_start_epoch": [25, 35, 45],
    "hdbscan_min_cluster_size": [3, 4, 6],
    "hdbscan_min_samples": [1, 2, 4],
    "masking_rate": [0.1, 0.2, 0.3],
    "masked_recon_weight": [0.5, 0.75, 1.0],
}

PAIRWISE_SWEEP: List[Tuple[str, List[Any], str, List[Any]]] = [
    ("lr", [5e-4, 1e-3, 2e-3], "batch_size", [128, 256, 512]),
    ("cluster_density_alpha", [0.4, 0.6, 0.8], "rare_triplet_weight", [0.05, 0.1, 0.2]),
    ("hdbscan_min_cluster_size", [3, 4, 6], "hdbscan_min_samples", [1, 2, 4]),
]

BATCH_CORRECTION_SWEEP: Dict[str, Dict[str, Any]] = {
    "batch_off": {
        "use_batch_conditioning": False,
        "adversarial_batch_weight": 0.0,
        "mmd_batch_weight": 0.0,
    },
    "dann_light": {
        "use_batch_conditioning": True,
        "batch_correction_key": "batch",
        "adversarial_batch_weight": 0.1,
        "adversarial_lambda": 1.0,
        "adversarial_start_epoch": 10,
        "adversarial_ramp_epochs": 20,
        "mmd_batch_weight": 0.0,
    },
    "dann_strong": {
        "use_batch_conditioning": True,
        "batch_correction_key": "batch",
        "adversarial_batch_weight": 0.2,
        "adversarial_lambda": 1.0,
        "adversarial_start_epoch": 10,
        "adversarial_ramp_epochs": 20,
        "mmd_batch_weight": 0.0,
    },
    "mmd_light": {
        "use_batch_conditioning": True,
        "batch_correction_key": "batch",
        "adversarial_batch_weight": 0.0,
        "mmd_batch_weight": 0.05,
    },
    "mmd_strong": {
        "use_batch_conditioning": True,
        "batch_correction_key": "batch",
        "adversarial_batch_weight": 0.0,
        "mmd_batch_weight": 0.1,
    },
    "hybrid_dann_mmd": {
        "use_batch_conditioning": True,
        "batch_correction_key": "batch",
        "adversarial_batch_weight": 0.1,
        "adversarial_lambda": 1.0,
        "adversarial_start_epoch": 10,
        "adversarial_ramp_epochs": 20,
        "mmd_batch_weight": 0.05,
    },
}

DANN_BASE_OVERRIDES: Dict[str, Any] = {
    # Conservative DANN anchor: weaker signal + later start to reduce clustering drift.
    "use_batch_conditioning": True,
    "batch_correction_key": "batch",
    "adversarial_batch_weight": 0.05,
    "adversarial_lambda": 0.5,
    "adversarial_start_epoch": 30,
    "adversarial_ramp_epochs": 40,
    "mmd_batch_weight": 0.0,
}

DANN_SINGLE_PARAM_SWEEP: Dict[str, List[Any]] = {
    "adversarial_batch_weight": [0.02, 0.05, 0.08, 0.12],
    "adversarial_lambda": [0.25, 0.5, 0.75, 1.0],
    "adversarial_start_epoch": [10, 20, 30, 40],
    "adversarial_ramp_epochs": [10, 20, 40, 60],
    "weight_exponent": [0.2, 0.3, 0.4, 0.5],
    "rare_triplet_weight": [0.05, 0.1, 0.15],
    "rare_triplet_start_epoch": [25, 35, 45],
    "cluster_density_alpha": [0.4, 0.6, 0.8],
    "lr": [5e-4, 1e-3, 2e-3],
    "batch_size": [128, 256, 512],
    "masking_rate": [0.1, 0.2, 0.3],
    "masked_recon_weight": [0.5, 0.75, 1.0],
    "z_dim": [64, 128, 256],
}

DANN_PAIRWISE_SWEEP: List[Tuple[str, List[Any], str, List[Any]]] = [
    (
        "adversarial_batch_weight",
        [0.02, 0.05, 0.08, 0.12],
        "adversarial_start_epoch",
        [10, 20, 30, 40],
    ),
    (
        "adversarial_batch_weight",
        [0.02, 0.05, 0.08, 0.12],
        "adversarial_lambda",
        [0.25, 0.5, 0.75, 1.0],
    ),
    (
        "adversarial_batch_weight",
        [0.02, 0.05, 0.08, 0.12],
        "rare_triplet_weight",
        [0.05, 0.1, 0.15],
    ),
    (
        "adversarial_batch_weight",
        [0.02, 0.05, 0.08, 0.12],
        "weight_exponent",
        [0.2, 0.3, 0.4, 0.5],
    ),
    (
        "adversarial_start_epoch",
        [10, 20, 30, 40],
        "adversarial_ramp_epochs",
        [10, 20, 40, 60],
    ),
    (
        "adversarial_batch_weight",
        [0.02, 0.05, 0.08, 0.12],
        "cluster_density_alpha",
        [0.4, 0.6, 0.8],
    ),
]

DANN_CONTROL_SWEEP: Dict[str, Dict[str, Any]] = {
    # Controls around DANN anchor to diagnose interactions.
    "dann_triplet_off": {"rare_triplet_weight": 0.0},
    "dann_nb_off_mse": {"reconstruction_distribution": "mse"},
    "dann_weighted_uniform": {
        "min_cell_weight": 1.0,
        "max_cell_weight": 1.0,
        "rare_triplet_min_weight": 0.0,
    },
    "dann_plus_mmd_light": {"mmd_batch_weight": 0.05},
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI args for hyperparameter search."""
    p = argparse.ArgumentParser(
        description="Comprehensive scRAW hyperparameter search (always metrics-only)."
    )
    p.add_argument("--preset", required=True, choices=["baron_best", "pancreas_best"])
    p.add_argument("--data", required=True, help="Input .h5ad path")
    p.add_argument("--output-root", required=True, help="Root output directory for this search")
    p.add_argument("--device", default="cpu", help="cpu|cuda|mps|auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--groups",
        default="baseline,single,pairwise,batch",
        help="Comma list among: baseline,single,pairwise,batch,dann",
    )
    p.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Optional hard cap on number of runs (0 = no cap).",
    )
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip runs that already contain results/analysis_results.csv (default: true).",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Force rerun even when previous metrics exist.",
    )
    p.add_argument(
        "--run-loss-ablation",
        action="store_true",
        default=True,
        dest="run_loss_ablation",
        help="After search, run a dedicated loss ablation study from best config (default: on).",
    )
    p.add_argument(
        "--no-loss-ablation",
        action="store_false",
        dest="run_loss_ablation",
        help="Disable post-search loss ablation study.",
    )
    return p.parse_args(argv)


def _value_to_token(value: Any) -> str:
    """Convert a value to a filesystem-friendly token."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.6g}"
        return text.replace("-", "m").replace(".", "p")
    return str(value).replace("/", "_").replace(" ", "_")


def _as_cli_value(value: Any) -> str:
    """Format a Python value to `--param key=value` text."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _metric_float(row: Dict[str, str], key: str) -> float:
    """Read metric key from CSV row as float."""
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def _score_from_row(row: Dict[str, Any]) -> float:
    """Primary ranking score used in summaries."""
    vals = [row.get("NMI"), row.get("ARI"), row.get("ACC"), row.get("F1_Macro"), row.get("BalancedACC")]
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _subprocess_env() -> Dict[str, str]:
    """Build environment for subprocess CLI calls."""
    env = os.environ.copy()
    src_dir = Path(__file__).resolve().parents[1]
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_dir) if not prev else f"{src_dir}:{prev}"
    env.setdefault("NUMBA_DISABLE_JIT", "1")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return env


def _build_plan(args: argparse.Namespace) -> Tuple[List[RunSpec], Dict[str, Any]]:
    """Build all RunSpec entries for selected search groups."""
    selected_groups = {g.strip().lower() for g in args.groups.split(",") if g.strip()}
    allowed = {"baseline", "single", "pairwise", "batch", "dann"}
    unknown = selected_groups - allowed
    if unknown:
        raise ValueError(f"Unknown groups: {sorted(unknown)}")

    baseline_params = dict(get_preset(args.preset).algorithm_params)
    plan: List[RunSpec] = []

    if "baseline" in selected_groups:
        plan.append(RunSpec(group="00_baseline", name="baseline", overrides={}))

    if "single" in selected_groups:
        for param_name, values in SINGLE_PARAM_SWEEP.items():
            baseline_value = baseline_params.get(param_name)
            for value in values:
                if baseline_value == value:
                    continue
                run_name = f"{param_name}_{_value_to_token(value)}"
                plan.append(
                    RunSpec(
                        group=f"01_single_param/{param_name}",
                        name=run_name,
                        overrides={param_name: value},
                    )
                )

    if "pairwise" in selected_groups:
        for p1, vals1, p2, vals2 in PAIRWISE_SWEEP:
            baseline_pair = (baseline_params.get(p1), baseline_params.get(p2))
            for v1 in vals1:
                for v2 in vals2:
                    if baseline_pair == (v1, v2):
                        continue
                    name = f"{p1}_{_value_to_token(v1)}__{p2}_{_value_to_token(v2)}"
                    plan.append(
                        RunSpec(
                            group=f"02_pairwise/{p1}__{p2}",
                            name=name,
                            overrides={p1: v1, p2: v2},
                        )
                    )

    if "batch" in selected_groups:
        for scenario, overrides in BATCH_CORRECTION_SWEEP.items():
            if all(baseline_params.get(k) == v for k, v in overrides.items()):
                continue
            plan.append(
                RunSpec(
                    group="03_batch_correction",
                    name=scenario,
                    overrides=dict(overrides),
                )
            )

    if "dann" in selected_groups:
        dann_base = dict(DANN_BASE_OVERRIDES)

        plan.append(
            RunSpec(
                group="04_dann_focus/00_anchor",
                name="dann_anchor",
                overrides=dict(dann_base),
            )
        )

        for param_name, values in DANN_SINGLE_PARAM_SWEEP.items():
            baseline_value = dann_base.get(param_name, baseline_params.get(param_name))
            for value in values:
                if baseline_value == value:
                    continue
                run_name = f"{param_name}_{_value_to_token(value)}"
                overrides = dict(dann_base)
                overrides[param_name] = value
                plan.append(
                    RunSpec(
                        group=f"04_dann_focus/01_single/{param_name}",
                        name=run_name,
                        overrides=overrides,
                    )
                )

        for p1, vals1, p2, vals2 in DANN_PAIRWISE_SWEEP:
            baseline_pair = (
                dann_base.get(p1, baseline_params.get(p1)),
                dann_base.get(p2, baseline_params.get(p2)),
            )
            for v1 in vals1:
                for v2 in vals2:
                    if baseline_pair == (v1, v2):
                        continue
                    name = f"{p1}_{_value_to_token(v1)}__{p2}_{_value_to_token(v2)}"
                    overrides = dict(dann_base)
                    overrides[p1] = v1
                    overrides[p2] = v2
                    plan.append(
                        RunSpec(
                            group=f"04_dann_focus/02_pairwise/{p1}__{p2}",
                            name=name,
                            overrides=overrides,
                        )
                    )

        for scenario, extra in DANN_CONTROL_SWEEP.items():
            overrides = dict(dann_base)
            overrides.update(extra)
            plan.append(
                RunSpec(
                    group="04_dann_focus/03_controls",
                    name=scenario,
                    overrides=overrides,
                )
            )

    # Stable order for reproducibility.
    plan.sort(key=lambda x: (x.group, x.name))
    if args.max_runs and args.max_runs > 0:
        plan = plan[: int(args.max_runs)]
    return plan, baseline_params


def _load_overrides_json(raw: Any) -> Dict[str, Any]:
    """Parse overrides JSON string to dict safely."""
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return dict(parsed)


def _build_loss_ablation_plan(reference_overrides: Dict[str, Any]) -> List[RunSpec]:
    """Build post-search loss ablation plan from best run overrides."""
    ref = dict(reference_overrides)
    plan: List[RunSpec] = []

    triplet_off = {"rare_triplet_weight": 0.0}
    nb_off = {"reconstruction_distribution": "mse"}
    weighted_uniform = {
        "min_cell_weight": 1.0,
        "max_cell_weight": 1.0,
        # Keep triplet anchors active while weighted reconstruction is neutralized.
        "rare_triplet_min_weight": 0.0,
    }
    weighted_density_only = {"cluster_density_alpha": 0.0}
    weighted_cluster_only = {"cluster_density_alpha": 1.0}
    dann_off = {
        "use_batch_conditioning": False,
        "adversarial_batch_weight": 0.0,
        "mmd_batch_weight": 0.0,
    }
    ref_adv_weight = float(ref.get("adversarial_batch_weight", 0.0) or 0.0)
    dann_on = {
        "use_batch_conditioning": True,
        "batch_correction_key": str(ref.get("batch_correction_key", "batch")),
        "adversarial_batch_weight": ref_adv_weight if ref_adv_weight > 0.0 else 0.2,
        "adversarial_lambda": float(ref.get("adversarial_lambda", 1.0) or 1.0),
        "adversarial_start_epoch": int(ref.get("adversarial_start_epoch", 10) or 10),
        "adversarial_ramp_epochs": int(ref.get("adversarial_ramp_epochs", 20) or 20),
        "mmd_batch_weight": 0.0,
    }

    def add(group: str, name: str, updates: Dict[str, Any]) -> None:
        merged = dict(ref)
        merged.update(updates)
        plan.append(RunSpec(group=group, name=name, overrides=merged))

    add("00_reference", "reference_best", {})

    add("01_single_component", "ablate_triplet", triplet_off)
    add("01_single_component", "ablate_nb_reconstruction", nb_off)
    add("01_single_component", "ablate_weighted_uniform", weighted_uniform)
    add("01_single_component", "ablate_weighted_cluster_component", weighted_density_only)
    add("01_single_component", "ablate_weighted_density_component", weighted_cluster_only)
    add("01_single_component", "dann_forced_off", dann_off)
    add("01_single_component", "dann_forced_on", dann_on)

    add("02_pairwise_component", "ablate_triplet_nb", {**triplet_off, **nb_off})
    add("02_pairwise_component", "ablate_triplet_weighted", {**triplet_off, **weighted_uniform})
    add("02_pairwise_component", "ablate_nb_weighted", {**nb_off, **weighted_uniform})
    add("02_pairwise_component", "ablate_triplet_dann", {**triplet_off, **dann_off})
    add("02_pairwise_component", "ablate_nb_dann", {**nb_off, **dann_off})
    add("02_pairwise_component", "ablate_weighted_dann", {**weighted_uniform, **dann_off})

    add("03_higher_order", "ablate_triplet_nb_weighted", {**triplet_off, **nb_off, **weighted_uniform})
    add("03_higher_order", "ablate_triplet_nb_dann", {**triplet_off, **nb_off, **dann_off})
    add(
        "03_higher_order",
        "ablate_triplet_nb_weighted_dann",
        {**triplet_off, **nb_off, **weighted_uniform, **dann_off},
    )

    plan.sort(key=lambda x: (x.group, x.name))
    return plan


def _build_cmd(
    args: argparse.Namespace,
    data_path: Path,
    run_dir: Path,
    spec: RunSpec,
) -> List[str]:
    """Build one scRAW CLI command."""
    cmd = [
        args.python_bin,
        "-m",
        "scraw_dedicated.cli",
        "--preset",
        args.preset,
        "--data",
        str(data_path),
        "--output",
        str(run_dir),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--metrics-only",
        "--capture-snapshots",
        "off",
    ]
    for key, value in sorted(spec.overrides.items()):
        cmd.extend(["--param", f"{key}={_as_cli_value(value)}"])
    return cmd


def _read_metrics_from_run(run_dir: Path) -> Dict[str, Any]:
    """Read standard metrics from one completed run output."""
    csv_path = run_dir / "results" / "analysis_results.csv"
    if not csv_path.exists():
        return {
            "status": "missing",
            "NMI": np.nan,
            "ARI": np.nan,
            "ACC": np.nan,
            "F1_Macro": np.nan,
            "BalancedACC": np.nan,
            "RareACC": np.nan,
            "Silhouette": np.nan,
            "n_clusters_found": np.nan,
            "runtime": np.nan,
        }

    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {
            "status": "empty",
            "NMI": np.nan,
            "ARI": np.nan,
            "ACC": np.nan,
            "F1_Macro": np.nan,
            "BalancedACC": np.nan,
            "RareACC": np.nan,
            "Silhouette": np.nan,
            "n_clusters_found": np.nan,
            "runtime": np.nan,
        }

    row = rows[0]
    return {
        "status": "ok",
        "NMI": _metric_float(row, "NMI"),
        "ARI": _metric_float(row, "ARI"),
        "ACC": _metric_float(row, "ACC"),
        "F1_Macro": _metric_float(row, "F1_Macro"),
        "BalancedACC": _metric_float(row, "BalancedACC"),
        "RareACC": _metric_float(row, "RareACC"),
        "Silhouette": _metric_float(row, "Silhouette"),
        "n_clusters_found": _metric_float(row, "n_clusters_found"),
        "runtime": _metric_float(row, "runtime"),
    }


def _run_loss_ablation(
    args: argparse.Namespace,
    data_path: Path,
    output_root: Path,
    baseline_params: Dict[str, Any],
    ranked_rows: List[Dict[str, Any]],
    env: Dict[str, str],
) -> Dict[str, Any]:
    """Run post-search loss ablation from best overall search configuration."""
    best_overall: Optional[Dict[str, Any]] = None
    for row in ranked_rows:
        score = float(row.get("score", np.nan))
        if np.isfinite(score):
            best_overall = row
            break

    if best_overall is None:
        return {"status": "skipped_no_finite_best_score"}

    reference_overrides = _load_overrides_json(best_overall.get("overrides_json", "{}"))
    ablation_plan = _build_loss_ablation_plan(reference_overrides)
    if not ablation_plan:
        return {"status": "skipped_empty_ablation_plan"}

    ablation_root = output_root / "ablation_loss_impact"
    runs_root = ablation_root / "runs"
    logs_root = ablation_root / "logs"
    summaries_root = ablation_root / "summaries"
    meta_root = ablation_root / "meta"
    for d in (ablation_root, runs_root, logs_root, summaries_root, meta_root):
        d.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": "hyperparam_search",
        "preset": args.preset,
        "data": str(data_path),
        "seed": args.seed,
        "device": args.device,
        "metrics_only_forced": True,
        "capture_snapshots": "off",
        "reference_search_run": {
            "group": best_overall.get("group"),
            "name": best_overall.get("name"),
            "run_dir": best_overall.get("run_dir"),
            "score": best_overall.get("score"),
            "overrides": reference_overrides,
        },
        "baseline_params": baseline_params,
        "n_planned_runs": len(ablation_plan),
        "runs": [
            {"group": spec.group, "name": spec.name, "overrides": spec.overrides}
            for spec in ablation_plan
        ],
    }
    manifest_path = meta_root / "loss_ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows: List[Dict[str, Any]] = []
    failures = 0

    print("\nStarting post-search loss ablation:")
    print(
        f"  reference = {best_overall.get('group')}/{best_overall.get('name')} "
        f"(score={best_overall.get('score')})"
    )
    for idx, spec in enumerate(ablation_plan, start=1):
        run_dir = runs_root / spec.group / spec.name
        log_file = logs_root / spec.group / f"{spec.name}.log"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = _build_cmd(args, data_path, run_dir, spec)
        cmd_text = " ".join(shlex.quote(c) for c in cmd)
        print(f"[ABL {idx:03d}/{len(ablation_plan):03d}] {spec.group}/{spec.name}")
        print(f"  {cmd_text}")

        if args.skip_existing and (run_dir / "results" / "analysis_results.csv").exists():
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
        }
        row["score"] = _score_from_row(row)
        rows.append(row)

    fields = [
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
    ]

    all_csv = summaries_root / "loss_ablation_all_runs.csv"
    with all_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    ranked = sorted(
        rows,
        key=lambda r: (
            -float(r["score"]) if np.isfinite(r["score"]) else float("inf"),
            str(r["group"]),
            str(r["name"]),
        ),
    )
    ranked_csv = summaries_root / "loss_ablation_ranked_by_score.csv"
    with ranked_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in ranked:
            writer.writerow(row)

    reference_row = next(
        (r for r in rows if r.get("group") == "00_reference" and r.get("name") == "reference_best"),
        None,
    )
    delta_rows: List[Dict[str, Any]] = []
    if reference_row is not None:
        ref_score = float(reference_row.get("score", np.nan))
        ref_nmi = float(reference_row.get("NMI", np.nan))
        ref_ari = float(reference_row.get("ARI", np.nan))
        ref_acc = float(reference_row.get("ACC", np.nan))
        ref_f1 = float(reference_row.get("F1_Macro", np.nan))
        ref_bacc = float(reference_row.get("BalancedACC", np.nan))
        ref_rare = float(reference_row.get("RareACC", np.nan))
        ref_sil = float(reference_row.get("Silhouette", np.nan))
        ref_k = float(reference_row.get("n_clusters_found", np.nan))

        for row in rows:
            score = float(row.get("score", np.nan))
            nmi = float(row.get("NMI", np.nan))
            ari = float(row.get("ARI", np.nan))
            acc = float(row.get("ACC", np.nan))
            f1 = float(row.get("F1_Macro", np.nan))
            bacc = float(row.get("BalancedACC", np.nan))
            rare = float(row.get("RareACC", np.nan))
            sil = float(row.get("Silhouette", np.nan))
            k_val = float(row.get("n_clusters_found", np.nan))

            delta_rows.append(
                {
                    "group": row.get("group"),
                    "name": row.get("name"),
                    "status": row.get("status"),
                    "score": score,
                    "delta_score_vs_reference": score - ref_score
                    if np.isfinite(score) and np.isfinite(ref_score)
                    else np.nan,
                    "delta_NMI_vs_reference": nmi - ref_nmi
                    if np.isfinite(nmi) and np.isfinite(ref_nmi)
                    else np.nan,
                    "delta_ARI_vs_reference": ari - ref_ari
                    if np.isfinite(ari) and np.isfinite(ref_ari)
                    else np.nan,
                    "delta_ACC_vs_reference": acc - ref_acc
                    if np.isfinite(acc) and np.isfinite(ref_acc)
                    else np.nan,
                    "delta_F1_vs_reference": f1 - ref_f1
                    if np.isfinite(f1) and np.isfinite(ref_f1)
                    else np.nan,
                    "delta_BalancedACC_vs_reference": bacc - ref_bacc
                    if np.isfinite(bacc) and np.isfinite(ref_bacc)
                    else np.nan,
                    "delta_RareACC_vs_reference": rare - ref_rare
                    if np.isfinite(rare) and np.isfinite(ref_rare)
                    else np.nan,
                    "delta_Silhouette_vs_reference": sil - ref_sil
                    if np.isfinite(sil) and np.isfinite(ref_sil)
                    else np.nan,
                    "delta_n_clusters_vs_reference": k_val - ref_k
                    if np.isfinite(k_val) and np.isfinite(ref_k)
                    else np.nan,
                    "run_dir": row.get("run_dir"),
                }
            )

    delta_fields = [
        "group",
        "name",
        "status",
        "score",
        "delta_score_vs_reference",
        "delta_NMI_vs_reference",
        "delta_ARI_vs_reference",
        "delta_ACC_vs_reference",
        "delta_F1_vs_reference",
        "delta_BalancedACC_vs_reference",
        "delta_RareACC_vs_reference",
        "delta_Silhouette_vs_reference",
        "delta_n_clusters_vs_reference",
        "run_dir",
    ]
    delta_csv = summaries_root / "loss_ablation_delta_vs_reference.csv"
    with delta_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=delta_fields)
        writer.writeheader()
        for row in delta_rows:
            writer.writerow(row)

    summary = {
        "status": "ok",
        "n_runs": len(rows),
        "n_failures": failures,
        "manifest_json": str(manifest_path),
        "all_runs_csv": str(all_csv),
        "ranked_csv": str(ranked_csv),
        "delta_vs_reference_csv": str(delta_csv),
        "output_root": str(ablation_root),
    }
    summary_path = meta_root / "loss_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point for full hyperparameter search."""
    args = parse_args(argv)

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    output_root = Path(args.output_root).expanduser().resolve()
    runs_root = output_root / "runs"
    logs_root = output_root / "logs"
    summaries_root = output_root / "summaries"
    meta_root = output_root / "meta"
    for d in (output_root, runs_root, logs_root, summaries_root, meta_root):
        d.mkdir(parents=True, exist_ok=True)

    plan, baseline_params = _build_plan(args)
    if not plan:
        raise ValueError("Search plan is empty after filters.")

    manifest = {
        "preset": args.preset,
        "data": str(data_path),
        "seed": args.seed,
        "device": args.device,
        "groups": args.groups,
        "metrics_only_forced": True,
        "capture_snapshots": "off",
        "baseline_params": baseline_params,
        "n_planned_runs": len(plan),
        "runs": [
            {"group": spec.group, "name": spec.name, "overrides": spec.overrides}
            for spec in plan
        ],
    }
    (meta_root / "search_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows: List[Dict[str, Any]] = []
    env = _subprocess_env()
    failures = 0

    for idx, spec in enumerate(plan, start=1):
        run_dir = runs_root / spec.group / spec.name
        log_file = logs_root / spec.group / f"{spec.name}.log"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = _build_cmd(args, data_path, run_dir, spec)
        cmd_text = " ".join(shlex.quote(c) for c in cmd)
        print(f"[{idx:03d}/{len(plan):03d}] {spec.group}/{spec.name}")
        print(f"  {cmd_text}")

        if args.skip_existing and (run_dir / "results" / "analysis_results.csv").exists():
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
        }
        row["score"] = _score_from_row(row)
        rows.append(row)

    all_runs_csv = summaries_root / "all_runs_metrics.csv"
    fields = [
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
    ]
    with all_runs_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    ranked = sorted(
        rows,
        key=lambda r: (
            -float(r["score"]) if np.isfinite(r["score"]) else float("inf"),
            str(r["group"]),
            str(r["name"]),
        ),
    )
    ranked_csv = summaries_root / "ranked_by_score.csv"
    with ranked_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in ranked:
            writer.writerow(row)

    best_by_group: List[Dict[str, Any]] = []
    for group in sorted({r["group"] for r in rows}):
        candidates = [r for r in rows if r["group"] == group and np.isfinite(r["score"])]
        if not candidates:
            continue
        best = sorted(candidates, key=lambda r: -float(r["score"]))[0]
        best_by_group.append(best)

    best_csv = summaries_root / "best_by_group.csv"
    with best_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in best_by_group:
            writer.writerow(row)

    run_info = {
        "n_runs": len(rows),
        "n_failures": failures,
        "all_runs_csv": str(all_runs_csv),
        "ranked_csv": str(ranked_csv),
        "best_by_group_csv": str(best_csv),
    }

    if args.run_loss_ablation:
        ablation_summary = _run_loss_ablation(
            args=args,
            data_path=data_path,
            output_root=output_root,
            baseline_params=baseline_params,
            ranked_rows=ranked,
            env=env,
        )
        run_info["loss_ablation"] = ablation_summary
        if ablation_summary.get("status") == "ok":
            failures += int(ablation_summary.get("n_failures", 0) or 0)
    else:
        run_info["loss_ablation"] = {"status": "disabled"}

    run_info["n_total_failures_including_ablation"] = failures
    (meta_root / "search_summary.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(f" - {all_runs_csv}")
    print(f" - {ranked_csv}")
    print(f" - {best_csv}")
    print(f" - {meta_root / 'search_manifest.json'}")
    print(f" - {meta_root / 'search_summary.json'}")
    if isinstance(run_info.get("loss_ablation"), dict):
        ab_status = run_info["loss_ablation"].get("status")
        if ab_status == "ok":
            print(f" - {run_info['loss_ablation'].get('all_runs_csv')}")
            print(f" - {run_info['loss_ablation'].get('ranked_csv')}")
            print(f" - {run_info['loss_ablation'].get('delta_vs_reference_csv')}")
            print(f" - {run_info['loss_ablation'].get('summary_json')}")
        else:
            print(f"Loss ablation: {ab_status}")
    print(f"Failures: {failures}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
