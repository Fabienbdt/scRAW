#!/usr/bin/env python3
"""Multi-seed robustness runner for scRAW dedicated presets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Construit et retourne le parseur d'arguments CLI pour ce script.
    
    
    Args:
        argv: Paramètre d'entrée `argv` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    p = argparse.ArgumentParser(description="Run multi-seed robustness with scRAW dedicated presets.")
    p.add_argument("--preset", required=True, choices=["baron_best", "pancreas_best"])
    p.add_argument("--data", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--seeds", default="11,22,33,42,77")
    p.add_argument("--device", default="auto")
    p.add_argument("--metrics-only", action="store_true")
    p.add_argument("--unsupervised", action="store_true")
    p.add_argument("--dann", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--batch-key", default=None)
    p.add_argument("--capture-snapshots", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--snapshot-interval", type=int, default=None)
    p.add_argument("--param", action="append", default=[])
    p.add_argument("--preprocess", action="append", default=[])
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _metric_value(row: Dict[str, str], key: str) -> float:
    """Helper interne: metric value.
    
    
    Args:
        row: Paramètre d'entrée `row` utilisé dans cette étape du pipeline.
        key: Paramètre d'entrée `key` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")


def _read_metrics(path: Path) -> Dict[str, float]:
    """Helper interne: read metrics.
    
    
    Args:
        path: Paramètre d'entrée `path` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    metrics_path = path / "results" / "analysis_results.csv"
    if not metrics_path.exists():
        return {
            "status": "missing",
            "NMI": np.nan,
            "ARI": np.nan,
            "ACC": np.nan,
            "F1_Macro": np.nan,
            "BalancedACC": np.nan,
            "RareACC": np.nan,
            "Silhouette": np.nan,
        }

    with metrics_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
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
        }

    r = rows[0]
    return {
        "status": "ok",
        "NMI": _metric_value(r, "NMI"),
        "ARI": _metric_value(r, "ARI"),
        "ACC": _metric_value(r, "ACC"),
        "F1_Macro": _metric_value(r, "F1_Macro"),
        "BalancedACC": _metric_value(r, "BalancedACC"),
        "RareACC": _metric_value(r, "RareACC"),
        "Silhouette": _metric_value(r, "Silhouette"),
    }


def _build_run_cmd(args: argparse.Namespace, seed: int, run_output: Path) -> List[str]:
    """Helper interne: build run cmd.
    
    
    Args:
        args: Paramètre d'entrée `args` utilisé dans cette étape du pipeline.
        seed: Paramètre d'entrée `seed` utilisé dans cette étape du pipeline.
        run_output: Paramètre d'entrée `run_output` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    cmd = [
        sys.executable,
        "-m",
        "scraw_dedicated.cli",
        "--preset",
        args.preset,
        "--data",
        args.data,
        "--output",
        str(run_output),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--dann",
        args.dann,
        "--capture-snapshots",
        args.capture_snapshots,
    ]
    if args.snapshot_interval is not None:
        cmd.extend(["--snapshot-interval", str(args.snapshot_interval)])
    if args.metrics_only:
        cmd.append("--metrics-only")
    if args.unsupervised:
        cmd.append("--unsupervised")
    if args.batch_key:
        cmd.extend(["--batch-key", args.batch_key])
    for item in args.param:
        cmd.extend(["--param", item])
    for item in args.preprocess:
        cmd.extend(["--preprocess", item])
    return cmd


def _subprocess_env() -> Dict[str, str]:
    """Helper interne: subprocess env.
    
    
    Args:
        Aucun argument explicite en dehors du contexte objet.
    
    Returns:
        Valeur calculée par la fonction.
    """
    env = os.environ.copy()
    src_dir = Path(__file__).resolve().parents[1]
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_dir) if not prev else f"{src_dir}:{prev}"
    return env


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Point d'entrée principal appelé lors de l'exécution du script.
    
    
    Args:
        argv: Paramètre d'entrée `argv` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    args = parse_args(argv)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("No seeds provided")

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    failed = 0

    for seed in seeds:
        run_dir = out_root / f"seed_{seed}"
        cmd = _build_run_cmd(args, seed, run_dir)
        print(f"\n=== seed={seed} ===")
        print("Command:")
        print(" ".join(shlex.quote(x) for x in cmd))

        if args.dry_run:
            rc = 0
        else:
            rc = subprocess.run(cmd, env=_subprocess_env()).returncode

        metrics = _read_metrics(run_dir)
        status = metrics["status"] if rc == 0 else f"run_failed_{rc}"
        if rc != 0:
            failed += 1

        row = {"seed": seed, "status": status}
        row.update(
            {
                k: metrics[k]
                for k in [
                    "NMI",
                    "ARI",
                    "ACC",
                    "F1_Macro",
                    "BalancedACC",
                    "RareACC",
                    "Silhouette",
                ]
            }
        )
        rows.append(row)

    per_seed_csv = out_root / "seed_metrics.csv"
    with per_seed_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "status",
                "NMI",
                "ARI",
                "ACC",
                "F1_Macro",
                "BalancedACC",
                "RareACC",
                "Silhouette",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows if str(r["status"]) == "ok"]
    metrics = ["NMI", "ARI", "ACC", "F1_Macro", "BalancedACC", "RareACC", "Silhouette"]
    aggregate = {
        "preset": args.preset,
        "data": args.data,
        "seeds": seeds,
        "n_ok": len(ok),
        "n_failed": failed,
        "stats": {},
    }

    for m in metrics:
        vals = np.array([float(r[m]) for r in ok], dtype=float) if ok else np.array([], dtype=float)
        aggregate["stats"][m] = {
            "mean": float(np.nanmean(vals)) if vals.size else float("nan"),
            "std": float(np.nanstd(vals)) if vals.size else float("nan"),
            "min": float(np.nanmin(vals)) if vals.size else float("nan"),
            "max": float(np.nanmax(vals)) if vals.size else float("nan"),
        }

    if ok:
        aggregate["score_mean_NMI_ARI_ACC"] = float(
            np.nanmean(
                [
                    aggregate["stats"]["NMI"]["mean"],
                    aggregate["stats"]["ARI"]["mean"],
                    aggregate["stats"]["ACC"]["mean"],
                ]
            )
        )
        aggregate["score_stability_NMI_ARI_ACC"] = float(
            1.0
            - np.nanmean(
                [
                    aggregate["stats"]["NMI"]["std"],
                    aggregate["stats"]["ARI"]["std"],
                    aggregate["stats"]["ACC"]["std"],
                ]
            )
        )
    else:
        aggregate["score_mean_NMI_ARI_ACC"] = float("nan")
        aggregate["score_stability_NMI_ARI_ACC"] = float("nan")

    agg_json = out_root / "seed_aggregate.json"
    agg_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(" -", per_seed_csv)
    print(" -", agg_json)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
