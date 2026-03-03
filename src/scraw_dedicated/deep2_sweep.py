#!/usr/bin/env python3
"""Run the Baron Deep2 sweep configuration used for SCRBenchmark parity."""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


def _exp(name: str, args: List[str]) -> Tuple[str, List[str]]:
    """Helper interne: exp.
    
    
    Args:
        name: Paramètre d'entrée `name` utilisé dans cette étape du pipeline.
        args: Paramètre d'entrée `args` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    return name, args


DEEP2_EXPERIMENTS: List[Tuple[str, List[str]]] = [
    _exp("baseline_trip10_s35", []),
    _exp("baseline_true_unsupervised", ["--unsupervised"]),
    _exp("ablate_triplet", ["--param", "rare_triplet_weight=0.0"]),
    _exp("ablate_reco_cluster", ["--param", "cluster_density_alpha=0.0"]),
    _exp("ablate_reco_density", ["--param", "cluster_density_alpha=1.0"]),
    _exp(
        "best_dann_w020_with_evolution",
        [
            "--param",
            "use_batch_conditioning=true",
            "--param",
            "batch_correction_key=batch",
            "--param",
            "adversarial_batch_weight=0.2",
            "--param",
            "adversarial_lambda=1.0",
            "--param",
            "adversarial_start_epoch=10",
            "--param",
            "adversarial_ramp_epochs=20",
            "--param",
            "mmd_batch_weight=0.0",
            "--capture-snapshots",
            "on",
            "--snapshot-interval",
            "20",
        ],
    ),
    _exp(
        "ablate_weighted_reconstruction",
        [
            "--param",
            "min_cell_weight=1.0",
            "--param",
            "max_cell_weight=1.0",
            "--param",
            "rare_triplet_min_weight=0.0",
        ],
    ),
    _exp("ablate_nb_reconstruction", ["--param", "reconstruction_distribution=mse"]),
    _exp(
        "ablate_weighted_and_nb_triplet_only",
        [
            "--param",
            "min_cell_weight=1.0",
            "--param",
            "max_cell_weight=1.0",
            "--param",
            "rare_triplet_min_weight=0.0",
            "--param",
            "reconstruction_distribution=mse",
        ],
    ),
]


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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Construit et retourne le parseur d'arguments CLI pour ce script.
    
    
    Args:
        argv: Paramètre d'entrée `argv` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    p = argparse.ArgumentParser(description="Run Deep2 Baron sweep with scraw_dedicated")
    p.add_argument("--data", required=True, help="Path to baron_human_pancreas.h5ad")
    p.add_argument("--output-root", required=True, help="Output root directory")
    p.add_argument("--preset", default="baron_best", choices=["baron_best", "pancreas_best"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--snapshot-interval", type=int, default=20)
    p.add_argument("--only", default="", help="Comma-separated subset of experiment names")
    p.add_argument("--metrics-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _selected_experiments(only: str) -> List[Tuple[str, List[str]]]:
    """Helper interne: selected experiments.
    
    
    Args:
        only: Paramètre d'entrée `only` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    if not only.strip():
        return DEEP2_EXPERIMENTS
    wanted = {x.strip() for x in only.split(",") if x.strip()}
    return [e for e in DEEP2_EXPERIMENTS if e[0] in wanted]


def _run_one(
    name: str,
    extra_args: List[str],
    base_cmd: List[str],
    output_root: Path,
    log_dir: Path,
    dry_run: bool,
) -> int:
    """Helper interne: run one.
    
    
    Args:
        name: Paramètre d'entrée `name` utilisé dans cette étape du pipeline.
        extra_args: Paramètre d'entrée `extra_args` utilisé dans cette étape du pipeline.
        base_cmd: Paramètre d'entrée `base_cmd` utilisé dans cette étape du pipeline.
        output_root: Paramètre d'entrée `output_root` utilisé dans cette étape du pipeline.
        log_dir: Paramètre d'entrée `log_dir` utilisé dans cette étape du pipeline.
        dry_run: Paramètre d'entrée `dry_run` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    out_dir = output_root / name
    cmd = base_cmd + ["--output", str(out_dir)] + extra_args

    log_file = log_dir / f"{name}.log"
    print(f"\n--- {name} ---")
    print("Command:")
    print(" ".join(shlex.quote(x) for x in cmd))

    if dry_run:
        return 0

    with log_file.open("w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=_subprocess_env())
    return int(proc.returncode)


def _collect_summary(output_root: Path, experiments: List[Tuple[str, List[str]]]) -> Path:
    """Helper interne: collect summary.
    
    
    Args:
        output_root: Paramètre d'entrée `output_root` utilisé dans cette étape du pipeline.
        experiments: Paramètre d'entrée `experiments` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    rows: List[Dict[str, object]] = []

    for name, _ in experiments:
        csv_path = output_root / name / "results" / "analysis_results.csv"
        if not csv_path.exists():
            rows.append({"config": name, "status": "missing"})
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            rows.append({"config": name, "status": "read_error"})
            continue

        if df.empty:
            rows.append({"config": name, "status": "empty"})
            continue

        s = df.iloc[0]
        rows.append(
            {
                "config": name,
                "NMI": s.get("NMI"),
                "ARI": s.get("ARI"),
                "ACC": s.get("ACC"),
                "F1_Macro": s.get("F1_Macro"),
                "BalancedACC": s.get("BalancedACC"),
                "RareACC": s.get("RareACC"),
                "Silhouette": s.get("Silhouette"),
                "n_clusters_found": s.get("n_clusters_found"),
                "runtime_s": s.get("runtime"),
                "status": "ok",
            }
        )

    summary_path = output_root / "summary.csv"
    fields = [
        "config",
        "NMI",
        "ARI",
        "ACC",
        "F1_Macro",
        "BalancedACC",
        "RareACC",
        "Silhouette",
        "n_clusters_found",
        "runtime_s",
        "status",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return summary_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Point d'entrée principal appelé lors de l'exécution du script.
    
    
    Args:
        argv: Paramètre d'entrée `argv` utilisé dans cette étape du pipeline.
    
    Returns:
        Valeur calculée par la fonction.
    """
    args = parse_args(argv)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    experiments = _selected_experiments(args.only)
    if not experiments:
        raise ValueError("No experiment selected. Check --only.")

    base_cmd = [
        sys.executable,
        "-m",
        "scraw_dedicated.cli",
        "--preset",
        args.preset,
        "--data",
        str(Path(args.data).expanduser().resolve()),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--capture-snapshots",
        "on",
        "--snapshot-interval",
        str(args.snapshot_interval),
    ]
    if args.metrics_only:
        base_cmd.append("--metrics-only")

    failed = 0
    for name, extra in experiments:
        rc = _run_one(
            name=name,
            extra_args=extra,
            base_cmd=base_cmd,
            output_root=output_root,
            log_dir=log_dir,
            dry_run=args.dry_run,
        )
        if rc != 0:
            failed += 1
            print(f"FAIL: {name} (exit={rc})")

    summary_path = _collect_summary(output_root, experiments)
    print("\nSummary:", summary_path)
    print("Failures:", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
