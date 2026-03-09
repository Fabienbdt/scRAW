#!/usr/bin/env python3
"""Top-k multi-seed refinement for scRAW Optuna ultra-search outputs.

This script re-runs the best configurations found by Optuna with multiple seeds
to improve robustness of the selected final configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .presets import PRESETS


SCORE_WEIGHTS: Dict[str, float] = {
    "ARI": 0.30,
    "NMI": 0.25,
    "F1_Macro": 0.20,
    "BalancedACC": 0.15,
    "RareACC": 0.10,
}


def _as_cli_value(value: Any) -> str:
    """Convert python value to CLI scalar."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _safe_float(value: Any) -> float:
    """Parse float or NaN."""
    try:
        x = float(value)
    except Exception:
        return float("nan")
    if not np.isfinite(x):
        return float("nan")
    return float(x)


def _selected_method(requested: str, target_clusters: int) -> str:
    """Map requested final clustering to final comparison method name."""
    req = str(requested).strip().lower()
    if req == "leiden":
        return f"leiden_target{int(target_clusters)}_final"
    return "hdbscan_final"


def _score(metrics: Dict[str, Any]) -> float:
    """Weighted clustering score."""
    total = 0.0
    for k, w in SCORE_WEIGHTS.items():
        v = _safe_float(metrics.get(k))
        if np.isnan(v):
            v = 0.0
        total += float(w) * float(v)
    return float(total)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    """Read CSV rows."""
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _build_env(output_root: Path) -> Dict[str, str]:
    """Build subprocess env with writable cache dirs."""
    env = os.environ.copy()
    src_dir = Path(__file__).resolve().parents[1]
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_dir) if not prev else f"{src_dir}:{prev}"

    cache_root = output_root / ".cache"
    numba_dir = cache_root / "numba"
    mpl_dir = cache_root / "mpl"
    xdg_dir = cache_root / "xdg_cache"
    numba_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("NUMBA_CACHE_DIR", str(numba_dir))
    env.setdefault("MPLCONFIGDIR", str(mpl_dir))
    env.setdefault("XDG_CACHE_HOME", str(xdg_dir))
    env.setdefault("NUMBA_DISABLE_JIT", "1")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return env


def _seed_sequence(base_seed: int, n_seeds: int, seed_step: int) -> List[int]:
    """Deterministic seed sequence."""
    n = max(1, int(n_seeds))
    step = max(1, int(seed_step))
    return [int(base_seed) + i * step for i in range(n)]


def _build_cmd(
    python_bin: str,
    preset: str,
    data_path: Path,
    run_dir: Path,
    device: str,
    seed: int,
    overrides: Dict[str, Any],
    batch_key: Optional[str],
) -> List[str]:
    """Build one scRAW CLI command in metrics-only mode."""
    cmd: List[str] = [
        python_bin,
        "-m",
        "scraw_dedicated.cli",
        "--preset",
        preset,
        "--data",
        str(data_path),
        "--output",
        str(run_dir),
        "--seed",
        str(int(seed)),
        "--device",
        str(device),
        "--metrics-only",
        "--save-processed-data",
        "off",
        "--capture-snapshots",
        "off",
        "--dann",
        "auto",
    ]
    if batch_key:
        cmd.extend(["--batch-key", str(batch_key)])
    for k, v in sorted(overrides.items()):
        cmd.extend(["--param", f"{k}={_as_cli_value(v)}"])
    return cmd


def _read_analysis_metrics(run_dir: Path) -> Dict[str, Any]:
    """Read standard analysis metrics."""
    rows = _read_csv(run_dir / "results" / "analysis_results.csv")
    if not rows:
        return {}
    row = rows[0]
    return {
        "runtime": _safe_float(row.get("runtime")),
        "ACC": _safe_float(row.get("ACC")),
    }


def _read_final_metrics(run_dir: Path, requested: str, target_clusters: int) -> Dict[str, Any]:
    """Read final clustering metrics with fallback to HDBSCAN if needed."""
    method = _selected_method(requested, target_clusters)
    rows = _read_csv(run_dir / "results" / "clustering_final" / "final_clustering_comparison.csv")
    by_method: Dict[str, Dict[str, str]] = {}
    for row in rows:
        m = str(row.get("method", "")).strip()
        if m:
            by_method[m] = row

    chosen = by_method.get(method)
    used_method = method
    fallback = False
    if chosen is None and method != "hdbscan_final":
        chosen = by_method.get("hdbscan_final")
        used_method = "hdbscan_final"
        fallback = chosen is not None

    if chosen is None:
        return {"used_method": used_method, "fallback": fallback}

    out: Dict[str, Any] = {
        "used_method": used_method,
        "fallback": fallback,
        "ARI": _safe_float(chosen.get("ARI")),
        "NMI": _safe_float(chosen.get("NMI")),
        "ACC": _safe_float(chosen.get("ACC")),
        "F1_Macro": _safe_float(chosen.get("F1_Macro")),
        "BalancedACC": _safe_float(chosen.get("BalancedACC")),
        "RareACC": _safe_float(chosen.get("RareACC")),
        "n_clusters_found": _safe_float(chosen.get("n_clusters_found")),
        "resolution": _safe_float(chosen.get("resolution")),
    }
    return out


def _mean(rows: List[Dict[str, Any]], key: str) -> float:
    """Mean of finite values."""
    vals = [float(r.get(key, np.nan)) for r in rows if np.isfinite(float(r.get(key, np.nan)))]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _pick_topk(all_trials_csv: Path, top_k: int) -> List[Dict[str, Any]]:
    """Pick top-k candidate trials from all_trials.csv."""
    rows = _read_csv(all_trials_csv)
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        score = _safe_float(row.get("score_mean"))
        trial_num = row.get("trial")
        ov = row.get("overrides_json")
        if np.isnan(score) or trial_num is None or ov is None:
            continue
        try:
            trial_int = int(trial_num)
            overrides = json.loads(str(ov))
        except Exception:
            continue
        if not isinstance(overrides, dict):
            continue
        cleaned.append(
            {
                "trial": trial_int,
                "score_mean": score,
                "final_clustering_requested": str(row.get("final_clustering_requested", "hdbscan")),
                "dann_enabled": row.get("dann_enabled"),
                "overrides": overrides,
            }
        )

    cleaned.sort(key=lambda r: (-float(r["score_mean"]), int(r["trial"])))
    if int(top_k) > 0:
        cleaned = cleaned[: int(top_k)]
    return cleaned


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI args."""
    p = argparse.ArgumentParser(description="Refine top-k Optuna scRAW trials with multi-seed reruns.")
    p.add_argument("--search-root", required=True, help="Optuna search root (contains summaries/all_trials.csv)")
    p.add_argument("--preset", required=True, choices=sorted(PRESETS.keys()))
    p.add_argument("--data", required=True, help="Input .h5ad file")
    p.add_argument("--output-root", default=None, help="Refinement output root")
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--device", default="auto")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--seed-step", type=int, default=97)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--target-clusters", type=int, default=14)
    p.add_argument("--batch-key", default=None)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entrypoint."""
    args = _parse_args(argv)

    search_root = Path(args.search_root).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    all_trials_csv = search_root / "summaries" / "all_trials.csv"
    if not all_trials_csv.exists():
        raise FileNotFoundError(f"Missing all_trials.csv: {all_trials_csv}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else (search_root / "refine_topk_multiseed").resolve()
    )
    runs_root = output_root / "runs"
    logs_root = output_root / "logs"
    summaries_root = output_root / "summaries"
    meta_root = output_root / "meta"
    for d in (runs_root, logs_root, summaries_root, meta_root):
        d.mkdir(parents=True, exist_ok=True)

    env = _build_env(output_root)
    candidates = _pick_topk(all_trials_csv, top_k=int(args.top_k))
    if not candidates:
        raise RuntimeError(f"No valid candidate rows found in {all_trials_csv}")

    seeds = _seed_sequence(args.base_seed, args.n_seeds, args.seed_step)
    manifest = {
        "source_search_root": str(search_root),
        "preset": args.preset,
        "data": str(data_path),
        "device": args.device,
        "python_bin": args.python_bin,
        "base_seed": int(args.base_seed),
        "seed_step": int(args.seed_step),
        "seeds": [int(s) for s in seeds],
        "n_seeds": int(len(seeds)),
        "top_k": int(args.top_k),
        "target_clusters": int(args.target_clusters),
        "score_weights": SCORE_WEIGHTS,
        "cluster_count_penalty": "none",
        "dry_run": bool(args.dry_run),
        "skip_existing": bool(args.skip_existing),
        "candidates": [
            {
                "rank": i + 1,
                "trial": int(c["trial"]),
                "score_mean_optuna": float(c["score_mean"]),
                "final_clustering_requested": c["final_clustering_requested"],
            }
            for i, c in enumerate(candidates)
        ],
        "timestamp": datetime.now().isoformat(),
    }
    _write_json(meta_root / "manifest.json", manifest)

    seed_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for rank, cand in enumerate(candidates, start=1):
        trial_id = int(cand["trial"])
        requested_clustering = str(cand["final_clustering_requested"])
        overrides = dict(cand["overrides"])
        candidate_id = f"rank_{rank:03d}_trial_{trial_id:04d}"

        for seed in seeds:
            run_dir = runs_root / candidate_id / f"seed_{int(seed)}"
            log_file = logs_root / candidate_id / f"seed_{int(seed)}.log"
            run_dir.mkdir(parents=True, exist_ok=True)
            log_file.parent.mkdir(parents=True, exist_ok=True)

            cmd = _build_cmd(
                python_bin=args.python_bin,
                preset=args.preset,
                data_path=data_path,
                run_dir=run_dir,
                device=args.device,
                seed=int(seed),
                overrides=overrides,
                batch_key=args.batch_key,
            )
            cmd_str = " ".join(shlex.quote(c) for c in cmd)

            status = "ok"
            if args.dry_run:
                status = "dry_run"
                log_file.write_text(cmd_str + "\n", encoding="utf-8")
            else:
                analysis_csv = run_dir / "results" / "analysis_results.csv"
                if args.skip_existing and analysis_csv.exists():
                    status = "existing"
                else:
                    with log_file.open("w") as fh:
                        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
                    if int(proc.returncode) != 0:
                        status = f"failed_{int(proc.returncode)}"

            final_metrics = _read_final_metrics(
                run_dir=run_dir,
                requested=requested_clustering,
                target_clusters=int(args.target_clusters),
            )
            analysis_metrics = _read_analysis_metrics(run_dir)
            metrics = dict(final_metrics)
            if not np.isfinite(_safe_float(metrics.get("ACC"))):
                metrics["ACC"] = analysis_metrics.get("ACC")
            metrics["runtime"] = analysis_metrics.get("runtime", float("nan"))

            score = _score(metrics)
            if status.startswith("failed") or status == "dry_run":
                score = 0.0

            row: Dict[str, Any] = {
                "candidate_rank": int(rank),
                "candidate_id": candidate_id,
                "trial": int(trial_id),
                "seed": int(seed),
                "status": status,
                "requested_method": _selected_method(requested_clustering, int(args.target_clusters)),
                "used_method": str(metrics.get("used_method", "")),
                "fallback_to_hdbscan": int(bool(metrics.get("fallback", False))),
                "score": float(score),
                "ARI": _safe_float(metrics.get("ARI")),
                "NMI": _safe_float(metrics.get("NMI")),
                "ACC": _safe_float(metrics.get("ACC")),
                "F1_Macro": _safe_float(metrics.get("F1_Macro")),
                "BalancedACC": _safe_float(metrics.get("BalancedACC")),
                "RareACC": _safe_float(metrics.get("RareACC")),
                "n_clusters_found": _safe_float(metrics.get("n_clusters_found")),
                "resolution": _safe_float(metrics.get("resolution")),
                "runtime": _safe_float(metrics.get("runtime")),
                "run_dir": str(run_dir),
                "log_file": str(log_file),
                "command": cmd_str,
                "overrides_json": json.dumps(overrides, sort_keys=True),
            }
            seed_rows.append(row)

    seed_fields = [
        "candidate_rank",
        "candidate_id",
        "trial",
        "seed",
        "status",
        "requested_method",
        "used_method",
        "fallback_to_hdbscan",
        "score",
        "ARI",
        "NMI",
        "ACC",
        "F1_Macro",
        "BalancedACC",
        "RareACC",
        "n_clusters_found",
        "resolution",
        "runtime",
        "run_dir",
        "log_file",
        "command",
        "overrides_json",
    ]
    with (summaries_root / "all_seed_runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=seed_fields)
        writer.writeheader()
        for row in seed_rows:
            writer.writerow(row)

    aggregated: List[Dict[str, Any]] = []
    for rank, cand in enumerate(candidates, start=1):
        trial_id = int(cand["trial"])
        candidate_id = f"rank_{rank:03d}_trial_{trial_id:04d}"
        sub = [r for r in seed_rows if str(r.get("candidate_id")) == candidate_id]
        if not sub:
            continue
        agg: Dict[str, Any] = {
            "candidate_rank": int(rank),
            "candidate_id": candidate_id,
            "trial": int(trial_id),
            "score_mean_optuna": float(cand["score_mean"]),
            "n_seed_runs": int(len(sub)),
            "n_seed_ok": int(sum(1 for r in sub if str(r.get("status", "")).startswith(("ok", "existing")))),
            "fallback_count": int(sum(int(r.get("fallback_to_hdbscan", 0)) for r in sub)),
            "score_mean": _mean(sub, "score"),
            "score_std": float(
                np.std([float(r.get("score", 0.0)) for r in sub])
            ),
            "ARI_mean": _mean(sub, "ARI"),
            "NMI_mean": _mean(sub, "NMI"),
            "ACC_mean": _mean(sub, "ACC"),
            "F1_Macro_mean": _mean(sub, "F1_Macro"),
            "BalancedACC_mean": _mean(sub, "BalancedACC"),
            "RareACC_mean": _mean(sub, "RareACC"),
            "n_clusters_found_mean": _mean(sub, "n_clusters_found"),
            "resolution_mean": _mean(sub, "resolution"),
            "runtime_mean": _mean(sub, "runtime"),
            "overrides_json": sub[0].get("overrides_json", "{}"),
        }
        aggregated.append(agg)

    aggregated.sort(
        key=lambda r: (
            -float(r.get("score_mean", np.nan)) if np.isfinite(float(r.get("score_mean", np.nan))) else float("inf"),
            int(r.get("trial", 10**9)),
        )
    )

    agg_fields = [
        "candidate_rank",
        "candidate_id",
        "trial",
        "score_mean_optuna",
        "n_seed_runs",
        "n_seed_ok",
        "fallback_count",
        "score_mean",
        "score_std",
        "ARI_mean",
        "NMI_mean",
        "ACC_mean",
        "F1_Macro_mean",
        "BalancedACC_mean",
        "RareACC_mean",
        "n_clusters_found_mean",
        "resolution_mean",
        "runtime_mean",
        "overrides_json",
    ]
    with (summaries_root / "aggregated_topk.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        for row in aggregated:
            writer.writerow(row)

    summary = {
        "status": "ok",
        "output_root": str(output_root),
        "search_root": str(search_root),
        "n_candidates": int(len(candidates)),
        "n_seed_runs": int(len(seed_rows)),
        "elapsed_seconds": round(float(time.time() - t0), 1),
        "best": aggregated[0] if aggregated else None,
        "all_seed_runs_csv": str(summaries_root / "all_seed_runs.csv"),
        "aggregated_csv": str(summaries_root / "aggregated_topk.csv"),
        "timestamp": datetime.now().isoformat(),
    }
    _write_json(meta_root / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
