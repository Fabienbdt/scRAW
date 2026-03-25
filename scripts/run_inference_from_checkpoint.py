#!/usr/bin/env python3
"""Replay scRAW inference from a saved autoencoder checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scraw import run_inference_from_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scRAW preprocessing + encoding + clustering from saved weights only.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a saved config JSON, typically config/config_used.json from the training run.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the saved autoencoder.pt checkpoint to replay.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where inference-only outputs will be written.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional dataset override. Useful when config_used.json stores a relative data path.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional runtime device override, for example cuda or cpu.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_inference_from_checkpoint(
        config=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        data_path=args.data_path,
        device=args.device,
    )
    print(f"mode = {result['mode']}")
    print(f"strict_repro = {result['config']['runtime']['strict_repro']}")
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(result["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
