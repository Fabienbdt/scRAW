#!/usr/bin/env python3
"""Wrapper script for top-k multi-seed refinement after Optuna ultra-search."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scraw_dedicated.optuna_refine_topk import main


if __name__ == "__main__":
    raise SystemExit(main())
