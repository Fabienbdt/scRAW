#!/usr/bin/env python3
"""Wrapper script to regenerate figure-rich ablation runs."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scraw_dedicated.regenerate_ablation_figures import main


if __name__ == "__main__":
    raise SystemExit(main())
