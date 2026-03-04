#!/usr/bin/env python3
"""Wrapper script for comprehensive hyperparameter search."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scraw_dedicated.hyperparam_search import main


if __name__ == "__main__":
    raise SystemExit(main())
