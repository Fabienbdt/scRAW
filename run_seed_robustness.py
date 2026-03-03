#!/usr/bin/env python3
"""Compatibility wrapper for multi-seed robustness CLI."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scraw_dedicated.seed_robustness import main

if __name__ == "__main__":
    raise SystemExit(main())
