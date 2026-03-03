#!/usr/bin/env python3
"""Wrapper script for a single scRAW run."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scraw_dedicated.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
