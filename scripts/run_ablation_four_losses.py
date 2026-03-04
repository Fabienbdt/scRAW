#!/usr/bin/env python3
"""Wrapper for lightweight 4-loss ablation with shared epoch-30 latent."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scraw_dedicated.ablation_four_losses import main


if __name__ == "__main__":
    raise SystemExit(main())
