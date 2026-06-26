"""Regenerate dissertation figure assets from structured outputs."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dbsface.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["figures"]))
