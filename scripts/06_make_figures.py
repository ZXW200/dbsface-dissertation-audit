"""Regenerate dissertation figure assets from structured outputs."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dbsface.cli import main as cli_main
from generate_additional_dissertation_figures import main as additional_figure_main


if __name__ == "__main__":
    status = cli_main(["figures"])
    if status:
        raise SystemExit(status)
    raise SystemExit(additional_figure_main())
