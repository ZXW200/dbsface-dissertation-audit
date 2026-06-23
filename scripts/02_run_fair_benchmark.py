"""Run the clean matched-seed benchmark used as the main comparison."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dbsface._bootstrap import add_package_module_dir

add_package_module_dir()

from dbsface.experiments.run_fair_benchmark_multiseed import main


if __name__ == "__main__":
    raise SystemExit(main())
