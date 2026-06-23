"""Run linear, CNN-AEV, and Grad-CAM ROI explanation outputs."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dbsface._bootstrap import add_package_module_dir

add_package_module_dir()

from dbsface.explain.run_cnn_sklearn_method_comparison import main


if __name__ == "__main__":
    raise SystemExit(main())
