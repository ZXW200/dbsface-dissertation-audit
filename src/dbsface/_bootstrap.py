"""Runtime path helpers for the submission package."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def add_project_paths(change_cwd: bool = True) -> Path:
    root = package_root()
    src_dir = root / "src"
    module_dir = Path(__file__).resolve().parent

    paths = [src_dir, module_dir]
    paths.extend(path for path in module_dir.iterdir() if path.is_dir())

    for path in paths:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)

    if change_cwd:
        os.chdir(root)

    return root
