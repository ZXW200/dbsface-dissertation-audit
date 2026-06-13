"""Runtime path helper for script-style modules.

The public entry-point scripts add ``src`` to ``sys.path`` before importing the
package. This helper is retained for compatibility with the final six command
wrappers.
"""

from __future__ import annotations

import sys
from pathlib import Path


def add_package_module_dir() -> None:
    src_dir = Path(__file__).resolve().parents[1]
    src_dir_text = str(src_dir)
    if src_dir_text not in sys.path:
        sys.path.insert(0, src_dir_text)


