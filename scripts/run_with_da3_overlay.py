#!/usr/bin/env python3
"""Run a Python entry point with optional DA3 paths appended safely.

``PYTHONPATH`` prepends directories and can accidentally shadow the validated
CUDA Torch build when a dependency overlay contains ``torch``.  This bootstrap
appends the colon-separated ``MAGICIAN_DA3_APPEND_PATHS`` entries only after
the active environment has initialized its normal site-packages.
"""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_with_da3_overlay.py ENTRYPOINT [ARGS ...]")
    paths = os.environ.get("MAGICIAN_DA3_APPEND_PATHS", "")
    for raw_path in paths.split(os.pathsep):
        if raw_path and raw_path not in sys.path:
            sys.path.append(raw_path)
    entrypoint = Path(sys.argv[1]).resolve()
    working_directory = str(Path.cwd())
    if working_directory not in sys.path:
        sys.path.insert(0, working_directory)
    sys.argv = [str(entrypoint), *sys.argv[2:]]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
