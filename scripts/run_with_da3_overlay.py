#!/usr/bin/env python3
"""Run a Python entry point with optional DA3 paths appended safely.

``PYTHONPATH`` prepends directories and can accidentally shadow the validated
CUDA Torch build when a dependency overlay contains ``torch``.  This bootstrap
appends the colon-separated ``MAGICIAN_DA3_APPEND_PATHS`` entries only after
the active environment has initialized its normal site-packages.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys


def _append_overlay_paths() -> list[Path]:
    paths = os.environ.get("MAGICIAN_DA3_APPEND_PATHS", "")
    appended = []
    for raw_path in paths.split(os.pathsep):
        if raw_path and raw_path not in sys.path:
            sys.path.append(raw_path)
            appended.append(Path(raw_path).resolve())
    return appended


def _sha256_python_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _probe_da3_import(appended: list[Path]) -> None:
    import_output = io.StringIO()
    import_errors = io.StringIO()
    with contextlib.redirect_stdout(import_output), contextlib.redirect_stderr(
        import_errors
    ):
        import depth_anything_3.api as da3_api

    origin = Path(da3_api.__file__).resolve()
    package_root = origin.parent
    source_root = appended[0] if appended else None
    source_bound = False
    if source_root is not None:
        try:
            origin.relative_to(source_root)
            source_bound = True
        except ValueError:
            pass
    print(
        json.dumps(
            {
                "origin": str(origin),
                "package_root": str(package_root),
                "python_tree_sha256": _sha256_python_tree(package_root),
                "source_root": str(source_root) if source_root is not None else None,
                "source_bound": source_bound,
                "import_stdout": import_output.getvalue(),
                "import_stderr": import_errors.getvalue(),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: run_with_da3_overlay.py ENTRYPOINT [ARGS ...] | --probe-da3-import"
        )
    appended = _append_overlay_paths()
    if sys.argv[1] == "--probe-da3-import":
        _probe_da3_import(appended)
        return
    entrypoint = Path(sys.argv[1]).resolve()
    working_directory = str(Path.cwd())
    if working_directory not in sys.path:
        sys.path.insert(0, working_directory)
    sys.argv = [str(entrypoint), *sys.argv[2:]]
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
