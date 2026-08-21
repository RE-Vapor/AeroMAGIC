#!/usr/bin/env python3
"""Validate a raw PAN-15 bundle and publish its candidate Contract bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from macarons.utility.ue5_capture_adapter import (
    adapt_pan15_raw_manifest,
    raw_capture_summary,
)
from macarons.utility.ue5_observation_contract import (
    load_bundle,
    validate_bundle,
    write_bundle,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    bundle = adapt_pan15_raw_manifest(args.raw_manifest)
    manifest = write_bundle(bundle, args.output)
    summary = {
        "task": "PAN-15",
        "result": "PASS",
        "raw": raw_capture_summary(args.raw_manifest),
        "canonical": validate_bundle(load_bundle(manifest)).as_dict(),
        "canonical_manifest": "canonical_bundle/manifest.json",
        "canonical_manifest_sha256": sha256(manifest),
        "geometry_status": bundle.provenance["geometry_status"],
    }
    summary_path = manifest.parent / "adapter_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
