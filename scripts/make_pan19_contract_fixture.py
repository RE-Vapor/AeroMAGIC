#!/usr/bin/env python3
"""Generate and verify the deterministic PAN-19 Contract v1 fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from macarons.utility.ue5_observation_contract import (
    BundleValidationError,
    load_bundle,
    make_synthetic_plane_bundle,
    validate_bundle,
    write_bundle,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=5)
    parser.add_argument("--plane-camera-z-m", type=float, default=2.0)
    args = parser.parse_args()
    bundle = make_synthetic_plane_bundle(
        image_size=args.image_size,
        plane_camera_z_m=args.plane_camera_z_m,
    )
    manifest = write_bundle(bundle, args.output)
    summary = validate_bundle(load_bundle(manifest)).as_dict()
    negative_manifest = manifest.parent / "negative_missing_down_manifest.json"
    negative_payload = json.loads(manifest.read_text(encoding="utf-8"))
    negative_payload["faces"] = negative_payload["faces"][:-1]
    negative_payload["face_names"] = negative_payload["face_names"][:-1]
    negative_manifest.write_text(
        json.dumps(negative_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        load_bundle(negative_manifest)
    except BundleValidationError as error:
        negative_result = {"result": "PASS", "rejection": str(error)}
    else:
        raise RuntimeError("negative missing-face fixture was unexpectedly accepted")
    summary.update(
        {
            "task": "PAN-19",
            "manifest": str(manifest),
            "manifest_sha256": sha256(manifest),
            "negative_missing_face_manifest": str(negative_manifest),
            "negative_missing_face_manifest_sha256": sha256(negative_manifest),
            "negative_missing_face": negative_result,
            "result": "PASS",
        }
    )
    summary_path = manifest.parent / "validation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
