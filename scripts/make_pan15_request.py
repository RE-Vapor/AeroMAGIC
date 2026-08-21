#!/usr/bin/env python3
"""Create one immutable file-exchange request for PAN-15."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=("analytic", "hkust"))
    parser.add_argument("--level", required=True)
    parser.add_argument("--position-actor-label")
    parser.add_argument("--position-ue-cm", nargs=3, type=float)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.position_actor_label and args.position_ue_cm is not None:
        parser.error("position actor label and explicit UE position are mutually exclusive")
    if args.position_ue_cm is not None and not all(
        math.isfinite(value) for value in args.position_ue_cm
    ):
        parser.error("explicit UE position must contain three finite values")
    payload = {
        "schema_version": "pan15.capture-request.v1",
        "scenario": args.scenario,
        "level_path": args.level,
        "position_actor_label": args.position_actor_label,
        "position_ue_cm": args.position_ue_cm,
        "request_id": args.request_id,
        "frame_id": args.frame_id,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
