#!/usr/bin/env python3
"""Create one immutable file-exchange request for PAN-15."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=("analytic", "hkust"))
    parser.add_argument("--level", required=True)
    parser.add_argument("--position-actor-label")
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = {
        "schema_version": "pan15.capture-request.v1",
        "scenario": args.scenario,
        "level_path": args.level,
        "position_actor_label": args.position_actor_label,
        "request_id": args.request_id,
        "frame_id": args.frame_id,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
