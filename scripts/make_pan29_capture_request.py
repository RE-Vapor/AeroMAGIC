#!/usr/bin/env python3
"""Create one immutable PAN-29 UE5 capture request from a replay plan row."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


LEVEL_PATH = "/Game/PAN13_Derived/HKUST_ZUp_QA"


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def build_capture_request(*, plan_path: Path, observation_id: int) -> dict[str, Any]:
    plan_path = Path(plan_path).expanduser().resolve()
    plan = _read_json(plan_path)
    if plan.get("schema_version") != "pan29.ue5-postrun-replay.v1":
        raise ValueError("unexpected PAN-29 replay plan schema")
    rows = plan.get("observations")
    if not isinstance(rows, list):
        raise ValueError("replay plan observations must be a list")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("observation_id") == observation_id]
    if len(matches) != 1:
        raise ValueError(f"expected one replay row for observation {observation_id}")
    row = matches[0]
    ue_position = [float(value) for value in row.get("ue_position_cm", ())]
    planner_position = [float(value) for value in row.get("planner_position_scene_units", ())]
    if len(ue_position) != 3 or len(planner_position) != 3 or not all(
        math.isfinite(value) for value in (*ue_position, *planner_position)
    ):
        raise ValueError("replay row positions must contain three finite values")
    timestamp_ns = row.get("source_capture_timestamp_ns")
    timestamp_utc = row.get("source_capture_timestamp_utc")
    if type(timestamp_ns) is not int or timestamp_ns <= 0 or not isinstance(timestamp_utc, str):
        raise ValueError("replay row lacks an exact source timestamp")
    metrics = plan.get("source", {}).get("metrics", {})
    metrics_sha = metrics.get("sha256") if isinstance(metrics, Mapping) else None
    if not isinstance(metrics_sha, str) or len(metrics_sha) != 64:
        raise ValueError("replay plan lacks its source metrics SHA256")
    return {
        "schema_version": "pan15.capture-request.v1",
        "scenario": "hkust",
        "level_path": LEVEL_PATH,
        "position_actor_label": None,
        "position_ue_cm": ue_position,
        "request_id": str(row["replay_request_id"]),
        "frame_id": str(row["replay_frame_id"]),
        "replay_source": {
            "schema_version": "pan29.replay-source.v1",
            "observation_id": observation_id,
            "source_bundle_id": observation_id,
            "source_capture_timestamp_ns": timestamp_ns,
            "source_capture_timestamp_utc": timestamp_utc,
            "planner_position_scene_units": planner_position,
            "requested_position_ue_cm": ue_position,
            "source_metrics_sha256": metrics_sha,
            "within_pan13_conservative_fly_volume": bool(
                row.get("within_pan13_conservative_fly_volume")
            ),
            "planner_input_unchanged": True,
            "ue5_role": "post_run_visualization_only",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--observation-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite capture request: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_capture_request(plan_path=args.plan, observation_id=args.observation_id)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"observation_id": args.observation_id, "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
