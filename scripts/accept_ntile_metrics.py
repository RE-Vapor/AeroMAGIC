#!/usr/bin/env python3
"""Fail closed on generic N-tile gate or full-run telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _load_metrics(path: Path) -> Mapping[str, Any]:
    candidates = [path] if path.is_file() else sorted(path.glob("*.online.json"))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one online metrics file, found {len(candidates)}")
    return json.loads(candidates[0].read_text(encoding="utf-8"))


def evaluate(
    manifest: Mapping[str, Any], metrics: Mapping[str, Any], *, mode: str
) -> Mapping[str, Any]:
    expected_partition = manifest["tile_partition"]
    tiled = metrics.get("tile_metrics")
    checks = {
        "scene_matches": metrics.get("scene") == manifest.get("scene"),
        "run_id_matches": metrics.get("run", {}).get("run_id")
        == manifest.get("run_id") + ("_gate" if mode == "gate" else ""),
        "tile_metrics_present": isinstance(tiled, Mapping),
    }
    coverage = [] if not isinstance(tiled, Mapping) else tiled.get("coverage", [])
    checks["partition_matches"] = bool(tiled) and tiled.get("partition") == expected_partition
    checks["coverage_nonempty"] = bool(coverage)
    tolerance = 1e-8
    checks["raw_recombination_exact"] = bool(tiled) and float(
        tiled.get("recombination_max_abs_raw_error", float("inf"))
    ) <= tolerance
    checks["normalized_recombination_exact"] = bool(tiled) and float(
        tiled.get("recombination_max_abs_normalized_error", float("inf"))
    ) <= tolerance
    frames = metrics.get("frames", [])
    first_gate = frames[0].get("sensor_range_gate") if frames else None
    checks["range_gate_accepted"] = bool(first_gate) and first_gate.get("accepted") is True
    checks["first_frame_has_partial_points"] = bool(frames) and int(
        frames[0].get("partial_point_count", 0)
    ) >= 1
    if mode == "full":
        final_tiles = coverage[-1].get("tiles", {}) if coverage else {}
        checks["all_tiles_have_reference_points"] = set(final_tiles) == set(
            expected_partition["tile_ids"]
        ) and all(int(tile.get("reference_points", 0)) > 0 for tile in final_tiles.values())
        checks["all_tiles_have_covered_points"] = bool(final_tiles) and all(
            int(tile.get("covered_points", 0)) > 0 for tile in final_tiles.values()
        )
    accepted = all(checks.values())
    return {
        "schema_version": 1,
        "mode": mode,
        "accepted": accepted,
        "checks": checks,
        "tile_partition": expected_partition,
        "observation_count": len(metrics.get("trajectory", {}).get("positions", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--mode", choices=("gate", "full"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    result = evaluate(manifest, _load_metrics(Path(args.metrics)), mode=args.mode)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(output)
    if not result["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
