#!/usr/bin/env python3
"""Validate an isolated two-observation seam-gate run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--runner-return-code", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    frame = metrics["frames"][0]
    planning = metrics["cross_tile"]["planning"][0]
    seam_collision = planning["seam_segment_collision"]

    try:
        import torch

        capture = torch.load(
            Path(metrics["capture_dir"]) / "0.pt",
            map_location="cpu",
            weights_only=False,
        )
        rgb_finite = bool(torch.isfinite(capture["rgb"]).all())
        renderer_mask_pixels = int(capture["mask"].sum())
        renderer_depth = capture["zbuf"]
        finite_depth = torch.isfinite(renderer_depth) & capture["mask"]
        renderer_depth_finite_pixels = int(finite_depth.sum())
    except (ImportError, FileNotFoundError, KeyError, RuntimeError) as error:
        rgb_finite = False
        renderer_mask_pixels = 0
        renderer_depth_finite_pixels = 0
        capture_error = repr(error)
    else:
        capture_error = None

    depth_stats = frame["depth_scene_units"]
    checks = {
        "runner_return_code_zero": args.runner_return_code == 0,
        "start_explicitly_free": manifest["start_occupied"] is False,
        "two_observations_completed": metrics["trajectory"]["observation_count"] == 2,
        "frame_zero_has_valid_planning_depth": frame["planning_pixels"] > 0,
        "frame_zero_depth_stats_finite": all(
            value is None or math.isfinite(value) for value in depth_stats.values()
        ),
        "frame_zero_rgb_finite": rgb_finite,
        "frame_zero_renderer_mask_nonempty": renderer_mask_pixels > 0,
        "frame_zero_renderer_depth_nonempty": renderer_depth_finite_pixels > 0,
        "seam_gt_mesh_collision_clear": seam_collision["gt_mesh"] is False,
        "seam_predicted_point_cloud_collision_clear": (
            seam_collision["predicted_point_cloud"] is False
        ),
        "frame_zero_generated_tile_two_candidate": (
            planning["candidate_summary"]["generated_tile_2_count"] > 0
        ),
        "frame_zero_has_legal_candidate": planning["candidate_summary"]["legal_count"] > 0,
    }
    if manifest["label"] == "S":
        checks["S_direct_crossing_candidate_generated"] = planning[
            "direct_crossing_candidate_generated"
        ] is True
        checks["S_direct_crossing_candidate_legal"] = planning[
            "direct_crossing_candidate_legal"
        ] is True
    result = {
        "schema_version": 1,
        "label": manifest["label"],
        "accepted": all(checks.values()),
        "checks": checks,
        "start": {
            "grid_index": manifest["start_grid_index"],
            "pose_scene_units_and_degrees": manifest[
                "start_pose_scene_units_and_degrees"
            ],
            "occupied": manifest["start_occupied"],
        },
        "depth": {
            "planning_pixels": frame["planning_pixels"],
            "valid_pixels": frame["valid_pixels"],
            "scene_units": depth_stats,
            "renderer_mask_pixels": renderer_mask_pixels,
            "renderer_finite_depth_pixels": renderer_depth_finite_pixels,
            "rgb_finite": rgb_finite,
            "capture_error": capture_error,
        },
        "seam_segment_collision": seam_collision,
        "candidate_summary": planning["candidate_summary"],
    }
    output = Path(args.output)
    _write_json(output, result)
    print(output)
    if not result["accepted"]:
        raise SystemExit("cross-tile gate failed")


if __name__ == "__main__":
    main()
