#!/usr/bin/env python3
"""Aggregate machine-readable 12-NW-6C-5 online/diagnostic results."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": fmean(values),
        "population_stddev": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def _nested(value: Any, *keys: str) -> Any:
    for key in keys:
        if value is None:
            return None
        value = value.get(key)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.results_root).resolve()
    output = Path(args.output).resolve()
    rows = []
    for online_path in sorted(root.rglob("*.online.json")):
        online = _read(online_path)
        if online.get("scene") != "12-NW-6C-5":
            continue
        diagnostic_path = online_path.with_name(
            online_path.name.replace(".online.json", ".diagnostic.json")
        )
        diagnostic = _read(diagnostic_path) if diagnostic_path.exists() else None
        coverage = online.get("coverage", [])
        frames = online.get("frames", [])
        aggregate = diagnostic.get("aggregate") if diagnostic else None
        rows.append(
            {
                "run_id": online.get("run", {}).get("run_id"),
                "planner": online.get("planner"),
                "source": frames[0].get("source") if frames else None,
                "start_index": online.get("start_index"),
                "observations": online.get("trajectory", {}).get("observation_count"),
                "final_normalized_coverage": coverage[-1].get("normalized") if coverage else None,
                "final_raw_coverage": coverage[-1].get("raw") if coverage else None,
                "best_normalized_coverage": max(
                    (item.get("normalized", 0.0) for item in coverage), default=None
                ),
                "final_point_count": online.get("trajectory", {}).get("final_point_count"),
                "path_length_meters": online.get("trajectory", {}).get("path_length_meters"),
                "provider_seconds": online.get("latency", {}).get("provider_seconds"),
                "trajectory_seconds": online.get("latency", {}).get("trajectory_seconds"),
                "peak_allocated_mib": online.get("cuda", {}).get("peak_allocated_mib"),
                "peak_reserved_mib": online.get("cuda", {}).get("peak_reserved_mib"),
                "mean_planning_ratio": (
                    sum(frame.get("planning_ratio", 0.0) for frame in frames) / len(frames)
                    if frames
                    else None
                ),
                "cache_hits": sum(bool(frame.get("cache_hit")) for frame in frames),
                "pose_conditioning_fallbacks": [
                    {
                        "frame_id": frame.get("frame_id"),
                        "reason": frame.get("pose_conditioning_fallback"),
                    }
                    for frame in frames
                    if frame.get("pose_conditioning_fallback")
                ],
                "diagnostic": {
                    "paired_pixels": aggregate.get("paired_pixels"),
                    "depth_meters": aggregate.get("depth_meters"),
                    "paired_ray_geometry_meters": aggregate.get(
                        "paired_ray_geometry_meters"
                    ),
                    "confidence": aggregate.get("confidence"),
                    "test_fit_diagnostic_only": aggregate.get(
                        "test_fit_diagnostic_only"
                    ),
                }
                if aggregate
                else None,
                "online_only": online.get("online_only"),
                "renderer_gt_read_online": online.get("renderer_gt_read"),
                "diagnostic_feedback_to_online_planner": (
                    diagnostic.get("feedback_to_online_planner") if diagnostic else None
                ),
                "online_json": str(online_path.relative_to(root)),
                "diagnostic_json": (
                    str(diagnostic_path.relative_to(root)) if diagnostic else None
                ),
            }
        )
    groups = defaultdict(list)
    for row in rows:
        groups[row["run_id"]].append(row)
    aggregates = {}
    for run_id, run_rows in sorted(groups.items()):
        run_rows.sort(key=lambda item: item["start_index"])
        diagnostic_fields = {
            "depth_mae_meters": ("depth_meters", "mae"),
            "depth_rmse_meters": ("depth_meters", "rmse"),
            "depth_abs_rel": ("depth_meters", "abs_rel"),
            "depth_delta_1_25": ("depth_meters", "delta_1_25"),
            "ray_geometry_mae_meters": ("paired_ray_geometry_meters", "mae"),
            "confidence_error_pearson": (
                "confidence",
                "pearson_confidence_vs_abs_error",
            ),
            "test_fit_scale_only_factor": (
                "test_fit_diagnostic_only",
                "scale_only_factor",
            ),
        }
        diagnostic_summary = {
            name: _distribution(values)
            for name, path in diagnostic_fields.items()
            if (
                values := [
                    value
                    for row in run_rows
                    if (value := _nested(row["diagnostic"], *path)) is not None
                ]
            )
        }
        diagnostic_summary["paired_pixels_total"] = sum(
            _nested(row["diagnostic"], "paired_pixels") or 0 for row in run_rows
        )
        aggregates[run_id] = {
            "planner": run_rows[0]["planner"],
            "source": run_rows[0]["source"],
            "start_indices": [row["start_index"] for row in run_rows],
            "observations": sorted({row["observations"] for row in run_rows}),
            "final_normalized_coverage": _distribution(
                [row["final_normalized_coverage"] for row in run_rows]
            ),
            "final_raw_coverage": _distribution(
                [row["final_raw_coverage"] for row in run_rows]
            ),
            "final_point_count": _distribution(
                [row["final_point_count"] for row in run_rows]
            ),
            "path_length_meters": _distribution(
                [row["path_length_meters"] for row in run_rows]
            ),
            "provider_seconds": _distribution(
                [row["provider_seconds"] for row in run_rows]
            ),
            "trajectory_seconds": _distribution(
                [row["trajectory_seconds"] for row in run_rows]
            ),
            "peak_allocated_mib": _distribution(
                [row["peak_allocated_mib"] for row in run_rows]
            ),
            "pose_conditioning_fallback_count": sum(
                len(row["pose_conditioning_fallbacks"]) for row in run_rows
            ),
            "post_run_diagnostic_per_start": diagnostic_summary,
        }
    main_groups = {
        run_id: aggregate
        for run_id, aggregate in aggregates.items()
        if run_id.startswith("main_")
    }
    fairness_checks = {
        "main_run_ids": sorted(main_groups),
        "expected_main_run_ids_present": set(main_groups)
        == {
            "main_scone_gt",
            "main_scone_da3",
            "main_magician_gt",
            "main_magician_da3",
        },
        "each_main_run_has_starts_0_through_4": all(
            aggregate["start_indices"] == [0, 1, 2, 3, 4]
            for aggregate in main_groups.values()
        ),
        "each_main_run_has_101_observations": all(
            aggregate["observations"] == [101]
            for aggregate in main_groups.values()
        ),
        "online_renderer_gt_reads_absent": all(
            row["renderer_gt_read_online"] is False for row in rows
        ),
        "diagnostic_feedback_absent": all(
            row["diagnostic_feedback_to_online_planner"] is False
            for row in rows
            if row["diagnostic"] is not None
        ),
    }
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scene": "12-NW-6C-5",
        "scope": "single_scene_only",
        "results_root": str(root),
        "run_count": len(rows),
        "runs": rows,
        "aggregates": aggregates,
        "fairness_checks": fairness_checks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
