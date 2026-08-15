#!/usr/bin/env python3
"""Derive a bounded runtime sensor range from scene and gate evidence."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def _load(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new_json(path: Path, value: Any) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite calibration report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be positive.")
    return float(value)


def _bounds(value: Mapping[str, Sequence[float]], scale: float) -> list[list[float]]:
    minimum = value.get("minimum", value.get("min"))
    maximum = value.get("maximum", value.get("max"))
    if minimum is None or maximum is None or len(minimum) != 3 or len(maximum) != 3:
        raise ValueError("bounds must contain three-dimensional minimum/maximum values.")
    result = [
        [float(coordinate) * scale for coordinate in minimum],
        [float(coordinate) * scale for coordinate in maximum],
    ]
    if any(low >= high for low, high in zip(*result)):
        raise ValueError("every bounds maximum must exceed its minimum.")
    return result


def _corners(bounds: Sequence[Sequence[float]]) -> list[tuple[float, float, float]]:
    return list(itertools.product(*zip(bounds[0], bounds[1])))


def _maximum_corner_distance(
    left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]
) -> float:
    return max(math.dist(a, b) for a in _corners(left) for b in _corners(right))


def derive_sensor_range(
    *,
    scene: str,
    assembly: Mapping[str, Any],
    settings: Mapping[str, Any],
    params: Mapping[str, Any],
    experiment: Mapping[str, Any],
    gate_evidence: Sequence[Mapping[str, Any]],
    depth_quantile_label: str = "p90",
    depth_quantile: float = 0.9,
    safety_margin_fraction: float = 0.1,
    rounding_quantum: float = 5.0,
) -> Mapping[str, Any]:
    if not gate_evidence:
        raise ValueError("at least one gate-evidence file is required.")
    if not 0.0 < float(depth_quantile) <= 1.0:
        raise ValueError("depth_quantile must be in (0, 1].")
    if not 0.0 <= float(safety_margin_fraction) <= 1.0:
        raise ValueError("safety_margin_fraction must be in [0, 1].")
    rounding_quantum = _positive(rounding_quantum, "rounding_quantum")

    scene_scale = _positive(
        params["_data"]["scene_scale_factor"], "scene_scale_factor"
    )
    configured_range = _positive(
        params["_camera_management"]["sensor_range"], "sensor_range"
    )
    znear = _positive(params["_depth_module"]["znear"], "znear")
    zfar = _positive(params["_depth_module"]["zfar"], "zfar")
    if zfar <= znear:
        raise ValueError("zfar must exceed znear.")

    assembly_bounds = assembly["output"]["bounds"]
    renderer_bounds = _bounds(assembly_bounds, scene_scale)
    camera_bounds = _bounds(
        {
            "minimum": settings["camera"]["x_min"],
            "maximum": settings["camera"]["x_max"],
        },
        scene_scale,
    )
    geometry_upper_bound = _maximum_corner_distance(camera_bounds, renderer_bounds)
    hard_cap = min(zfar, geometry_upper_bound)

    evidence_rows = []
    for evidence in gate_evidence:
        depth = evidence["depth"]["scene_units"]
        observed = _positive(
            depth[depth_quantile_label],
            f"gate depth {depth_quantile_label}",
        )
        evidence_rows.append(
            {
                "label": evidence.get("label"),
                "depth_quantile_scene_units": observed,
                "depth_min_scene_units": depth.get("min"),
                "depth_median_scene_units": depth.get("median"),
                "depth_max_scene_units": depth.get("max"),
                "valid_pixels": evidence["depth"].get("valid_pixels"),
            }
        )
    controlling_depth = max(row["depth_quantile_scene_units"] for row in evidence_rows)
    unrounded = controlling_depth * (1.0 + float(safety_margin_fraction))
    recommended = math.ceil(unrounded / rounding_quantum) * rounding_quantum
    if recommended > hard_cap:
        raise ValueError(
            "calibrated sensor range exceeds the geometry/renderer hard cap: "
            f"recommended={recommended:.3f}, cap={hard_cap:.3f}"
        )

    manifest_scale = _positive(
        assembly["shared_transform"]["scale"], "assembly shared_transform.scale"
    )
    scene_units_per_meter = _positive(
        experiment["da3_scene_units_per_meter"][scene],
        f"da3_scene_units_per_meter.{scene}",
    )
    return {
        "schema_version": 1,
        "scene": scene,
        "unit_contract": {
            "threshold_unit": "runtime_scene_unit",
            "assembly_source_to_obj_scale": manifest_scale,
            "magician_obj_to_runtime_scale": scene_scale,
            "assembly_source_to_runtime_scale": manifest_scale * scene_scale,
            "runtime_scene_units_per_meter": scene_units_per_meter,
            "distance_consumers": {
                "sensor_range": "Euclidean runtime scene units",
                "renderer_znear_zfar": "view-depth runtime scene units",
                "carving_tolerance": "signed-distance runtime scene units",
                "coverage_surface_epsilon": "runtime scene units",
                "collision": "runtime scene coordinates; no range threshold",
            },
        },
        "geometry": {
            "renderer_bounds_runtime_scene_units": renderer_bounds,
            "camera_lattice_bounds_runtime_scene_units": camera_bounds,
            "renderer_bbox_diagonal_scene_units": math.dist(
                renderer_bounds[0], renderer_bounds[1]
            ),
            "camera_to_renderer_corner_upper_bound_scene_units": geometry_upper_bound,
        },
        "observed_depth_distribution": {
            "quantile_label": depth_quantile_label,
            "quantile": float(depth_quantile),
            "evidence": evidence_rows,
            "controlling_depth_scene_units": controlling_depth,
        },
        "calibration": {
            "formula": (
                "ceil(max(start_depth_quantile) * (1 + safety_margin_fraction) "
                "/ rounding_quantum) * rounding_quantum"
            ),
            "configured_sensor_range_scene_units": configured_range,
            "safety_margin_fraction": float(safety_margin_fraction),
            "rounding_quantum_scene_units": rounding_quantum,
            "unrounded_recommendation_scene_units": unrounded,
            "recommended_sensor_range_scene_units": recommended,
            "renderer_znear_scene_units": znear,
            "renderer_zfar_scene_units": zfar,
            "hard_cap_formula": (
                "min(renderer_zfar, camera_to_renderer_corner_upper_bound)"
            ),
            "hard_cap_scene_units": hard_cap,
            "requires_override": recommended != configured_range,
            "failure_condition": "recommended_sensor_range > hard_cap",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--assembly-manifest", required=True)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--params-config", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--gate-evidence", action="append", required=True)
    parser.add_argument("--depth-quantile-label", default="p90")
    parser.add_argument("--depth-quantile", type=float, default=0.9)
    parser.add_argument("--safety-margin-fraction", type=float, default=0.1)
    parser.add_argument("--rounding-quantum", type=float, default=5.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = derive_sensor_range(
        scene=args.scene,
        assembly=_load(Path(args.assembly_manifest)),
        settings=_load(Path(args.settings)),
        params=_load(Path(args.params_config)),
        experiment=_load(Path(args.experiment_config)),
        gate_evidence=[_load(Path(path)) for path in args.gate_evidence],
        depth_quantile_label=args.depth_quantile_label,
        depth_quantile=args.depth_quantile,
        safety_margin_fraction=args.safety_margin_fraction,
        rounding_quantum=args.rounding_quantum,
    )
    output = Path(args.output)
    _write_new_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
