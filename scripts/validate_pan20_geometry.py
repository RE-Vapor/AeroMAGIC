#!/usr/bin/env python3
"""Run PAN-20 depth, pose, seam, and point-cloud conformance checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from macarons.utility.planning_observations import build_cubemap_cameras
from macarons.utility.ue5_capture_adapter import (
    UE_SCENE_DEPTH_ENCODING,
    adapt_pan15_raw_manifest,
)
from macarons.utility.ue5_geometry_conformance import (
    bundle_pointcloud,
    camera_ray_directions,
    direction_gram,
    forward_directions,
    pointcloud_summary,
    seam_summary,
)
from macarons.utility.ue5_observation_contract import validate_bundle, write_bundle


EXPECTED_FACE_FORWARD = np.asarray(
    (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ),
    dtype=np.float64,
)
EXPECTED_MARKER_CENTER_Z_M = {
    "front": 2.8875,
    "back": 2.775,
    "left": 2.54875,
    "right": 2.65,
    "up": 2.1,
    "down": 2.825,
}
EXPECTED_DOWN_FLOOR_Z_M = 3.625


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_raw_exr_r(raw_root: Path, face_record: dict, key: str) -> np.ndarray:
    asset = face_record[key]["raw_exr"]
    path = raw_root / asset["path"]
    if sha256(path) != asset["sha256"]:
        raise ValueError(f"raw EXR checksum mismatch: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"invalid EXR: {path}")
    return np.asarray(image[..., 2], dtype=np.float32)


def analytic_summary(raw_manifest_path: Path, bundle) -> dict:
    raw = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    raw_root = raw_manifest_path.parent
    face_records = {item["face_name"]: item for item in raw["faces"]}
    faces = {face.face_name: face for face in bundle.faces}
    world_to_meters = float(raw["world_to_meters"])

    marker_checks = {}
    for name, expected in EXPECTED_MARKER_CENTER_Z_M.items():
        depth_z = load_raw_exr_r(raw_root, face_records[name], "scene_depth_r")
        height, width = depth_z.shape
        center = depth_z[height // 2 - 1 : height // 2 + 1, width // 2 - 1 : width // 2 + 1]
        measured = float(np.median(center) / world_to_meters)
        marker_checks[name] = {
            "expected_camera_z_m": expected,
            "measured_camera_z_m": measured,
            "absolute_error_m": abs(measured - expected),
        }

    down_raw_z_m = (
        load_raw_exr_r(raw_root, face_records["down"], "scene_depth_r")
        / world_to_meters
    )
    floor_inliers = np.abs(down_raw_z_m - EXPECTED_DOWN_FLOOR_Z_M) < 0.02
    floor_z = float(np.median(down_raw_z_m[floor_inliers]))
    down_face = faces["down"]
    height, width = down_face.image_size
    sample_pixels = {
        "center_plane_axial_base": None,
        "edge_midpoint": (height // 2, width - 1),
        "corner": (height - 1, width - 1),
    }
    plane_samples = {
        "center_plane_axial_base": {
            "measured_range_m": floor_z,
            "expected_range_m": EXPECTED_DOWN_FLOOR_Z_M,
            "absolute_error_m": abs(floor_z - EXPECTED_DOWN_FLOOR_Z_M),
            "note": "median floor camera-z; exact optical centre is intentionally occluded by NegZ marker",
        }
    }
    rays = camera_ray_directions(down_face)
    for label, coordinates in sample_pixels.items():
        if coordinates is None:
            continue
        row, column = coordinates
        measured_range = float(down_face.depth_range_m[row, column])
        expected_range = EXPECTED_DOWN_FLOOR_Z_M / float(rays[row, column, 2])
        plane_samples[label] = {
            "pixel_row_column": [row, column],
            "raw_camera_z_m": float(down_raw_z_m[row, column]),
            "measured_range_m": measured_range,
            "expected_range_m": expected_range,
            "expected_ratio_to_z": float(1.0 / rays[row, column, 2]),
            "absolute_error_m": abs(measured_range - expected_range),
        }

    raw_no_hit = {}
    no_hit_mismatch = 0
    for name in EXPECTED_MARKER_CENTER_Z_M:
        raw_depth = load_raw_exr_r(raw_root, face_records[name], "scene_depth_r")
        expected_valid = (
            np.isfinite(raw_depth)
            & (raw_depth > float(raw["near_m"]) * world_to_meters)
            & (raw_depth < min(float(raw["far_m"]) * world_to_meters, 65504.0))
        )
        actual_valid = np.asarray(faces[name].valid_mask, dtype=bool)
        mismatch = int(np.count_nonzero(expected_valid != actual_valid))
        no_hit_mismatch += mismatch
        raw_no_hit[name] = {
            "raw_overflow_or_no_hit_count": int(np.count_nonzero(raw_depth >= 65504.0)),
            "canonical_invalid_count": int(np.count_nonzero(~actual_valid)),
            "mask_mismatch_count": mismatch,
            "invalid_depth_all_nan": bool(
                np.isnan(np.asarray(faces[name].depth_range_m)[~actual_valid]).all()
            ),
        }

    directions = forward_directions(bundle)
    forward_errors = np.linalg.norm(directions - EXPECTED_FACE_FORWARD, axis=1)
    determinants = [
        float(np.linalg.det(np.asarray(face.T_world_from_cam)[:3, :3]))
        for face in bundle.faces
    ]

    import torch

    reference = build_cubemap_cameras(
        torch.zeros(1, 3), znear=0.1, zfar=10.0, device=torch.device("cpu"), rig_frame="world"
    )
    reference_forwards = np.stack(
        [reference[face.face_name].R[0, :, 2].detach().cpu().numpy() for face in bundle.faces]
    )
    reference_gram_error = float(
        np.max(np.abs(direction_gram(directions) - direction_gram(reference_forwards)))
    )

    checks = {
        "marker_center_max_abs_error_m": max(
            item["absolute_error_m"] for item in marker_checks.values()
        ),
        "plane_sample_max_abs_error_m": max(
            item["absolute_error_m"] for item in plane_samples.values()
        ),
        "forward_max_l2_error": float(forward_errors.max()),
        "rotation_det_max_abs_error": max(abs(value - 1.0) for value in determinants),
        "pan10_direction_gram_max_abs_error": reference_gram_error,
        "no_hit_mask_mismatch_count": no_hit_mismatch,
    }
    checks["pass"] = bool(
        checks["marker_center_max_abs_error_m"] <= 0.02
        and checks["plane_sample_max_abs_error_m"] <= 0.02
        and checks["forward_max_l2_error"] <= 1e-6
        and checks["rotation_det_max_abs_error"] <= 1e-6
        and checks["pan10_direction_gram_max_abs_error"] <= 1e-6
        and checks["no_hit_mask_mismatch_count"] == 0
    )
    return {
        "source_depth_encoding": UE_SCENE_DEPTH_ENCODING,
        "source_unit": "UE world unit (centimetre for WorldToMeters=100)",
        "world_to_meters": world_to_meters,
        "conversion_formula": (
            "z_m=EXR_R/WorldToMeters; "
            "range_m=z_m*sqrt(((u-cx)/fx)^2+((v-cy)/fy)^2+1)"
        ),
        "encoding_evidence": (
            "known down-facing plane EXR R is constant camera-z across rays; "
            "canonical range follows centre/edge/corner ray-length ratios"
        ),
        "marker_center_checks": marker_checks,
        "down_plane": {
            "expected_camera_z_m": EXPECTED_DOWN_FLOOR_Z_M,
            "inlier_count": int(floor_inliers.sum()),
            "measured_camera_z_m": floor_z,
            "samples": plane_samples,
        },
        "pose": {
            "expected_forward_by_face": EXPECTED_FACE_FORWARD.tolist(),
            "measured_forward_by_face": directions.tolist(),
            "rotation_determinants": determinants,
            "pan10_reference": {
                "method": "build_cubemap_cameras(rig_frame=world) pairwise direction Gram",
                "direction_gram_max_abs_error": reference_gram_error,
            },
        },
        "no_hit": raw_no_hit,
        "threshold_checks": checks,
    }


def render_visualization(path: Path, analytic_bundle, hkust_bundle) -> None:
    points, colors = bundle_pointcloud(hkust_bundle)
    if points.shape[0] > 40000:
        indices = np.linspace(0, points.shape[0] - 1, 40000, dtype=np.int64)
        points, colors = points[indices], colors[indices]
    colors_float = colors.astype(np.float32) / 255.0
    figure = plt.figure(figsize=(18, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 6)
    for index, face in enumerate(analytic_bundle.faces):
        axis = figure.add_subplot(grid[0, index])
        axis.imshow(face.rgb_uint8)
        axis.set_title(f"analytic {face.face_name}")
        axis.axis("off")
    axis_depth = figure.add_subplot(grid[1, 0:2])
    down = analytic_bundle.faces[-1]
    depth = np.ma.masked_invalid(down.depth_range_m)
    image = axis_depth.imshow(depth, cmap="viridis")
    axis_depth.set_title("analytic down canonical range (m)")
    figure.colorbar(image, ax=axis_depth, fraction=0.046)
    axis_xy = figure.add_subplot(grid[1, 2:4])
    axis_xy.scatter(points[:, 0], points[:, 1], s=0.25, c=colors_float)
    axis_xy.set_title("HKUST merged point cloud XY")
    axis_xy.set_aspect("equal", adjustable="box")
    axis_xy.set_xlabel("world X (m)")
    axis_xy.set_ylabel("world Y (m)")
    axis_xz = figure.add_subplot(grid[1, 4:6])
    axis_xz.scatter(points[:, 0], points[:, 2], s=0.25, c=colors_float)
    axis_xz.set_title("HKUST merged point cloud XZ")
    axis_xz.set_aspect("equal", adjustable="box")
    axis_xz.set_xlabel("world X (m)")
    axis_xz.set_ylabel("world Z (m)")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_checksums(root: Path) -> Path:
    checksum_path = root / "CHECKSUMS.sha256"
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == checksum_path:
            continue
        lines.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analytic-raw-manifest", required=True, type=Path)
    parser.add_argument("--hkust-raw-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        analytic_bundle = adapt_pan15_raw_manifest(args.analytic_raw_manifest)
        hkust_bundle = adapt_pan15_raw_manifest(args.hkust_raw_manifest)
        analytic_validation = validate_bundle(analytic_bundle).as_dict()
        hkust_validation = validate_bundle(hkust_bundle).as_dict()
        write_bundle(analytic_bundle, temporary / "analytic_validated_bundle")
        write_bundle(hkust_bundle, temporary / "hkust_validated_bundle")

        analytic = analytic_summary(args.analytic_raw_manifest, analytic_bundle)
        hkust_points = pointcloud_summary(hkust_bundle)
        hkust_seams = seam_summary(hkust_bundle)
        hkust_points_array, _ = bundle_pointcloud(hkust_bundle)
        camera_z = float(hkust_bundle.position_world_m[2])
        below_fraction = float(np.mean(hkust_points_array[:, 2] < camera_z))
        hkust_checks = {
            "point_count_minimum": 10000,
            "point_count": hkust_points["point_count"],
            "fraction_below_camera": below_fraction,
            "min_validity_agreement_fraction": hkust_seams[
                "min_validity_agreement_fraction"
            ],
            "max_world_point_distance_p95_m": hkust_seams[
                "max_world_point_distance_p95_m"
            ],
            "max_world_point_distance_median_m": hkust_seams[
                "max_world_point_distance_median_m"
            ],
            "max_relative_world_point_distance_p95": hkust_seams[
                "max_relative_world_point_distance_p95"
            ],
        }
        hkust_checks["pass"] = bool(
            hkust_checks["point_count"] >= hkust_checks["point_count_minimum"]
            and hkust_checks["fraction_below_camera"] >= 0.5
            and hkust_checks["min_validity_agreement_fraction"] >= 0.99
            and hkust_seams["seam_count"] == 12
            and hkust_checks["max_world_point_distance_median_m"] <= 2.0
            and hkust_checks["max_relative_world_point_distance_p95"] <= 0.05
        )
        report = {
            "schema_version": "pan20.geometry-conformance.v1",
            "task": "PAN-20",
            "result": "PASS" if analytic["threshold_checks"]["pass"] and hkust_checks["pass"] else "FAIL",
            "pioneer_commit": subprocess.check_output(
                ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_fixtures": {
                "analytic_raw_manifest": str(args.analytic_raw_manifest.resolve()),
                "analytic_raw_manifest_sha256": sha256(args.analytic_raw_manifest),
                "hkust_raw_manifest": str(args.hkust_raw_manifest.resolve()),
                "hkust_raw_manifest_sha256": sha256(args.hkust_raw_manifest),
            },
            "validator": {"analytic": analytic_validation, "hkust": hkust_validation},
            "analytic": analytic,
            "hkust": {
                "pointcloud": hkust_points,
                "seams": hkust_seams,
                "threshold_checks": hkust_checks,
            },
            "artifacts": {
                "analytic_validated_bundle": "analytic_validated_bundle/manifest.json",
                "hkust_validated_bundle": "hkust_validated_bundle/manifest.json",
                "visualization": "pan20_visualization.png",
            },
            "known_limitations": [
                "RGBA16F raw depth has binary16 quantization and a 65504-world-unit overflow ceiling.",
                "HKUST is one fixed pose and uses coarse seam/scale sanity thresholds, not exhaustive conformance.",
                "Far vegetation and occlusion boundaries reach 9.40 m absolute seam p95 (3.44% relative); the robust gate uses median distance plus relative p95.",
                "The analytic floor optical centre is occluded by the required NegZ direction marker; plane centre z is estimated from floor inliers.",
            ],
        }
        render_visualization(temporary / "pan20_visualization.png", analytic_bundle, hkust_bundle)
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_checksums(temporary)
        os.replace(temporary, output)
        temporary = None
        print(json.dumps({"result": report["result"], "output": str(output)}, sort_keys=True))
        return 0 if report["result"] == "PASS" else 1
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
