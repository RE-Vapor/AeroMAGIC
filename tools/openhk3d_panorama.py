#!/usr/bin/env python3
"""Generate a deterministic GT equirectangular panorama pilot with Blender."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence


PANORAMA_SCHEMA = "openhk3d-panorama-v1"


class PanoramaError(RuntimeError):
    """Raised when panorama generation inputs are ambiguous or unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> Dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise PanoramaError("Cannot read JSON {}: {}".format(path, exc)) from exc
    if not isinstance(value, dict):
        raise PanoramaError("Expected a JSON object in {}".format(path))
    return value


def _find_scene_obj(scene_root: Path) -> Path:
    candidates = sorted(scene_root.glob("*.obj"))
    if len(candidates) != 1:
        raise PanoramaError(
            "Expected exactly one OBJ directly in {}; found {}".format(
                scene_root, len(candidates)
            )
        )
    return candidates[0]


def pose_index_to_pose(camera: Dict[str, object], pose_index: Sequence[int]) -> Dict[str, object]:
    if len(pose_index) != 5:
        raise PanoramaError("Camera pose index must contain five integers")
    counts = [int(camera[key]) for key in ("pose_l", "pose_w", "pose_h")]
    angle_counts = [int(camera[key]) for key in ("pose_n_theta", "pose_n_azim")]
    indices = [int(value) for value in pose_index]
    limits = counts + angle_counts
    if any(index < 0 or index >= limit for index, limit in zip(indices, limits)):
        raise PanoramaError(
            "Camera pose index {} is outside grid limits {}".format(indices, limits)
        )
    minimum = [float(value) for value in camera["x_min"]]
    maximum = [float(value) for value in camera["x_max"]]
    position = [
        minimum[axis]
        + (indices[axis] + 0.5) * (maximum[axis] - minimum[axis]) / counts[axis]
        for axis in range(3)
    ]
    elevation = -90.0 + 180.0 * (1 + indices[3]) / (angle_counts[0] + 1)
    azimuth = 360.0 * indices[4] / angle_counts[1]
    return {
        "pose_index": indices,
        "position_world_xyz": position,
        "elevation_degrees": elevation,
        "azimuth_degrees": azimuth,
    }


def _occupied_spatial_indices(scene_root: Path) -> set:
    occupied_path = scene_root / "occupied_pose.json"
    if not occupied_path.is_file():
        raise PanoramaError("Missing occupied_pose.json in {}".format(scene_root))
    value = _load_json(occupied_path)
    indices = value.get("X_idx")
    flags = value.get("occupied")
    if not isinstance(indices, list) or not isinstance(flags, list) or len(indices) != len(flags):
        raise PanoramaError("occupied_pose.json has inconsistent X_idx/occupied arrays")
    return {
        tuple(int(component) for component in index)
        for index, occupied in zip(indices, flags)
        if bool(occupied)
    }


def build_render_config(args: argparse.Namespace) -> Dict[str, object]:
    scene_root = Path(args.scene_dir).expanduser().resolve()
    if not scene_root.is_dir():
        raise PanoramaError("Scene directory does not exist: {}".format(scene_root))
    settings_path = scene_root / "settings.json"
    settings = _load_json(settings_path)
    camera = settings.get("camera")
    if not isinstance(camera, dict):
        raise PanoramaError("settings.json has no camera object")
    starts = camera.get("start_positions")
    if not isinstance(starts, list) or not starts:
        raise PanoramaError("settings.json has no start_positions")
    if args.pose_limit is not None:
        if args.pose_limit <= 0:
            raise PanoramaError("--pose-limit must be positive")
        starts = starts[: args.pose_limit]
    poses = [pose_index_to_pose(camera, value) for value in starts]
    occupied = _occupied_spatial_indices(scene_root)
    invalid = [
        pose["pose_index"]
        for pose in poses
        if tuple(pose["pose_index"][:3]) in occupied
    ]
    if invalid:
        raise PanoramaError("Configured start positions are occupied: {}".format(invalid))
    if args.width <= 0 or args.height <= 0 or args.width != 2 * args.height:
        raise PanoramaError("Equirectangular output must have a positive 2:1 resolution")
    if args.samples <= 0:
        raise PanoramaError("--samples must be positive")
    if args.max_depth <= 0:
        raise PanoramaError("--max-depth must be positive")

    scene_obj = _find_scene_obj(scene_root)
    orchestrator_path = Path(__file__).resolve()
    backend_path = orchestrator_path.with_name("openhk3d_blender_panorama.py")
    if not backend_path.is_file():
        raise PanoramaError("Missing Blender panorama backend: {}".format(backend_path))
    assembly_manifest = scene_root / "assembly-manifest.json"
    source = {
        "scene_name": scene_root.name,
        "obj_name": scene_obj.name,
        "obj_sha256": _sha256(scene_obj),
        "settings_sha256": _sha256(settings_path),
        "occupied_pose_sha256": _sha256(scene_root / "occupied_pose.json"),
        "assembly_manifest_sha256": (
            _sha256(assembly_manifest) if assembly_manifest.is_file() else None
        ),
    }
    scope = {
        "schema_version": PANORAMA_SCHEMA,
        "batch_id": args.batch_id,
        "implementation": {
            "orchestrator_sha256": _sha256(orchestrator_path),
            "blender_backend_sha256": _sha256(backend_path),
        },
        "source": source,
        "camera_model": {
            "projection": "equirectangular",
            "horizontal_fov_degrees": 360.0,
            "vertical_fov_degrees": 180.0,
            "resolution": [args.width, args.height],
            "orientation_policy": "world_up_yaw_only",
            "longitude_zero": "MAGICIAN_start_azimuth_center_ray",
            "depth_representation": "radial_distance_from_camera_center",
            "depth_units": "MAGICIAN_world_units",
        },
        "renderer": {
            "engine": "CYCLES",
            "compute_device": args.compute_device,
            "samples": args.samples,
            "max_depth": args.max_depth,
        },
        "poses": poses,
    }
    return {
        "schema_version": PANORAMA_SCHEMA,
        "batch_id": args.batch_id,
        "scope_fingerprint_algorithm": "canonical-json-v1+sha256",
        "scope_fingerprint": _canonical_sha256(scope),
        "scope": scope,
        "scene_dir": str(scene_root),
        "obj_path": str(scene_obj),
        "poses": poses,
        "width": args.width,
        "height": args.height,
        "samples": args.samples,
        "compute_device": args.compute_device,
        "max_depth": args.max_depth,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render RGB and GT range-depth equirectangular panoramas"
    )
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--blender", required=True)
    parser.add_argument("--batch-id", default="OHK3D-W2-PANO-PILOT-001")
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument(
        "--compute-device",
        choices=("CPU", "CUDA", "OPTIX", "HIP", "METAL"),
        default="CPU",
    )
    parser.add_argument("--max-depth", type=float, default=1000.0)
    parser.add_argument("--pose-limit", type=int)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise PanoramaError("Output already exists: {}".format(output))
    blender = Path(args.blender).expanduser().resolve()
    if not blender.is_file() or not os.access(str(blender), os.X_OK):
        raise PanoramaError("Blender executable is not available: {}".format(blender))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".{}-".format(output.name), dir=str(output.parent)))
    try:
        config = build_render_config(args)
        config["output_dir"] = str(staging)
        config_path = staging / "render-config.json"
        _write_json(config_path, config)
        blender_script = Path(__file__).with_name("openhk3d_blender_panorama.py").resolve()
        if not blender_script.is_file():
            raise PanoramaError("Missing Blender panorama backend: {}".format(blender_script))
        subprocess.run(
            [
                str(blender),
                "--background",
                "--python",
                str(blender_script),
                "--",
                "--config",
                str(config_path),
            ],
            check=True,
        )
        manifest_path = staging / "panorama-manifest.json"
        manifest = _load_json(manifest_path)
        if manifest.get("status") != "passed":
            raise PanoramaError("Panorama backend did not report status=passed")
        config_path.unlink()
        os.replace(str(staging), str(output))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print("Generated panorama dataset: {}".format(output))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PanoramaError, subprocess.CalledProcessError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
