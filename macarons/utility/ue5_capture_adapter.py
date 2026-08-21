"""Adapter from PAN-15 raw UE files to the PAN-19 canonical contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2

from .ue5_observation_contract import (
    CANONICAL_DEPTH_ENCODING,
    SCHEMA_VERSION,
    CanonicalFace,
    CanonicalObservationBundle,
    camera_z_to_ray_range,
    validate_bundle,
)


UE_SCENE_DEPTH_ENCODING = (
    "ue5_scs_scene_depth_camera_z_world_units_rgba16f_exr_r"
)
# Compatibility name for PAN-15 callers; PAN-20 validated the final meaning.
UE_SCENE_DEPTH_CANDIDATE_ENCODING = UE_SCENE_DEPTH_ENCODING


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_path(root: Path, asset: Mapping[str, Any]) -> Path:
    path = root / asset["path"]
    if _sha256(path) != asset["sha256"]:
        raise ValueError(f"raw asset checksum mismatch: {asset['path']}")
    if path.stat().st_size != int(asset["bytes"]):
        raise ValueError(f"raw asset size mismatch: {asset['path']}")
    return path


def _load_array(root: Path, asset: Mapping[str, Any], dtype: Any) -> np.ndarray:
    path = _verified_path(root, asset)
    shape = tuple(int(value) for value in asset["shape"])
    array = np.fromfile(path, dtype=dtype)
    if array.size != int(np.prod(shape)):
        raise ValueError(f"raw asset element count mismatch: {asset['path']}")
    return array.reshape(shape)


def _load_exr_r(root: Path, asset: Mapping[str, Any], image_size: Any) -> np.ndarray:
    path = _verified_path(root, asset)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    expected = tuple(int(value) for value in image_size)
    if image is None or image.ndim != 3 or image.shape[:2] != expected:
        actual = None if image is None else image.shape
        raise ValueError(f"invalid raw EXR shape for {asset['path']}: {actual}")
    if image.shape[2] < 3:
        raise ValueError(f"raw EXR lacks R channel: {asset['path']}")
    # OpenCV returns OpenEXR channels in BGRA order.
    return np.asarray(image[..., 2], dtype=np.float32)


def adapt_pan15_raw_manifest(manifest_path: Path) -> CanonicalObservationBundle:
    """Build a Contract v1 bundle while preserving raw provenance.

    PAN-20 established that SCS_SCENE_DEPTH exported through the RGBA16F EXR
    R channel is optical-axis camera-z in UE world units.  It is converted to
    metres and then to Euclidean ray range with the per-face pixel intrinsics.
    """

    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "pan15.ue5-raw-rgbd.v1":
        raise ValueError("unexpected PAN-15 raw schema_version")
    world_to_meters = float(raw["world_to_meters"])
    if world_to_meters <= 0:
        raise ValueError("WorldToMeters must be positive")
    faces = []
    for item in raw["faces"]:
        rgb = _load_array(root, item["rgb_uint8"], np.uint8)
        scene_depth_record = item["scene_depth_r"]
        device_depth_record = item["device_depth_r"]
        scene_depth = _load_exr_r(
            root, scene_depth_record["raw_exr"], item["image_size"]
        )
        _verified_path(root, device_depth_record["raw_exr"])
        near_world = float(raw["near_m"]) * world_to_meters
        far_world = min(float(raw["far_m"]) * world_to_meters, 65504.0)
        valid_mask = (
            np.isfinite(scene_depth)
            & (scene_depth > near_world)
            & (scene_depth < far_world)
        )
        height, width = (int(value) for value in item["image_size"])
        fov_radians = np.deg2rad(float(item["fov_degrees"]))
        focal = 0.5 * width / np.tan(0.5 * fov_radians)
        canonical_K = np.asarray(
            (
                (focal, 0.0, 0.5 * (width - 1.0)),
                (0.0, focal, 0.5 * (height - 1.0)),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        depth_camera_z_m = np.full(scene_depth.shape, np.nan, dtype=np.float32)
        depth_camera_z_m[valid_mask] = scene_depth[valid_mask] / world_to_meters
        depth_range_m = camera_z_to_ray_range(
            depth_camera_z_m, canonical_K, valid_mask
        )
        faces.append(
            CanonicalFace(
                face_name=item["face_name"],
                request_id=item["request_id"],
                frame_id=item["frame_id"],
                capture_timestamp_ns=int(item["capture_timestamp_ns"]),
                rgb_uint8=rgb,
                depth_range_m=depth_range_m,
                valid_mask=valid_mask.astype(np.bool_, copy=False),
                K_pixel=canonical_K,
                T_world_from_cam=np.asarray(
                    item["T_world_from_cam"], dtype=np.float64
                ),
                image_size=tuple(item["image_size"]),
                fov_degrees=float(item["fov_degrees"]),
                metadata={
                    "source_depth_encoding": UE_SCENE_DEPTH_ENCODING,
                    "canonical_depth_encoding": CANONICAL_DEPTH_ENCODING,
                    "conversion_formula": (
                        "z_m=EXR_R/WorldToMeters; "
                        "range_m=z_m*sqrt(((u-cx)/fx)^2+((v-cy)/fy)^2+1)"
                    ),
                    "raw_scene_depth_exr_sha256": scene_depth_record["raw_exr"]["sha256"],
                    "raw_device_depth_exr_sha256": device_depth_record["raw_exr"]["sha256"],
                    "raw_device_depth_exr_path": device_depth_record["raw_exr"]["path"],
                    "valid_mask_rule": "finite and near < EXR_R < min(far,65504)",
                    "raw_candidate_K_pixel": item["K_pixel"],
                    "intrinsics_correction": (
                        "PAN-20 UE pixel-centre model: fx=fy=N/(2*tan(FOV/2)); "
                        "cx=cy=(N-1)/2"
                    ),
                    "coordinate_conversion": "UE left-handed world to right-handed world by Y flip",
                },
            )
        )
    first_pose = np.asarray(faces[0].T_world_from_cam, dtype=np.float64)
    bundle = CanonicalObservationBundle(
        schema_version=SCHEMA_VERSION,
        request_id=raw["request_id"],
        frame_id=raw["frame_id"],
        capture_timestamp_ns=int(raw["capture_timestamp_ns"]),
        position_world_m=tuple(float(value) for value in first_pose[:3, 3]),
        faces=tuple(faces),
        provenance={
            "task": "PAN-15",
            "scenario": raw["scenario"],
            "raw_manifest": "raw_bundle/manifest.json",
            "raw_manifest_sha256": _sha256(manifest_path),
            "engine_version": raw["engine_version"],
            "project_commit": raw["project_commit"],
            "capture_config_sha256": raw["capture_config_sha256"],
            "source_depth_encoding": UE_SCENE_DEPTH_ENCODING,
            "canonical_depth_encoding": CANONICAL_DEPTH_ENCODING,
            "geometry_status": "validated_by_PAN-20",
        },
    )
    validate_bundle(bundle)
    return bundle


def raw_capture_summary(manifest_path: Path) -> Dict[str, Any]:
    raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return {
        "scenario": raw["scenario"],
        "face_count": len(raw["faces"]),
        "raw_bundle_bytes": sum(
            int(face[key]["bytes"])
            for face in raw["faces"]
            for key in ("rgb_uint8", "rgb_png", "python_readback_candidate_mask")
        )
        + sum(
            int(face[key][nested]["bytes"])
            for face in raw["faces"]
            for key in ("scene_depth_r", "device_depth_r")
            for nested in ("raw_exr", "python_readback")
        ),
        "capture_seconds": raw["timings_seconds"][
            "total_capture_readback_serialization_write"
        ],
        "python_readback_candidate_valid_counts": {
            face["face_name"]: face["python_readback_candidate_mask"]["valid_count"]
            for face in raw["faces"]
        },
        "python_readback_candidate_no_hit_counts": {
            face["face_name"]: face["python_readback_candidate_mask"]["no_hit_count"]
            for face in raw["faces"]
        },
    }


__all__ = [
    "UE_SCENE_DEPTH_ENCODING",
    "UE_SCENE_DEPTH_CANDIDATE_ENCODING",
    "adapt_pan15_raw_manifest",
    "raw_capture_summary",
]
