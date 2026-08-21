"""Minimal UE5-to-PIONEER six-face observation contract.

The contract is intentionally small.  It defines the validated boundary at
which a complete UE5 capture may enter the existing PAN-10 six-face fusion;
it does not replace that fusion, visibility union, gain accounting, or the
planner.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = "pioneer.ue5-observation.v1"
FACE_NAMES: Tuple[str, ...] = (
    "front",
    "back",
    "left",
    "right",
    "up",
    "down",
)
SOURCE_DEPTH_CAMERA_Z = "camera_z_m"
CANONICAL_DEPTH_ENCODING = "euclidean_ray_range_m"


class BundleValidationError(ValueError):
    """Raised before mapping state is touched when a bundle is invalid."""


def _as_numpy(value: Any) -> np.ndarray:
    """Convert NumPy or torch-like values without importing torch."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CanonicalFace:
    """One canonical square perspective face."""

    face_name: str
    request_id: str
    frame_id: str
    capture_timestamp_ns: int
    rgb_uint8: np.ndarray
    depth_range_m: np.ndarray
    valid_mask: np.ndarray
    K_pixel: np.ndarray
    T_world_from_cam: np.ndarray
    image_size: Tuple[int, int]
    fov_degrees: float = 90.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalObservationBundle:
    """Exactly six faces captured at one position and one world epoch."""

    schema_version: str
    request_id: str
    frame_id: str
    capture_timestamp_ns: int
    position_world_m: Tuple[float, float, float]
    faces: Tuple[CanonicalFace, ...]
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class BundleValidationSummary:
    schema_version: str
    request_id: str
    frame_id: str
    face_count: int
    image_size: Tuple[int, int]
    fov_degrees: float
    valid_depth_count: int
    no_hit_count: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "frame_id": self.frame_id,
            "face_count": self.face_count,
            "image_size": list(self.image_size),
            "fov_degrees": self.fov_degrees,
            "valid_depth_count": self.valid_depth_count,
            "no_hit_count": self.no_hit_count,
        }


def camera_z_to_ray_range(
    depth_camera_z_m: Any,
    K_pixel: Any,
    valid_mask: Any,
) -> np.ndarray:
    """Convert optical-axis depth to Euclidean range from the optical centre.

    For pixel ``(u, v)``, the conversion is
    ``range = z * sqrt(((u-cx)/fx)^2 + ((v-cy)/fy)^2 + 1)``.
    Invalid/no-hit pixels are always represented by NaN at the contract edge.
    """

    depth = _as_numpy(depth_camera_z_m).astype(np.float64, copy=False).squeeze()
    mask = _as_numpy(valid_mask).astype(bool, copy=False).squeeze()
    intrinsic = _as_numpy(K_pixel).astype(np.float64, copy=False).squeeze()
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise BundleValidationError("camera-z depth and mask must have one HxW shape")
    if intrinsic.shape != (3, 3):
        raise BundleValidationError("K_pixel must have shape 3x3")
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    if not all(math.isfinite(value) and value > 0 for value in (fx, fy)):
        raise BundleValidationError("K_pixel focal lengths must be finite and positive")
    rows, columns = np.indices(depth.shape, dtype=np.float64)
    scale = np.sqrt(
        ((columns - cx) / fx) ** 2 + ((rows - cy) / fy) ** 2 + 1.0
    )
    output = np.full(depth.shape, np.nan, dtype=np.float32)
    usable = mask & np.isfinite(depth) & (depth > 0.0)
    output[usable] = (depth[usable] * scale[usable]).astype(np.float32)
    return output


def _validate_pose(transform: np.ndarray, label: str) -> None:
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise BundleValidationError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise BundleValidationError(f"{label} must be a rigid homogeneous transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise BundleValidationError(f"{label} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise BundleValidationError(f"{label} rotation must have determinant +1")


def validate_bundle(bundle: CanonicalObservationBundle) -> BundleValidationSummary:
    """Validate a complete bundle before any map or frame-counter mutation."""

    if not isinstance(bundle, CanonicalObservationBundle):
        raise BundleValidationError("bundle must be a CanonicalObservationBundle")
    if bundle.schema_version != SCHEMA_VERSION:
        raise BundleValidationError(
            f"schema_version must be {SCHEMA_VERSION!r}"
        )
    if not bundle.request_id or not bundle.frame_id:
        raise BundleValidationError("request_id and frame_id must be non-empty")
    if type(bundle.capture_timestamp_ns) is not int or bundle.capture_timestamp_ns <= 0:
        raise BundleValidationError("capture_timestamp_ns must be a positive integer")
    position = np.asarray(bundle.position_world_m, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise BundleValidationError("position_world_m must contain three finite values")
    if not isinstance(bundle.provenance, Mapping) or not bundle.provenance:
        raise BundleValidationError("provenance must be a non-empty mapping")

    names = tuple(face.face_name for face in bundle.faces)
    if names != FACE_NAMES:
        raise BundleValidationError(
            f"faces must be exactly ordered as {FACE_NAMES}; received {names}"
        )

    reference_size: Optional[Tuple[int, int]] = None
    reference_fov: Optional[float] = None
    reference_K: Optional[np.ndarray] = None
    valid_count = 0
    no_hit_count = 0
    for face in bundle.faces:
        label = f"face {face.face_name}"
        if (
            face.request_id != bundle.request_id
            or face.frame_id != bundle.frame_id
            or face.capture_timestamp_ns != bundle.capture_timestamp_ns
        ):
            raise BundleValidationError(
                f"{label} request/frame/timestamp does not match bundle epoch"
            )
        if len(face.image_size) != 2:
            raise BundleValidationError(f"{label} image_size must be (height, width)")
        image_size = tuple(int(value) for value in face.image_size)
        if image_size[0] < 2 or image_size[0] != image_size[1]:
            raise BundleValidationError(f"{label} must be a square image of size >= 2")
        if reference_size is None:
            reference_size = image_size
        elif image_size != reference_size:
            raise BundleValidationError("all six faces must share image_size")
        fov = float(face.fov_degrees)
        if not math.isclose(fov, 90.0, abs_tol=1e-6):
            raise BundleValidationError(f"{label} fov_degrees must be 90")
        if reference_fov is None:
            reference_fov = fov
        elif not math.isclose(fov, reference_fov, abs_tol=1e-6):
            raise BundleValidationError("all six faces must share fov_degrees")

        height, width = image_size
        rgb = _as_numpy(face.rgb_uint8)
        depth = _as_numpy(face.depth_range_m)
        mask = _as_numpy(face.valid_mask)
        intrinsic = _as_numpy(face.K_pixel).astype(np.float64, copy=False)
        transform = _as_numpy(face.T_world_from_cam).astype(np.float64, copy=False)
        if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
            raise BundleValidationError(f"{label} rgb_uint8 must be HxWx3 uint8")
        if depth.shape != (height, width) or not np.issubdtype(depth.dtype, np.floating):
            raise BundleValidationError(f"{label} depth_range_m must be floating HxW")
        if mask.shape != (height, width) or mask.dtype != np.bool_:
            raise BundleValidationError(f"{label} valid_mask must be boolean HxW")
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise BundleValidationError(f"{label} K_pixel must be finite 3x3")
        if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            raise BundleValidationError(f"{label} K_pixel focal lengths must be positive")
        if not np.allclose(intrinsic[2], (0.0, 0.0, 1.0), atol=1e-6):
            raise BundleValidationError(f"{label} K_pixel last row is invalid")
        if reference_K is None:
            reference_K = intrinsic
        elif not np.allclose(intrinsic, reference_K, atol=1e-6):
            raise BundleValidationError("all six faces must share K_pixel")
        _validate_pose(transform, f"{label} T_world_from_cam")
        if not np.allclose(transform[:3, 3], position, atol=1e-5):
            raise BundleValidationError("all six faces must share the bundle optical centre")
        if not (np.isfinite(depth[mask]) & (depth[mask] > 0.0)).all():
            raise BundleValidationError(f"{label} valid depth must be finite and positive")
        if not np.isnan(depth[~mask]).all():
            raise BundleValidationError(
                f"{label} invalid/no-hit depth must be represented by NaN"
            )
        valid_count += int(mask.sum())
        no_hit_count += int((~mask).sum())

    assert reference_size is not None and reference_fov is not None
    return BundleValidationSummary(
        schema_version=bundle.schema_version,
        request_id=bundle.request_id,
        frame_id=bundle.frame_id,
        face_count=len(bundle.faces),
        image_size=reference_size,
        fov_degrees=reference_fov,
        valid_depth_count=valid_count,
        no_hit_count=no_hit_count,
    )


def _world_from_opencv_camera(face: Any) -> np.ndarray:
    metadata = getattr(face, "metadata", {})
    try:
        rotation_world_to_camera = _as_numpy(
            metadata["R_opencv_world_to_camera"]
        ).astype(np.float64, copy=False).squeeze()
        translation_world_to_camera = _as_numpy(
            metadata["T_opencv_world_to_camera"]
        ).astype(np.float64, copy=False).reshape(3)
    except KeyError as error:
        raise BundleValidationError(
            f"PAN-10 face {getattr(face, 'name', '?')} lacks OpenCV extrinsics metadata"
        ) from error
    if rotation_world_to_camera.shape != (3, 3):
        raise BundleValidationError("PAN-10 OpenCV rotation must be 3x3")
    rotation_camera_to_world = rotation_world_to_camera.T
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = rotation_camera_to_world
    output[:3, 3] = -rotation_camera_to_world @ translation_world_to_camera
    return output


def adapt_pan10_observation_bundle(
    bundle: Any,
    *,
    request_id: Optional[str] = None,
    frame_id: Optional[str] = None,
    capture_timestamp_ns: Optional[int] = None,
) -> CanonicalObservationBundle:
    """Adapt an existing PAN-10 camera-z cubemap without changing its fusion."""

    bundle_id = int(getattr(bundle, "bundle_id"))
    metadata = getattr(bundle, "metadata", {})
    resolved_request_id = request_id or f"pan10-request-{bundle_id:06d}"
    resolved_frame_id = frame_id or f"pan10-frame-{bundle_id:06d}"
    resolved_timestamp = capture_timestamp_ns or int(
        metadata.get("capture_timestamp_unix_ns", 0)
    )
    position = _as_numpy(getattr(bundle, "center")).astype(np.float64).reshape(3)
    faces = []
    for face in getattr(bundle, "faces"):
        face_metadata = getattr(face, "metadata", {})
        intrinsic = _as_numpy(face_metadata.get("K_pixel")).astype(
            np.float64, copy=False
        ).squeeze()
        depth_z = _as_numpy(getattr(face, "depth_z")).squeeze()
        valid_mask = _as_numpy(getattr(face, "valid_mask")).astype(
            bool, copy=False
        ).squeeze()
        rgb = _as_numpy(getattr(face, "rgb")).squeeze()
        if rgb.dtype != np.uint8:
            rgb = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        depth_range = camera_z_to_ray_range(depth_z, intrinsic, valid_mask)
        height, width = depth_range.shape
        faces.append(
            CanonicalFace(
                face_name=str(getattr(face, "name")),
                request_id=resolved_request_id,
                frame_id=resolved_frame_id,
                capture_timestamp_ns=resolved_timestamp,
                rgb_uint8=rgb,
                depth_range_m=depth_range,
                valid_mask=valid_mask.astype(np.bool_, copy=False),
                K_pixel=intrinsic,
                T_world_from_cam=_world_from_opencv_camera(face),
                image_size=(height, width),
                fov_degrees=float(getattr(face, "fov_degrees", 90.0)),
                metadata={
                    "source_depth_encoding": SOURCE_DEPTH_CAMERA_Z,
                    "canonical_depth_encoding": CANONICAL_DEPTH_ENCODING,
                    "depth_conversion_formula": (
                        "range=z*sqrt(((u-cx)/fx)^2+((v-cy)/fy)^2+1)"
                    ),
                    "pan10_depth_source": face_metadata.get("depth_source"),
                    "pan10_rig_frame": face_metadata.get("rig_frame"),
                },
            )
        )
    canonical = CanonicalObservationBundle(
        schema_version=SCHEMA_VERSION,
        request_id=resolved_request_id,
        frame_id=resolved_frame_id,
        capture_timestamp_ns=resolved_timestamp,
        position_world_m=tuple(float(value) for value in position),
        faces=tuple(faces),
        provenance={
            "adapter": "adapt_pan10_observation_bundle",
            "source_bundle_id": bundle_id,
            "source_observation_mode": metadata.get("observation_mode"),
            "source_depth_encoding": SOURCE_DEPTH_CAMERA_Z,
            "canonical_depth_encoding": CANONICAL_DEPTH_ENCODING,
        },
    )
    validate_bundle(canonical)
    return canonical


def _canonical_face_transforms(position: Sequence[float]) -> Mapping[str, np.ndarray]:
    """Return a right-handed OpenCV-style fixed six-face rig."""

    directions = {
        "front": ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        "back": ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
        "left": ((0, 0, 1), (0, -1, 0), (1, 0, 0)),
        "right": ((0, 0, -1), (0, -1, 0), (-1, 0, 0)),
        "up": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
        "down": ((-1, 0, 0), (0, 0, -1), (0, -1, 0)),
    }
    result = {}
    for name, columns in directions.items():
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(columns, dtype=np.float64).T
        transform[:3, 3] = np.asarray(position, dtype=np.float64)
        result[name] = transform
    return result


def make_synthetic_plane_bundle(
    *,
    image_size: int = 5,
    plane_camera_z_m: float = 2.0,
    request_id: str = "pan19-synthetic-request",
    frame_id: str = "pan19-synthetic-frame",
    capture_timestamp_ns: int = 1_725_000_000_000_000_000,
) -> CanonicalObservationBundle:
    """Build a deterministic CPU-only analytic plane fixture."""

    if image_size < 3 or image_size % 2 == 0:
        raise ValueError("synthetic fixture image_size must be odd and >= 3")
    if not math.isfinite(plane_camera_z_m) or plane_camera_z_m <= 0:
        raise ValueError("plane_camera_z_m must be finite and positive")
    focal = 0.5 * (image_size - 1)
    intrinsic = np.asarray(
        [[focal, 0.0, focal], [0.0, focal, focal], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    mask = np.ones((image_size, image_size), dtype=np.bool_)
    depth_z = np.full((image_size, image_size), plane_camera_z_m, dtype=np.float32)
    range_m = camera_z_to_ray_range(depth_z, intrinsic, mask)
    position = (1.0, 2.0, 3.0)
    transforms = _canonical_face_transforms(position)
    faces = []
    for index, name in enumerate(FACE_NAMES):
        rgb = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        rgb[..., index % 3] = np.uint8(48 + index * 32)
        faces.append(
            CanonicalFace(
                face_name=name,
                request_id=request_id,
                frame_id=frame_id,
                capture_timestamp_ns=capture_timestamp_ns,
                rgb_uint8=rgb,
                depth_range_m=range_m.copy(),
                valid_mask=mask.copy(),
                K_pixel=intrinsic.copy(),
                T_world_from_cam=transforms[name],
                image_size=(image_size, image_size),
                metadata={
                    "fixture": "analytic_frontoparallel_plane",
                    "plane_camera_z_m": plane_camera_z_m,
                    "source_depth_encoding": SOURCE_DEPTH_CAMERA_Z,
                    "canonical_depth_encoding": CANONICAL_DEPTH_ENCODING,
                },
            )
        )
    output = CanonicalObservationBundle(
        schema_version=SCHEMA_VERSION,
        request_id=request_id,
        frame_id=frame_id,
        capture_timestamp_ns=capture_timestamp_ns,
        position_world_m=position,
        faces=tuple(faces),
        provenance={
            "task": "PAN-19",
            "fixture": "analytic_frontoparallel_plane",
            "source_depth_encoding": SOURCE_DEPTH_CAMERA_Z,
            "canonical_depth_encoding": CANONICAL_DEPTH_ENCODING,
        },
    )
    validate_bundle(output)
    return output


def write_bundle(bundle: CanonicalObservationBundle, destination: Path) -> Path:
    """Atomically publish one checksum-backed file/manifest bundle."""

    validate_bundle(bundle)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite bundle: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(destination.parent))
    )
    try:
        serialized_faces = []
        for face in bundle.faces:
            assets = {}
            for field_name in ("rgb_uint8", "depth_range_m", "valid_mask"):
                relative = Path("faces") / face.face_name / f"{field_name}.npy"
                path = temporary / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as stream:
                    np.save(stream, _as_numpy(getattr(face, field_name)), allow_pickle=False)
                assets[field_name] = {
                    "path": relative.as_posix(),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            serialized_faces.append(
                {
                    "face_name": face.face_name,
                    "request_id": face.request_id,
                    "frame_id": face.frame_id,
                    "capture_timestamp_ns": face.capture_timestamp_ns,
                    "image_size": list(face.image_size),
                    "fov_degrees": face.fov_degrees,
                    "K_pixel": _as_numpy(face.K_pixel).tolist(),
                    "T_world_from_cam": _as_numpy(face.T_world_from_cam).tolist(),
                    "assets": assets,
                    "metadata": dict(face.metadata),
                }
            )
        manifest = {
            "schema_version": bundle.schema_version,
            "request_id": bundle.request_id,
            "frame_id": bundle.frame_id,
            "capture_timestamp_ns": bundle.capture_timestamp_ns,
            "position_world_m": list(bundle.position_world_m),
            "face_names": list(FACE_NAMES),
            "faces": serialized_faces,
            "provenance": dict(bundle.provenance),
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(str(temporary), str(destination))
        temporary = None
        return destination / "manifest.json"
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def load_bundle(manifest_path: Path) -> CanonicalObservationBundle:
    """Load and checksum-verify a bundle before returning it."""

    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    faces = []
    for item in payload["faces"]:
        arrays = {}
        for field_name, asset in item["assets"].items():
            path = root / asset["path"]
            if _sha256(path) != asset["sha256"]:
                raise BundleValidationError(f"checksum mismatch: {asset['path']}")
            with path.open("rb") as stream:
                arrays[field_name] = np.load(stream, allow_pickle=False)
        faces.append(
            CanonicalFace(
                face_name=item["face_name"],
                request_id=item["request_id"],
                frame_id=item["frame_id"],
                capture_timestamp_ns=item["capture_timestamp_ns"],
                rgb_uint8=arrays["rgb_uint8"],
                depth_range_m=arrays["depth_range_m"],
                valid_mask=arrays["valid_mask"],
                K_pixel=np.asarray(item["K_pixel"], dtype=np.float64),
                T_world_from_cam=np.asarray(
                    item["T_world_from_cam"], dtype=np.float64
                ),
                image_size=tuple(item["image_size"]),
                fov_degrees=item["fov_degrees"],
                metadata=item.get("metadata", {}),
            )
        )
    bundle = CanonicalObservationBundle(
        schema_version=payload["schema_version"],
        request_id=payload["request_id"],
        frame_id=payload["frame_id"],
        capture_timestamp_ns=payload["capture_timestamp_ns"],
        position_world_m=tuple(payload["position_world_m"]),
        faces=tuple(faces),
        provenance=payload["provenance"],
    )
    validate_bundle(bundle)
    return bundle


__all__ = [
    "BundleValidationError",
    "BundleValidationSummary",
    "CANONICAL_DEPTH_ENCODING",
    "CanonicalFace",
    "CanonicalObservationBundle",
    "FACE_NAMES",
    "SCHEMA_VERSION",
    "SOURCE_DEPTH_CAMERA_Z",
    "adapt_pan10_observation_bundle",
    "camera_z_to_ray_range",
    "load_bundle",
    "make_synthetic_plane_bundle",
    "validate_bundle",
    "write_bundle",
]
