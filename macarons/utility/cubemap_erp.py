"""Deterministic RGB cubemap-to-equirectangular projection.

Face labels and tuple order are deliberately ignored.  Every equirectangular
pixel defines one world-space ray; the ray is transformed into each face
camera, projected with that face's pixel intrinsic matrix, and sampled from
the most forward-facing camera that contains it.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np


class CubemapProjectionError(ValueError):
    """Raised when six RGB faces cannot form a deterministic panorama."""


@dataclass(frozen=True)
class _ProjectionFace:
    rgb: np.ndarray
    intrinsic: np.ndarray
    rotation_world_from_cam: np.ndarray
    stable_key: str


def _as_projection_faces(bundle: Any) -> Tuple[_ProjectionFace, ...]:
    try:
        source_faces = tuple(bundle.faces)
    except (AttributeError, TypeError) as error:
        raise CubemapProjectionError("bundle must expose exactly six faces") from error
    if len(source_faces) != 6:
        raise CubemapProjectionError("bundle must contain exactly six RGB faces")

    records = []
    expected_side = None
    expected_center = None
    for index, face in enumerate(source_faces):
        label = f"face[{index}]"
        rgb = np.asarray(getattr(face, "rgb_uint8", None))
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise CubemapProjectionError(f"{label} RGB must be uint8 HxWx3")
        height, width = rgb.shape[:2]
        if height <= 0 or height != width:
            raise CubemapProjectionError(f"{label} RGB must be square")
        image_size = tuple(getattr(face, "image_size", ()))
        if image_size != (height, width):
            raise CubemapProjectionError(
                f"{label} image_size must match its RGB dimensions"
            )
        if expected_side is None:
            expected_side = height
        elif height != expected_side:
            raise CubemapProjectionError("all six RGB faces must have one square size")

        intrinsic = np.asarray(getattr(face, "K_pixel", None), dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise CubemapProjectionError(f"{label} K_pixel must be finite 3x3")
        if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
            raise CubemapProjectionError(f"{label} K_pixel focal lengths must be positive")
        if not np.allclose(intrinsic[2], (0.0, 0.0, 1.0), atol=1e-9):
            raise CubemapProjectionError(f"{label} K_pixel last row is invalid")

        transform = np.asarray(
            getattr(face, "T_world_from_cam", None), dtype=np.float64
        )
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise CubemapProjectionError(
                f"{label} T_world_from_cam must be finite 4x4"
            )
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
            raise CubemapProjectionError(
                f"{label} T_world_from_cam must be homogeneous"
            )
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise CubemapProjectionError(
                f"{label} T_world_from_cam rotation must be orthonormal"
            )
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
            raise CubemapProjectionError(
                f"{label} T_world_from_cam rotation must have determinant +1"
            )
        center = transform[:3, 3]
        if expected_center is None:
            expected_center = center
        elif not np.allclose(center, expected_center, atol=1e-6):
            raise CubemapProjectionError("all six faces must share one camera centre")

        # The key makes exact seam ties independent of caller-provided ordering
        # without assigning any semantic meaning to a face name.
        digest = hashlib.sha256()
        digest.update(np.round(rotation, decimals=12).astype("<f8").tobytes())
        digest.update(np.round(intrinsic, decimals=12).astype("<f8").tobytes())
        digest.update(np.ascontiguousarray(rgb).tobytes())
        records.append(
            _ProjectionFace(
                rgb=rgb,
                intrinsic=intrinsic,
                rotation_world_from_cam=rotation,
                stable_key=digest.hexdigest(),
            )
        )

    return tuple(sorted(records, key=lambda record: record.stable_key))


def cubemap_rgb_to_equirectangular(
    bundle: Any,
    *,
    output_height: int | None = None,
) -> np.ndarray:
    """Project six canonical RGB faces into a uint8 ``H x 2H`` panorama.

    The equirectangular convention follows PIONEER's fixed world cubemap:
    longitude ``[-pi, pi)`` runs left-to-right, latitude
    ``(+pi/2, -pi/2)`` runs top-to-bottom, ``+Y`` is up, and ``+Z``/front is
    the image centre.  The right quarter points toward ``-X``/right and the
    horizontal wrap seam points toward ``-Z``/back.

    Nearest-neighbour sampling is intentional: the same bundle and requested
    height produce byte-identical output, including geometric seam ties.
    """

    faces = _as_projection_faces(bundle)
    if output_height is None:
        output_height = int(faces[0].rgb.shape[0])
    if isinstance(output_height, bool) or not isinstance(output_height, int):
        raise CubemapProjectionError("output_height must be an integer")
    if output_height < 2:
        raise CubemapProjectionError("output_height must be at least 2")

    height = output_height
    width = 2 * height
    longitudes = ((np.arange(width, dtype=np.float64) + 0.5) / width - 0.5) * (
        2.0 * np.pi
    )
    cos_longitude = np.cos(longitudes)
    sin_longitude = np.sin(longitudes)
    output = np.empty((height, width, 3), dtype=np.uint8)

    # Row blocks bound temporary memory for paper-scale previews while keeping
    # all ray selection and sampling vectorised.
    rows_per_block = min(height, 128)
    boundary_tolerance = 1e-9
    score_tolerance = 1e-12
    for row_start in range(0, height, rows_per_block):
        row_stop = min(row_start + rows_per_block, height)
        rows = np.arange(row_start, row_stop, dtype=np.float64)
        latitudes = (0.5 - (rows + 0.5) / height) * np.pi
        cos_latitude = np.cos(latitudes)[:, None]
        sin_latitude = np.sin(latitudes)[:, None]
        world_rays = np.empty((row_stop - row_start, width, 3), dtype=np.float64)
        world_rays[..., 0] = -cos_latitude * sin_longitude[None, :]
        world_rays[..., 1] = sin_latitude
        world_rays[..., 2] = cos_latitude * cos_longitude[None, :]

        best_score = np.full(world_rays.shape[:2], -np.inf, dtype=np.float64)
        block = np.zeros(world_rays.shape[:2] + (3,), dtype=np.uint8)
        for face in faces:
            # Row-vector form of d_cam = R_world_from_cam.T @ d_world.
            camera_rays = world_rays @ face.rotation_world_from_cam
            projected = camera_rays @ face.intrinsic.T
            camera_z = projected[..., 2]
            finite_forward = np.isfinite(projected).all(axis=-1) & (camera_z > 0.0)
            safe_z = np.where(finite_forward, camera_z, 1.0)
            pixel_u = projected[..., 0] / safe_z
            pixel_v = projected[..., 1] / safe_z
            face_height, face_width = face.rgb.shape[:2]
            contains = (
                finite_forward
                & (pixel_u >= -0.5 - boundary_tolerance)
                & (pixel_u <= face_width - 0.5 + boundary_tolerance)
                & (pixel_v >= -0.5 - boundary_tolerance)
                & (pixel_v <= face_height - 0.5 + boundary_tolerance)
            )
            choose = contains & (camera_z > best_score + score_tolerance)
            if not np.any(choose):
                continue
            sample_u = np.floor(np.where(contains, pixel_u, 0.0) + 0.5).astype(
                np.int64
            )
            sample_v = np.floor(np.where(contains, pixel_v, 0.0) + 0.5).astype(
                np.int64
            )
            np.clip(sample_u, 0, face_width - 1, out=sample_u)
            np.clip(sample_v, 0, face_height - 1, out=sample_v)
            sampled = face.rgb[sample_v, sample_u]
            block[choose] = sampled[choose]
            best_score[choose] = camera_z[choose]

        if not np.isfinite(best_score).all():
            missing = int(np.count_nonzero(~np.isfinite(best_score)))
            raise CubemapProjectionError(
                f"six face poses/intrinsics leave {missing} ERP rays uncovered"
            )
        output[row_start:row_stop] = block

    return output
