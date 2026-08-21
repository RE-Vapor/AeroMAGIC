"""Validated PAN-19 canonical bundle adapter into the existing PAN-10 path.

This module changes representation only.  PAN-10 remains authoritative for
six-face fusion, seam ownership, proxy visibility union, and deduplication.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from pytorch3d.renderer import FoVPerspectiveCameras

from .planning_observations import ObservationBundle, PerspectiveFaceObservation
from .ue5_observation_contract import (
    CanonicalObservationBundle,
    validate_bundle,
)


UE_CONTRACT_TO_MAGICIAN = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)
PYTORCH3D_FROM_OPENCV = np.diag([-1.0, -1.0, 1.0])
ADAPTER_VERSION = "pan21-ue5-canonical-to-pan10-v1"


def canonical_bundle_to_pan10(
    bundle: CanonicalObservationBundle,
    *,
    device: Any,
    scene_units_per_meter: float,
    bundle_id: int,
    znear: float = 0.1,
    zfar: float = 650.0,
) -> ObservationBundle:
    """Convert one validated UE bundle without reimplementing PAN-10 fusion."""

    validate_bundle(bundle)
    if not math.isfinite(scene_units_per_meter) or scene_units_per_meter <= 0.0:
        raise ValueError("scene_units_per_meter must be finite and positive")
    if type(bundle_id) is not int or bundle_id < 0:
        raise ValueError("bundle_id must be a non-negative integer")
    resolved_device = torch.device(device)
    faces = []
    for canonical in bundle.faces:
        height, width = canonical.image_size
        intrinsic = np.asarray(canonical.K_pixel, dtype=np.float64)
        fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
        rows, columns = np.indices((height, width), dtype=np.float64)
        ray_scale = np.sqrt(
            ((columns - cx) / fx) ** 2
            + ((rows - cy) / fy) ** 2
            + 1.0
        )
        valid = np.asarray(canonical.valid_mask, dtype=bool)
        depth_z = np.zeros((height, width), dtype=np.float32)
        depth_range = np.asarray(canonical.depth_range_m, dtype=np.float64)
        depth_z[valid] = (
            depth_range[valid] / ray_scale[valid] * scene_units_per_meter
        ).astype(np.float32)

        canonical_pose = np.asarray(canonical.T_world_from_cam, dtype=np.float64)
        rotation_world_from_cv = canonical_pose[:3, :3]
        center_contract_m = canonical_pose[:3, 3]
        rotation_world_from_cv = UE_CONTRACT_TO_MAGICIAN @ rotation_world_from_cv
        center = (
            UE_CONTRACT_TO_MAGICIAN
            @ center_contract_m
            * scene_units_per_meter
        )
        rotation_world_from_pytorch3d = (
            rotation_world_from_cv @ PYTORCH3D_FROM_OPENCV
        )
        rotation = torch.as_tensor(
            rotation_world_from_pytorch3d,
            dtype=torch.float32,
            device=resolved_device,
        ).reshape(1, 3, 3)
        center_tensor = torch.as_tensor(
            center, dtype=torch.float32, device=resolved_device
        ).reshape(1, 3)
        translation = -torch.bmm(center_tensor.unsqueeze(1), rotation).squeeze(1)

        # PyTorch3D's raster grid spans pixel centres from +1 to -1.  UE's
        # physical 90-degree frustum reaches half a pixel beyond that grid, so
        # use the centre-to-centre effective FOV to preserve every UE ray.
        effective_fov_x = math.degrees(2.0 * math.atan((width - 1.0) / (2.0 * fx)))
        effective_fov_y = math.degrees(2.0 * math.atan((height - 1.0) / (2.0 * fy)))
        if not math.isclose(effective_fov_x, effective_fov_y, abs_tol=1e-6):
            raise ValueError("canonical face must use square-pixel intrinsics")
        camera = FoVPerspectiveCameras(
            R=rotation,
            T=translation,
            znear=float(znear),
            zfar=float(zfar),
            fov=float(effective_fov_y),
            aspect_ratio=1.0,
            device=resolved_device,
        )
        faces.append(
            PerspectiveFaceObservation(
                name=canonical.face_name,
                camera=camera,
                rgb=torch.as_tensor(
                    np.asarray(canonical.rgb_uint8, dtype=np.float32) / 255.0,
                    dtype=torch.float32,
                    device=resolved_device,
                ).unsqueeze(0),
                depth_z=torch.as_tensor(
                    depth_z, dtype=torch.float32, device=resolved_device
                ).reshape(1, height, width, 1),
                valid_mask=torch.as_tensor(
                    valid, dtype=torch.bool, device=resolved_device
                ).reshape(1, height, width, 1),
                R=rotation,
                T=translation,
                K=torch.as_tensor(
                    intrinsic, dtype=torch.float32, device=resolved_device
                ).reshape(1, 3, 3),
                camera_center=center_tensor,
                fov_degrees=float(canonical.fov_degrees),
                metadata={
                    "adapter_version": ADAPTER_VERSION,
                    "depth_source": "UE5",
                    "rgb_source": "UE5",
                    "znear": float(znear),
                    "zfar": float(zfar),
                    "physical_fov_degrees": float(canonical.fov_degrees),
                    "pytorch3d_effective_pixel_center_fov_degrees": effective_fov_y,
                    "scene_units_per_meter": float(scene_units_per_meter),
                    "world_axis_transform": "ue-contract-rh-to-magician-xyz-v1",
                    "request_id": bundle.request_id,
                    "frame_id": bundle.frame_id,
                    "depth_cache_metadata": {
                        "adapter_version": ADAPTER_VERSION,
                        "source_revision": bundle.provenance.get("source_revision"),
                        "source_tree_sha256": bundle.provenance.get("source_tree_sha256"),
                    },
                },
            )
        )

    center = faces[0].camera_center
    return ObservationBundle(
        bundle_id=bundle_id,
        center=center,
        faces=tuple(faces),
        capture_seconds=0.0,
        metadata={
            "observation_mode": "cubemap6",
            "depth_source": "UE5",
            "rgb_source": "UE5",
            "rig_frame": "world",
            "extrinsics_version": "pan21-ue5-contract-world-v1",
            "adapter_version": ADAPTER_VERSION,
            "request_id": bundle.request_id,
            "frame_id": bundle.frame_id,
            "capture_timestamp_unix_ns": bundle.capture_timestamp_ns,
            "renderer_zbuf_read": False,
            "depth_inference_count": 0,
            "depth_provider_event_count": 6,
            "artifact_committed": True,
            "artifact_transaction_version": "pan19-canonical-manifest-v1",
        },
    )


__all__ = [
    "ADAPTER_VERSION",
    "UE_CONTRACT_TO_MAGICIAN",
    "canonical_bundle_to_pan10",
]
