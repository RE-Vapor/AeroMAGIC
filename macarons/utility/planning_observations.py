"""Full-sphere planning observations represented as a six-face cubemap.

This module deliberately sits beside the legacy single-view camera helpers.  It
does not change their storage or projection conventions.  A cubemap is one
planning observation: all six perspective faces share an optical centre, use a
90 degree field of view, and are fused before proxy state is updated.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from pytorch3d.renderer import (
    AmbientLights,
    FoVPerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
    SoftPhongShader,
    look_at_view_transform,
)
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
from torchvision.transforms import functional as vision_functional


CUBEMAP_FACE_NAMES: Tuple[str, ...] = (
    "front",
    "back",
    "left",
    "right",
    "up",
    "down",
)

# Directions are expressed in PyTorch3D view coordinates: +X is left, +Y is
# up, and +Z points into the scene.  The polar faces use an explicit up vector
# to avoid the look-at singularity and keep their seams deterministic.
_FACE_AXES: Mapping[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    "front": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "back": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "left": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "right": ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "up": ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    "down": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
}


@dataclass
class PerspectiveFaceObservation:
    """One square perspective face belonging to an observation bundle."""

    name: str
    camera: FoVPerspectiveCameras
    rgb: torch.Tensor
    depth_z: torch.Tensor
    valid_mask: torch.Tensor
    R: torch.Tensor
    T: torch.Tensor
    K: torch.Tensor
    camera_center: torch.Tensor
    fov_degrees: float = 90.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def image_height(self) -> int:
        return int(self.depth_z.shape[-3])

    @property
    def image_width(self) -> int:
        return int(self.depth_z.shape[-2])

    @property
    def zbuf(self) -> torch.Tensor:
        """Compatibility alias for code which calls rendered depth ``zbuf``."""

        return self.depth_z

    def frame_dict(self, *, cpu: bool = False) -> Dict[str, Any]:
        """Return the face in the repository's existing serialized-frame shape."""

        def maybe_cpu(value: torch.Tensor) -> torch.Tensor:
            value = value.detach()
            return value.cpu() if cpu else value

        return {
            "rgb": maybe_cpu(self.rgb),
            "zbuf": maybe_cpu(self.depth_z),
            "mask": maybe_cpu(self.valid_mask),
            "R": maybe_cpu(self.R),
            "T": maybe_cpu(self.T),
            "K": maybe_cpu(self.K),
            "camera_center": maybe_cpu(self.camera_center),
            "face_name": self.name,
            "fov_degrees": float(self.fov_degrees),
            "image_height": self.image_height,
            "image_width": self.image_width,
            **self.metadata,
        }


@dataclass
class ObservationBundle:
    """A single full-sphere observation made from exactly six faces."""

    bundle_id: int
    center: torch.Tensor
    faces: Tuple[PerspectiveFaceObservation, ...]
    capture_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = tuple(face.name for face in self.faces)
        if names != CUBEMAP_FACE_NAMES:
            raise ValueError(
                "Cubemap faces must be ordered as "
                f"{CUBEMAP_FACE_NAMES}; received {names}."
            )
        centers = torch.cat(
            [face.camera_center.reshape(1, 3) for face in self.faces], dim=0
        )
        reference = self.center.reshape(1, 3).expand_as(centers)
        if not torch.allclose(centers, reference, rtol=1e-5, atol=1e-5):
            raise ValueError("All cubemap faces must share one optical centre.")

    @property
    def render_count(self) -> int:
        return len(self.faces)

    @property
    def face_cameras(self) -> Dict[str, FoVPerspectiveCameras]:
        return {face.name: face.camera for face in self.faces}

    @property
    def depth_maps(self) -> Dict[str, torch.Tensor]:
        return {face.name: face.depth_z for face in self.faces}

    def face(self, name: str) -> PerspectiveFaceObservation:
        for face in self.faces:
            if face.name == name:
                return face
        raise KeyError(f"Unknown cubemap face: {name}")


def _single_center(center: Any, device: Any) -> torch.Tensor:
    center_tensor = torch.as_tensor(center, device=device)
    if not center_tensor.is_floating_point():
        center_tensor = center_tensor.float()
    if center_tensor.numel() != 3:
        raise ValueError("A cubemap must have exactly one 3D optical centre.")
    return center_tensor.reshape(1, 3)


def _scalar(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    if torch.is_tensor(value):
        return float(value.reshape(-1)[0].detach().cpu().item())
    return float(value)


def _synchronize_if_cuda(device: Any) -> None:
    """Synchronize only CUDA devices so elapsed times include queued kernels."""

    if not torch.cuda.is_available():
        return
    resolved = torch.device(f"cuda:{device}") if isinstance(device, int) else torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)


def _view_to_world_axes(
    axes: torch.Tensor,
    reference_camera: Optional[FoVPerspectiveCameras],
) -> torch.Tensor:
    if reference_camera is None:
        return axes
    if int(reference_camera.R.shape[0]) != 1:
        raise ValueError("reference_camera must contain exactly one camera.")
    # PyTorch3D transforms row-vector world directions as world @ R.  Its
    # inverse therefore maps a view-space direction as view @ R^T.
    return axes @ reference_camera.R[0].transpose(-1, -2)


def build_cubemap_cameras(
    center: Any,
    znear: float,
    zfar: float,
    device: Any,
    *,
    reference_camera: Optional[FoVPerspectiveCameras] = None,
) -> Dict[str, FoVPerspectiveCameras]:
    """Build six square 90-degree cameras around one optical centre.

    If ``reference_camera`` is supplied, ``front`` is aligned with that
    camera's optical axis.  Without it, the face definitions use the global
    PyTorch3D view axes.  Full-sphere coverage is identical in either case.
    """

    eye = _single_center(center, device)
    if znear <= 0:
        raise ValueError("znear must be positive.")
    if zfar <= znear:
        raise ValueError("zfar must be greater than znear.")

    cameras: Dict[str, FoVPerspectiveCameras] = {}
    for name in CUBEMAP_FACE_NAMES:
        direction_values, up_values = _FACE_AXES[name]
        axes = torch.tensor(
            [direction_values, up_values], dtype=eye.dtype, device=eye.device
        )
        axes = _view_to_world_axes(axes, reference_camera)
        direction = torch.nn.functional.normalize(axes[0], dim=0).reshape(1, 3)
        up = torch.nn.functional.normalize(axes[1], dim=0).reshape(1, 3)
        R, T = look_at_view_transform(eye=eye, at=eye + direction, up=up)
        cameras[name] = FoVPerspectiveCameras(
            R=R,
            T=T,
            znear=float(znear),
            zfar=float(zfar),
            fov=90.0,
            aspect_ratio=1.0,
            degrees=True,
            device=device,
        )
    return cameras


def cubemap_pixel_intrinsics(
    image_height: int,
    image_width: int,
    *,
    fov_degrees: float = 90.0,
    device: Any = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a conventional 3x3 pixel-space intrinsic matrix."""

    if image_height < 2 or image_width < 2:
        raise ValueError("Cubemap faces must be at least 2x2 pixels.")
    if image_height != image_width:
        raise ValueError("Cubemap perspective faces must be square.")
    focal = 0.5 * image_width / math.tan(math.radians(fov_degrees) / 2.0)
    cx = (image_width - 1.0) / 2.0
    cy = (image_height - 1.0) / 2.0
    return torch.tensor(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    ).reshape(1, 3, 3)


def _build_square_renderer(
    face_size: int,
    camera: FoVPerspectiveCameras,
    device: Any,
    ambient_light_intensity: float,
    max_faces_per_bin: int,
) -> MeshRendererWithFragments:
    raster_settings = RasterizationSettings(
        image_size=(face_size, face_size),
        max_faces_per_bin=max_faces_per_bin,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    lights = AmbientLights(
        ambient_color=(
            (
                ambient_light_intensity,
                ambient_light_intensity,
                ambient_light_intensity,
            ),
        ),
        device=device,
    )
    return MeshRendererWithFragments(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=camera, lights=lights),
    )


def capture_cubemap_observation(
    camera: Any,
    mesh: Any,
    face_size: int,
    ambient_light_intensity: float = 1.0,
    *,
    rgb_provider: Any = None,
    save_frame: bool = True,
    dir_path: Optional[str] = None,
    save_png: bool = True,
    max_faces_per_bin: int = 200000,
) -> ObservationBundle:
    """Render and optionally persist one six-face GT-mesh observation.

    ``rgb_provider`` is an explicit provider boundary for a future UE5/remote
    capture adapter.  This implementation accepts only ``None`` and therefore
    cannot accidentally label a local mesh render as a UE5 observation.

    ``camera.n_frames_captured`` is increased once after all six faces succeed.
    """

    if rgb_provider is not None:
        raise NotImplementedError(
            "Cubemap RGB providers are not implemented; use rgb_provider=None "
            "for the GT mesh renderer."
        )
    if not isinstance(face_size, int) or face_size < 2:
        raise ValueError("face_size must be an integer >= 2.")
    if camera.fov_camera is None:
        raise ValueError("camera.fov_camera must be initialized before capture.")

    device = camera.device
    center = camera.fov_camera.get_camera_center()
    znear = _scalar(getattr(camera.fov_camera, "znear", None), 1.0)
    zfar = _scalar(getattr(camera, "zfar", None), 100.0)
    face_cameras = build_cubemap_cameras(
        center,
        znear,
        zfar,
        device,
        reference_camera=camera.fov_camera,
    )
    renderer = _build_square_renderer(
        face_size,
        face_cameras[CUBEMAP_FACE_NAMES[0]],
        device,
        ambient_light_intensity,
        max_faces_per_bin,
    )
    K = cubemap_pixel_intrinsics(
        face_size,
        face_size,
        device=device,
        dtype=center.dtype,
    )
    bundle_id = int(camera.n_frames_captured)
    _synchronize_if_cuda(device)
    capture_started = time.perf_counter()
    faces = []

    with torch.no_grad():
        for name in CUBEMAP_FACE_NAMES:
            face_camera = face_cameras[name]
            rgba, fragments = renderer(mesh, cameras=face_camera)
            rgb_chw = rgba[..., :3].permute(0, 3, 1, 2)
            rgb_chw = vision_functional.adjust_contrast(
                rgb_chw, float(getattr(camera, "contrast_factor", 1.0))
            )
            rgb = rgb_chw.permute(0, 2, 3, 1).contiguous()
            depth = fragments.zbuf.contiguous()
            valid_mask = torch.isfinite(depth) & (depth > 0.0)
            faces.append(
                PerspectiveFaceObservation(
                    name=name,
                    camera=face_camera,
                    rgb=rgb,
                    depth_z=depth,
                    valid_mask=valid_mask,
                    R=face_camera.R,
                    T=face_camera.T,
                    K=K.clone(),
                    camera_center=center.clone(),
                    metadata={
                        "bundle_id": bundle_id,
                        "zfar": zfar,
                        "znear": znear,
                        "observation_mode": "cubemap6",
                        "source": "gt_mesh",
                    },
                )
            )

    _synchronize_if_cuda(device)
    capture_seconds = time.perf_counter() - capture_started
    bundle = ObservationBundle(
        bundle_id=bundle_id,
        center=center.clone(),
        faces=tuple(faces),
        capture_seconds=capture_seconds,
        metadata={
            "observation_mode": "cubemap6",
            "source": "gt_mesh",
            "face_size": face_size,
            "face_fov_degrees": 90.0,
            "render_count": len(faces),
        },
    )

    output_root = dir_path if dir_path is not None else camera.save_dir_path
    if save_frame and output_root is not None:
        bundle_dir = os.path.join(output_root, f"{bundle_id:06d}")
        os.makedirs(bundle_dir, exist_ok=True)
        for face in bundle.faces:
            torch.save(
                face.frame_dict(cpu=True),
                os.path.join(bundle_dir, f"{face.name}.pt"),
            )
        torch.save(
            {
                "bundle_id": bundle_id,
                "face_names": CUBEMAP_FACE_NAMES,
                "camera_center": center.detach().cpu(),
                "capture_seconds": capture_seconds,
                **bundle.metadata,
            },
            os.path.join(bundle_dir, "bundle.pt"),
        )
        if save_png:
            image_root = os.path.join(os.path.dirname(output_root), "imgs", f"{bundle_id:06d}")
            os.makedirs(image_root, exist_ok=True)
            for face in bundle.faces:
                image = face.rgb[0].permute(2, 0, 1).detach().cpu().clamp(0.0, 1.0)
                vision_functional.to_pil_image(image).save(
                    os.path.join(image_root, f"{face.name}.png")
                )

    # The bundle, not each face, is the unit of observation and history.
    camera.n_frames_captured += 1
    camera.last_observation_bundle = bundle
    return bundle


def ndc_pixel_grid(
    image_height: int,
    image_width: int,
    *,
    device: Any,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the repository's PyTorch3D NDC grid for arbitrary H/W."""

    if image_height < 2 or image_width < 2:
        raise ValueError("Projection grids must be at least 2x2 pixels.")
    minimum = min(image_height, image_width)
    rows = torch.arange(image_height, device=device, dtype=dtype)
    columns = torch.arange(image_width, device=device, dtype=dtype)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    ndc_x = image_width / minimum - 2.0 * column_grid / (minimum - 1.0)
    ndc_y = image_height / minimum - 2.0 * row_grid / (minimum - 1.0)
    return torch.stack((ndc_x, ndc_y), dim=-1)


def unproject_depth_to_world(
    depth_map: torch.Tensor,
    face_camera: FoVPerspectiveCameras,
) -> torch.Tensor:
    """Unproject a depth tensor of shape HxW, HxWx1, or 1xHxWx1."""

    depth = _depth_2d(depth_map)
    height, width = depth.shape
    ndc_xy = ndc_pixel_grid(
        height,
        width,
        device=depth.device,
        dtype=depth.dtype,
    )
    ndc_points = torch.cat((ndc_xy, depth.unsqueeze(-1)), dim=-1).reshape(1, -1, 3)
    return face_camera.unproject_points(
        ndc_points, scaled_depth_input=False
    ).reshape(height, width, 3)


def _depth_2d(depth_map: torch.Tensor) -> torch.Tensor:
    depth = depth_map
    if depth.ndim == 4:
        if depth.shape[0] != 1 or depth.shape[-1] != 1:
            raise ValueError("4D depth maps must have shape 1xHxWx1.")
        depth = depth[0, ..., 0]
    elif depth.ndim == 3:
        if depth.shape[-1] == 1:
            depth = depth[..., 0]
        elif depth.shape[0] == 1:
            depth = depth[0]
        else:
            raise ValueError("3D depth maps must have shape HxWx1 or 1xHxW.")
    elif depth.ndim != 2:
        raise ValueError("Depth maps must be 2D, 3D, or 4D tensors.")
    return depth


def _mask_2d(mask: torch.Tensor) -> torch.Tensor:
    return _depth_2d(mask).bool()


def _ndc_bounds(image_height: int, image_width: int) -> Tuple[float, float, float, float]:
    minimum = min(image_height, image_width)
    max_x = image_width / minimum
    min_x = max_x - 2.0 * (image_width - 1.0) / (minimum - 1.0)
    max_y = image_height / minimum
    min_y = max_y - 2.0 * (image_height - 1.0) / (minimum - 1.0)
    return min_x, max_x, min_y, max_y


def _project_points(
    points: torch.Tensor,
    face_camera: FoVPerspectiveCameras,
    image_height: int,
    image_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    projections = face_camera.get_full_projection_transform().transform_points(points)
    view_points = face_camera.get_world_to_view_transform().transform_points(points)
    min_x, max_x, min_y, max_y = _ndc_bounds(image_height, image_width)
    in_fov = (
        (projections[:, 0] >= min_x)
        & (projections[:, 0] <= max_x)
        & (projections[:, 1] >= min_y)
        & (projections[:, 1] <= max_y)
        & (view_points[:, 2] > 0.0)
    )
    return projections, view_points, in_fov


def _nearest_pixel_indices(
    projections: torch.Tensor,
    image_height: int,
    image_width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    minimum = min(image_height, image_width)
    column = torch.round(
        (image_width / minimum - projections[:, 0]) * (minimum - 1.0) / 2.0
    ).long()
    row = torch.round(
        (image_height / minimum - projections[:, 1]) * (minimum - 1.0) / 2.0
    ).long()
    return row.clamp(0, image_height - 1), column.clamp(0, image_width - 1)


def _voxel_mean(
    points: torch.Tensor,
    features: torch.Tensor,
    voxel_size: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if points.numel() == 0:
        return points.reshape(0, 3), features.reshape(0, features.shape[-1])
    voxel_indices = torch.floor(points / voxel_size).to(torch.int64)
    _, inverse = torch.unique(voxel_indices, dim=0, return_inverse=True)
    n_voxels = int(inverse.max().item()) + 1
    counts = torch.zeros(n_voxels, 1, dtype=points.dtype, device=points.device)
    counts.index_add_(
        0,
        inverse,
        torch.ones(points.shape[0], 1, dtype=points.dtype, device=points.device),
    )
    point_sums = torch.zeros(n_voxels, 3, dtype=points.dtype, device=points.device)
    point_sums.index_add_(0, inverse, points)
    feature_sums = torch.zeros(
        n_voxels,
        features.shape[-1],
        dtype=features.dtype,
        device=features.device,
    )
    feature_sums.index_add_(0, inverse, features)
    return point_sums / counts, feature_sums / counts.to(features.dtype)


def _face_projection_margin(
    projections: torch.Tensor,
    in_fov: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    min_x, max_x, min_y, max_y = _ndc_bounds(image_height, image_width)
    normalized_x = 2.0 * (projections[:, 0] - min_x) / (max_x - min_x) - 1.0
    normalized_y = 2.0 * (projections[:, 1] - min_y) / (max_y - min_y) - 1.0
    margin = 1.0 - torch.maximum(normalized_x.abs(), normalized_y.abs())
    return torch.where(in_fov, margin, torch.full_like(margin, -torch.inf))


def _proxy_union_and_signed_distance(
    bundle: ObservationBundle,
    proxy_points: torch.Tensor,
    sensor_range: float,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    n_points = int(proxy_points.shape[0])
    if n_points == 0:
        empty_mask = torch.zeros(0, dtype=torch.bool, device=proxy_points.device)
        empty_owner = torch.zeros(0, dtype=torch.long, device=proxy_points.device)
        return proxy_points.reshape(0, 3), empty_mask, None, empty_owner

    margins = []
    projections_by_face = []
    view_points_by_face = []
    range_mask = torch.linalg.norm(
        proxy_points - bundle.center.reshape(1, 3), dim=-1
    ) < sensor_range
    for face in bundle.faces:
        depth = _depth_2d(face.depth_z)
        projections, view_points, in_fov = _project_points(
            proxy_points, face.camera, *depth.shape
        )
        in_fov = in_fov & range_mask
        margins.append(
            _face_projection_margin(projections, in_fov, *depth.shape)
        )
        projections_by_face.append(projections)
        view_points_by_face.append(view_points)

    margin_stack = torch.stack(margins, dim=0)
    union_mask = torch.isfinite(margin_stack).any(dim=0)
    owner = torch.argmax(margin_stack, dim=0)
    if not union_mask.any():
        return proxy_points[union_mask], union_mask, None, owner

    signed_distance_all = torch.zeros(
        n_points, dtype=proxy_points.dtype, device=proxy_points.device
    )
    for face_index, face in enumerate(bundle.faces):
        owned = union_mask & (owner == face_index)
        if not owned.any():
            continue
        depth = _depth_2d(face.depth_z)
        valid = _mask_2d(face.valid_mask) & torch.isfinite(depth) & (depth > 0.0)
        row, column = _nearest_pixel_indices(
            projections_by_face[face_index][owned], *depth.shape
        )
        sampled_depth = depth[row, column]
        sampled_valid = valid[row, column]
        zfar = float(face.metadata.get("zfar", sensor_range))
        sampled_depth = torch.where(
            sampled_valid,
            sampled_depth,
            torch.full_like(sampled_depth, 1.1 * zfar),
        )
        signed_distance_all[owned] = (
            view_points_by_face[face_index][owned, 2] - sampled_depth
        )
    return (
        proxy_points[union_mask],
        union_mask,
        signed_distance_all[union_mask].reshape(-1, 1),
        owner,
    )


def process_cubemap_observation(
    *,
    bundle: ObservationBundle,
    proxy_points: torch.Tensor,
    gathering_factor: float,
    sensor_range: float,
    voxel_size: float,
    device: Any,
) -> Dict[str, Any]:
    """Fuse six real faces into one mapping/proxy update payload.

    The returned core keys match ``process_planning_depth_frame``.  Proxy
    points which lie on face seams receive one deterministic owner (maximum
    distance to the face boundary), so each proxy contributes one signed
    distance and one occupancy observation per bundle.
    """

    if not 0.0 < gathering_factor <= 1.0:
        raise ValueError("gathering_factor must be in the interval (0, 1].")
    if sensor_range <= 0.0:
        raise ValueError("sensor_range must be positive.")
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive.")
    if proxy_points.device != torch.device(device):
        proxy_points = proxy_points.to(device)

    geometry_started = time.perf_counter()
    raw_points = []
    raw_features = []
    face_valid_pixel_counts: Dict[str, int] = {}
    face_sampled_point_counts: Dict[str, int] = {}

    for face in bundle.faces:
        depth = _depth_2d(face.depth_z).to(device)
        mask = _mask_2d(face.valid_mask).to(device)
        mask = mask & torch.isfinite(depth) & (depth > 0.0) & (depth < sensor_range)
        world = unproject_depth_to_world(depth, face.camera)
        colors = face.rgb.to(device)[0]
        points = world[mask]
        features = colors[mask]
        face_valid_pixel_counts[face.name] = int(mask.sum().item())
        if points.shape[0] > 0:
            sample_count = max(1, int(points.shape[0] * gathering_factor))
            indices = torch.randperm(points.shape[0], device=points.device)[:sample_count]
            points = points[indices]
            features = features[indices]
        face_sampled_point_counts[face.name] = int(points.shape[0])
        raw_points.append(points)
        raw_features.append(features)

    if raw_points:
        concatenated_points = torch.cat(raw_points, dim=0)
        concatenated_features = torch.cat(raw_features, dim=0)
    else:
        concatenated_points = torch.zeros(0, 3, device=device)
        concatenated_features = torch.zeros(0, 3, device=device)
    part_pc, part_pc_features = _voxel_mean(
        concatenated_points, concatenated_features, voxel_size
    )
    fov_proxy_points, fov_proxy_mask, signed_distances, proxy_owner = (
        _proxy_union_and_signed_distance(bundle, proxy_points, sensor_range)
    )
    geometry_seconds = time.perf_counter() - geometry_started

    raw_count = int(concatenated_points.shape[0])
    unique_count = int(part_pc.shape[0])
    stats = {
        "observation_mode": "cubemap6",
        "bundle_id": int(bundle.bundle_id),
        "bundle_count": 1,
        "face_count": len(bundle.faces),
        "face_names": list(CUBEMAP_FACE_NAMES),
        "face_size": int(bundle.faces[0].image_height),
        "real_face_render_count": int(bundle.render_count),
        "capture_seconds": float(bundle.capture_seconds),
        "geometry_seconds": float(geometry_seconds),
        "raw_point_count": raw_count,
        "raw_sampled_point_count": raw_count,
        "unique_point_count": unique_count,
        "deduplicated_point_count": raw_count - unique_count,
        "voxel_size": float(voxel_size),
        "proxy_union_count": int(fov_proxy_mask.sum().item()),
        "face_point_counts": [
            face_sampled_point_counts[name] for name in CUBEMAP_FACE_NAMES
        ],
        "face_valid_pixel_counts": face_valid_pixel_counts,
        "face_sampled_point_counts": face_sampled_point_counts,
    }
    return {
        "part_pc": part_pc,
        "part_pc_features": part_pc_features,
        "fov_proxy_points": fov_proxy_points,
        "fov_proxy_mask": fov_proxy_mask,
        "sgn_dists": signed_distances,
        "X_cam": bundle.center.reshape(1, 3),
        "planning_masks": {face.name: face.valid_mask for face in bundle.faces},
        "depth_bundle": bundle,
        "bundle": bundle,
        "bundle_stats": stats,
        "proxy_face_owner": proxy_owner,
        "capture_seconds": float(bundle.capture_seconds),
        "geometry_seconds": float(geometry_seconds),
    }


def visible_mask_from_depth_map(
    points: torch.Tensor,
    face_camera: FoVPerspectiveCameras,
    depth_map: torch.Tensor,
    depth_tolerance: float = 1.0,
) -> torch.Tensor:
    """Test point visibility against one rendered z-depth map."""

    if depth_tolerance < 0.0:
        raise ValueError("depth_tolerance must be non-negative.")
    if points.shape[0] == 0:
        return torch.zeros(0, dtype=torch.bool, device=points.device)
    depth = _depth_2d(depth_map).to(points.device)
    projections, view_points, in_fov = _project_points(
        points, face_camera, *depth.shape
    )
    visible = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    indices = torch.where(in_fov)[0]
    if indices.numel() == 0:
        return visible
    row, column = _nearest_pixel_indices(projections[indices], *depth.shape)
    rendered_depth = depth[row, column]
    has_surface = torch.isfinite(rendered_depth) & (rendered_depth > 0.0)
    depth_match = view_points[indices, 2] <= rendered_depth + depth_tolerance
    visible[indices[has_surface & depth_match]] = True
    return visible


def _ordered_face_values(
    values: Any,
    *,
    label: str,
) -> Tuple[Any, ...]:
    if isinstance(values, Mapping):
        missing = [name for name in CUBEMAP_FACE_NAMES if name not in values]
        if missing:
            raise ValueError(f"{label} is missing cubemap faces: {missing}")
        return tuple(values[name] for name in CUBEMAP_FACE_NAMES)
    ordered = tuple(values)
    if len(ordered) != len(CUBEMAP_FACE_NAMES):
        raise ValueError(
            f"{label} must contain {len(CUBEMAP_FACE_NAMES)} entries; "
            f"received {len(ordered)}."
        )
    return ordered


def visible_union_from_depth_maps(
    points: torch.Tensor,
    face_cameras: Any,
    depth_maps: Any,
    depth_tolerance: float = 1.0,
) -> torch.Tensor:
    """OR point visibility across six real or imagined cubemap depth maps.

    ``face_cameras`` and ``depth_maps`` may each be either a face-name mapping
    or a six-element sequence in :data:`CUBEMAP_FACE_NAMES` order.
    """

    ordered_cameras = _ordered_face_values(face_cameras, label="face_cameras")
    ordered_depths = _ordered_face_values(depth_maps, label="depth_maps")
    visible_union = torch.zeros(
        points.shape[0], dtype=torch.bool, device=points.device
    )
    for face_camera, depth_map in zip(ordered_cameras, ordered_depths):
        visible_union |= visible_mask_from_depth_map(
            points,
            face_camera,
            depth_map,
            depth_tolerance=depth_tolerance,
        )
    return visible_union


__all__ = [
    "CUBEMAP_FACE_NAMES",
    "ObservationBundle",
    "PerspectiveFaceObservation",
    "build_cubemap_cameras",
    "capture_cubemap_observation",
    "cubemap_pixel_intrinsics",
    "ndc_pixel_grid",
    "process_cubemap_observation",
    "unproject_depth_to_world",
    "visible_mask_from_depth_map",
    "visible_union_from_depth_maps",
]
