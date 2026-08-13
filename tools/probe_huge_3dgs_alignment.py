#!/usr/bin/env python3
"""Run a single-view HUGE 3DGS/MAGICIAN camera and geometry feasibility probe.

This is deliberately not an observation provider and does not invoke the planner.
It renders one fixed MAGICIAN start pose at 128x128 from the official HUGE PLY,
then compares the two 3DGS depth definitions and alpha support with the converted
mesh z-buffer.  The script writes a machine-readable report, exact output arrays,
and a compact visual diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw
from pytorch3d.renderer import FoVPerspectiveCameras

# Make the checked-out package importable when this file is invoked as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from macarons.utility.gaussian_utils import (  # noqa: E402
    convert_camera_from_gs_to_pytorch3d,
    convert_camera_from_pytorch3d_to_gs,
)
from macarons.utility.macarons_utils import (  # noqa: E402
    Camera,
    Settings,
    get_rgb_renderer,
    load_scene,
)


LOGGER = logging.getLogger("huge_3dgs_probe")
REQUIRED_PLY_PROPERTIES = (
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "filter_3D",
)


@dataclass(frozen=True)
class PlyHeader:
    vertex_count: int
    data_offset: int
    dtype: np.dtype
    properties: tuple[str, ...]


@dataclass
class GaussianBuffers:
    xyz: torch.Tensor
    features_dc: torch.Tensor
    opacity: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ply", required=True, type=Path, help="Official point_cloud_utm50.ply"
    )
    parser.add_argument(
        "--scene-dir", required=True, type=Path, help="Converted MAGICIAN scene"
    )
    parser.add_argument("--landmarks", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--start-number", type=int, default=0)
    parser.add_argument("--znear", type=float, default=1.0)
    parser.add_argument("--zfar", type=float, default=1600.0)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--alpha-threshold", type=float, default=0.1)
    parser.add_argument("--kernel-size", type=float, default=0.0)
    parser.add_argument(
        "--frustum-center-margin",
        type=float,
        default=0.15,
        help=(
            "Extra image fraction retained on every side before rasterization; a "
            "3-sigma scale bound is added per Gaussian"
        ),
    )
    parser.add_argument("--minimum-overlap-pixels", type=int, default=500)
    parser.add_argument("--minimum-mask-iou", type=float, default=0.5)
    parser.add_argument(
        "--maximum-depth-median-relative-error", type=float, default=0.1
    )
    parser.add_argument("--maximum-depth-p90-relative-error", type=float, default=0.25)
    parser.add_argument("--maximum-camera-center-error-m", type=float, default=1e-4)
    parser.add_argument("--maximum-camera-rotation-error", type=float, default=1e-5)
    parser.add_argument(
        "--maximum-landmark-projection-error-px", type=float, default=0.01
    )
    return parser


def setup_logging(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "probe.log"
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)
    return log_path


def sha256_file(path: Path, chunk_size: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_binary_ply_header(path: Path) -> PlyHeader:
    scalar_types = {
        "char": "i1",
        "uchar": "u1",
        "int8": "i1",
        "uint8": "u1",
        "short": "<i2",
        "ushort": "<u2",
        "int16": "<i2",
        "uint16": "<u2",
        "int": "<i4",
        "uint": "<u4",
        "int32": "<i4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    vertex_count: int | None = None
    properties: list[tuple[str, str]] = []
    current_element: str | None = None
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError(f"not a PLY file: {path}")
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError("PLY header ended before end_header")
            line = raw.decode("ascii").strip()
            if line == "format binary_little_endian 1.0":
                continue
            if line.startswith("format "):
                raise ValueError(f"unsupported PLY format: {line}")
            fields = line.split()
            if fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
                current_element = "vertex"
            elif fields and fields[0] == "element":
                current_element = fields[1]
            elif fields and fields[0] == "property" and current_element == "vertex":
                if len(fields) != 3 or fields[1] == "list":
                    raise ValueError(f"unsupported vertex property: {line}")
                if fields[1] not in scalar_types:
                    raise ValueError(f"unsupported PLY scalar type: {fields[1]}")
                properties.append((fields[2], scalar_types[fields[1]]))
            elif line == "end_header":
                data_offset = handle.tell()
                break
    if vertex_count is None:
        raise ValueError("PLY has no vertex element")
    names = tuple(name for name, _ in properties)
    missing = sorted(set(REQUIRED_PLY_PROPERTIES) - set(names))
    if missing:
        raise ValueError(f"PLY is missing required properties: {missing}")
    if names != REQUIRED_PLY_PROPERTIES:
        raise ValueError(
            "unexpected PLY vertex schema/order; refusing to guess record layout: "
            f"{names}"
        )
    dtype = np.dtype(properties, align=False)
    expected_size = data_offset + vertex_count * dtype.itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"PLY size mismatch: header implies {expected_size} bytes, file has {actual_size}"
        )
    return PlyHeader(vertex_count, data_offset, dtype, names)


def _copy_columns(
    records: np.memmap,
    names: Iterable[str],
    start: int,
    stop: int,
    device: torch.device,
) -> torch.Tensor:
    array = np.stack([records[name][start:stop] for name in names], axis=1)
    return torch.from_numpy(array).to(device=device, dtype=torch.float32)


def load_official_gaussians(
    path: Path,
    header: PlyHeader,
    device: torch.device,
    chunk_size: int,
) -> GaussianBuffers:
    """Load every official PLY Gaussian and apply RaDe-GS inference activations."""
    count = header.vertex_count
    records = np.memmap(
        path,
        mode="r",
        dtype=header.dtype,
        offset=header.data_offset,
        shape=(count,),
    )
    xyz = torch.empty((count, 3), dtype=torch.float32, device=device)
    features_dc = torch.empty((count, 1, 3), dtype=torch.float32, device=device)
    opacity = torch.empty((count, 1), dtype=torch.float32, device=device)
    scales = torch.empty((count, 3), dtype=torch.float32, device=device)
    rotations = torch.empty((count, 4), dtype=torch.float32, device=device)

    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        xyz[start:stop].copy_(
            _copy_columns(records, ("x", "y", "z"), start, stop, device)
        )
        features_dc[start:stop, 0].copy_(
            _copy_columns(records, ("f_dc_0", "f_dc_1", "f_dc_2"), start, stop, device)
        )
        raw_scale = _copy_columns(
            records, ("scale_0", "scale_1", "scale_2"), start, stop, device
        )
        raw_opacity = _copy_columns(records, ("opacity",), start, stop, device)
        filter_3d = _copy_columns(records, ("filter_3D",), start, stop, device)
        raw_rotation = _copy_columns(
            records, ("rot_0", "rot_1", "rot_2", "rot_3"), start, stop, device
        )

        unfiltered_scales = torch.exp(raw_scale)
        scale_squares = torch.square(unfiltered_scales)
        filtered_scale_squares = scale_squares + torch.square(filter_3d)
        scales[start:stop].copy_(torch.sqrt(filtered_scale_squares))
        filter_coefficient = torch.sqrt(
            scale_squares.prod(dim=1) / filtered_scale_squares.prod(dim=1)
        )
        opacity[start:stop].copy_(
            torch.sigmoid(raw_opacity) * filter_coefficient.unsqueeze(1)
        )
        rotations[start:stop].copy_(torch.nn.functional.normalize(raw_rotation, dim=1))

        if start == 0 or stop == count or start // chunk_size % 10 == 0:
            LOGGER.info("loaded %d/%d Gaussians", stop, count)
    del records
    return GaussianBuffers(xyz, features_dc, opacity, scales, rotations)


def prefilter_gaussians_for_view(
    buffers: GaussianBuffers,
    gs_camera: Any,
    chunk_size: int,
    center_margin: float,
) -> tuple[GaussianBuffers, dict[str, Any]]:
    """Conservatively compact Gaussians before the memory-heavy CUDA rasterizer.

    RaDe-GS itself rejects centers at view-space z <= 0.2.  Its filter-building
    path also uses a 15% image margin.  We retain that center margin and expand it
    further by a 3-sigma screen-space upper bound based on each Gaussian's largest
    filtered scale.  This is deterministic culling, not sampling.
    """
    if center_margin < 0:
        raise ValueError("frustum-center-margin must be non-negative")
    count = buffers.xyz.shape[0]
    device = buffers.xyz.device
    keep = torch.empty(count, dtype=torch.bool, device=device)
    tan_fovx = math.tan(float(gs_camera.FoVx) * 0.5)
    tan_fovy = math.tan(float(gs_camera.FoVy) * 0.5)
    center_limit = 1.0 + 2.0 * center_margin
    sigma_multiplier = 3.0
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        xyz = buffers.xyz[start:stop]
        points_h = torch.cat(
            (xyz, torch.ones((stop - start, 1), dtype=xyz.dtype, device=device)), dim=1
        )
        view = points_h @ gs_camera.world_view_transform
        clip = points_h @ gs_camera.full_proj_transform
        ndc_xy = clip[:, :2] / torch.clamp(clip[:, 3:4], min=1e-7)
        z = view[:, 2]
        safe_z = torch.clamp(z, min=0.2)
        maximum_scale = buffers.scales[start:stop].amax(dim=1)
        x_over_z = view[:, 0] / safe_z
        y_over_z = view[:, 1] / safe_z
        radius_x_ndc = (
            sigma_multiplier
            * maximum_scale
            / (safe_z * tan_fovx)
            * torch.sqrt(1.0 + torch.square(x_over_z))
        )
        radius_y_ndc = (
            sigma_multiplier
            * maximum_scale
            / (safe_z * tan_fovy)
            * torch.sqrt(1.0 + torch.square(y_over_z))
        )
        keep[start:stop] = (
            (z > 0.2)
            & torch.isfinite(ndc_xy).all(dim=1)
            & (torch.abs(ndc_xy[:, 0]) <= center_limit + radius_x_ndc)
            & (torch.abs(ndc_xy[:, 1]) <= center_limit + radius_y_ndc)
        )
    kept_count = int(keep.sum().item())
    if kept_count == 0:
        raise RuntimeError("conservative view prefilter rejected every Gaussian")
    LOGGER.info(
        "conservative view prefilter retained %d/%d Gaussians (%.2f%%)",
        kept_count,
        count,
        kept_count * 100.0 / count,
    )

    # Compact one field at a time so the full source and all compacted copies are
    # never simultaneously resident.  empty_cache returns each released source
    # allocation before the rasterizer requests its much larger geometry buffers.
    buffers.xyz = buffers.xyz[keep].contiguous()
    torch.cuda.empty_cache()
    buffers.features_dc = buffers.features_dc[keep].contiguous()
    torch.cuda.empty_cache()
    buffers.opacity = buffers.opacity[keep].contiguous()
    torch.cuda.empty_cache()
    buffers.scales = buffers.scales[keep].contiguous()
    torch.cuda.empty_cache()
    buffers.rotations = buffers.rotations[keep].contiguous()
    del keep
    torch.cuda.empty_cache()
    return buffers, {
        "method": "deterministic conservative view-frustum compaction; no sampling",
        "input_gaussians": count,
        "retained_gaussians": kept_count,
        "retained_fraction": kept_count / count,
        "near_center_z_m": 0.2,
        "center_margin_each_image_side": center_margin,
        "ndc_center_limit": center_limit,
        "scale_bound_sigma": sigma_multiplier,
    }


def read_landmarks(path: Path) -> tuple[np.ndarray, list[str]]:
    points: list[list[float]] = []
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 4:
            fields = line.split(maxsplit=3)
        if len(fields) < 4:
            raise ValueError(f"invalid landmark row: {line!r}")
        points.append([float(fields[0]), float(fields[1]), float(fields[2])])
        labels.append(fields[3].strip())
    if len(points) < 3:
        raise ValueError(f"expected at least three landmarks, found {len(points)}")
    return np.asarray(points, dtype=np.float32), labels


def quantiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"median": None, "p90": None, "p95": None, "max": None}
    return {
        "median": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def mask_metrics(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, float | int]:
    intersection = int(np.logical_and(reference, candidate).sum())
    union = int(np.logical_or(reference, candidate).sum())
    reference_count = int(reference.sum())
    candidate_count = int(candidate.sum())
    return {
        "intersection_pixels": intersection,
        "union_pixels": union,
        "reference_pixels": reference_count,
        "candidate_pixels": candidate_count,
        "iou": float(intersection / union) if union else 1.0,
        "mesh_recall": float(intersection / reference_count)
        if reference_count
        else 0.0,
        "gaussian_recall": float(intersection / candidate_count)
        if candidate_count
        else 0.0,
    }


def depth_metrics(
    mesh_depth: np.ndarray,
    gaussian_depth: np.ndarray,
    overlap: np.ndarray,
) -> dict[str, Any]:
    valid = (
        overlap
        & np.isfinite(mesh_depth)
        & np.isfinite(gaussian_depth)
        & (mesh_depth > 0)
        & (gaussian_depth > 0)
    )
    mesh = mesh_depth[valid]
    gaussian = gaussian_depth[valid]
    absolute = np.abs(gaussian - mesh)
    relative = absolute / np.maximum(np.abs(mesh), 1e-6)
    ratio = gaussian / np.maximum(mesh, 1e-6)
    return {
        "overlap_pixels": int(valid.sum()),
        "absolute_error_m": quantiles(absolute),
        "relative_error": quantiles(relative),
        "gaussian_to_mesh_depth_ratio": quantiles(ratio),
        "mesh_depth_m": quantiles(mesh),
        "gaussian_depth_m": quantiles(gaussian),
    }


def git_facts(root: Path) -> dict[str, Any]:
    def command(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    return {
        "head": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "status_porcelain": command("status", "--short"),
    }


def _to_uint8_rgb(array: np.ndarray) -> np.ndarray:
    return np.clip(np.moveaxis(array, 0, -1), 0.0, 1.0).astype(np.float32) * 255.0


def _gray(array: np.ndarray) -> np.ndarray:
    values = np.clip(array, 0.0, 1.0)
    return np.repeat((values[..., None] * 255.0).astype(np.uint8), 3, axis=2)


def _depth_visual(
    depth: np.ndarray, valid: np.ndarray, low: float, high: float
) -> np.ndarray:
    normalized = np.zeros_like(depth, dtype=np.float32)
    normalized[valid] = np.clip((depth[valid] - low) / max(high - low, 1e-6), 0.0, 1.0)
    red = normalized
    green = 1.0 - np.abs(normalized * 2.0 - 1.0)
    blue = 1.0 - normalized
    rgb = np.stack((red, green, blue), axis=-1)
    rgb[~valid] = 0.0
    return (rgb * 255.0).astype(np.uint8)


def save_visualization(
    output_path: Path,
    rgb: np.ndarray,
    alpha: np.ndarray,
    expected_depth: np.ndarray,
    median_depth: np.ndarray,
    mesh_depth: np.ndarray,
    mesh_mask: np.ndarray,
    gaussian_mask: np.ndarray,
) -> None:
    all_depth = np.concatenate(
        [
            expected_depth[gaussian_mask],
            median_depth[gaussian_mask],
            mesh_depth[mesh_mask],
        ]
    )
    all_depth = all_depth[np.isfinite(all_depth) & (all_depth > 0)]
    low, high = np.quantile(all_depth, [0.02, 0.98]) if all_depth.size else (0.0, 1.0)
    overlap = mesh_mask & gaussian_mask
    expected_relative = np.zeros_like(mesh_depth, dtype=np.float32)
    median_relative = np.zeros_like(mesh_depth, dtype=np.float32)
    expected_relative[overlap] = np.abs(
        expected_depth[overlap] - mesh_depth[overlap]
    ) / np.maximum(mesh_depth[overlap], 1e-6)
    median_relative[overlap] = np.abs(
        median_depth[overlap] - mesh_depth[overlap]
    ) / np.maximum(mesh_depth[overlap], 1e-6)
    mask_overlay = np.zeros((*mesh_mask.shape, 3), dtype=np.uint8)
    mask_overlay[mesh_mask & ~gaussian_mask] = (255, 64, 64)
    mask_overlay[gaussian_mask & ~mesh_mask] = (64, 64, 255)
    mask_overlay[overlap] = (64, 220, 64)

    panels = [
        ("3DGS RGB", _to_uint8_rgb(rgb).astype(np.uint8)),
        ("alpha", _gray(alpha)),
        ("mesh zbuf", _depth_visual(mesh_depth, mesh_mask, float(low), float(high))),
        ("mask: green overlap", mask_overlay),
        (
            "expected depth",
            _depth_visual(expected_depth, gaussian_mask, float(low), float(high)),
        ),
        (
            "median depth",
            _depth_visual(median_depth, gaussian_mask, float(low), float(high)),
        ),
        ("expected rel err", _depth_visual(expected_relative, overlap, 0.0, 0.5)),
        ("median rel err", _depth_visual(median_relative, overlap, 0.0, 0.5)),
    ]
    height, width = mesh_mask.shape
    label_height = 22
    canvas = Image.new(
        "RGB", (4 * width, 2 * (height + label_height)), color=(20, 20, 20)
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(panels):
        row, column = divmod(index, 4)
        x, y = column * width, row * (height + label_height)
        canvas.paste(Image.fromarray(panel), (x, y + label_height))
        draw.text((x + 4, y + 4), label, fill=(240, 240, 240))
    canvas.save(output_path)


def build_fixed_camera(
    scene_dir: Path,
    device: torch.device,
    image_height: int,
    image_width: int,
    start_number: int,
    zfar: float,
) -> tuple[Any, torch.Tensor, torch.Tensor, Any, Any]:
    settings_dict = json.loads(
        (scene_dir / "settings.json").read_text(encoding="utf-8")
    )
    occupied = torch.load(scene_dir / "occupied_pose.pt", map_location="cpu")
    settings = Settings(settings_dict, device=device, scene_scale_factor=1.0)
    base_camera = FoVPerspectiveCameras(device=device, zfar=zfar)
    renderer = get_rgb_renderer(
        image_height=image_height,
        image_width=image_width,
        ambient_light_intensity=1.0,
        cameras=base_camera,
        device=device,
    )
    camera = Camera(
        x_min=settings.camera.x_min,
        x_max=settings.camera.x_max,
        pose_l=settings.camera.pose_l,
        pose_w=settings.camera.pose_w,
        pose_h=settings.camera.pose_h,
        pose_n_elev=settings.camera.pose_n_elev,
        pose_n_azim=settings.camera.pose_n_azim,
        n_interpolation_steps=1,
        zfar=zfar,
        renderer=renderer,
        device=device,
        contrast_factor=settings.camera.contrast_factor,
        occupied_pose_data={key: value.to(device) for key, value in occupied.items()},
    )
    if not 0 <= start_number < len(settings.camera.start_positions):
        raise IndexError(
            f"start-number {start_number} outside [0, {len(settings.camera.start_positions)})"
        )
    start_index = settings.camera.start_positions[start_number]
    camera.initialize_camera(start_index)
    pose, _ = camera.get_pose_from_idx(start_index)
    center, angles, fov_camera = camera.get_camera_parameters_from_pose(pose)
    fov_camera.K = fov_camera.get_projection_transform().get_matrix().transpose(-1, -2)
    return settings, start_index, center, angles, fov_camera


def camera_and_landmark_report(
    fov_magic: Any,
    center_magic: torch.Tensor,
    transform_huge_to_magic: torch.Tensor,
    landmarks_huge_np: np.ndarray,
    labels: list[str],
    image_height: int,
    image_width: int,
    znear: float,
    zfar: float,
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    q = transform_huge_to_magic[:3, :3]
    r_huge = q.transpose(0, 1) @ fov_magic.R[0]
    p3d_huge = FoVPerspectiveCameras(
        device=device,
        R=r_huge.unsqueeze(0),
        T=fov_magic.T,
        K=fov_magic.K,
        znear=znear,
        zfar=zfar,
    )
    gs_camera = convert_camera_from_pytorch3d_to_gs(
        p3d_huge, height=image_height, width=image_width, device=device
    )[0]
    gs_camera.znear = znear
    gs_camera.zfar = zfar
    roundtrip_huge = convert_camera_from_gs_to_pytorch3d([gs_camera], device=device)
    r_magic_roundtrip = q @ roundtrip_huge.R[0]
    center_huge_expected = center_magic[0] @ q
    center_huge_roundtrip = roundtrip_huge.get_camera_center()[0]
    center_magic_roundtrip = center_huge_roundtrip @ q.transpose(0, 1)

    landmarks_huge = torch.from_numpy(landmarks_huge_np).to(device=device)
    landmarks_magic = landmarks_huge @ q.transpose(0, 1)
    image_size = torch.tensor(
        [[image_height, image_width]], dtype=torch.float32, device=device
    )
    p3d_screen = fov_magic.transform_points_screen(
        landmarks_magic.unsqueeze(0), image_size=image_size
    )[0]
    roundtrip_screen = roundtrip_huge.transform_points_screen(
        landmarks_huge.unsqueeze(0), image_size=image_size
    )[0]
    points_h = torch.cat(
        (landmarks_huge, torch.ones((len(landmarks_huge), 1), device=device)), dim=1
    )
    clip = points_h @ gs_camera.full_proj_transform
    ndc_xy = clip[:, :2] / clip[:, 3:4]
    raster_xy = torch.stack(
        (
            (ndc_xy[:, 0] + 1.0) * image_width / 2.0 - 0.5,
            (ndc_xy[:, 1] + 1.0) * image_height / 2.0 - 0.5,
        ),
        dim=1,
    )
    aligned_raster_xy = raster_xy + 0.5
    p3d_view_depth = fov_magic.get_world_to_view_transform().transform_points(
        landmarks_magic
    )[:, 2]
    gs_view_depth = (points_h @ gs_camera.world_view_transform)[:, 2]
    visible = (
        (p3d_view_depth >= znear)
        & (p3d_view_depth <= zfar)
        & (p3d_screen[:, 0] >= 0)
        & (p3d_screen[:, 0] < image_width)
        & (p3d_screen[:, 1] >= 0)
        & (p3d_screen[:, 1] < image_height)
    )
    visible_indices = torch.where(visible)[0]
    if len(visible_indices) >= 3:
        optical_distance = torch.linalg.norm(
            p3d_screen[visible_indices, :2]
            - torch.tensor([image_width / 2.0, image_height / 2.0], device=device),
            dim=1,
        )
        selected = visible_indices[torch.argsort(optical_distance)[:3]]
    else:
        selected = torch.argsort(torch.abs(p3d_screen[:, 2]))[:3]

    landmarks: list[dict[str, Any]] = []
    for raw_index in selected.detach().cpu().tolist():
        projection_error = torch.linalg.norm(
            aligned_raster_xy[raw_index] - p3d_screen[raw_index, :2]
        )
        roundtrip_error = torch.linalg.norm(
            roundtrip_screen[raw_index, :2] - p3d_screen[raw_index, :2]
        )
        landmarks.append(
            {
                "index": raw_index,
                "label": labels[raw_index],
                "huge_xyz": landmarks_huge[raw_index].detach().cpu().tolist(),
                "magic_xyz": landmarks_magic[raw_index].detach().cpu().tolist(),
                "in_frame": bool(visible[raw_index].item()),
                "pytorch3d_pixel_xy": p3d_screen[raw_index, :2].detach().cpu().tolist(),
                "gs_raster_pixel_center_xy": raster_xy[raw_index]
                .detach()
                .cpu()
                .tolist(),
                "gs_aligned_pixel_xy": aligned_raster_xy[raw_index]
                .detach()
                .cpu()
                .tolist(),
                "gs_to_pytorch3d_projection_error_px": float(projection_error.item()),
                "roundtrip_projection_error_px": float(roundtrip_error.item()),
                "pytorch3d_view_depth_m": float(p3d_view_depth[raw_index].item()),
                "gs_view_depth_m": float(gs_view_depth[raw_index].item()),
                "view_depth_abs_error_m": float(
                    torch.abs(
                        p3d_view_depth[raw_index] - gs_view_depth[raw_index]
                    ).item()
                ),
            }
        )

    report = {
        "frame_mapping": {
            "equation_column_vectors": "x_magic = Q @ x_huge",
            "Q": q.detach().cpu().tolist(),
            "determinant": float(torch.linalg.det(q).item()),
            "camera_equation_row_vectors": "R_huge = Q.T @ R_magic; T_huge = T_magic",
        },
        "pytorch3d_magic": {
            "R": fov_magic.R[0].detach().cpu().tolist(),
            "T": fov_magic.T[0].detach().cpu().tolist(),
            "K": fov_magic.K[0].detach().cpu().tolist(),
            "center_m": center_magic[0].detach().cpu().tolist(),
        },
        "pytorch3d_huge": {
            "R": r_huge.detach().cpu().tolist(),
            "T": p3d_huge.T[0].detach().cpu().tolist(),
            "center_m": p3d_huge.get_camera_center()[0].detach().cpu().tolist(),
        },
        "gaussian_splatting": {
            "R": gs_camera.R.detach().cpu().tolist(),
            "T": gs_camera.T.detach().cpu().tolist(),
            "FoVx_rad": float(gs_camera.FoVx),
            "FoVy_rad": float(gs_camera.FoVy),
            "focal_x_px": float(gs_camera.focal_x),
            "focal_y_px": float(gs_camera.focal_y),
            "center_huge_m": gs_camera.camera_center.detach().cpu().tolist(),
        },
        "round_trip": {
            "rotation_magic_max_abs_error": float(
                torch.max(torch.abs(r_magic_roundtrip - fov_magic.R[0])).item()
            ),
            "translation_max_abs_error_m": float(
                torch.max(torch.abs(roundtrip_huge.T[0] - fov_magic.T[0])).item()
            ),
            "center_huge_max_abs_error_m": float(
                torch.max(
                    torch.abs(center_huge_roundtrip - center_huge_expected)
                ).item()
            ),
            "center_magic_max_abs_error_m": float(
                torch.max(torch.abs(center_magic_roundtrip - center_magic[0])).item()
            ),
            "pixel_center_note": (
                "GS raster coordinates index pixel centers at (n+0.5); +0.5 is applied only "
                "for comparison with PyTorch3D screen coordinates"
            ),
        },
        "landmark_count": len(landmarks_huge_np),
        "visible_landmark_count": int(visible.sum().item()),
        "selected_landmarks": landmarks,
    }
    return gs_camera, report


def render_gaussians(
    buffers: GaussianBuffers,
    gs_camera: Any,
    image_height: int,
    image_width: int,
    kernel_size: float,
) -> dict[str, torch.Tensor]:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    settings = GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=math.tan(float(gs_camera.FoVx) * 0.5),
        tanfovy=math.tan(float(gs_camera.FoVy) * 0.5),
        kernel_size=kernel_size,
        bg=torch.zeros(3, dtype=torch.float32, device=buffers.xyz.device),
        scale_modifier=1.0,
        viewmatrix=gs_camera.world_view_transform,
        projmatrix=gs_camera.full_proj_transform,
        sh_degree=0,
        campos=gs_camera.camera_center,
        prefiltered=False,
        require_depth=True,
        require_coord=True,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=settings)
    means2d = torch.zeros_like(buffers.xyz)
    outputs = rasterizer(
        means3D=buffers.xyz,
        means2D=means2d,
        shs=buffers.features_dc,
        colors_precomp=None,
        opacities=buffers.opacity,
        scales=buffers.scales,
        rotations=buffers.rotations,
        cov3D_precomp=None,
    )
    (
        rgb,
        radii,
        expected_coord,
        median_coord,
        expected_depth,
        median_depth,
        alpha,
        normal,
    ) = outputs
    return {
        "rgb": rgb,
        "radii": radii,
        "expected_coord": expected_coord,
        "median_coord": median_coord,
        "expected_depth": expected_depth,
        "median_depth": median_depth,
        "alpha": alpha,
        "normal": normal,
    }


def run_probe(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    ply = args.ply.resolve()
    scene_dir = args.scene_dir.resolve()
    landmarks_path = args.landmarks.resolve()
    for path in (ply, scene_dir, landmarks_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if (args.image_height, args.image_width) != (128, 128):
        raise ValueError(
            "this feasibility probe is intentionally restricted to 128x128"
        )
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(
            "the official 102M-Gaussian probe requires an available CUDA device"
        )
    device = torch.device(args.device)
    cuda_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    torch.cuda.set_device(cuda_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(cuda_index)

    manifest = json.loads(
        (scene_dir / "conversion_manifest.json").read_text(encoding="utf-8")
    )
    transform = torch.tensor(
        manifest["coordinate_system"]["T_huge_to_magic"],
        dtype=torch.float32,
        device=device,
    )
    expected_ply = next(
        (
            entry
            for entry in manifest["inputs"]
            if Path(entry["path"]).name == ply.name
            and entry["size_bytes"] == ply.stat().st_size
        ),
        None,
    )
    LOGGER.info("hashing official PLY (%d bytes)", ply.stat().st_size)
    hash_started = time.perf_counter()
    actual_sha256 = sha256_file(ply)
    hash_seconds = time.perf_counter() - hash_started
    if expected_ply and actual_sha256 != expected_ply["sha256"]:
        raise ValueError(
            f"PLY SHA-256 mismatch: {actual_sha256} != {expected_ply['sha256']}"
        )
    header = parse_binary_ply_header(ply)
    LOGGER.info(
        "validated PLY schema: %d vertices, %d-byte records",
        header.vertex_count,
        header.dtype.itemsize,
    )

    mesh_started = time.perf_counter()
    obj_paths = sorted(scene_dir.glob("*.obj"))
    if len(obj_paths) != 1:
        raise RuntimeError(f"expected exactly one scene OBJ, found {len(obj_paths)}")
    mesh = load_scene(str(obj_paths[0]), scene_scale_factor=1.0, device=device)
    mesh_load_seconds = time.perf_counter() - mesh_started
    settings, start_index, center, angles, fov_magic = build_fixed_camera(
        scene_dir,
        device,
        args.image_height,
        args.image_width,
        args.start_number,
        args.zfar,
    )
    landmarks_np, labels = read_landmarks(landmarks_path)
    gs_camera, camera_report = camera_and_landmark_report(
        fov_magic,
        center,
        transform,
        landmarks_np,
        labels,
        args.image_height,
        args.image_width,
        args.znear,
        args.zfar,
        device,
    )

    mesh_render_started = time.perf_counter()
    renderer = get_rgb_renderer(
        image_height=args.image_height,
        image_width=args.image_width,
        ambient_light_intensity=1.0,
        cameras=fov_magic,
        device=device,
    )
    fragments = renderer.rasterizer(mesh, cameras=fov_magic)
    torch.cuda.synchronize(device)
    mesh_render_seconds = time.perf_counter() - mesh_render_started
    mesh_depth = fragments.zbuf[0, :, :, 0].detach().cpu().numpy()
    mesh_mask = fragments.pix_to_face[0, :, :, 0].detach().cpu().numpy() >= 0

    LOGGER.info("loading all %d official Gaussians", header.vertex_count)
    gaussian_load_started = time.perf_counter()
    buffers = load_official_gaussians(ply, header, device, args.chunk_size)
    torch.cuda.synchronize(device)
    gaussian_load_seconds = time.perf_counter() - gaussian_load_started
    prefilter_started = time.perf_counter()
    buffers, prefilter_report = prefilter_gaussians_for_view(
        buffers,
        gs_camera,
        args.chunk_size,
        args.frustum_center_margin,
    )
    torch.cuda.synchronize(device)
    gaussian_prefilter_seconds = time.perf_counter() - prefilter_started
    LOGGER.info("rendering one %dx%d 3DGS view", args.image_width, args.image_height)
    render_started = time.perf_counter()
    with torch.no_grad():
        rendered = render_gaussians(
            buffers,
            gs_camera,
            args.image_height,
            args.image_width,
            args.kernel_size,
        )
    torch.cuda.synchronize(device)
    gaussian_render_seconds = time.perf_counter() - render_started

    rgb = rendered["rgb"].detach().cpu().numpy()
    alpha = rendered["alpha"].squeeze().detach().cpu().numpy()
    expected_depth = rendered["expected_depth"].squeeze().detach().cpu().numpy()
    median_depth = rendered["median_depth"].squeeze().detach().cpu().numpy()
    gaussian_mask = (
        (alpha >= args.alpha_threshold)
        & np.isfinite(expected_depth)
        & np.isfinite(median_depth)
        & (expected_depth > 0)
        & (median_depth > 0)
    )
    output_shapes = {
        "rgb": list(rgb.shape),
        "expected_depth": list(expected_depth.shape),
        "median_depth": list(median_depth.shape),
        "alpha": list(alpha.shape),
        "mesh_zbuf": list(mesh_depth.shape),
    }
    expected_shapes = {
        "rgb": [3, args.image_height, args.image_width],
        "expected_depth": [args.image_height, args.image_width],
        "median_depth": [args.image_height, args.image_width],
        "alpha": [args.image_height, args.image_width],
        "mesh_zbuf": [args.image_height, args.image_width],
    }
    finite_outputs = {
        "rgb": bool(np.isfinite(rgb).all()),
        "expected_depth_on_valid": bool(
            np.isfinite(expected_depth[gaussian_mask]).all()
        ),
        "median_depth_on_valid": bool(np.isfinite(median_depth[gaussian_mask]).all()),
        "alpha": bool(np.isfinite(alpha).all()),
        "mesh_zbuf_on_valid": bool(np.isfinite(mesh_depth[mesh_mask]).all()),
    }

    flips = {
        "identity": gaussian_mask,
        "horizontal": np.fliplr(gaussian_mask),
        "vertical": np.flipud(gaussian_mask),
        "horizontal_vertical": np.flipud(np.fliplr(gaussian_mask)),
    }
    mask_alignment = {
        name: mask_metrics(mesh_mask, value) for name, value in flips.items()
    }
    identity_iou = float(mask_alignment["identity"]["iou"])
    best_orientation = max(
        mask_alignment, key=lambda name: float(mask_alignment[name]["iou"])
    )
    identity_orientation_best = (
        identity_iou
        >= max(float(value["iou"]) for value in mask_alignment.values()) - 1e-9
    )
    overlap = mesh_mask & gaussian_mask
    expected_alignment = depth_metrics(mesh_depth, expected_depth, overlap)
    median_alignment = depth_metrics(mesh_depth, median_depth, overlap)

    camera_roundtrip = camera_report["round_trip"]
    landmark_errors = [
        item["gs_to_pytorch3d_projection_error_px"]
        for item in camera_report["selected_landmarks"]
    ]
    camera_pass = bool(
        camera_roundtrip["center_magic_max_abs_error_m"]
        <= args.maximum_camera_center_error_m
        and camera_roundtrip["rotation_magic_max_abs_error"]
        <= args.maximum_camera_rotation_error
        and camera_report["visible_landmark_count"] >= 3
        and max(landmark_errors, default=float("inf"))
        <= args.maximum_landmark_projection_error_px
    )
    render_pass = bool(
        output_shapes == expected_shapes
        and all(finite_outputs.values())
        and int(gaussian_mask.sum()) > 0
        and int(mesh_mask.sum()) > 0
    )

    def depth_mode_pass(metrics: dict[str, Any]) -> bool:
        relative = metrics["relative_error"]
        return bool(
            metrics["overlap_pixels"] >= args.minimum_overlap_pixels
            and relative["median"] is not None
            and relative["p90"] is not None
            and relative["median"] <= args.maximum_depth_median_relative_error
            and relative["p90"] <= args.maximum_depth_p90_relative_error
        )

    expected_depth_pass = depth_mode_pass(expected_alignment)
    median_depth_pass = depth_mode_pass(median_alignment)
    geometry_alignment_pass = bool(
        identity_orientation_best
        and identity_iou >= args.minimum_mask_iou
        and (expected_depth_pass or median_depth_pass)
    )
    overall_pass = camera_pass and render_pass and geometry_alignment_pass

    output_dir = args.output_dir.resolve()
    arrays_path = output_dir / "probe_outputs.npz"
    np.savez_compressed(
        arrays_path,
        rgb=rgb.astype(np.float32),
        expected_depth=expected_depth.astype(np.float32),
        median_depth=median_depth.astype(np.float32),
        alpha=alpha.astype(np.float32),
        mesh_zbuf=mesh_depth.astype(np.float32),
        mesh_mask=mesh_mask,
        gaussian_mask=gaussian_mask,
    )
    overview_path = output_dir / "probe_overview.png"
    save_visualization(
        overview_path,
        rgb,
        alpha,
        expected_depth,
        median_depth,
        mesh_depth,
        mesh_mask,
        gaussian_mask,
    )
    cuda_free, cuda_total = torch.cuda.mem_get_info(device)
    report: dict[str, Any] = {
        "schema_version": 1,
        "probe": "HUGE 1_office official 3DGS single-view alignment",
        "scope": {
            "planner_connected": False,
            "observation_provider_implemented": False,
            "gt_color_decoupled": False,
            "formal_256x456_smoke_run": False,
            "view_count": 1,
            "resolution": [args.image_height, args.image_width],
        },
        "result": "PASS" if overall_pass else "FAIL",
        "checks": {
            "camera_round_trip_and_landmarks": camera_pass,
            "official_ply_render": render_pass,
            "mesh_3dgs_geometry_alignment": geometry_alignment_pass,
            "expected_depth_alignment": expected_depth_pass,
            "median_depth_alignment": median_depth_pass,
        },
        "thresholds_fixed_before_run": {
            "alpha": args.alpha_threshold,
            "frustum_center_margin_each_image_side": args.frustum_center_margin,
            "minimum_overlap_pixels": args.minimum_overlap_pixels,
            "minimum_mask_iou": args.minimum_mask_iou,
            "maximum_depth_median_relative_error": args.maximum_depth_median_relative_error,
            "maximum_depth_p90_relative_error": args.maximum_depth_p90_relative_error,
            "maximum_camera_center_error_m": args.maximum_camera_center_error_m,
            "maximum_camera_rotation_error": args.maximum_camera_rotation_error,
            "maximum_landmark_projection_error_px": args.maximum_landmark_projection_error_px,
            "depth_rule": "either expected or median depth must satisfy both relative-error bounds",
        },
        "inputs": {
            "ply": str(ply),
            "ply_size_bytes": ply.stat().st_size,
            "ply_sha256": actual_sha256,
            "manifest_expected_ply_sha256": expected_ply["sha256"]
            if expected_ply
            else None,
            "ply_schema": list(header.properties),
            "gaussian_count": header.vertex_count,
            "scene_dir": str(scene_dir),
            "mesh_obj": str(obj_paths[0]),
            "landmarks": str(landmarks_path),
            "landmark_sha256": sha256_file(landmarks_path),
        },
        "fixed_view": {
            "start_number": args.start_number,
            "start_index": start_index.detach().cpu().tolist(),
            "center_magic_m": center[0].detach().cpu().tolist(),
            "angles_elevation_azimuth_deg": angles[0].detach().cpu().tolist(),
            "znear_m": args.znear,
            "zfar_m": args.zfar,
        },
        "camera": camera_report,
        "outputs": {
            "shapes": output_shapes,
            "finite": finite_outputs,
            "alpha": {
                "min": float(np.min(alpha)),
                "max": float(np.max(alpha)),
                "mean": float(np.mean(alpha)),
                "valid_pixels": int(gaussian_mask.sum()),
            },
            "rgb": {
                "min": float(np.min(rgb)),
                "max": float(np.max(rgb)),
                "mean": float(np.mean(rgb)),
            },
            "visible_gaussians": int((rendered["radii"] > 0).sum().item()),
            "prefilter": prefilter_report,
            "artifacts": [
                arrays_path.name,
                overview_path.name,
                "probe.log",
                "probe_report.json",
            ],
        },
        "mesh": {
            "vertices": int(mesh.verts_packed().shape[0]),
            "faces": int(mesh.faces_packed().shape[0]),
            "visible_pixels": int(mesh_mask.sum()),
            "visible_zbuf_m": quantiles(mesh_depth[mesh_mask]),
            "settings_camera_envelope": {
                "min": settings.camera.x_min.detach().cpu().tolist(),
                "max": settings.camera.x_max.detach().cpu().tolist(),
            },
        },
        "alignment": {
            "mask_by_orientation": mask_alignment,
            "best_orientation": best_orientation,
            "identity_orientation_best": identity_orientation_best,
            "expected_depth": expected_alignment,
            "median_depth": median_alignment,
        },
        "timing_seconds": {
            "sha256": hash_seconds,
            "mesh_load": mesh_load_seconds,
            "mesh_render": mesh_render_seconds,
            "gaussian_load_and_activation": gaussian_load_seconds,
            "gaussian_view_prefilter": gaussian_prefilter_seconds,
            "gaussian_render": gaussian_render_seconds,
            "total": time.perf_counter() - started,
        },
        "gpu": {
            "logical_device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "name": torch.cuda.get_device_name(device),
            "total_bytes": int(cuda_total),
            "free_bytes_after_probe": int(cuda_free),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "repository": git_facts(repo_root),
        },
    }
    return report


def main() -> int:
    args = build_parser().parse_args()
    log_path = setup_logging(args.output_dir.resolve())
    repo_root = REPO_ROOT
    report_path = args.output_dir.resolve() / "probe_report.json"
    try:
        LOGGER.info("starting constrained one-view feasibility probe")
        report = run_probe(args, repo_root)
        LOGGER.info("probe result: %s", report["result"])
        exit_code = 0 if report["result"] == "PASS" else 2
    except Exception as exc:
        LOGGER.exception("probe failed before completing all checks")
        report = {
            "schema_version": 1,
            "probe": "HUGE 1_office official 3DGS single-view alignment",
            "result": "ERROR",
            "scope": {
                "planner_connected": False,
                "observation_provider_implemented": False,
                "gt_color_decoupled": False,
                "formal_256x456_smoke_run": False,
                "view_count": 1,
                "resolution": [args.image_height, args.image_width],
            },
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "software": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "repository": git_facts(repo_root),
            },
        }
        exit_code = 1
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("wrote %s and %s", report_path, log_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
