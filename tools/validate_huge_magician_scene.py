#!/usr/bin/env python3
"""Exercise a converted HUGE scene through MAGICIAN's unmodified loader/camera code."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import time
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch3d.renderer import FoVPerspectiveCameras

from macarons.testers.magician_planning import compute_magician_trajectory
from macarons.utility.macarons_utils import (
    Camera,
    Settings,
    get_rgb_renderer,
    load_scene,
)


def json_safe_error(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def tensor_is_finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-height", type=int, default=128)
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--zfar", type=float, default=750.0)
    parser.add_argument("--report-path", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    scene_dir = args.scene_dir.resolve()
    manifest = json.loads((scene_dir / "conversion_manifest.json").read_text(encoding="utf-8"))
    settings_dict = json.loads((scene_dir / "settings.json").read_text(encoding="utf-8"))
    occupied = torch.load(scene_dir / "occupied_pose.pt", map_location="cpu")
    obj_paths = sorted(scene_dir.glob("*.obj"))
    if len(obj_paths) != 1:
        raise RuntimeError(f"expected exactly one root OBJ, found {len(obj_paths)}")
    obj_path = obj_paths[0]

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {args.device}, but CUDA is unavailable")
    device = torch.device(args.device)
    cuda_index: int | None = None
    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(cuda_index)
        torch.cuda.reset_peak_memory_stats(cuda_index)

    report: dict[str, Any] = {
        "scene_dir": str(scene_dir),
        "device": str(device),
        "scene_scale_factor": 1.0,
        "geometry_loader": {},
        "camera": {},
        "depth_mask": {},
        "rgb": {},
        "occupancy": {},
        "candidate_poses": {},
        "known_risks": {},
        "compatibility": {},
    }

    mesh = load_scene(str(obj_path), scene_scale_factor=1.0, device=device)
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    report["geometry_loader"] = {
        "passed": bool(len(verts) and len(faces) and tensor_is_finite(verts)),
        "vertices": len(verts),
        "faces": len(faces),
        "verts_finite": tensor_is_finite(verts),
        "faces_finite": tensor_is_finite(faces.float()),
        "textures_type": type(mesh.textures).__name__ if mesh.textures is not None else None,
        "aabb_min": verts.min(dim=0).values.detach().cpu().tolist(),
        "aabb_max": verts.max(dim=0).values.detach().cpu().tolist(),
    }

    settings = Settings(settings_dict, device=device, scene_scale_factor=1.0)
    base_camera = FoVPerspectiveCameras(device=device, zfar=args.zfar)
    renderer = get_rgb_renderer(
        image_height=args.image_height,
        image_width=args.image_width,
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
        zfar=args.zfar,
        renderer=renderer,
        device=device,
        contrast_factor=settings.camera.contrast_factor,
        occupied_pose_data={key: value.to(device) for key, value in occupied.items()},
    )

    start = settings.camera.start_positions[0]
    camera.initialize_camera(start)
    pose, _ = camera.get_pose_from_idx(start)
    center, angles, fov = camera.get_camera_parameters_from_pose(pose)
    recovered_center = fov.get_camera_center()
    center_error = float(torch.max(torch.abs(center - recovered_center)).item())
    elevation = float(angles[0, 0].item())
    azimuth = float(angles[0, 1].item())
    elev_rad = np.deg2rad(elevation)
    azim_rad = np.deg2rad(azimuth)
    expected_forward = torch.tensor(
        [
            np.cos(elev_rad) * np.sin(azim_rad),
            np.sin(elev_rad),
            np.cos(elev_rad) * np.cos(azim_rad),
        ],
        device=device,
        dtype=fov.R.dtype,
    )
    pytorch3d_forward = fov.R[0, :, 2]
    forward_error = float(torch.max(torch.abs(expected_forward - pytorch3d_forward)).item())
    report["camera"] = {
        "start_index": start.detach().cpu().tolist(),
        "center": center[0].detach().cpu().tolist(),
        "angles_elevation_azimuth_deg": angles[0].detach().cpu().tolist(),
        "center_round_trip_max_abs_error": center_error,
        "forward_convention_max_abs_error": forward_error,
        "R_shape": list(fov.R.shape),
        "T_shape": list(fov.T.shape),
        "R_finite": tensor_is_finite(fov.R),
        "T_finite": tensor_is_finite(fov.T),
        "znear": float(fov.znear.item()),
        "zfar": float(fov.zfar.item()),
    }

    all_starts = []
    for start_number, start_index in enumerate(settings.camera.start_positions):
        start_pose, _ = camera.get_pose_from_idx(start_index)
        start_center, start_angles, start_fov = camera.get_camera_parameters_from_pose(start_pose)
        start_fragments = renderer.rasterizer(mesh, cameras=start_fov)
        start_mask = start_fragments.pix_to_face >= 0
        start_visible_depth = start_fragments.zbuf[start_mask]
        in_bounds = bool(
            torch.all(start_center[0] >= settings.camera.x_min).item()
            and torch.all(start_center[0] <= settings.camera.x_max).item()
        )
        target = torch.tensor(
            manifest["camera"]["target_magic"], device=device, dtype=start_center.dtype
        )
        to_target = target - start_center[0]
        to_target /= torch.linalg.norm(to_target)
        start_elevation = float(start_angles[0, 0].item())
        start_azimuth = float(start_angles[0, 1].item())
        elevation_rad = np.deg2rad(start_elevation)
        azimuth_rad = np.deg2rad(start_azimuth)
        start_forward = torch.tensor(
            [
                np.cos(elevation_rad) * np.sin(azimuth_rad),
                np.sin(elevation_rad),
                np.cos(elevation_rad) * np.cos(azimuth_rad),
            ],
            device=device,
            dtype=start_center.dtype,
        )
        all_starts.append(
            {
                "number": start_number,
                "index": start_index.detach().cpu().tolist(),
                "center": start_center[0].detach().cpu().tolist(),
                "in_envelope": in_bounds,
                "occupied": bool(camera.check_if_pose_is_occupied(start_index, input_type="idx")),
                "forward_dot_target": float(torch.dot(start_forward, to_target).item()),
                "visible_pixels": int(start_mask.sum().item()),
                "visible_depth_finite": (
                    tensor_is_finite(start_visible_depth) if len(start_visible_depth) else False
                ),
                "visible_depth_min": (
                    float(start_visible_depth.min().item()) if len(start_visible_depth) else None
                ),
                "visible_depth_max": (
                    float(start_visible_depth.max().item()) if len(start_visible_depth) else None
                ),
            }
        )
    report["camera"]["all_starts"] = all_starts
    report["camera"]["all_starts_passed"] = all(
        entry["in_envelope"]
        and not entry["occupied"]
        and entry["forward_dot_target"] > 0.9
        and entry["visible_pixels"] > 0
        and entry["visible_depth_finite"]
        and entry["visible_depth_max"] <= args.zfar
        for entry in all_starts
    )

    fragments = renderer.rasterizer(mesh, cameras=fov)
    depth = fragments.zbuf
    mask = fragments.pix_to_face >= 0
    visible_depth = depth[mask]
    report["depth_mask"] = {
        "passed": bool(
            mask.any().item() and len(visible_depth) and tensor_is_finite(visible_depth)
        ),
        "depth_shape": list(depth.shape),
        "mask_shape": list(mask.shape),
        "visible_pixels": int(mask.sum().item()),
        "visible_depth_finite": tensor_is_finite(visible_depth) if len(visible_depth) else False,
        "visible_depth_min": float(visible_depth.min().item()) if len(visible_depth) else None,
        "visible_depth_max": float(visible_depth.max().item()) if len(visible_depth) else None,
    }

    rgb_error = None
    try:
        rgb, captured_depth = camera.capture_image(mesh, save_frame=False)
        report["rgb"] = {
            "passed": bool(
                tensor_is_finite(rgb)
                and tensor_is_finite(captured_depth[captured_depth >= 0])
            ),
            "shape": list(rgb.shape),
            "depth_shape": list(captured_depth.shape),
            "finite": tensor_is_finite(rgb),
        }
    except Exception as exc:  # preserve the exact upstream failure as evidence
        rgb_error = json_safe_error(exc)
        report["rgb"] = {"passed": False, "error": rgb_error}

    current_occupied = camera.check_if_pose_is_occupied(start, input_type="idx")
    x_idx = occupied["X_idx"]
    occupied_values = occupied["occupied"]
    report["occupancy"] = {
        "dtype_X_idx": str(x_idx.dtype),
        "dtype_occupied": str(occupied_values.dtype),
        "X_idx_shape": list(x_idx.shape),
        "occupied_shape": list(occupied_values.shape),
        "occupied_count": int(occupied_values.sum().item()),
        "free_count": int((~occupied_values).sum().item()),
        "start_is_occupied": bool(current_occupied),
    }

    neighbors = camera.get_neighboring_poses(start)
    valid_neighbors = camera.get_valid_neighbors(neighbors, mesh)
    report["candidate_poses"] = {
        "neighbor_count": len(neighbors),
        "reported_valid_count": len(valid_neighbors),
        "nonzero": bool(len(valid_neighbors)),
    }

    load_source = inspect.getsource(load_scene)
    neighbor_source = inspect.getsource(Camera.get_valid_neighbors)
    trajectory_source = inspect.getsource(compute_magician_trajectory)
    trajectory_tree = ast.parse(textwrap.dedent(trajectory_source))
    trajectory_loaded_names = {
        node.id
        for node in ast.walk(trajectory_tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    report["known_risks"] = {
        "multi_texture_branch_selected": manifest["materials"]["diffuse_texture_count"] > 1,
        "multi_texture_bare_torch_tensor_call_present": "torch.tensor()" in load_source,
        "neighbor_validity_bypass_present": "if True:" in neighbor_source
        and "check_if_pose_is_valid" in neighbor_source,
        "use_perfect_depth_map_used_in_trajectory_body": (
            "use_perfect_depth_map" in trajectory_loaded_names
        ),
        "compute_collision_used_in_trajectory_body": "compute_collision" in trajectory_loaded_names,
    }

    geometry_ok = report["geometry_loader"]["passed"]
    depth_ok = report["depth_mask"]["passed"]
    candidates_ok = report["candidate_poses"]["nonzero"]
    rgb_expected = bool(manifest["materials"]["rgb_mesh_compatible"])
    rgb_ok = report["rgb"]["passed"]
    all_starts_ok = report["camera"]["all_starts_passed"]
    fatal = not (geometry_ok and depth_ok and candidates_ok and all_starts_ok) or (
        rgb_expected and not rgb_ok
    )
    if geometry_ok and depth_ok and candidates_ok and rgb_ok:
        status = "PASS"
    elif geometry_ok and depth_ok and candidates_ok and not rgb_expected and not rgb_ok:
        status = "PARTIAL"
    else:
        status = "BLOCKED"
    report["compatibility"] = {
        "status": status,
        "file_format": "PASS" if geometry_ok else "FAIL",
        "magician_geometry_loader": "PASS" if geometry_ok else "FAIL",
        "magician_depth_and_mask_rasterizer": "PASS" if depth_ok else "FAIL",
        "magician_rgb_harness": "PASS" if rgb_ok else "FAIL",
        "bounded_planning_loop": "NOT_RUN" if not rgb_ok else "NOT_RUN_BY_THIS_VALIDATOR",
        "no_ground_truth_leakage_online_planning": "NOT_VALIDATED",
        "reason": rgb_error,
    }
    report["elapsed_seconds"] = time.perf_counter() - started
    if device.type == "cuda":
        assert cuda_index is not None
        report["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated(cuda_index))

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
