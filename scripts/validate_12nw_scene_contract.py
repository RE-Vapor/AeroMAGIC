#!/usr/bin/env python3
"""Validate the 12-NW-6C-5 transform with the real CUDA rasterizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import trimesh

from macarons.testers.magician_planning import setup_test_camera
from macarons.utility.macarons_utils import Settings, load_params, load_scene
from macarons.utility.scene_transform import (
    resolve_scene_mesh_transform,
    transform_scene_vertices,
)


SCENE = "12-NW-6C-5"


def _bounds(vertices) -> list[list[float]]:
    if hasattr(vertices, "detach"):
        vertices = vertices.detach().cpu().numpy()
    return [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()]


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/test/test_da3_12-nw-6c-5_real_mesh_config.json",
    )
    parser.add_argument("--output", default="results/12-nw-6c-5_phase4/scene_gate.json")
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    root = ROOT
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    scene_dir = root / config["dataset_path"] / SCENE
    mesh_path = next(scene_dir.glob("*.obj"))
    settings_dict = json.loads((scene_dir / "settings.json").read_text(encoding="utf-8"))
    params = load_params(root / "configs/macarons" / config["params_name"])
    device = torch.device(f"cuda:{config['numGPU']}")
    torch.cuda.set_device(device)
    settings = Settings(settings_dict, device, params.scene_scale_factor)
    transform = resolve_scene_mesh_transform(config, SCENE)

    mesh = load_scene(
        str(mesh_path),
        params.scene_scale_factor,
        device,
        mesh_transform=transform,
    )
    collision_mesh = trimesh.load(mesh_path, process=False, force="mesh")
    raw_vertices = np.asarray(collision_mesh.vertices).copy()
    pre_runtime_vertices = transform_scene_vertices(
        raw_vertices,
        transform,
        scene_scale_factor=1.0,
    )
    collision_mesh.vertices = transform_scene_vertices(
        raw_vertices,
        transform,
        scene_scale_factor=params.scene_scale_factor,
    )
    mesh_vertices = mesh.verts_list()[0]
    collision_vertices = np.asarray(collision_mesh.vertices)
    render_bounds = np.asarray(_bounds(mesh_vertices))
    collision_bounds = np.asarray(_bounds(collision_vertices))

    capture_root = output_path.parent / "scene_gate_capture" / f"start_{args.start_index}"
    frames_dir = capture_root / "frames"
    (capture_root / "imgs").mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    start_positions = settings.camera.start_positions
    if args.start_index < 0 or args.start_index >= len(start_positions):
        parser.error(f"start-index must be in [0, {len(start_positions) - 1}]")
    camera = setup_test_camera(
        params,
        mesh,
        collision_mesh.ray,
        start_positions[args.start_index],
        settings,
        None,
        device,
        str(frames_dir),
    )
    frame = torch.load(frames_dir / "0.pt", map_location=device, weights_only=False)
    depth = frame["zbuf"]
    mask = frame["mask"]
    rgb = frame["rgb"]
    finite_depth = torch.isfinite(depth) & mask
    settings_min = settings.scene.x_min
    settings_max = settings.scene.x_max
    inside = ((mesh_vertices >= settings_min) & (mesh_vertices <= settings_max)).all(dim=1)
    sha256 = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    settings_sha256 = hashlib.sha256((scene_dir / "settings.json").read_bytes()).hexdigest()
    rotation = np.eye(3)[list(transform["axis_order"])]
    rotation *= np.asarray(transform["axis_signs"])[:, None]

    result = {
        "schema_version": 1,
        "scene": SCENE,
        "mesh": {
            "path": str(mesh_path.relative_to(root)),
            "sha256": sha256,
            "settings_sha256": settings_sha256,
            "vertex_count": len(mesh_vertices),
            "face_count": int(mesh.faces_list()[0].shape[0]),
            "transform": transform,
            "axis_transform_determinant": float(np.linalg.det(rotation)),
            "raw_bounds": _bounds(raw_vertices),
            "raw_extents": np.ptp(raw_vertices, axis=0).tolist(),
            "transformed_pre_runtime_bounds": _bounds(pre_runtime_vertices),
            "transformed_pre_runtime_extents": np.ptp(
                pre_runtime_vertices, axis=0
            ).tolist(),
            "renderer_bounds_scene_units": render_bounds.tolist(),
            "collision_bounds_scene_units": collision_bounds.tolist(),
            "renderer_collision_max_abs_bound_difference": float(
                np.max(np.abs(render_bounds - collision_bounds))
            ),
            "fraction_inside_coverage_reference_bounds": float(inside.float().mean()),
        },
        "coverage_reference": {
            "bounds_scene_units": [
                settings_min.detach().cpu().tolist(),
                settings_max.detach().cpu().tolist(),
            ],
            "same_transformed_mesh": True,
        },
        "metric_scale": {
            "scene_units_per_meter": config["da3_scene_units_per_meter"][SCENE],
            "primary_evidence": (
                "Tile indices +304/+147 align with the measured 150-unit horizontal "
                "grid intervals [304*150,305*150] and [147*150,148*150]."
            ),
            "independent_cross_check": (
                "After the explicit 0.1 preprocessing and existing x10 runtime scale, "
                "all mesh extents fit the independently supplied settings.json envelope."
            ),
            "eiffel_scale_reused": False,
        },
        "camera": {
            "start_index": args.start_index,
            "start_grid_index": start_positions[args.start_index].detach().cpu().tolist(),
            "position_scene_units": camera.X_cam_history[0].detach().cpu().tolist(),
            "orientation_degrees": camera.V_cam_history[0].detach().cpu().tolist(),
            "bounds_scene_units": [
                settings.camera.x_min.detach().cpu().tolist(),
                settings.camera.x_max.detach().cpu().tolist(),
            ],
        },
        "rasterizer": {
            "backend": "PyTorch3D CUDA MeshRasterizer",
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "rgb_all_finite": bool(torch.isfinite(rgb).all()),
            "mask_pixel_count": int(mask.sum()),
            "mask_fraction": float(mask.float().mean()),
            "finite_depth_pixel_count": int(finite_depth.sum()),
            "depth_min_scene_units": float(depth[finite_depth].min()),
            "depth_max_scene_units": float(depth[finite_depth].max()),
            "depth_mean_scene_units": float(depth[finite_depth].mean()),
        },
        "gate_passed": bool(
            mask.any()
            and finite_depth.any()
            and torch.isfinite(rgb).all()
            and inside.all()
            and np.max(np.abs(render_bounds - collision_bounds)) < 1e-4
            and abs(np.linalg.det(rotation) - 1.0) < 1e-8
        ),
    }
    _write_json(output_path, result)
    print(output_path)
    if not result["gate_passed"]:
        raise SystemExit("scene contract gate failed")


if __name__ == "__main__":
    main()
