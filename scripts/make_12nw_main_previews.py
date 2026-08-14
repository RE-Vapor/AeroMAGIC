#!/usr/bin/env python3
"""Render five-start and representative-start previews for a main run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lmdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from macarons.utility.scene_transform import (
    resolve_scene_mesh_transform,
    transform_scene_vertices,
)


COLORS = ["#68ddff", "#ffcc66", "#ff6b8a", "#8ee28e", "#bd9cff"]


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sample(values: np.ndarray, limit: int) -> np.ndarray:
    return values[:: max(1, len(values) // limit)]


def _mesh_vertices(config: dict, scene: str) -> np.ndarray:
    scene_dir = ROOT / config["dataset_path"] / scene
    mesh_path = next(scene_dir.glob("*.obj"))
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    params = _read(ROOT / "configs/macarons" / config["params_name"])
    transform = resolve_scene_mesh_transform(config, scene)
    return np.asarray(
        transform_scene_vertices(
            np.asarray(mesh.vertices),
            transform,
            scene_scale_factor=params["_data"]["scene_scale_factor"],
        )
    )


def _metrics(run_root: Path) -> list[dict]:
    values = [_read(path) for path in sorted((run_root / "metrics").glob("*.online.json"))]
    return sorted(values, key=lambda item: item["start_index"])


def _load_trajectory(config: dict, scene: str, start_index: int) -> dict:
    lmdb_name = config.get("lmdb_dir_name") or config["scone_lmdb_dir_name"]
    env = lmdb.open(
        str(ROOT / "results/scene_exploration" / lmdb_name),
        readonly=True,
        lock=False,
    )
    with env.begin() as transaction:
        payload = transaction.get(f"{scene}/{start_index}".encode("utf-8"))
    env.close()
    if payload is None:
        raise KeyError(f"missing LMDB trajectory {scene}/{start_index}")
    return pickle.loads(payload)


def _five_start_preview(
    run_root: Path,
    config: dict,
    scene: str,
    metrics: list[dict],
    mesh_vertices: np.ndarray,
    output: Path,
) -> None:
    plt.style.use("dark_background")
    fig, (ax_path, ax_cov) = plt.subplots(1, 2, figsize=(19, 8.5), facecolor="#0d1015")
    mesh_sample = _sample(mesh_vertices, 30000)
    ax_path.scatter(
        mesh_sample[:, 0], mesh_sample[:, 2], s=0.25, c="#8c96a8", alpha=0.18
    )
    for index, metric in enumerate(metrics):
        positions = np.asarray(metric["trajectory"]["positions"])
        coverage = 100 * np.asarray([item["normalized"] for item in metric["coverage"]])
        label = f"start {metric['start_index']} — {coverage[-1]:.2f}%"
        color = COLORS[index % len(COLORS)]
        ax_path.plot(positions[:, 0], positions[:, 2], lw=2.0, color=color, label=label)
        ax_path.scatter(positions[0, 0], positions[0, 2], s=45, color=color)
        ax_cov.plot(np.arange(1, len(coverage) + 1), coverage, lw=2.2, color=color, label=label)
    run_id = metrics[0]["run"]["run_id"]
    source = metrics[0]["frames"][0]["source"]
    planner = metrics[0]["planner"].upper()
    fig.suptitle(
        f"{scene} — {planner}/{source} — five-start production preview",
        fontsize=22,
    )
    ax_path.set_title("GT mesh reference + camera trajectories (top view)")
    ax_path.set_xlabel("x (scene units = meters)")
    ax_path.set_ylabel("z (scene units = meters)")
    ax_path.axis("equal")
    ax_path.grid(alpha=0.15)
    ax_path.legend(fontsize=10)
    ax_cov.set_title("Normalized coverage by observation")
    ax_cov.set_xlabel("observation")
    ax_cov.set_ylabel("coverage (%)")
    ax_cov.set_xlim(1, max(len(metric["coverage"]) for metric in metrics))
    ax_cov.set_ylim(bottom=0)
    ax_cov.grid(alpha=0.2)
    ax_cov.legend(fontsize=10)
    fig.text(
        0.5,
        0.015,
        f"{run_id} • starts 0–4 • seeds 8/9 • beam 10×10 • mapping 2× • collision=true",
        ha="center",
        color="#aeb8c8",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _representative_preview(
    config: dict,
    scene: str,
    metric: dict,
    mesh_vertices: np.ndarray,
    trajectory: dict,
    output: Path,
) -> None:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(23, 14), facecolor="#0d1015")
    grid = fig.add_gridspec(2, 10, height_ratios=[2.4, 4.8], hspace=0.35, wspace=0.4)
    capture_root = Path(metric["capture_dir"]).parent
    observations = len(metric["coverage"])
    selected = sorted({0, observations // 4, observations // 2, 3 * observations // 4, observations - 1})
    for column, frame_id in enumerate(selected):
        axis = fig.add_subplot(grid[0, column * 2 : column * 2 + 2])
        axis.imshow(plt.imread(capture_root / "imgs" / f"{frame_id}.png"))
        axis.set_title(f"observation {frame_id + 1}")
        axis.axis("off")

    points = np.asarray(trajectory["points"])
    saved_colors = trajectory.get("points_color")
    if saved_colors is None:
        height = points[:, 1]
        height_range = np.ptp(height)
        normalized_height = (
            (height - height.min()) / height_range
            if height_range > 0
            else np.zeros_like(height)
        )
        point_colors = plt.colormaps["viridis"](normalized_height)[:, :3]
    else:
        point_colors = np.clip(np.asarray(saved_colors), 0, 1)
    stride = max(1, len(points) // 25000)
    points = points[::stride]
    point_colors = point_colors[::stride]
    positions = np.asarray(metric["trajectory"]["positions"])
    mesh_sample = _sample(mesh_vertices, 30000)

    ax_top = fig.add_subplot(grid[1, 0:4])
    ax_top.scatter(mesh_sample[:, 0], mesh_sample[:, 2], s=0.2, c="#8c96a8", alpha=0.15)
    ax_top.scatter(points[:, 0], points[:, 2], s=0.5, c=point_colors, alpha=0.45)
    ax_top.plot(positions[:, 0], positions[:, 2], color="#ff4238", lw=2.2)
    ax_top.scatter(*positions[0, [0, 2]], s=75, c="#24d85b", label="start")
    ax_top.scatter(*positions[-1, [0, 2]], s=75, c="#ffd21a", label="end")
    ax_top.set_title("GT reference + reconstructed cloud + path")
    ax_top.axis("equal")
    ax_top.grid(alpha=0.15)
    ax_top.legend()

    ax_3d = fig.add_subplot(grid[1, 4:7], projection="3d")
    ax_3d.scatter(points[:, 0], points[:, 2], points[:, 1], s=0.4, c=point_colors, alpha=0.4)
    ax_3d.plot(positions[:, 0], positions[:, 2], positions[:, 1], color="#ff4238", lw=2.0)
    ax_3d.view_init(elev=25, azim=-55)
    ax_3d.set_title("Reconstructed cloud (oblique)")
    ax_3d.set_axis_off()

    coverage = 100 * np.asarray([item["normalized"] for item in metric["coverage"]])
    ax_cov = fig.add_subplot(grid[1, 7:10])
    ax_cov.plot(np.arange(1, len(coverage) + 1), coverage, color="#68ddff", lw=2.5)
    ax_cov.fill_between(np.arange(1, len(coverage) + 1), coverage, alpha=0.12, color="#68ddff")
    ax_cov.set_title("Normalized coverage")
    ax_cov.set_xlabel("observation")
    ax_cov.set_ylabel("coverage (%)")
    ax_cov.set_ylim(bottom=0)
    ax_cov.grid(alpha=0.2)

    source = metric["frames"][0]["source"]
    planner = metric["planner"].upper()
    fig.suptitle(
        f"{scene} — {planner}/{source} — representative start {metric['start_index']}",
        fontsize=24,
    )
    fig.text(
        0.5,
        0.015,
        (
            f"{len(coverage)} observations • {metric['trajectory']['final_point_count']:,} points • "
            f"final normalized coverage {coverage[-1]:.2f}% • "
            f"path {metric['trajectory']['path_length_meters']:.2f} m"
        ),
        ha="center",
        color="#aeb8c8",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene")
    parser.add_argument("--representative-start", type=int, default=0)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = _read(run_root / "config.json")
    configured_scenes = config.get("test_scenes", [])
    if len(configured_scenes) != 1:
        raise ValueError("preview requires exactly one configured scene")
    scene = args.scene or configured_scenes[0]
    if configured_scenes != [scene]:
        raise ValueError(f"configured scene {configured_scenes[0]} does not match {scene}")
    metrics = _metrics(run_root)
    if [metric["start_index"] for metric in metrics] != [0, 1, 2, 3, 4]:
        raise ValueError("preview requires completed starts 0 through 4")
    mesh_vertices = _mesh_vertices(config, scene)
    run_id = metrics[0]["run"]["run_id"]
    _five_start_preview(
        run_root,
        config,
        scene,
        metrics,
        mesh_vertices,
        output_dir / f"{run_id}_five_trajectory_preview.png",
    )
    representative = next(
        metric for metric in metrics if metric["start_index"] == args.representative_start
    )
    trajectory = _load_trajectory(config, scene, args.representative_start)
    _representative_preview(
        config,
        scene,
        representative,
        mesh_vertices,
        trajectory,
        output_dir / f"{run_id}_start_{args.representative_start}_representative.png",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
