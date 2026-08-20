#!/usr/bin/env python3
"""Render one evidence-rich preview from a completed PIONEER trajectory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Mapping, Optional, Sequence

import lmdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


FACE_NAMES = ("front", "back", "left", "right", "up", "down")
DEFAULT_MAX_BUNDLE_ROWS = 6
BUNDLE_TRANSACTION_VERSION = "pioneer-bundle-commit-v1"


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(paths: Sequence[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_lmdb(lmdb_path: Path, key: str) -> Mapping[str, Any]:
    if not lmdb_path.is_dir():
        raise FileNotFoundError(f"LMDB directory does not exist: {lmdb_path}")
    environment = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=1,
    )
    try:
        with environment.begin() as transaction:
            payload = transaction.get(key.encode("utf-8"))
    finally:
        environment.close()
    if payload is None:
        raise KeyError(f"LMDB has no trajectory {key}")
    value = pickle.loads(payload)
    if not isinstance(value, Mapping):
        raise ValueError(f"LMDB trajectory {key} must be a mapping")
    return value


def _as_xyz(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite Nx3 array")
    return array


def _point_colors(points: np.ndarray, value: Any) -> np.ndarray:
    if value is not None:
        colors = np.asarray(value, dtype=np.float64)
        if colors.shape == points.shape and np.isfinite(colors).all():
            return np.clip(colors, 0.0, 1.0)
    height = points[:, 1]
    span = float(np.ptp(height))
    normalized = (height - height.min()) / span if span else np.zeros_like(height)
    return plt.colormaps["viridis"](normalized)[:, :3]


def _sample_indices(length: int, limit: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.int64)
    if length <= limit:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, num=limit, dtype=np.int64)


def _resolve_bundle_images(
    metrics: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], Path, list[list[Path]], list[Path]]:
    pioneer = metrics.get("pioneer_observation")
    if not isinstance(pioneer, Mapping):
        raise ValueError("metrics.pioneer_observation is required")
    bundles = pioneer.get("bundles")
    if not isinstance(bundles, list) or not bundles:
        raise ValueError("metrics must contain at least one PIONEER bundle")
    if int(pioneer.get("bundle_count", -1)) != len(bundles):
        raise ValueError("bundle_count does not match bundle telemetry")
    if int(pioneer.get("real_face_render_count", -1)) != 6 * len(bundles):
        raise ValueError("real_face_render_count must equal six times bundle_count")

    capture_dir = Path(str(metrics.get("capture_dir", ""))).expanduser().resolve()
    if capture_dir.name != "frames":
        raise ValueError("metrics.capture_dir must point to the trajectory frames directory")
    images_root = capture_dir.parent / "imgs"
    commit_root = capture_dir.parent / ".pioneer_bundle_commits"
    run = metrics.get("run") if isinstance(metrics.get("run"), Mapping) else {}
    requires_transaction = str(run.get("depth_source", "GT")).upper() != "GT"
    image_rows: list[list[Path]] = []
    commit_markers: list[Path] = []
    seen_ids: set[int] = set()
    for bundle in bundles:
        if not isinstance(bundle, Mapping):
            raise ValueError("each PIONEER bundle must be an object")
        bundle_id = int(bundle.get("bundle_id", -1))
        if bundle_id < 0 or bundle_id in seen_ids:
            raise ValueError("bundle_id values must be unique non-negative integers")
        seen_ids.add(bundle_id)
        if int(bundle.get("face_count", -1)) != 6:
            raise ValueError(f"bundle {bundle_id} must contain six faces")
        if tuple(bundle.get("face_names") or ()) != FACE_NAMES:
            raise ValueError(f"bundle {bundle_id} face order is not canonical")
        row = [images_root / f"{bundle_id:06d}" / f"{face}.png" for face in FACE_NAMES]
        missing = [str(path) for path in row if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing PIONEER face images: " + ", ".join(missing))
        transaction_version = bundle.get("artifact_transaction_version")
        if requires_transaction or transaction_version is not None:
            if (
                transaction_version != BUNDLE_TRANSACTION_VERSION
                or bundle.get("artifact_committed") is not True
            ):
                raise ValueError(
                    f"bundle {bundle_id} lacks a committed artifact transaction"
                )
            marker_path = commit_root / f"{bundle_id:06d}.json"
            if not marker_path.is_file():
                raise FileNotFoundError(
                    f"missing PIONEER bundle commit marker: {marker_path}"
                )
            marker = _read_json(marker_path)
            expected_frame_names = {"bundle.pt", *(f"{name}.pt" for name in FACE_NAMES)}
            expected_image_names = {f"{name}.png" for name in FACE_NAMES}
            frame_hashes = marker.get("frame_sha256")
            image_hashes = marker.get("image_sha256")
            if (
                marker.get("transaction_version") != BUNDLE_TRANSACTION_VERSION
                or int(marker.get("bundle_id", -1)) != bundle_id
                or tuple(marker.get("face_names") or ()) != FACE_NAMES
                or marker.get("png_committed") is not True
                or not isinstance(frame_hashes, Mapping)
                or set(frame_hashes) != expected_frame_names
                or not isinstance(image_hashes, Mapping)
                or set(image_hashes) != expected_image_names
            ):
                raise ValueError(f"invalid PIONEER bundle commit marker: {marker_path}")
            for filename, expected_sha in frame_hashes.items():
                artifact = capture_dir / f"{bundle_id:06d}" / filename
                if not artifact.is_file() or _sha256(artifact) != expected_sha:
                    raise ValueError(f"bundle frame hash mismatch: {artifact}")
            for filename, expected_sha in image_hashes.items():
                artifact = images_root / f"{bundle_id:06d}" / filename
                if not artifact.is_file() or _sha256(artifact) != expected_sha:
                    raise ValueError(f"bundle image hash mismatch: {artifact}")
            commit_markers.append(marker_path)
        image_rows.append(row)
    return bundles, images_root, image_rows, commit_markers


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def generate_preview(
    *,
    metrics_path: Path,
    lmdb_path: Path,
    output_path: Path,
    lmdb_key: Optional[str] = None,
    max_points: int = 25000,
    max_bundle_rows: int = DEFAULT_MAX_BUNDLE_ROWS,
    dpi: int = 120,
) -> Mapping[str, Any]:
    """Validate completed artifacts and render a single comprehensive PNG."""

    metrics_path = metrics_path.expanduser().resolve()
    lmdb_path = lmdb_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if output_path.suffix.lower() != ".png":
        raise ValueError("preview output must use a .png suffix")
    if max_points <= 0 or max_bundle_rows <= 0 or dpi <= 0:
        raise ValueError("max_points, max_bundle_rows, and dpi must be positive")

    metrics = _read_json(metrics_path)
    if str(metrics.get("planner", "")).lower() != "pioneer":
        raise ValueError("preview requires planner=pioneer metrics")
    run = metrics.get("run")
    if not isinstance(run, Mapping) or run.get("planning_observation_mode") != "cubemap6":
        raise ValueError("preview requires planning_observation_mode=cubemap6")
    scene = str(metrics.get("scene", ""))
    start_index = int(metrics.get("start_index", -1))
    if not scene or start_index < 0:
        raise ValueError("metrics scene and start_index are required")

    bundles, images_root, image_rows, commit_markers = _resolve_bundle_images(metrics)
    capture_dir = Path(str(metrics["capture_dir"])).expanduser().resolve()
    if len(bundles) > 1 and max_bundle_rows < 2:
        raise ValueError(
            "max_bundle_rows must be at least two for a multi-bundle run"
        )
    display_indices = _sample_indices(len(bundles), max_bundle_rows)
    display_index_set = set(int(index) for index in display_indices)
    displayed_bundles = [bundles[int(index)] for index in display_indices]
    key = lmdb_key or f"{scene}/{start_index}"
    trajectory = _load_lmdb(lmdb_path, key)
    points = _as_xyz(trajectory.get("points"), "LMDB points")
    if not len(points):
        raise ValueError("LMDB trajectory contains no reconstructed points")
    positions = _as_xyz(metrics.get("trajectory", {}).get("positions"), "trajectory positions")
    if len(positions) != len(bundles):
        raise ValueError("trajectory observation count must equal PIONEER bundle count")
    colors = _point_colors(points, trajectory.get("points_color"))

    coverage_rows = metrics.get("coverage")
    if not isinstance(coverage_rows, list) or len(coverage_rows) != len(bundles):
        raise ValueError("coverage length must equal PIONEER bundle count")
    coverage = np.asarray([float(row["normalized"]) for row in coverage_rows])
    if not np.isfinite(coverage).all():
        raise ValueError("coverage must be finite")

    face_size = int(run.get("pioneer_face_size") or 0)
    loaded_images: list[list[np.ndarray]] = []
    for row_index, row in enumerate(image_rows):
        loaded_row = []
        for path in row:
            image = plt.imread(path)
            if image.ndim != 3 or image.shape[2] not in (3, 4):
                raise ValueError(f"face image must be RGB/RGBA: {path}")
            if face_size and image.shape[:2] != (face_size, face_size):
                raise ValueError(f"face image size disagrees with metrics: {path}")
            if row_index in display_index_set:
                loaded_row.append(image[:, :, :3])
        if row_index in display_index_set:
            loaded_images.append(loaded_row)

    plt.style.use("dark_background")
    figure = plt.figure(
        figsize=(24, 2.65 * len(displayed_bundles) + 7.0),
        facecolor="#0d1015",
    )
    grid = figure.add_gridspec(
        len(displayed_bundles) + 2,
        12,
        height_ratios=[2.4] * len(displayed_bundles) + [3.0, 3.0],
        hspace=0.3,
        wspace=0.12,
    )
    for row_index, (bundle_index, bundle, images) in enumerate(
        zip(display_indices, displayed_bundles, loaded_images)
    ):
        point_counts = list(bundle.get("face_point_counts") or [None] * 6)
        for face_index, (face, image) in enumerate(zip(FACE_NAMES, images)):
            axis = figure.add_subplot(
                grid[row_index, face_index * 2 : face_index * 2 + 2]
            )
            axis.imshow(image)
            count = point_counts[face_index] if face_index < len(point_counts) else None
            suffix = f" · {int(count):,} pts" if count is not None else ""
            axis.set_title(f"{face}{suffix}", fontsize=10)
            axis.axis("off")
            if face_index == 0:
                axis.text(
                    -0.08,
                    0.5,
                    f"obs {int(bundle_index) + 1}\nbundle {int(bundle['bundle_id']):06d}",
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    color="#aeb8c8",
                    fontsize=10,
                )

    sample = _sample_indices(len(points), max_points)
    sampled_points = points[sample]
    sampled_colors = colors[sample]

    top_axis = figure.add_subplot(grid[len(displayed_bundles) :, 0:4])
    top_axis.scatter(
        sampled_points[:, 0],
        sampled_points[:, 2],
        s=0.7,
        c=sampled_colors,
        alpha=0.55,
        rasterized=True,
    )
    top_axis.plot(positions[:, 0], positions[:, 2], "-o", color="#ff554f", lw=2.3)
    top_axis.scatter(*positions[0, [0, 2]], s=80, c="#24d85b", label="start")
    top_axis.scatter(*positions[-1, [0, 2]], s=80, c="#ffd21a", label="end")
    top_axis.set_title("Reconstructed cloud + camera path (top view)")
    top_axis.set_xlabel("world x")
    top_axis.set_ylabel("world z")
    top_axis.axis("equal")
    top_axis.grid(alpha=0.15)
    top_axis.legend()

    spatial_axis = figure.add_subplot(
        grid[len(displayed_bundles) :, 4:8], projection="3d"
    )
    spatial_axis.scatter(
        sampled_points[:, 0],
        sampled_points[:, 2],
        sampled_points[:, 1],
        s=0.6,
        c=sampled_colors,
        alpha=0.5,
        rasterized=True,
    )
    spatial_axis.plot(
        positions[:, 0], positions[:, 2], positions[:, 1], color="#ff554f", lw=2.3
    )
    spatial_axis.view_init(elev=25, azim=-55)
    spatial_axis.set_title("Reconstructed cloud (oblique)")
    spatial_axis.set_axis_off()

    coverage_axis = figure.add_subplot(grid[len(displayed_bundles) :, 8:12])
    observation_ids = np.arange(1, len(coverage) + 1)
    coverage_percent = 100.0 * coverage
    coverage_axis.plot(observation_ids, coverage_percent, "-o", color="#68ddff", lw=2.6)
    coverage_axis.fill_between(
        observation_ids, coverage_percent, color="#68ddff", alpha=0.12
    )
    coverage_axis.set_title("Normalized coverage")
    coverage_axis.set_xlabel("observation bundle")
    coverage_axis.set_ylabel("coverage (%)")
    coverage_axis.set_xticks(observation_ids)
    coverage_axis.set_ylim(bottom=0)
    coverage_axis.grid(alpha=0.2)

    pioneer = metrics["pioneer_observation"]
    latency = metrics.get("latency") or {}
    cuda = metrics.get("cuda") or {}
    trajectory_metrics = metrics.get("trajectory") or {}
    run_id = str(run.get("run_id") or "PIONEER")
    depth_source = str(run.get("depth_source") or "GT").strip().upper()
    figure.suptitle(
        f"{scene} — PIONEER cubemap6/{depth_source} — "
        f"{len(bundles)} full-sphere observations",
        fontsize=22,
        y=0.995,
    )
    figure.text(
        0.5,
        0.012,
        (
            f"{run_id} · {len(points):,} points · final coverage {coverage_percent[-1]:.3f}% · "
            f"path {float(trajectory_metrics.get('path_length_scene_units', 0.0)):.3f} scene units · "
            f"real {int(pioneer['real_face_render_count'])} face renders · "
            f"candidate {int(pioneer.get('imagined_candidate_face_render_count', 0))} face renders · "
            f"trajectory {float(latency.get('trajectory_seconds', 0.0)):.3f}s · "
            f"peak reserved {float(cuda.get('peak_reserved_mib', 0.0)):.1f} MiB"
        ),
        ha="center",
        color="#aeb8c8",
        fontsize=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f".{output_path.stem}.tmp.png")
    figure.savefig(
        temporary_output,
        dpi=dpi,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
    )
    plt.close(figure)
    temporary_output.replace(output_path)
    with Image.open(output_path) as preview_image:
        preview_width, preview_height = preview_image.size
        preview_mode = preview_image.mode

    image_paths = [path for row in image_rows for path in row]
    data_file = lmdb_path / "data.mdb"
    sidecar = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "preview": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "width": preview_width,
            "height": preview_height,
            "mode": preview_mode,
        },
        "sources": {
            "metrics": {"path": str(metrics_path), "sha256": _sha256(metrics_path)},
            "lmdb": {
                "path": str(lmdb_path),
                "data_sha256": _sha256(data_file) if data_file.is_file() else None,
                "key": key,
            },
            "capture_images_root": str(images_root),
            "capture_images_tree_sha256": _tree_sha256(image_paths, images_root),
            "capture_image_count": len(image_paths),
            "bundle_commit_marker_count": len(commit_markers),
            "bundle_commit_markers_tree_sha256": (
                _tree_sha256(commit_markers, capture_dir.parent)
                if commit_markers
                else None
            ),
        },
        "summary": {
            "planner": "pioneer",
            "scene": scene,
            "start_index": start_index,
            "depth_source": depth_source,
            "bundle_count": len(bundles),
            "displayed_bundle_count": len(displayed_bundles),
            "displayed_bundle_ids": [
                int(bundle["bundle_id"]) for bundle in displayed_bundles
            ],
            "face_count": len(image_paths),
            "face_names": list(FACE_NAMES),
            "reconstructed_point_count": len(points),
            "final_normalized_coverage": float(coverage[-1]),
            "real_face_render_count": int(pioneer["real_face_render_count"]),
            "imagined_candidate_face_render_count": int(
                pioneer.get("imagined_candidate_face_render_count", 0)
            ),
        },
    }
    _atomic_json(output_path.with_suffix(".json"), sidecar)
    return sidecar


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--lmdb", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lmdb-key")
    parser.add_argument("--max-points", type=int, default=25000)
    parser.add_argument(
        "--max-bundle-rows",
        type=int,
        default=DEFAULT_MAX_BUNDLE_ROWS,
        help=(
            "maximum uniformly sampled cubemap rows to draw; all bundle images "
            "are still decoded, validated, and hashed"
        ),
    )
    parser.add_argument("--dpi", type=int, default=120)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = generate_preview(
        metrics_path=args.metrics,
        lmdb_path=args.lmdb,
        output_path=args.output,
        lmdb_key=args.lmdb_key,
        max_points=args.max_points,
        max_bundle_rows=args.max_bundle_rows,
        dpi=args.dpi,
    )
    print(json.dumps(result["preview"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
