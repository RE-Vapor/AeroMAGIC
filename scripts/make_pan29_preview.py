#!/usr/bin/env python3
"""Create a five-row Planner-cubemap plus same-pose UE5-ERP preview."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


OBSERVATION_IDS = (0, 5, 10, 14, 19)
FACE_NAMES = ("front", "back", "left", "right", "up", "down")
BACKGROUND = (13, 16, 21)
PANEL = (24, 29, 38)
TEXT = (232, 237, 245)
MUTED = (170, 184, 204)
ACCENT = (104, 221, 255)
ORANGE = (255, 177, 66)
GREEN = (64, 218, 116)
RED = (255, 91, 86)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        key, separator, value = line.partition("=")
        if not separator or key in values:
            raise ValueError(f"invalid or duplicate status line: {line!r}")
        values[key] = value
    return values


def _tree_sha256(paths: Sequence[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _fit_range(values: np.ndarray, low: float, high: float) -> np.ndarray:
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum == minimum:
        return np.full_like(values, 0.5 * (low + high), dtype=np.float64)
    return low + (values - minimum) / (maximum - minimum) * (high - low)


def _draw_trajectory_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    positions: np.ndarray,
    selected_ids: Sequence[int],
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=18, fill=PANEL)
    draw.text(
        (left + 24, top + 18),
        f"{len(positions)}-observation Planner path (top view)",
        font=_font(24, bold=True),
        fill=TEXT,
    )
    plot = (left + 54, top + 72, right - 28, bottom - 42)
    x_values = _fit_range(positions[:, 0], plot[0], plot[2])
    z_values = _fit_range(positions[:, 2], plot[3], plot[1])
    points = [(float(x), float(z)) for x, z in zip(x_values, z_values)]
    draw.line(points, fill=RED, width=5, joint="curve")
    selected = set(selected_ids)
    for index, point in enumerate(points):
        radius = 9 if index in selected else 4
        color = ORANGE if index in selected else MUTED
        draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=color,
        )
    draw.text((plot[0], bottom - 32), "orange = UE5 replayed IDs", font=_font(18), fill=MUTED)


def _draw_coverage_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    coverage: np.ndarray,
    selected_ids: Sequence[int],
    task: str,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=18, fill=PANEL)
    draw.text(
        (left + 24, top + 18),
        f"{task} source normalized coverage",
        font=_font(24, bold=True),
        fill=TEXT,
    )
    plot = (left + 54, top + 72, right - 28, bottom - 42)
    xs = np.linspace(plot[0], plot[2], len(coverage))
    max_value = max(1.0, float(coverage.max()))
    ys = plot[3] - coverage / max_value * (plot[3] - plot[1])
    points = [(float(x), float(y)) for x, y in zip(xs, ys)]
    draw.line(points, fill=ACCENT, width=5, joint="curve")
    for index in selected_ids:
        x_value, y_value = points[index]
        draw.ellipse((x_value - 8, y_value - 8, x_value + 8, y_value + 8), fill=ORANGE)
    draw.text(
        (plot[0], bottom - 32),
        f"final {100.0 * float(coverage[-1]):.3f}% · unchanged by UE5 replay",
        font=_font(18),
        fill=MUTED,
    )


def generate_pan29_preview(
    *, metrics_path: Path, replay_result_path: Path, output_path: Path
) -> dict[str, Any]:
    metrics_path = Path(metrics_path).expanduser().resolve()
    replay_result_path = Path(replay_result_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    sidecar_path = output_path.with_suffix(".json")
    commit_path = output_path.with_suffix(".commit.json")
    if output_path.suffix.lower() != ".png":
        raise ValueError("PAN-29 preview output must use a .png suffix")
    for target in (output_path, sidecar_path, commit_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite preview artifact: {target}")
    if replay_result_path.name != "replay_result.json":
        raise ValueError("preview requires the published replay_result.json")

    metrics = _read_json(metrics_path)
    result = _read_json(replay_result_path)
    selected_ids = tuple(int(value) for value in result.get("observation_ids") or ())
    if (
        result.get("schema_version") != "pan29.ue5-postrun-replay-result.v1"
        or result.get("result") != "PASS"
        or len(selected_ids) != 5
        or tuple(sorted(selected_ids)) != selected_ids
        or len(set(selected_ids)) != 5
        or int(result.get("replay_count", -1)) != 5
        or result.get("planner_input_unchanged") is not True
    ):
        raise ValueError("PAN-29 replay result is not an exact five-observation PASS")
    run_manifest_record = result.get("run_manifest")
    if not isinstance(run_manifest_record, Mapping):
        raise ValueError("PAN-29 replay result lacks its run manifest")
    run_manifest_path = Path(
        str(run_manifest_record.get("path", ""))
    ).expanduser().resolve()
    if (
        not run_manifest_path.is_file()
        or _sha256(run_manifest_path) != run_manifest_record.get("sha256")
    ):
        raise ValueError("PAN-29 run manifest changed after validation")
    status_path = run_manifest_path.parent / "status.txt"
    if not status_path.is_file():
        raise ValueError("PAN-29 run status is missing")
    status = _read_key_values(status_path)
    if status.get("snapshot_integrity_preflight") != "PASS":
        raise ValueError("PAN-29 snapshot preflight did not pass")
    if status.get("snapshot_integrity_postflight") != "PASS":
        raise ValueError("PAN-29 snapshot postflight did not pass")
    if status.get("exit_code") not in (None, "0"):
        raise ValueError("PAN-29 run status is not successful")
    plan_record = result.get("plan")
    if not isinstance(plan_record, Mapping):
        raise ValueError("PAN-29 replay result lacks its plan")
    plan_path = Path(str(plan_record.get("path", ""))).expanduser().resolve()
    if not plan_path.is_file() or _sha256(plan_path) != plan_record.get("sha256"):
        raise ValueError("PAN-29 replay plan changed after validation")
    plan = _read_json(plan_path)
    if tuple(plan.get("selected_bundle_ids") or ()) != selected_ids:
        raise ValueError("replay result observation IDs differ from the plan")
    task = str(plan.get("task") or "PAN-29")
    source = plan.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("PAN-29 replay plan lacks source provenance")
    metrics_record = source.get("metrics")
    if (
        not isinstance(metrics_record, Mapping)
        or Path(str(metrics_record.get("path", ""))).expanduser().resolve() != metrics_path
        or _sha256(metrics_path) != metrics_record.get("sha256")
    ):
        raise ValueError("source metrics no longer match the replay plan")
    if metrics.get("planner") != "pioneer" or metrics.get("scene") != "HKUST":
        raise ValueError("preview requires the HKUST PIONEER metrics")
    run = metrics.get("run")
    pioneer = metrics.get("pioneer_observation")
    trajectory = metrics.get("trajectory")
    if not all(isinstance(value, Mapping) for value in (run, pioneer, trajectory)):
        raise ValueError("metrics lack run/pioneer/trajectory records")
    assert isinstance(run, Mapping) and isinstance(pioneer, Mapping) and isinstance(trajectory, Mapping)
    if str(source.get("depth_source", "")).upper() != "GT" or run.get("planning_observation_mode") != "cubemap6":
        raise ValueError("preview source must be the GT cubemap6 run")
    bundles = pioneer.get("bundles")
    source_observation_count = int(trajectory.get("observation_count", -1))
    if source_observation_count <= selected_ids[-1]:
        raise ValueError("preview source observation count is too small")
    if (
        not isinstance(bundles, list)
        or len(bundles) != source_observation_count
        or int(pioneer.get("bundle_count", -1)) != source_observation_count
    ):
        raise ValueError("preview source bundle count is inconsistent")
    by_id = {int(row["bundle_id"]): row for row in bundles if isinstance(row, Mapping)}
    if tuple(by_id) != tuple(range(source_observation_count)):
        raise ValueError("source bundle IDs must be contiguous from zero")
    positions = np.asarray(trajectory.get("positions"), dtype=np.float64)
    if positions.shape != (source_observation_count, 3) or not np.isfinite(positions).all():
        raise ValueError("source trajectory positions have an invalid shape")
    coverage_rows = metrics.get("coverage")
    if not isinstance(coverage_rows, list) or len(coverage_rows) != source_observation_count:
        raise ValueError("source coverage count differs from the trajectory")
    coverage = np.asarray([float(row["normalized"]) for row in coverage_rows])
    if not np.isfinite(coverage).all():
        raise ValueError("source coverage must be finite")

    capture_root = Path(str(source.get("capture_root", ""))).expanduser().resolve()
    images_root = capture_root / "imgs"
    all_image_paths = [
        images_root / f"{bundle_id:06d}" / f"{face}.png"
        for bundle_id in range(source_observation_count)
        for face in FACE_NAMES
    ]
    decoded_source: dict[tuple[int, str], Image.Image] = {}
    for bundle_id in range(source_observation_count):
        row = by_id[bundle_id]
        if int(row.get("face_count", -1)) != 6 or tuple(row.get("face_names") or ()) != FACE_NAMES:
            raise ValueError(f"source bundle {bundle_id} is not canonical cubemap6")
        for face in FACE_NAMES:
            path = images_root / f"{bundle_id:06d}" / f"{face}.png"
            if not path.is_file():
                raise FileNotFoundError(f"source face image is missing: {path}")
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                if rgb.width != rgb.height:
                    raise ValueError(f"source face image is not square: {path}")
                if bundle_id in selected_ids:
                    decoded_source[(bundle_id, face)] = rgb.copy()
    source_tree = _tree_sha256(all_image_paths, images_root)
    original_preview = source.get("original_preview")
    if not isinstance(original_preview, Mapping):
        raise ValueError("replay plan lacks original preview provenance")
    original_path = Path(str(original_preview.get("path", ""))).expanduser().resolve()
    original_sidecar_path = Path(
        str(original_preview.get("sidecar_path", ""))
    ).expanduser().resolve()
    if not original_path.is_file() or _sha256(original_path) != original_preview.get("sha256"):
        raise ValueError("original preview changed")
    if (
        not original_sidecar_path.is_file()
        or _sha256(original_sidecar_path) != original_preview.get("sidecar_sha256")
    ):
        raise ValueError("original preview sidecar changed")
    original_sidecar = _read_json(original_sidecar_path)
    original_sources = original_sidecar.get("sources")
    if (
        not isinstance(original_sources, Mapping)
        or int(original_sources.get("capture_image_count", -1))
        != 6 * source_observation_count
        or original_sources.get("capture_images_tree_sha256") != source_tree
    ):
        raise ValueError("source face tree differs from the accepted original preview")

    replays = result.get("replays")
    if not isinstance(replays, list) or [row.get("observation_id") for row in replays if isinstance(row, Mapping)] != list(selected_ids):
        raise ValueError("replay result rows are missing or reordered")
    erp_images: dict[int, Image.Image] = {}
    erp_paths = []
    for row in replays:
        assert isinstance(row, Mapping)
        observation_id = int(row["observation_id"])
        erp = row.get("erp")
        if not isinstance(erp, Mapping):
            raise ValueError(f"replay {observation_id} lacks ERP provenance")
        path = Path(str(erp.get("path", ""))).expanduser().resolve()
        if not path.is_file() or _sha256(path) != erp.get("sha256"):
            raise ValueError(f"replay ERP changed: {path}")
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if rgb.width != 2 * rgb.height:
                raise ValueError(f"replay ERP is not 2:1: {path}")
            erp_images[observation_id] = rgb.copy()
        erp_paths.append(path)

    plan_rows = {
        int(row["observation_id"]): row
        for row in plan.get("observations", ())
        if isinstance(row, Mapping)
    }
    if tuple(plan_rows) != selected_ids:
        raise ValueError("replay plan rows are not the exact selected IDs")
    out_of_policy_ids = [
        observation_id
        for observation_id in selected_ids
        if not bool(plan_rows[observation_id]["within_pan13_conservative_fly_volume"])
    ]
    if result.get("out_of_pan13_fly_policy_ids") != out_of_policy_ids:
        raise ValueError("replay result fly-policy summary differs from the plan")
    for replay in replays:
        assert isinstance(replay, Mapping)
        observation_id = int(replay["observation_id"])
        if bool(replay.get("within_pan13_conservative_fly_volume")) != bool(
            plan_rows[observation_id]["within_pan13_conservative_fly_volume"]
        ):
            raise ValueError(f"replay {observation_id} fly-policy value differs from plan")

    width = 2240
    header_height = 132
    row_height = 270
    bottom_height = 390
    footer_height = 76
    height = header_height + len(selected_ids) * row_height + bottom_height + footer_height
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (36, 24),
        f"{task} · HKUST GT {source_observation_count}obs · Same-pose UE5 post-run replay",
        font=_font(38, bold=True),
        fill=TEXT,
    )
    draw.text(
        (38, 78),
        "Planner input — actual GT cubemap used online     |     UE5 replay — post-run visualization, not Planner input",
        font=_font(22),
        fill=ACCENT,
    )

    label_width = 248
    face_size = 236
    face_gap = 7
    erp_width = 472
    for row_index, observation_id in enumerate(selected_ids):
        top = header_height + row_index * row_height
        draw.rectangle((20, top + 6, width - 20, top + row_height - 8), fill=PANEL)
        row = plan_rows[observation_id]
        policy_color = GREEN if row["within_pan13_conservative_fly_volume"] else ORANGE
        draw.text((36, top + 30), f"ID {observation_id}", font=_font(30, bold=True), fill=TEXT)
        draw.text(
            (36, top + 72),
            f"step {observation_id + 1}/{source_observation_count}",
            font=_font(20),
            fill=MUTED,
        )
        pose = row["planner_position_scene_units"]
        draw.text((36, top + 110), f"P [{pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}]", font=_font(17), fill=MUTED)
        source_timestamp = str(row["source_capture_timestamp_utc"])
        timestamp_date, separator, timestamp_time = source_timestamp.partition("T")
        draw.text((36, top + 140), "source timestamp (UTC)", font=_font(14), fill=MUTED)
        draw.text((36, top + 162), timestamp_date, font=_font(14), fill=MUTED)
        draw.text(
            (36, top + 183),
            timestamp_time if separator else source_timestamp,
            font=_font(14),
            fill=MUTED,
        )
        policy_lines = (
            ("inside PAN13 fly policy",)
            if row["within_pan13_conservative_fly_volume"]
            else ("post-hoc replay", "outside PAN13 fly policy")
        )
        for line_index, policy_line in enumerate(policy_lines):
            draw.text(
                (36, top + 216 + 19 * line_index),
                policy_line,
                font=_font(14, bold=True),
                fill=policy_color,
            )

        x_cursor = label_width
        for face in FACE_NAMES:
            image = decoded_source[(observation_id, face)].resize(
                (face_size, face_size), Image.Resampling.LANCZOS
            )
            canvas.paste(image, (x_cursor, top + 12))
            draw.rectangle((x_cursor, top + 12, x_cursor + face_size, top + 42), fill=(0, 0, 0))
            draw.text((x_cursor + 9, top + 17), face, font=_font(17, bold=True), fill=TEXT)
            x_cursor += face_size + face_gap
        erp = erp_images[observation_id].resize(
            (erp_width, face_size), Image.Resampling.LANCZOS
        )
        canvas.paste(erp, (x_cursor + 6, top + 12))
        draw.rectangle((x_cursor + 6, top + 12, x_cursor + 6 + erp_width, top + 47), fill=(0, 0, 0))
        draw.text((x_cursor + 16, top + 18), "UE5 ERP · post-run only", font=_font(18, bold=True), fill=ORANGE)

    bottom_top = header_height + len(selected_ids) * row_height + 16
    _draw_trajectory_panel(
        canvas,
        draw,
        (28, bottom_top, width // 2 - 10, bottom_top + 330),
        positions,
        selected_ids,
    )
    _draw_coverage_panel(
        draw,
        (width // 2 + 10, bottom_top, width - 28, bottom_top + 330),
        coverage,
        selected_ids,
        task,
    )
    draw.text(
        (width // 2, height - 48),
        "UE5 and Planner use different visual assets/lighting; this preview does not claim RGB, depth, or coverage parity.",
        anchor="mm",
        font=_font(19),
        fill=MUTED,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".pan29-preview-", dir=str(output_path.parent)))
    try:
        temporary_png = temporary_dir / output_path.name
        canvas.save(temporary_png, format="PNG", optimize=False, compress_level=9)
        sidecar = {
            "schema_version": "pan29.preview.v1",
            "task": task,
            "preview": {
                "path": str(output_path),
                "sha256": _sha256(temporary_png),
                "width": width,
                "height": height,
                "mode": "RGB",
            },
            "sources": {
                "metrics": {"path": str(metrics_path), "sha256": _sha256(metrics_path)},
                "replay_plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
                "replay_result": {"path": str(replay_result_path), "sha256": _sha256(replay_result_path)},
                "original_preview": {
                    "path": str(original_path),
                    "sha256": _sha256(original_path),
                },
                "all_source_face_count": len(all_image_paths),
                "all_source_face_tree_sha256": source_tree,
                "displayed_source_face_count": 30,
                "displayed_source_face_tree_sha256": _tree_sha256(
                    [
                        images_root / f"{observation_id:06d}" / f"{face}.png"
                        for observation_id in selected_ids
                        for face in FACE_NAMES
                    ],
                    images_root,
                ),
                "ue5_erp_count": 5,
                "ue5_erp_tree_sha256": _tree_sha256(erp_paths, Path("/")),
            },
            "summary": {
                "scene": "HKUST",
                "source_depth": "GT",
                "source_observation_count": source_observation_count,
                "displayed_observation_ids": list(selected_ids),
                "planner_face_image_count": 30,
                "ue5_erp_count": 5,
                "planner_input_unchanged": True,
                "ue5_role": "post_run_visualization_only",
                "out_of_pan13_fly_policy_ids": out_of_policy_ids,
            },
        }
        temporary_sidecar = temporary_dir / sidecar_path.name
        temporary_sidecar.write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        commit = {
            "schema_version": "pan29.preview-commit.v1",
            "preview_path": str(output_path),
            "preview_sha256": _sha256(temporary_png),
            "sidecar_path": str(sidecar_path),
            "sidecar_sha256": _sha256(temporary_sidecar),
        }
        temporary_commit = temporary_dir / commit_path.name
        temporary_commit.write_text(
            json.dumps(commit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_png.replace(output_path)
        temporary_sidecar.replace(sidecar_path)
        temporary_commit.replace(commit_path)
        return sidecar
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--replay-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = generate_pan29_preview(
        metrics_path=args.metrics,
        replay_result_path=args.replay_result,
        output_path=args.output,
    )
    print(json.dumps(result["preview"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
