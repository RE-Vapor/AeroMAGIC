#!/usr/bin/env python3
"""Build a controlled three-column UE5 lighting comparison for PAN-31."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from macarons.utility.cubemap_erp import cubemap_rgb_to_equirectangular


OBSERVATION_IDS = (0, 12, 24, 36, 49)
FACE_NAMES = ("front", "back", "left", "right", "up", "down")
VARIANTS = (
    ("baseline", "A · PAN-30 baseline"),
    ("exposure", "B · exposure only · +1 EV"),
    ("relight", "C · +1 EV · balanced relight"),
)
BACKGROUND = (13, 16, 21)
PANEL = (24, 29, 38)
TEXT = (234, 238, 246)
MUTED = (164, 177, 197)
ACCENTS = {"baseline": (104, 221, 255), "exposure": (255, 177, 66), "relight": (64, 218, 116)}


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


def _read_status(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in values:
            raise ValueError(f"invalid status line: {line!r}")
        values[key] = value
    if (
        values.get("snapshot_integrity_preflight") != "PASS"
        or values.get("snapshot_integrity_postflight") != "PASS"
        or values.get("exit_code") != "0"
    ):
        raise ValueError(f"replay status is not a successful PASS: {path}")
    return values


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


@dataclass(frozen=True)
class ReplayEvidence:
    name: str
    label: str
    run_dir: Path
    result: Mapping[str, Any]
    plan: Mapping[str, Any]
    config: Mapping[str, Any]
    erps: Mapping[int, np.ndarray]
    masks: Mapping[int, np.ndarray]
    stats: Mapping[int, Mapping[str, float]]


def _mask_erp(canonical_root: Path) -> np.ndarray:
    manifest = _read_json(canonical_root / "manifest.json")
    faces = []
    rows = manifest.get("faces")
    if not isinstance(rows, list) or tuple(row.get("face_name") for row in rows) != FACE_NAMES:
        raise ValueError(f"canonical bundle is not ordered cubemap6: {canonical_root}")
    for row in rows:
        face_name = str(row["face_name"])
        mask_path = canonical_root / str(row["assets"]["valid_mask"]["path"])
        valid = np.load(mask_path, allow_pickle=False)
        if valid.shape != (256, 256) or valid.dtype != np.bool_:
            raise ValueError(f"unexpected valid mask: {mask_path}")
        rgb = np.repeat(valid[..., None].astype(np.uint8) * 255, 3, axis=2)
        faces.append(
            SimpleNamespace(
                rgb_uint8=rgb,
                image_size=(256, 256),
                K_pixel=np.asarray(row["K_pixel"], dtype=np.float64),
                T_world_from_cam=np.asarray(row["T_world_from_cam"], dtype=np.float64),
            )
        )
    return cubemap_rgb_to_equirectangular(SimpleNamespace(faces=faces), output_height=512)[..., 0] > 127


def _image_stats(rgb: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    if rgb.shape != (512, 1024, 3) or rgb.dtype != np.uint8 or valid.shape != rgb.shape[:2]:
        raise ValueError("PAN-31 ERP/mask shape mismatch")
    values = rgb.astype(np.float64) / 255.0
    luminance = 0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]
    geometry = luminance[valid]
    if geometry.size == 0:
        raise ValueError("PAN-31 ERP has no valid geometry pixels")
    return {
        "geometry_fraction": float(valid.mean()),
        "geometry_black_fraction_luma_le_0_05": float((geometry <= 0.05).mean()),
        "geometry_near_white_fraction_luma_ge_0_95": float((geometry >= 0.95).mean()),
        "geometry_luma_p05": float(np.quantile(geometry, 0.05)),
        "geometry_luma_p50": float(np.quantile(geometry, 0.50)),
        "geometry_luma_p95": float(np.quantile(geometry, 0.95)),
    }


def _load_replay(name: str, label: str, run_dir: Path) -> ReplayEvidence:
    run_dir = run_dir.expanduser().resolve()
    result_path = run_dir / "replay_result.json"
    result = _read_json(result_path)
    _read_status(run_dir / "status.txt")
    if (
        result.get("result") != "PASS"
        or tuple(result.get("observation_ids") or ()) != OBSERVATION_IDS
        or result.get("planner_input_unchanged") is not True
    ):
        raise ValueError(f"{name} is not the exact five-observation PASS")
    plan_record = result.get("plan")
    if not isinstance(plan_record, Mapping):
        raise ValueError(f"{name} lacks replay plan provenance")
    plan_path = Path(str(plan_record.get("path", ""))).expanduser().resolve()
    if plan_path != run_dir / "replay_plan.json" or _sha256(plan_path) != plan_record.get("sha256"):
        raise ValueError(f"{name} replay plan changed")
    plan = _read_json(plan_path)
    config = _read_json(run_dir / "capture_config.json")
    erps: dict[int, np.ndarray] = {}
    masks: dict[int, np.ndarray] = {}
    stats: dict[int, Mapping[str, float]] = {}
    for observation_id in OBSERVATION_IDS:
        erp_path = run_dir / "processed" / f"{observation_id:06d}" / "erp.png"
        with Image.open(erp_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        canonical = run_dir / "captures" / f"{observation_id:06d}" / "canonical_bundle"
        mask = _mask_erp(canonical)
        erps[observation_id] = rgb
        masks[observation_id] = mask
        stats[observation_id] = _image_stats(rgb, mask)
    return ReplayEvidence(name, label, run_dir, result, plan, config, erps, masks, stats)


def _canonical_plan_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in plan.get("observations") or ():
        rows.append(
            {
                "observation_id": row["observation_id"],
                "source_capture_timestamp_ns": row["source_capture_timestamp_ns"],
                "source_capture_timestamp_utc": row["source_capture_timestamp_utc"],
                "planner_position_scene_units": row["planner_position_scene_units"],
                "ue_position_cm": row["ue_position_cm"],
            }
        )
    return rows


def _validate_control(evidence: Mapping[str, ReplayEvidence]) -> None:
    baseline = evidence["baseline"]
    baseline_rows = _canonical_plan_rows(baseline.plan)
    baseline_metrics = baseline.plan["source"]["metrics"]["sha256"]
    expected_variants = {
        "exposure": "exposure_only_ev_plus_1",
        "relight": "balanced_relight_ev_plus_1",
    }
    for name, current in evidence.items():
        if _canonical_plan_rows(current.plan) != baseline_rows:
            raise ValueError(f"{name} changed observation identity, pose, or source time")
        if current.plan["source"]["metrics"]["sha256"] != baseline_metrics:
            raise ValueError(f"{name} changed the frozen Planner source")
        if name == "baseline":
            if current.config.get("lighting_ablation") is not None:
                raise ValueError("baseline unexpectedly contains a PAN-31 override")
            continue
        spec = current.config.get("lighting_ablation")
        if not isinstance(spec, Mapping) or spec.get("variant_id") != expected_variants[name]:
            raise ValueError(f"{name} lighting variant contract differs")
        if float(spec.get("capture_exposure_compensation_ev", 99.0)) != 1.0:
            raise ValueError(f"{name} did not use the shared +1 EV exposure")
        for observation_id in OBSERVATION_IDS:
            raw = _read_json(current.run_dir / "captures" / f"{observation_id:06d}" / "raw_bundle" / "manifest.json")
            applied = raw.get("lighting_ablation")
            if not isinstance(applied, Mapping) or applied.get("variant_id") != expected_variants[name] or applied.get("applied") is not True:
                raise ValueError(f"{name} observation {observation_id} lacks applied lighting provenance")
            faces = raw.get("faces")
            if not isinstance(faces, list) or len(faces) != 6:
                raise ValueError(f"{name} observation {observation_id} raw faces are incomplete")
            if any(float(face["capture_exposure"]["exposure_compensation_ev"]) != 1.0 for face in faces):
                raise ValueError(f"{name} observation {observation_id} exposure differs across faces")
    for observation_id in OBSERVATION_IDS:
        reference_mask = baseline.masks[observation_id]
        for name in ("exposure", "relight"):
            if not np.array_equal(evidence[name].masks[observation_id], reference_mask):
                raise ValueError(f"{name} changed ERP geometry mask at ID {observation_id}")


def generate_comparison(*, baseline_run: Path, exposure_run: Path, relight_run: Path, output: Path) -> dict[str, Any]:
    output = output.expanduser().resolve()
    sidecar = output.with_suffix(".json")
    commit = output.with_suffix(".commit.json")
    for target in (output, sidecar, commit):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite PAN-31 artifact: {target}")
    evidence = {
        name: _load_replay(name, label, run)
        for (name, label), run in zip(VARIANTS, (baseline_run, exposure_run, relight_run))
    }
    _validate_control(evidence)

    column_width = 690
    label_width = 180
    margin = 28
    header_height = 150
    row_height = 390
    footer_height = 210
    width = margin * 2 + label_width + column_width * 3
    height = header_height + row_height * len(OBSERVATION_IDS) + footer_height
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 24), "PAN-31 · Controlled UE5 lighting ablation", font=_font(40, bold=True), fill=TEXT)
    draw.text((margin, 78), "Same PAN-30 observation ID · pose · source timestamp · geometry; Planner unchanged", font=_font(24), fill=MUTED)
    for column, (name, label) in enumerate(VARIANTS):
        x = margin + label_width + column * column_width
        draw.text((x + 12, 116), label, font=_font(21, bold=True), fill=ACCENTS[name])

    for row_index, observation_id in enumerate(OBSERVATION_IDS):
        top = header_height + row_index * row_height
        draw.rounded_rectangle((margin, top + 8, width - margin, top + row_height - 10), radius=18, fill=PANEL)
        draw.text((margin + 18, top + 42), f"ID {observation_id}", font=_font(28, bold=True), fill=TEXT)
        plan_row = evidence["baseline"].plan["observations"][row_index]
        pose = plan_row["planner_position_scene_units"]
        draw.text((margin + 18, top + 86), f"P {pose[0]:.1f}", font=_font(16), fill=MUTED)
        draw.text((margin + 18, top + 112), f"  {pose[1]:.1f}", font=_font(16), fill=MUTED)
        draw.text((margin + 18, top + 138), f"  {pose[2]:.1f}", font=_font(16), fill=MUTED)
        for column, (name, _) in enumerate(VARIANTS):
            item = evidence[name]
            x = margin + label_width + column * column_width
            image = Image.fromarray(item.erps[observation_id]).resize((660, 330), Image.Resampling.LANCZOS)
            canvas.paste(image, (x + 10, top + 22))
            stats = item.stats[observation_id]
            text = (
                f"geom black≤5% {100*stats['geometry_black_fraction_luma_le_0_05']:.1f}%  "
                f"white≥95% {100*stats['geometry_near_white_fraction_luma_ge_0_95']:.2f}%  "
                f"P05/P50/P95 {stats['geometry_luma_p05']:.2f}/{stats['geometry_luma_p50']:.2f}/{stats['geometry_luma_p95']:.2f}"
            )
            draw.text((x + 14, top + 356), text, font=_font(13), fill=MUTED)

    summary_top = header_height + row_height * len(OBSERVATION_IDS) + 24
    aggregate: dict[str, dict[str, float]] = {}
    for column, (name, label) in enumerate(VARIANTS):
        rows = evidence[name].stats.values()
        aggregate[name] = {
            "mean_geometry_black_fraction_luma_le_0_05": float(np.mean([row["geometry_black_fraction_luma_le_0_05"] for row in rows])),
            "mean_geometry_near_white_fraction_luma_ge_0_95": float(np.mean([row["geometry_near_white_fraction_luma_ge_0_95"] for row in rows])),
            "mean_geometry_luma_p05": float(np.mean([row["geometry_luma_p05"] for row in rows])),
            "mean_geometry_luma_p50": float(np.mean([row["geometry_luma_p50"] for row in rows])),
            "mean_geometry_luma_p95": float(np.mean([row["geometry_luma_p95"] for row in rows])),
        }
        x = margin + label_width + column * column_width
        values = aggregate[name]
        draw.text((x + 10, summary_top), label, font=_font(19, bold=True), fill=ACCENTS[name])
        draw.text((x + 10, summary_top + 34), f"mean geometry black: {100*values['mean_geometry_black_fraction_luma_le_0_05']:.2f}%", font=_font(17), fill=TEXT)
        draw.text((x + 10, summary_top + 62), f"mean near-white: {100*values['mean_geometry_near_white_fraction_luma_ge_0_95']:.3f}%", font=_font(17), fill=TEXT)
        draw.text((x + 10, summary_top + 90), f"mean P05/P50/P95: {values['mean_geometry_luma_p05']:.3f} / {values['mean_geometry_luma_p50']:.3f} / {values['mean_geometry_luma_p95']:.3f}", font=_font(17), fill=TEXT)
    draw.text((margin, height - 46), "Controlled post-run visualization only · no Planner, trajectory, coverage, geometry, or timestamp changes", font=_font(18), fill=MUTED)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pan31-preview-", dir=str(output.parent)))
    try:
        candidate_png = temporary / output.name
        candidate_json = temporary / sidecar.name
        candidate_commit = temporary / commit.name
        canvas.save(candidate_png, format="PNG", optimize=False, compress_level=9)
        payload = {
            "schema_version": "pan31.ue5-lighting-comparison.v1",
            "result": "PASS",
            "observation_ids": list(OBSERVATION_IDS),
            "control": {
                "same_observation_id_pose_source_timestamp": True,
                "same_geometry_masks": True,
                "planner_input_unchanged": True,
                "post_run_visualization_only": True,
            },
            "variants": {
                name: {
                    "label": item.label,
                    "run_dir": str(item.run_dir),
                    "replay_result_sha256": _sha256(item.run_dir / "replay_result.json"),
                    "capture_config_sha256": _sha256(item.run_dir / "capture_config.json"),
                    "per_observation": {str(key): value for key, value in item.stats.items()},
                    "aggregate": aggregate[name],
                }
                for name, item in evidence.items()
            },
            "preview": {"path": str(output), "sha256": _sha256(candidate_png), "width": width, "height": height, "mode": "RGB"},
        }
        candidate_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        candidate_commit.write_text(
            json.dumps(
                {
                    "schema_version": "pan31.ue5-lighting-comparison-commit.v1",
                    "preview_sha256": _sha256(candidate_png),
                    "sidecar_sha256": _sha256(candidate_json),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(candidate_png, output)
        os.replace(candidate_json, sidecar)
        os.replace(candidate_commit, commit)
    finally:
        try:
            temporary.rmdir()
        except OSError:
            pass
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--exposure-run", type=Path, required=True)
    parser.add_argument("--relight-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = generate_comparison(
        baseline_run=args.baseline_run,
        exposure_run=args.exposure_run,
        relight_run=args.relight_run,
        output=args.output,
    )
    print(json.dumps({"result": result["result"], "preview": result["preview"]}, sort_keys=True))


if __name__ == "__main__":
    main()
