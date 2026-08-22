#!/usr/bin/env python3
"""Create a presentation-only Base Color + deterministic sky/cloud composite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw

from scripts.make_pan31_lighting_comparison import (
    BACKGROUND,
    MUTED,
    OBSERVATION_IDS,
    PANEL,
    TEXT,
    ReplayEvidence,
    _canonical_plan_rows,
    _font,
    _load_replay,
    _sha256,
)


LINEAR_ISSUE = "PAN-30"
LIT_VARIANT = "sky_diffuse_indirect_v1"
UNLIT_CONTRACT = {
    "mode": "material_base_color",
    "capture_source": "SCS_BASE_COLOR",
    "lighting_dependency": "none_unlit_diagnostic",
}
VARIANTS = (
    ("lit", "A · SkyLight + Lumen final color", (104, 221, 255)),
    ("base", "B · material Base Color · diagnostic", (255, 177, 66)),
    ("composite", "C · Base Color + sky/cloud composite", (64, 218, 116)),
)


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def procedural_sky_cloud_erp(*, height: int = 512) -> np.ndarray:
    """Return one world-fixed, seam-continuous presentation sky in ERP layout."""

    if isinstance(height, bool) or not isinstance(height, int) or height < 16:
        raise ValueError("sky ERP height must be an integer >= 16")
    width = 2 * height
    longitude = (np.arange(width, dtype=np.float64) + 0.5) / width * (2.0 * np.pi) - np.pi
    latitude = np.pi / 2.0 - (np.arange(height, dtype=np.float64) + 0.5) / height * np.pi
    lon, lat = np.meshgrid(longitude, latitude)
    cos_lat = np.cos(lat)
    direction = np.stack(
        (cos_lat * np.cos(lon), np.sin(lat), cos_lat * np.sin(lon)), axis=-1
    )
    vertical = direction[..., 1]

    horizon = np.asarray([0.70, 0.84, 0.95], dtype=np.float64)
    zenith = np.asarray([0.18, 0.48, 0.78], dtype=np.float64)
    lower = np.asarray([0.20, 0.39, 0.58], dtype=np.float64)
    upper_mix = np.clip(vertical, 0.0, 1.0) ** 0.55
    lower_mix = np.clip(-vertical, 0.0, 1.0) ** 0.8
    sky = horizon[None, None, :] * (1.0 - upper_mix[..., None]) + zenith[None, None, :] * upper_mix[..., None]
    sky = sky * (1.0 - lower_mix[..., None]) + lower[None, None, :] * lower_mix[..., None]

    x, y, z = direction[..., 0], direction[..., 1], direction[..., 2]
    noise = (
        0.34 * np.sin(3.1 * (1.4 * x + 0.3 * y - 0.8 * z) + 0.7)
        + 0.24 * np.sin(6.7 * (-0.2 * x + 0.8 * y + 1.1 * z) + 2.2)
        + 0.17 * np.sin(12.9 * (0.7 * x - 0.5 * y + 0.6 * z) + 4.1)
        + 0.12 * np.sin(24.3 * (-0.9 * x + 0.1 * y + 0.4 * z) + 1.4)
    )
    cloud_field = 0.5 + noise
    cloud_alpha = _smoothstep(0.48, 0.74, cloud_field)
    cloud_altitude = _smoothstep(-0.08, 0.12, vertical) * (1.0 - 0.3 * _smoothstep(0.78, 0.98, vertical))
    cloud_alpha *= cloud_altitude
    cloud_light = np.clip(0.72 + 0.24 * cloud_field, 0.62, 0.98)
    cloud_rgb = np.stack(
        (cloud_light, cloud_light * 0.98, cloud_light * 0.96), axis=-1
    )
    sky = sky * (1.0 - cloud_alpha[..., None]) + cloud_rgb * cloud_alpha[..., None]
    result = np.clip(np.rint(sky * 255.0), 1.0, 255.0).astype(np.uint8)
    return result


def composite_background(
    foreground: np.ndarray, valid_mask: np.ndarray, sky: np.ndarray
) -> np.ndarray:
    if (
        foreground.dtype != np.uint8
        or sky.dtype != np.uint8
        or valid_mask.dtype != np.bool_
        or foreground.shape != sky.shape
        or valid_mask.shape != foreground.shape[:2]
        or foreground.ndim != 3
        or foreground.shape[2] != 3
    ):
        raise ValueError("PAN-30 composite inputs have incompatible shape or dtype")
    result = sky.copy()
    result[valid_mask] = foreground[valid_mask]
    if not np.array_equal(result[valid_mask], foreground[valid_mask]):
        raise AssertionError("PAN-30 changed valid geometry pixels")
    return result


def _validate_inputs(lit: ReplayEvidence, base: ReplayEvidence) -> None:
    if _canonical_plan_rows(lit.plan) != _canonical_plan_rows(base.plan):
        raise ValueError("PAN-30 changed observation ID, pose, or source timestamp")
    if lit.plan["source"]["metrics"]["sha256"] != base.plan["source"]["metrics"]["sha256"]:
        raise ValueError("PAN-30 changed the frozen Planner source")
    lit_spec = lit.config.get("lighting_ablation")
    if not isinstance(lit_spec, Mapping) or lit_spec.get("variant_id") != LIT_VARIANT:
        raise ValueError("PAN-30 lit input is not the accepted PAN-34 treatment")
    if base.config.get("lighting_ablation") is not None or base.config.get("rgb_render_mode") != UNLIT_CONTRACT:
        raise ValueError("PAN-30 Base Color input lacks PAN-35 provenance")
    for observation_id in OBSERVATION_IDS:
        if not np.array_equal(lit.masks[observation_id], base.masks[observation_id]):
            raise ValueError(f"PAN-30 geometry mask changed at ID {observation_id}")
        raw = _read_json(
            base.run_dir / "captures" / f"{observation_id:06d}" / "raw_bundle" / "manifest.json"
        )
        if raw.get("rgb_render_mode") != UNLIT_CONTRACT:
            raise ValueError(f"PAN-30 Base Color manifest differs at ID {observation_id}")


def generate(*, lit_run: Path, base_run: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite PAN-30 output: {output_dir}")
    lit = _load_replay("lit", VARIANTS[0][1], lit_run)
    base = _load_replay("base", VARIANTS[1][1], base_run)
    _validate_inputs(lit, base)
    sky = procedural_sky_cloud_erp(height=512)
    composites = {
        observation_id: composite_background(
            base.erps[observation_id], base.masks[observation_id], sky
        )
        for observation_id in OBSERVATION_IDS
    }
    repeated_sky = procedural_sky_cloud_erp(height=512)
    if not np.array_equal(sky, repeated_sky):
        raise AssertionError("PAN-30 sky generation is not byte deterministic")

    per_observation: dict[str, Any] = {}
    for observation_id in OBSERVATION_IDS:
        valid = base.masks[observation_id]
        composite = composites[observation_id]
        if not np.array_equal(composite[valid], base.erps[observation_id][valid]):
            raise AssertionError("PAN-30 changed Base Color geometry")
        if not np.array_equal(composite[~valid], sky[~valid]):
            raise AssertionError("PAN-30 background does not match the frozen sky")
        per_observation[str(observation_id)] = {
            "valid_geometry_pixel_count": int(valid.sum()),
            "no_hit_background_pixel_count": int((~valid).sum()),
            "changed_valid_geometry_pixel_count": 0,
            "background_replaced_pixel_count": int((~valid).sum()),
        }

    column_width, label_width, margin = 690, 175, 26
    header_height, row_height, footer_height = 150, 390, 125
    width = 2 * margin + label_width + 3 * column_width
    height = header_height + len(OBSERVATION_IDS) * row_height + footer_height
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 24), "Linear PAN-30 · Base Color sky/cloud composite", font=_font(36, bold=True), fill=TEXT)
    draw.text((margin, 76), "Same ID · pose · timestamp · geometry; composite is presentation-only and not physically lit", font=_font(21), fill=MUTED)
    for column, (_, label, accent) in enumerate(VARIANTS):
        x = margin + label_width + column * column_width
        draw.text((x + 10, 116), label, font=_font(19, bold=True), fill=accent)

    for row_index, observation_id in enumerate(OBSERVATION_IDS):
        top = header_height + row_index * row_height
        draw.rounded_rectangle((margin, top + 8, width - margin, top + row_height - 10), radius=16, fill=PANEL)
        draw.text((margin + 16, top + 40), f"ID {observation_id}", font=_font(26, bold=True), fill=TEXT)
        pose = lit.plan["observations"][row_index]["planner_position_scene_units"]
        for index, value in enumerate(pose):
            draw.text((margin + 16, top + 82 + index * 25), f"{'P ' if index == 0 else '  '}{value:.1f}", font=_font(15), fill=MUTED)
        images = {
            "lit": lit.erps[observation_id],
            "base": base.erps[observation_id],
            "composite": composites[observation_id],
        }
        for column, (name, _, _) in enumerate(VARIANTS):
            x = margin + label_width + column * column_width
            image = Image.fromarray(images[name]).resize((660, 330), Image.Resampling.LANCZOS)
            canvas.paste(image, (x + 10, top + 22))
        draw.text((margin + 16, top + 190), "geometry\nbyte-exact\nin B and C", font=_font(14), fill=MUTED)

    footer_top = header_height + len(OBSERVATION_IDS) * row_height + 22
    draw.text((margin, footer_top), "C replaces only valid_mask=false background pixels with one deterministic world-fixed sky/cloud ERP.", font=_font(18, bold=True), fill=(64, 218, 116))
    draw.text((margin, footer_top + 35), "No relighting · no Planner input · no trajectory/coverage change · no photorealism claim", font=_font(17), fill=MUTED)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pan30-sky-cloud-", dir=str(output_dir.parent)))
    try:
        composite_dir = temporary / "composites"
        composite_dir.mkdir()
        sky_path = temporary / "sky_cloud_erp.png"
        preview_path = temporary / "pan30_sky_cloud_composite_preview.png"
        sidecar_path = temporary / "pan30_sky_cloud_composite_preview.json"
        commit_path = temporary / "pan30_sky_cloud_composite_preview.commit.json"
        Image.fromarray(sky).save(sky_path, format="PNG", optimize=False, compress_level=9)
        composite_records = {}
        for observation_id, image in composites.items():
            path = composite_dir / f"{observation_id:06d}.png"
            Image.fromarray(image).save(path, format="PNG", optimize=False, compress_level=9)
            composite_records[str(observation_id)] = {
                "path": str(path.relative_to(temporary)),
                "sha256": _sha256(path),
            }
        canvas.save(preview_path, format="PNG", optimize=False, compress_level=9)
        payload = {
            "schema_version": "pan30.base-color-sky-cloud-composite.v1",
            "linear_issue": LINEAR_ISSUE,
            "result": "PASS",
            "observation_ids": list(OBSERVATION_IDS),
            "control": {
                "same_observation_id_pose_source_timestamp": True,
                "same_geometry_masks": True,
                "valid_geometry_rgb_byte_exact": True,
                "background_only_replacement": True,
                "same_world_fixed_sky_for_all_observations": True,
                "byte_deterministic": True,
                "planner_input_unchanged": True,
                "post_run_presentation_composite": True,
                "physically_lit": False,
            },
            "source_runs": {
                "lit": {"path": str(lit.run_dir), "replay_result_sha256": _sha256(lit.run_dir / "replay_result.json")},
                "base_color": {"path": str(base.run_dir), "replay_result_sha256": _sha256(base.run_dir / "replay_result.json")},
            },
            "sky": {"path": "sky_cloud_erp.png", "sha256": _sha256(sky_path), "width": 1024, "height": 512, "generator": "world-direction periodic analytic clouds v1"},
            "composites": composite_records,
            "per_observation": per_observation,
            "preview": {"path": preview_path.name, "sha256": _sha256(preview_path), "width": width, "height": height},
        }
        sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        commit_path.write_text(
            json.dumps(
                {
                    "schema_version": "pan30.sky-cloud-preview-commit.v1",
                    "result": "PASS",
                    "preview_sha256": _sha256(preview_path),
                    "sidecar_sha256": _sha256(sidecar_path),
                    "sky_sha256": _sha256(sky_path),
                    "composite_sha256": {key: value["sha256"] for key, value in composite_records.items()},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    return {
        "result": "PASS",
        "linear_issue": LINEAR_ISSUE,
        "output_dir": str(output_dir),
        "preview_sha256": payload["preview"]["sha256"],
        "sky_sha256": payload["sky"]["sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lit-run", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(lit_run=args.lit_run, base_run=args.base_run, output_dir=args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
