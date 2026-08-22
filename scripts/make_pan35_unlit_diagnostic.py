#!/usr/bin/env python3
"""Compare UE5 relit RGB against the lighting-independent material base color."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
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


LIT_VARIANT = "sky_diffuse_indirect_v1"
UNLIT_CONTRACT = {
    "mode": "material_base_color",
    "capture_source": "SCS_BASE_COLOR",
    "lighting_dependency": "none_unlit_diagnostic",
}
VARIANTS = (
    ("lit", "A · SkyLight + Lumen final color", (104, 221, 255)),
    ("unlit", "B · UE5 material Base Color · unlit", (255, 177, 66)),
)


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _luminance(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.float64) / 255.0
    return 0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]


def _diagnose(lit: np.ndarray, unlit: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    lit_black = (_luminance(lit) <= 0.05) & valid
    unlit_black = (_luminance(unlit) <= 0.05) & valid
    count = int(lit_black.sum())
    if count == 0:
        raise ValueError("PAN-35 lit reference contains no black geometry pixels")
    persistent = int((lit_black & unlit_black).sum())
    recovered = int((lit_black & ~unlit_black).sum())
    return {
        "lit_black_geometry_pixel_count": count,
        "persistent_black_in_base_color_pixel_count": persistent,
        "recovered_without_lighting_pixel_count": recovered,
        "persistent_model_or_material_fraction_of_lit_black": persistent / count,
        "lighting_or_shading_fraction_of_lit_black": recovered / count,
    }


def _validate_control(evidence: Mapping[str, ReplayEvidence]) -> None:
    lit, unlit = evidence["lit"], evidence["unlit"]
    if _canonical_plan_rows(lit.plan) != _canonical_plan_rows(unlit.plan):
        raise ValueError("PAN-35 changed observation identity, pose, or source timestamp")
    if lit.plan["source"]["metrics"]["sha256"] != unlit.plan["source"]["metrics"]["sha256"]:
        raise ValueError("PAN-35 changed the frozen Planner source")
    lit_spec = lit.config.get("lighting_ablation")
    if not isinstance(lit_spec, Mapping) or lit_spec.get("variant_id") != LIT_VARIANT:
        raise ValueError("PAN-35 lit reference is not the accepted PAN-34 treatment")
    if unlit.config.get("lighting_ablation") is not None or unlit.config.get("rgb_render_mode") != UNLIT_CONTRACT:
        raise ValueError("PAN-35 treatment is not the exact unlit Base Color contract")
    for observation_id in OBSERVATION_IDS:
        if not np.array_equal(lit.masks[observation_id], unlit.masks[observation_id]):
            raise ValueError(f"PAN-35 changed geometry at ID {observation_id}")
        raw = _read_json(
            unlit.run_dir / "captures" / f"{observation_id:06d}" / "raw_bundle" / "manifest.json"
        )
        lighting = raw.get("lighting_ablation")
        if raw.get("rgb_render_mode") != UNLIT_CONTRACT or not isinstance(lighting, Mapping) or lighting.get("applied") is not False:
            raise ValueError(f"PAN-35 observation {observation_id} lacks unlit provenance")
        faces = raw.get("faces")
        if not isinstance(faces, list) or len(faces) != 6:
            raise ValueError(f"PAN-35 observation {observation_id} is not cubemap6")
        for face in faces:
            if face.get("rgb_capture_source") != "SCS_BASE_COLOR" or face.get("rgb_render_mode") != "material_base_color":
                raise ValueError(f"PAN-35 observation {observation_id} face is not Base Color")
            transform = face.get("rgb_post_read_exposure_transform")
            if not isinstance(transform, Mapping) or float(transform.get("exposure_ev", 99.0)) != 0.0:
                raise ValueError("PAN-35 Base Color was modified after readback")


def generate(*, lit_run: Path, unlit_run: Path, output: Path) -> dict[str, Any]:
    output = output.expanduser().resolve()
    sidecar = output.with_suffix(".json")
    commit = output.with_suffix(".commit.json")
    for target in (output, sidecar, commit):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite PAN-35 artifact: {target}")
    evidence = {
        "lit": _load_replay("lit", VARIANTS[0][1], lit_run),
        "unlit": _load_replay("unlit", VARIANTS[1][1], unlit_run),
    }
    _validate_control(evidence)
    diagnostic = {
        observation_id: _diagnose(
            evidence["lit"].erps[observation_id],
            evidence["unlit"].erps[observation_id],
            evidence["lit"].masks[observation_id],
        )
        for observation_id in OBSERVATION_IDS
    }
    lit_black_total = sum(row["lit_black_geometry_pixel_count"] for row in diagnostic.values())
    persistent_total = sum(row["persistent_black_in_base_color_pixel_count"] for row in diagnostic.values())
    aggregate = {
        "lit_black_geometry_pixel_count": lit_black_total,
        "persistent_black_in_base_color_pixel_count": persistent_total,
        "recovered_without_lighting_pixel_count": lit_black_total - persistent_total,
        "persistent_model_or_material_fraction_of_lit_black": persistent_total / lit_black_total,
        "lighting_or_shading_fraction_of_lit_black": (lit_black_total - persistent_total) / lit_black_total,
    }

    column_width, label_width, margin = 720, 180, 28
    header_height, row_height, footer_height = 150, 405, 190
    width = margin * 2 + label_width + 2 * column_width
    height = header_height + len(OBSERVATION_IDS) * row_height + footer_height
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 24), "PAN-35 · lighting vs material diagnostic", font=_font(38, bold=True), fill=TEXT)
    draw.text((margin, 78), "Same ID · pose · timestamp · geometry; Base Color bypasses all UE5 lighting", font=_font(22), fill=MUTED)
    for column, (_, label, accent) in enumerate(VARIANTS):
        x = margin + label_width + column * column_width
        draw.text((x + 12, 116), label, font=_font(21, bold=True), fill=accent)

    for row_index, observation_id in enumerate(OBSERVATION_IDS):
        top = header_height + row_index * row_height
        draw.rounded_rectangle((margin, top + 8, width - margin, top + row_height - 10), radius=18, fill=PANEL)
        draw.text((margin + 18, top + 42), f"ID {observation_id}", font=_font(28, bold=True), fill=TEXT)
        pose = evidence["lit"].plan["observations"][row_index]["planner_position_scene_units"]
        for index, value in enumerate(pose):
            draw.text((margin + 18, top + 86 + 26 * index), f"{'P ' if index == 0 else '  '}{value:.1f}", font=_font(16), fill=MUTED)
        for column, (name, _, _) in enumerate(VARIANTS):
            x = margin + label_width + column * column_width
            image = Image.fromarray(evidence[name].erps[observation_id]).resize((690, 345), Image.Resampling.LANCZOS)
            canvas.paste(image, (x + 10, top + 22))
            stats = evidence[name].stats[observation_id]
            draw.text(
                (x + 14, top + 374),
                f"geometry black≤5% {100*stats['geometry_black_fraction_luma_le_0_05']:.1f}%  P50 {stats['geometry_luma_p50']:.2f}",
                font=_font(14),
                fill=MUTED,
            )
        row = diagnostic[observation_id]
        draw.text(
            (margin + 18, top + 190),
            f"lit-black source:\n{100*row['lighting_or_shading_fraction_of_lit_black']:.1f}% lighting\n{100*row['persistent_model_or_material_fraction_of_lit_black']:.1f}% material",
            font=_font(15),
            fill=MUTED,
        )

    summary_top = header_height + len(OBSERVATION_IDS) * row_height + 24
    draw.text((margin, summary_top), "Diagnosis over all valid geometry pixels that were black in the lit render", font=_font(20, bold=True), fill=TEXT)
    draw.text((margin, summary_top + 40), f"Lighting/shading contribution: {100*aggregate['lighting_or_shading_fraction_of_lit_black']:.2f}%", font=_font(19), fill=(104, 221, 255))
    draw.text((margin, summary_top + 72), f"Persistent in Base Color (model/material): {100*aggregate['persistent_model_or_material_fraction_of_lit_black']:.2f}%", font=_font(19), fill=(255, 177, 66))
    draw.text((margin, height - 38), "Post-run diagnostic only · Base Color is not a photorealistic render · Planner and coverage unchanged", font=_font(16), fill=MUTED)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pan35-preview-", dir=str(output.parent)))
    try:
        candidate_png = temporary / output.name
        candidate_json = temporary / sidecar.name
        candidate_commit = temporary / commit.name
        canvas.save(candidate_png, format="PNG", optimize=False, compress_level=9)
        payload = {
            "schema_version": "pan35.ue5-unlit-material-diagnostic.v1",
            "result": "PASS",
            "observation_ids": list(OBSERVATION_IDS),
            "control": {
                "same_observation_id_pose_source_timestamp": True,
                "same_geometry_masks": True,
                "planner_input_unchanged": True,
                "post_run_diagnostic_only": True,
                "unlit_capture_source": "SCS_BASE_COLOR",
            },
            "aggregate_diagnosis": aggregate,
            "per_observation_diagnosis": {str(key): value for key, value in diagnostic.items()},
            "variants": {
                name: {
                    "label": label,
                    "run_dir": str(evidence[name].run_dir),
                    "replay_result_sha256": _sha256(evidence[name].run_dir / "replay_result.json"),
                    "per_observation_stats": {str(key): value for key, value in evidence[name].stats.items()},
                }
                for name, label, _ in VARIANTS
            },
        }
        candidate_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        candidate_commit.write_text(
            json.dumps(
                {
                    "schema_version": "pan35.preview-commit.v1",
                    "result": "PASS",
                    "png_sha256": _sha256(candidate_png),
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
        for child in temporary.iterdir():
            child.unlink()
        temporary.rmdir()
    return {
        "result": "PASS",
        "preview": {"path": str(output), "sha256": _sha256(output), "width": width, "height": height},
        "aggregate_diagnosis": aggregate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lit-run", type=Path, required=True)
    parser.add_argument("--unlit-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(lit_run=args.lit_run, unlit_run=args.unlit_run, output=args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
