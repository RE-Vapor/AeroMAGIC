#!/usr/bin/env python3
"""Build the controlled PAN-34 UE5 sky-diffuse/indirect-light comparison."""

from __future__ import annotations

import argparse
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


BASELINE_VARIANT = "ambient_fill_srgb_ev_plus_0_5_v1"
TREATMENT_VARIANT = "sky_diffuse_indirect_v1"
VARIANTS = (
    ("baseline", "A · PAN-32 ambient fill", (104, 221, 255)),
    ("indirect", "B · UE5 sky diffuse + Lumen GI", (255, 177, 66)),
)


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _same_float_list(actual: Any, expected: Any, label: str) -> None:
    if (
        not isinstance(actual, list)
        or not isinstance(expected, list)
        or len(actual) != len(expected)
        or not np.allclose(actual, expected, atol=1e-7, rtol=0.0)
    ):
        raise ValueError(f"{label} differs")


def _validate_control(evidence: Mapping[str, ReplayEvidence]) -> None:
    baseline = evidence["baseline"]
    treatment = evidence["indirect"]
    baseline_rows = _canonical_plan_rows(baseline.plan)
    if _canonical_plan_rows(treatment.plan) != baseline_rows:
        raise ValueError("PAN-34 changed observation identity, pose, or source timestamp")
    if treatment.plan["source"]["metrics"]["sha256"] != baseline.plan["source"]["metrics"]["sha256"]:
        raise ValueError("PAN-34 changed the frozen Planner source")

    baseline_spec = baseline.config.get("lighting_ablation")
    spec = treatment.config.get("lighting_ablation")
    if (
        not isinstance(baseline_spec, Mapping)
        or baseline_spec.get("variant_id") != BASELINE_VARIANT
        or not isinstance(spec, Mapping)
        or spec.get("variant_id") != TREATMENT_VARIANT
    ):
        raise ValueError("PAN-34 lighting variants are not the controlled pair")
    if (
        float(baseline_spec.get("post_read_exposure_transform_ev", 99.0)) != 0.5
        or float(spec.get("post_read_exposure_transform_ev", 99.0)) != 0.5
        or spec.get("post_read_shadow_lift") is not None
    ):
        raise ValueError("PAN-34 changed exposure or applied shadow postprocessing")

    directional_spec = spec.get("directional_light")
    sky_spec = spec.get("sky_light")
    gi_spec = spec.get("global_illumination")
    if not all(isinstance(value, Mapping) for value in (directional_spec, sky_spec, gi_spec)):
        raise ValueError("PAN-34 lacks UE5 indirect-lighting contracts")

    for observation_id in OBSERVATION_IDS:
        raw = _read_json(
            treatment.run_dir
            / "captures"
            / f"{observation_id:06d}"
            / "raw_bundle"
            / "manifest.json"
        )
        applied = raw.get("lighting_ablation")
        if (
            not isinstance(applied, Mapping)
            or applied.get("variant_id") != TREATMENT_VARIANT
            or applied.get("applied") is not True
            or applied.get("post_read_shadow_lift") is not None
            or applied.get("global_illumination") != gi_spec
            or applied.get("global_illumination_cvars_after")
            != {
                "r.DynamicGlobalIlluminationMethod": 1,
                "r.Lumen.DiffuseIndirect.Allow": 1,
                "r.Lumen.Reflections.Allow": 0,
            }
        ):
            raise ValueError(
                f"PAN-34 observation {observation_id} lacks applied Lumen provenance"
            )
        after = applied.get("after")
        if not isinstance(after, Mapping):
            raise ValueError(f"PAN-34 observation {observation_id} lacks a lighting snapshot")
        directional = after.get("directional_light")
        sky = after.get("sky_light")
        if (
            not isinstance(directional, Mapping)
            or float(directional.get("intensity", -1.0)) != float(directional_spec["intensity"])
            or float(directional.get("source_angle_degrees", -1.0))
            != float(directional_spec["source_angle_degrees"])
            or float(directional.get("indirect_lighting_intensity", -1.0))
            != float(directional_spec["indirect_lighting_intensity"])
            or not isinstance(sky, Mapping)
            or float(sky.get("intensity_scale", -1.0)) != float(sky_spec["intensity_scale"])
            or float(sky.get("indirect_lighting_intensity", -1.0))
            != float(sky_spec["indirect_lighting_intensity"])
            or sky.get("lower_hemisphere_is_solid_color")
            is not sky_spec["lower_hemisphere_is_solid_color"]
            or sky.get("real_time_capture") is not True
        ):
            raise ValueError(
                f"PAN-34 observation {observation_id} did not apply the UE5 lights"
            )
        _same_float_list(
            sky.get("lower_hemisphere_color_linear"),
            sky_spec["lower_hemisphere_color_linear"],
            f"PAN-34 observation {observation_id} lower-hemisphere color",
        )
        faces = raw.get("faces")
        if not isinstance(faces, list) or len(faces) != 6:
            raise ValueError(f"PAN-34 observation {observation_id} is not cubemap6")
        for face in faces:
            capture = face.get("capture_exposure")
            warmup = face.get("rgb_global_illumination_warmup")
            transform = face.get("rgb_post_read_exposure_transform")
            if (
                not isinstance(capture, Mapping)
                or capture.get("global_illumination") != gi_spec
                or capture.get("always_persist_rendering_state") is not True
                or not isinstance(warmup, Mapping)
                or int(warmup.get("capture_count", -1))
                != int(gi_spec["scene_capture_warmup_count"])
                or not isinstance(transform, Mapping)
                or float(transform.get("exposure_ev", 99.0)) != 0.5
            ):
                raise ValueError(
                    f"PAN-34 observation {observation_id} face {face.get('face_name')} "
                    "lacks the requested capture state"
                )

    for observation_id in OBSERVATION_IDS:
        if not np.array_equal(
            baseline.masks[observation_id], treatment.masks[observation_id]
        ):
            raise ValueError(f"PAN-34 changed ERP geometry at ID {observation_id}")


def _aggregate(item: ReplayEvidence) -> dict[str, float]:
    rows = list(item.stats.values())
    return {
        "mean_geometry_black_fraction_luma_le_0_05": float(
            np.mean([row["geometry_black_fraction_luma_le_0_05"] for row in rows])
        ),
        "mean_geometry_near_white_fraction_luma_ge_0_95": float(
            np.mean([row["geometry_near_white_fraction_luma_ge_0_95"] for row in rows])
        ),
        "mean_geometry_luma_p05": float(
            np.mean([row["geometry_luma_p05"] for row in rows])
        ),
        "mean_geometry_luma_p50": float(
            np.mean([row["geometry_luma_p50"] for row in rows])
        ),
        "mean_geometry_luma_p95": float(
            np.mean([row["geometry_luma_p95"] for row in rows])
        ),
    }


def generate_comparison(*, baseline_run: Path, indirect_run: Path, output: Path) -> dict[str, Any]:
    output = output.expanduser().resolve()
    sidecar = output.with_suffix(".json")
    commit = output.with_suffix(".commit.json")
    for target in (output, sidecar, commit):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite PAN-34 artifact: {target}")
    evidence = {
        "baseline": _load_replay("baseline", VARIANTS[0][1], baseline_run),
        "indirect": _load_replay("indirect", VARIANTS[1][1], indirect_run),
    }
    _validate_control(evidence)
    aggregate = {name: _aggregate(evidence[name]) for name, _, _ in VARIANTS}

    column_width = 720
    label_width = 180
    margin = 28
    header_height = 150
    row_height = 405
    footer_height = 200
    width = margin * 2 + label_width + column_width * 2
    height = header_height + row_height * len(OBSERVATION_IDS) + footer_height
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 24), "PAN-34 · UE5 sky diffuse and indirect light", font=_font(38, bold=True), fill=TEXT)
    draw.text((margin, 78), "Same ID · pose · source timestamp · geometry · exposure; Planner unchanged", font=_font(22), fill=MUTED)
    for column, (_, label, accent) in enumerate(VARIANTS):
        x = margin + label_width + column * column_width
        draw.text((x + 12, 116), label, font=_font(21, bold=True), fill=accent)

    for row_index, observation_id in enumerate(OBSERVATION_IDS):
        top = header_height + row_index * row_height
        draw.rounded_rectangle((margin, top + 8, width - margin, top + row_height - 10), radius=18, fill=PANEL)
        draw.text((margin + 18, top + 42), f"ID {observation_id}", font=_font(28, bold=True), fill=TEXT)
        pose = evidence["baseline"].plan["observations"][row_index]["planner_position_scene_units"]
        for index, value in enumerate(pose):
            draw.text((margin + 18, top + 86 + 26 * index), f"{'P ' if index == 0 else '  '}{value:.1f}", font=_font(16), fill=MUTED)
        for column, (name, _, _) in enumerate(VARIANTS):
            x = margin + label_width + column * column_width
            image = Image.fromarray(evidence[name].erps[observation_id]).resize((690, 345), Image.Resampling.LANCZOS)
            canvas.paste(image, (x + 10, top + 22))
            stats = evidence[name].stats[observation_id]
            text = (
                f"geom black≤5% {100*stats['geometry_black_fraction_luma_le_0_05']:.1f}%  "
                f"white≥95% {100*stats['geometry_near_white_fraction_luma_ge_0_95']:.2f}%  "
                f"P05/P50/P95 {stats['geometry_luma_p05']:.2f}/{stats['geometry_luma_p50']:.2f}/{stats['geometry_luma_p95']:.2f}"
            )
            draw.text((x + 14, top + 374), text, font=_font(13), fill=MUTED)

    summary_top = header_height + row_height * len(OBSERVATION_IDS) + 26
    for column, (name, label, accent) in enumerate(VARIANTS):
        x = margin + label_width + column * column_width
        values = aggregate[name]
        draw.text((x + 10, summary_top), label, font=_font(19, bold=True), fill=accent)
        draw.text((x + 10, summary_top + 34), f"mean geometry black: {100*values['mean_geometry_black_fraction_luma_le_0_05']:.2f}%", font=_font(17), fill=TEXT)
        draw.text((x + 10, summary_top + 62), f"mean near-white: {100*values['mean_geometry_near_white_fraction_luma_ge_0_95']:.3f}%", font=_font(17), fill=TEXT)
        draw.text((x + 10, summary_top + 90), f"mean P05/P50/P95: {values['mean_geometry_luma_p05']:.3f} / {values['mean_geometry_luma_p50']:.3f} / {values['mean_geometry_luma_p95']:.3f}", font=_font(17), fill=TEXT)
    draw.text((margin, height - 42), "UE5 re-rendered lighting · no shadow post-fill · no Planner, trajectory, coverage, geometry, or timestamp changes", font=_font(17), fill=MUTED)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pan34-preview-", dir=str(output.parent)))
    try:
        candidate_png = temporary / output.name
        candidate_json = temporary / sidecar.name
        candidate_commit = temporary / commit.name
        canvas.save(candidate_png, format="PNG", optimize=False, compress_level=9)
        payload = {
            "schema_version": "pan34.ue5-sky-indirect-comparison.v1",
            "result": "PASS",
            "observation_ids": list(OBSERVATION_IDS),
            "control": {
                "same_observation_id_pose_source_timestamp": True,
                "same_geometry_masks": True,
                "same_post_read_exposure": True,
                "ue5_sky_and_indirect_lighting_changed": True,
                "shadow_postprocessing_applied": False,
                "planner_input_unchanged": True,
                "post_run_visualization_only": True,
            },
            "variants": {
                name: {
                    "label": label,
                    "run_dir": str(evidence[name].run_dir),
                    "replay_result_sha256": _sha256(evidence[name].run_dir / "replay_result.json"),
                    "capture_config_sha256": _sha256(evidence[name].run_dir / "capture_config.json"),
                    "per_observation": {str(key): value for key, value in evidence[name].stats.items()},
                    "aggregate": aggregate[name],
                }
                for name, label, _ in VARIANTS
            },
            "preview": {
                "path": str(output),
                "sha256": _sha256(candidate_png),
                "width": width,
                "height": height,
                "mode": "RGB",
            },
        }
        candidate_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        candidate_commit.write_text(
            json.dumps(
                {
                    "schema_version": "pan34.ue5-sky-indirect-comparison-commit.v1",
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
    parser.add_argument("--indirect-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = generate_comparison(
        baseline_run=args.baseline_run,
        indirect_run=args.indirect_run,
        output=args.output,
    )
    print(json.dumps({"result": result["result"], "preview": result["preview"]}, sort_keys=True))


if __name__ == "__main__":
    main()
