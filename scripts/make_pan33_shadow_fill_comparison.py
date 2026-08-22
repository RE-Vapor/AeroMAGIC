#!/usr/bin/env python3
"""Build the controlled PAN-33 geometry-masked shadow-fill comparison."""

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
from scripts.process_pan29_replay_capture import SHADOW_FILL_SCHEMA


VARIANTS = (
    ("baseline", "A · PAN-32 ambient fill", (104, 221, 255)),
    ("light", "B · light geometry shadow fill", (255, 177, 66)),
    ("medium", "C · medium geometry shadow fill", (64, 218, 116)),
)
EXPECTED_VARIANTS = {
    "baseline": "ambient_fill_srgb_ev_plus_0_5_v1",
    "light": "shadow_fill_linear_0_006_v1",
    "medium": "shadow_fill_linear_0_012_v1",
}


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _shared_lighting_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "capture_exposure_compensation_ev": spec.get("capture_exposure_compensation_ev"),
        "post_read_exposure_transform_ev": spec.get("post_read_exposure_transform_ev"),
        "enable_manual_exposure_pipeline": spec.get("enable_manual_exposure_pipeline"),
        "directional_light": spec.get("directional_light"),
        "sky_light": spec.get("sky_light"),
    }


def _validate_control(evidence: Mapping[str, ReplayEvidence]) -> dict[str, float]:
    baseline = evidence["baseline"]
    baseline_rows = _canonical_plan_rows(baseline.plan)
    baseline_metrics = baseline.plan["source"]["metrics"]["sha256"]
    baseline_spec = baseline.config.get("lighting_ablation")
    if not isinstance(baseline_spec, Mapping) or baseline_spec.get("variant_id") != EXPECTED_VARIANTS["baseline"]:
        raise ValueError("PAN-33 baseline is not the accepted PAN-32 ambient-fill treatment")
    shared_contract = _shared_lighting_contract(baseline_spec)
    if (
        float(shared_contract["post_read_exposure_transform_ev"]) != 0.5
        or shared_contract["directional_light"] != {"intensity": 7.0, "source_angle_degrees": 3.0}
        or shared_contract["sky_light"] != {"intensity_scale": 3.0, "recapture_scene": True}
        or baseline_spec.get("post_read_shadow_lift") is not None
    ):
        raise ValueError("PAN-33 baseline lighting contract changed")

    strengths: dict[str, float] = {"baseline": 0.0}
    for name, current in evidence.items():
        if _canonical_plan_rows(current.plan) != baseline_rows:
            raise ValueError(f"{name} changed observation identity, pose, or source time")
        if current.plan["source"]["metrics"]["sha256"] != baseline_metrics:
            raise ValueError(f"{name} changed the frozen Planner source")
        spec = current.config.get("lighting_ablation")
        if not isinstance(spec, Mapping) or spec.get("variant_id") != EXPECTED_VARIANTS[name]:
            raise ValueError(f"{name} shadow-fill variant contract differs")
        if _shared_lighting_contract(spec) != shared_contract:
            raise ValueError(f"{name} changed exposure or UE5 lighting")
        shadow = spec.get("post_read_shadow_lift")
        if name == "baseline":
            if shadow is not None:
                raise ValueError("PAN-33 baseline unexpectedly applies shadow fill")
        else:
            if (
                not isinstance(shadow, Mapping)
                or shadow.get("method") != "geometry_masked_linear_toe_lift_v1"
                or float(shadow.get("cutoff_linear_luma", -1.0)) != 0.18
                or float(shadow.get("rolloff_power", -1.0)) != 2.0
            ):
                raise ValueError(f"{name} shadow-fill curve differs")
            strengths[name] = float(shadow.get("max_linear_lift", -1.0))

        for observation_id in OBSERVATION_IDS:
            raw = _read_json(current.run_dir / "captures" / f"{observation_id:06d}" / "raw_bundle" / "manifest.json")
            applied = raw.get("lighting_ablation")
            if not isinstance(applied, Mapping) or applied.get("variant_id") != EXPECTED_VARIANTS[name]:
                raise ValueError(f"{name} observation {observation_id} lacks applied variant provenance")
            if applied.get("post_read_shadow_lift") != shadow:
                raise ValueError(f"{name} observation {observation_id} applied a different shadow curve")
            faces = raw.get("faces")
            if not isinstance(faces, list) or len(faces) != 6:
                raise ValueError(f"{name} observation {observation_id} raw faces are incomplete")
            receipt = _read_json(
                current.run_dir
                / "processed"
                / f"{observation_id:06d}"
                / "receipt.json"
            )
            report = receipt.get("shadow_fill")
            if name == "baseline":
                if report is not None and report.get("applied") is not False:
                    raise ValueError(
                        f"baseline observation {observation_id} unexpectedly applies shadow fill"
                    )
                continue
            if (
                not isinstance(report, Mapping)
                or report.get("schema_version") != SHADOW_FILL_SCHEMA
                or report.get("applied") is not True
                or report.get("method") != shadow["method"]
                or float(report.get("max_linear_lift", -1.0))
                != float(shadow["max_linear_lift"])
                or float(report.get("cutoff_linear_luma", -1.0))
                != float(shadow["cutoff_linear_luma"])
                or float(report.get("rolloff_power", -1.0))
                != float(shadow["rolloff_power"])
                or int(report.get("total_shadow_lifted_pixel_count", 0)) <= 0
                or int(report.get("total_shadow_lift_changed_channel_count", 0)) <= 0
                or int(report.get("total_no_hit_changed_pixel_count", -1)) != 0
                or int(report.get("total_bright_region_changed_pixel_count", -1)) != 0
            ):
                raise ValueError(
                    f"{name} observation {observation_id} violates shadow-fill provenance"
                )
            report_faces = report.get("faces")
            if (
                not isinstance(report_faces, list)
                or [face.get("face_name") for face in report_faces]
                != [face.get("face_name") for face in faces]
            ):
                raise ValueError(
                    f"{name} observation {observation_id} shadow-fill faces differ"
                )
            for face in report_faces:
                valid = int(face.get("geometry_valid_pixel_count", -1))
                no_hit = int(face.get("no_hit_pixel_count", -1))
                eligible = int(face.get("eligible_shadow_pixel_count", -1))
                lifted = int(face.get("shadow_lifted_pixel_count", -1))
                if (
                    valid + no_hit != 256 * 256
                    or not 0 <= lifted <= eligible <= valid
                    or int(face.get("shadow_lift_changed_channel_count", -1)) < lifted
                    or int(face.get("no_hit_changed_pixel_count", -1)) != 0
                    or int(face.get("bright_region_changed_pixel_count", -1)) != 0
                ):
                    raise ValueError(
                        f"{name} observation {observation_id} face {face.get('face_name')} "
                        "violates the canonical geometry mask"
                    )

    if not 0.0 < strengths["light"] < strengths["medium"]:
        raise ValueError("PAN-33 shadow-fill strengths are not ordered")
    for observation_id in OBSERVATION_IDS:
        reference = baseline.masks[observation_id]
        for name in ("light", "medium"):
            if not np.array_equal(evidence[name].masks[observation_id], reference):
                raise ValueError(f"{name} changed ERP geometry mask at ID {observation_id}")
    return strengths


def _aggregate(item: ReplayEvidence) -> dict[str, float]:
    rows = item.stats.values()
    return {
        "mean_geometry_black_fraction_luma_le_0_05": float(np.mean([row["geometry_black_fraction_luma_le_0_05"] for row in rows])),
        "mean_geometry_near_white_fraction_luma_ge_0_95": float(np.mean([row["geometry_near_white_fraction_luma_ge_0_95"] for row in rows])),
        "mean_geometry_luma_p05": float(np.mean([row["geometry_luma_p05"] for row in rows])),
        "mean_geometry_luma_p50": float(np.mean([row["geometry_luma_p50"] for row in rows])),
        "mean_geometry_luma_p95": float(np.mean([row["geometry_luma_p95"] for row in rows])),
    }


def generate_comparison(*, baseline_run: Path, light_run: Path, medium_run: Path, output: Path) -> dict[str, Any]:
    output = output.expanduser().resolve()
    sidecar = output.with_suffix(".json")
    commit = output.with_suffix(".commit.json")
    for target in (output, sidecar, commit):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite PAN-33 artifact: {target}")
    evidence = {
        name: _load_replay(name, label, run)
        for (name, label, _), run in zip(VARIANTS, (baseline_run, light_run, medium_run))
    }
    strengths = _validate_control(evidence)

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
    draw.text((margin, 24), "PAN-33 · Geometry-masked shadow fill", font=_font(40, bold=True), fill=TEXT)
    draw.text((margin, 78), "Same ID · pose · source timestamp · geometry · exposure · UE5 lighting; Planner unchanged", font=_font(23), fill=MUTED)
    for column, (name, label, accent) in enumerate(VARIANTS):
        x = margin + label_width + column * column_width
        draw.text((x + 12, 116), label, font=_font(21, bold=True), fill=accent)

    for row_index, observation_id in enumerate(OBSERVATION_IDS):
        top = header_height + row_index * row_height
        draw.rounded_rectangle((margin, top + 8, width - margin, top + row_height - 10), radius=18, fill=PANEL)
        draw.text((margin + 18, top + 42), f"ID {observation_id}", font=_font(28, bold=True), fill=TEXT)
        pose = evidence["baseline"].plan["observations"][row_index]["planner_position_scene_units"]
        for index, value in enumerate(pose):
            prefix = "P " if index == 0 else "  "
            draw.text((margin + 18, top + 86 + 26 * index), f"{prefix}{value:.1f}", font=_font(16), fill=MUTED)
        for column, (name, _, _) in enumerate(VARIANTS):
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
    aggregate = {name: _aggregate(evidence[name]) for name, _, _ in VARIANTS}
    for column, (name, label, accent) in enumerate(VARIANTS):
        x = margin + label_width + column * column_width
        values = aggregate[name]
        draw.text((x + 10, summary_top), label, font=_font(19, bold=True), fill=accent)
        draw.text((x + 10, summary_top + 34), f"mean geometry black: {100*values['mean_geometry_black_fraction_luma_le_0_05']:.2f}%", font=_font(17), fill=TEXT)
        draw.text((x + 10, summary_top + 62), f"mean near-white: {100*values['mean_geometry_near_white_fraction_luma_ge_0_95']:.3f}%", font=_font(17), fill=TEXT)
        draw.text((x + 10, summary_top + 90), f"mean P05/P50/P95: {values['mean_geometry_luma_p05']:.3f} / {values['mean_geometry_luma_p50']:.3f} / {values['mean_geometry_luma_p95']:.3f}", font=_font(17), fill=TEXT)
    draw.text((margin, height - 46), "Visual post-run treatment only · no Planner, trajectory, coverage, geometry, background, or timestamp changes", font=_font(18), fill=MUTED)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".pan33-preview-", dir=str(output.parent)))
    try:
        candidate_png = temporary / output.name
        candidate_json = temporary / sidecar.name
        candidate_commit = temporary / commit.name
        payload = {
            "schema_version": "pan33.ue5-shadow-fill-comparison.v1",
            "result": "PASS",
            "observation_ids": list(OBSERVATION_IDS),
            "control": {
                "same_observation_id_pose_source_timestamp": True,
                "same_geometry_masks": True,
                "same_exposure_and_ue5_lighting": True,
                "no_hit_background_shadow_fill_unchanged": True,
                "bright_region_shadow_fill_unchanged": True,
                "planner_input_unchanged": True,
                "post_run_visualization_only": True,
            },
            "shadow_fill_max_linear_lift": strengths,
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
        }
        canvas.save(candidate_png, format="PNG", optimize=False, compress_level=9)
        payload["preview"] = {"path": str(output), "sha256": _sha256(candidate_png), "width": width, "height": height, "mode": "RGB"}
        candidate_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        candidate_commit.write_text(
            json.dumps(
                {
                    "schema_version": "pan33.ue5-shadow-fill-comparison-commit.v1",
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
    parser.add_argument("--light-run", type=Path, required=True)
    parser.add_argument("--medium-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = generate_comparison(
        baseline_run=args.baseline_run,
        light_run=args.light_run,
        medium_run=args.medium_run,
        output=args.output,
    )
    print(json.dumps({"result": result["result"], "preview": result["preview"]}, sort_keys=True))


if __name__ == "__main__":
    main()
