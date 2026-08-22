#!/usr/bin/env python3
"""Validate one UE5 replay capture and publish its PIONEER-oriented ERP."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from macarons.utility.cubemap_erp import cubemap_rgb_to_equirectangular
from macarons.utility.ue5_observation_contract import FACE_NAMES, load_bundle, validate_bundle


CONTRACT_TO_PIONEER = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)
SOURCE_SEMANTIC_SCHEMA = "pan29.source-bundle-semantic-validation.v1"
ERP_CONVENTION = {
    "projection": "equirectangular",
    "world_up": "+Y",
    "longitude_zero": "+Z/front",
    "image_right_quarter": "-X/right",
    "horizontal_wrap": "-Z/back",
    "sampling": "nearest-neighbour using actual T_world_from_cam and K_pixel",
}
POSITION_TOLERANCE_SCENE_UNITS = 2.0e-4  # 1 mm at 0.2 scene unit / metre.
BASIS_TOLERANCE = 1.0e-5
SHADOW_FILL_SCHEMA = "pan33.geometry-masked-shadow-fill.v1"


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


def _one_plan_row(plan: Mapping[str, Any], observation_id: int) -> Mapping[str, Any]:
    if plan.get("schema_version") != "pan29.ue5-postrun-replay.v1":
        raise ValueError("unexpected PAN-29 replay plan schema")
    rows = plan.get("observations")
    if not isinstance(rows, list):
        raise ValueError("replay plan observations must be a list")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("observation_id") == observation_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one replay row for observation {observation_id}")
    return matches[0]


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result


def _same_mapping(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} does not match the replay plan")


def _actual_timestamp_utc(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _source_basis_from_plan_row(
    row: Mapping[str, Any], source_position: np.ndarray
) -> tuple[dict[str, np.ndarray], str]:
    semantic = row.get("source_semantic_validation")
    if not isinstance(semantic, Mapping):
        raise ValueError("replay plan lacks source semantic validation")
    if (
        semantic.get("schema_version") != SOURCE_SEMANTIC_SCHEMA
        or semantic.get("result") != "PASS"
        or semantic.get("bundle_id") != row.get("observation_id")
        or semantic.get("source_capture_timestamp_ns")
        != row.get("source_capture_timestamp_ns")
        or semantic.get("source_capture_timestamp_utc")
        != row.get("source_capture_timestamp_utc")
    ):
        raise ValueError("replay plan source semantic identity is invalid")
    semantic_center = _array(
        semantic.get("camera_center_scene_units"),
        (3,),
        "source semantic camera center",
    )
    if float(np.max(np.abs(semantic_center - source_position))) > POSITION_TOLERANCE_SCENE_UNITS:
        raise ValueError("replay plan source semantic camera center differs from the pose")
    if float(semantic.get("camera_center_max_abs_error_scene_units", math.inf)) > POSITION_TOLERANCE_SCENE_UNITS:
        raise ValueError("replay plan source semantic center validation failed")
    if float(semantic.get("face_basis_max_abs_error", math.inf)) > BASIS_TOLERANCE:
        raise ValueError("replay plan source semantic basis validation failed")
    face_rows = semantic.get("faces")
    if not isinstance(face_rows, list) or tuple(
        face.get("face_name") for face in face_rows if isinstance(face, Mapping)
    ) != FACE_NAMES:
        raise ValueError("replay plan source semantic faces are incomplete or reordered")
    rotations: dict[str, np.ndarray] = {}
    hash_rows = []
    for face in face_rows:
        assert isinstance(face, Mapping)
        face_name = str(face["face_name"])
        rotation = _array(
            face.get("T_world_from_cam_rotation"),
            (3, 3),
            f"source semantic face {face_name} basis",
        )
        orthonormal_error = float(
            np.max(np.abs(rotation.T @ rotation - np.eye(3, dtype=np.float64)))
        )
        determinant_error = abs(float(np.linalg.det(rotation)) - 1.0)
        if max(orthonormal_error, determinant_error) > BASIS_TOLERANCE:
            raise ValueError(f"source basis for face {face_name} is not right-handed orthonormal")
        face_center = _array(
            face.get("camera_center_scene_units"),
            (3,),
            f"source semantic face {face_name} camera center",
        )
        if (
            float(np.max(np.abs(face_center - source_position)))
            > POSITION_TOLERANCE_SCENE_UNITS
            or float(face.get("camera_center_max_abs_error_scene_units", math.inf))
            > POSITION_TOLERANCE_SCENE_UNITS
            or float(face.get("basis_max_abs_error", math.inf)) > BASIS_TOLERANCE
        ):
            raise ValueError(f"source semantic face {face_name} validation failed")
        rotations[face_name] = rotation
        hash_rows.append(
            {"face_name": face_name, "T_world_from_cam_rotation": rotation.tolist()}
        )
    basis_sha256 = hashlib.sha256(
        json.dumps(hash_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return rotations, basis_sha256


def _pioneer_bundle_view(bundle: Any, scene_units_per_meter: float) -> Any:
    faces = []
    for face in bundle.faces:
        transform = np.asarray(face.T_world_from_cam, dtype=np.float64)
        converted = np.eye(4, dtype=np.float64)
        converted[:3, :3] = CONTRACT_TO_PIONEER @ transform[:3, :3]
        converted[:3, 3] = (
            CONTRACT_TO_PIONEER @ transform[:3, 3] * scene_units_per_meter
        )
        faces.append(replace(face, T_world_from_cam=converted))
    center = CONTRACT_TO_PIONEER @ np.asarray(bundle.position_world_m) * scene_units_per_meter
    return replace(bundle, position_world_m=tuple(center), faces=tuple(faces))


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.float64) / 255.0
    return np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )


def _linear_to_srgb_uint8(linear: np.ndarray) -> np.ndarray:
    values = np.clip(linear, 0.0, 1.0)
    encoded = np.where(
        values <= 0.0031308,
        12.92 * values,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )
    return np.rint(np.clip(encoded, 0.0, 1.0) * 255.0).astype(np.uint8)


def _apply_geometry_shadow_fill(bundle: Any, spec: Any) -> tuple[Any, dict[str, Any]]:
    if spec is None:
        return bundle, {
            "schema_version": SHADOW_FILL_SCHEMA,
            "applied": False,
            "method": None,
            "faces": [],
            "total_shadow_lifted_pixel_count": 0,
            "total_shadow_lift_changed_channel_count": 0,
        }
    if not isinstance(spec, Mapping) or spec.get("method") != "geometry_masked_linear_toe_lift_v1":
        raise ValueError("PAN-33 shadow-fill specification is invalid")
    max_lift = float(spec.get("max_linear_lift", -1.0))
    cutoff = float(spec.get("cutoff_linear_luma", -1.0))
    power = float(spec.get("rolloff_power", -1.0))
    if (
        not math.isfinite(max_lift)
        or not 0.0 < max_lift <= 0.1
        or not math.isfinite(cutoff)
        or not 0.0 < cutoff < 1.0
        or not math.isfinite(power)
        or not 1.0 <= power <= 8.0
    ):
        raise ValueError("PAN-33 shadow-fill parameters are out of range")

    faces = []
    reports = []
    total_lifted = 0
    total_channels = 0
    for face in bundle.faces:
        rgb = np.asarray(face.rgb_uint8, dtype=np.uint8)
        valid = np.asarray(face.valid_mask, dtype=np.bool_)
        linear = _srgb_to_linear(rgb)
        luma = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
        eligible = valid & (luma < cutoff)
        weight = np.zeros_like(luma, dtype=np.float64)
        weight[eligible] = np.power(1.0 - luma[eligible] / cutoff, power)
        lifted = np.clip(linear + max_lift * weight[..., None], 0.0, 1.0)
        transformed = rgb.copy()
        transformed[eligible] = _linear_to_srgb_uint8(lifted[eligible])
        changed = transformed != rgb
        changed_pixels = np.any(changed, axis=2)
        no_hit_changed = int(np.count_nonzero(changed_pixels & ~valid))
        bright_changed = int(np.count_nonzero(changed_pixels & valid & ~eligible))
        if no_hit_changed != 0 or bright_changed != 0:
            raise AssertionError("PAN-33 shadow fill escaped its geometry/shadow mask")
        lifted_count = int(np.count_nonzero(changed_pixels & eligible))
        channel_count = int(np.count_nonzero(changed))
        total_lifted += lifted_count
        total_channels += channel_count
        reports.append(
            {
                "face_name": face.face_name,
                "geometry_valid_pixel_count": int(np.count_nonzero(valid)),
                "no_hit_pixel_count": int(valid.size - np.count_nonzero(valid)),
                "eligible_shadow_pixel_count": int(np.count_nonzero(eligible)),
                "shadow_lifted_pixel_count": lifted_count,
                "shadow_lift_changed_channel_count": channel_count,
                "no_hit_changed_pixel_count": no_hit_changed,
                "bright_region_changed_pixel_count": bright_changed,
            }
        )
        faces.append(replace(face, rgb_uint8=transformed))
    if total_lifted <= 0 or total_channels <= 0:
        raise ValueError("PAN-33 shadow-fill treatment changed no geometry pixels")
    return replace(bundle, faces=tuple(faces)), {
        "schema_version": SHADOW_FILL_SCHEMA,
        "applied": True,
        "method": "geometry_masked_linear_toe_lift_v1",
        "max_linear_lift": max_lift,
        "cutoff_linear_luma": cutoff,
        "rolloff_power": power,
        "faces": reports,
        "total_shadow_lifted_pixel_count": total_lifted,
        "total_shadow_lift_changed_channel_count": total_channels,
        "total_no_hit_changed_pixel_count": int(sum(row["no_hit_changed_pixel_count"] for row in reports)),
        "total_bright_region_changed_pixel_count": int(sum(row["bright_region_changed_pixel_count"] for row in reports)),
    }


def _validate_replay_source(
    actual: Any, row: Mapping[str, Any], metrics_sha256: str
) -> None:
    if not isinstance(actual, Mapping):
        raise ValueError("capture lacks replay_source provenance")
    expected = {
        "schema_version": "pan29.replay-source.v1",
        "observation_id": row["observation_id"],
        "source_bundle_id": row["observation_id"],
        "source_capture_timestamp_ns": row["source_capture_timestamp_ns"],
        "source_capture_timestamp_utc": row["source_capture_timestamp_utc"],
        "planner_position_scene_units": row["planner_position_scene_units"],
        "requested_position_ue_cm": row["ue_position_cm"],
        "source_metrics_sha256": metrics_sha256,
        "within_pan13_conservative_fly_volume": row[
            "within_pan13_conservative_fly_volume"
        ],
        "planner_input_unchanged": True,
        "ue5_role": "post_run_visualization_only",
    }
    if dict(actual) != expected:
        raise ValueError("capture replay_source does not exactly match the plan")


def process_replay_capture(
    *,
    plan_path: Path,
    observation_id: int,
    request_path: Path,
    raw_manifest_path: Path,
    canonical_manifest_path: Path,
    output_dir: Path,
    erp_height: int = 512,
) -> dict[str, Any]:
    """Fail closed before atomically publishing one panorama and receipt."""

    paths = [
        Path(value).expanduser().resolve()
        for value in (plan_path, request_path, raw_manifest_path, canonical_manifest_path)
    ]
    plan_path, request_path, raw_manifest_path, canonical_manifest_path = paths
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite replay output: {output_dir}")
    if isinstance(erp_height, bool) or not isinstance(erp_height, int) or erp_height < 2:
        raise ValueError("erp_height must be an integer >= 2")

    plan = _read_json(plan_path)
    request = _read_json(request_path)
    raw = _read_json(raw_manifest_path)
    row = _one_plan_row(plan, observation_id)
    source = plan.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("replay plan lacks source provenance")
    metrics_record = source.get("metrics")
    if not isinstance(metrics_record, Mapping):
        raise ValueError("replay plan lacks metrics provenance")
    metrics_sha = str(metrics_record.get("sha256", ""))
    if len(metrics_sha) != 64:
        raise ValueError("replay plan metrics SHA256 is invalid")
    scene_units_per_meter = float(source.get("scene_units_per_meter", 0.0))
    if not math.isfinite(scene_units_per_meter) or scene_units_per_meter <= 0.0:
        raise ValueError("scene_units_per_meter must be finite and positive")

    if request.get("schema_version") != "pan15.capture-request.v1":
        raise ValueError("unexpected capture request schema")
    for key, expected in (
        ("request_id", row["replay_request_id"]),
        ("frame_id", row["replay_frame_id"]),
        ("position_ue_cm", row["ue_position_cm"]),
    ):
        _same_mapping(request.get(key), expected, f"request {key}")
    _validate_replay_source(request.get("replay_source"), row, metrics_sha)

    if raw.get("schema_version") != "pan15.ue5-raw-rgbd.v1":
        raise ValueError("unexpected raw capture schema")
    for key in ("request_id", "frame_id", "requested_position_ue_cm"):
        expected = request["position_ue_cm"] if key == "requested_position_ue_cm" else request[key]
        _same_mapping(raw.get(key), expected, f"raw {key}")
    _validate_replay_source(raw.get("replay_source"), row, metrics_sha)
    actual_timestamp = raw.get("capture_timestamp_ns")
    if type(actual_timestamp) is not int or actual_timestamp <= 0:
        raise ValueError("raw capture timestamp must be a positive integer")
    requested_position = _array(request["position_ue_cm"], (3,), "requested UE position")
    measured_ue_position = _array(
        raw.get("shared_optical_center_ue_cm"), (3,), "measured UE position"
    )
    ue_position_error = float(np.max(np.abs(measured_ue_position - requested_position)))
    if ue_position_error > 0.1:  # 1 mm in UE centimetres.
        raise ValueError(f"measured UE position differs from request: {ue_position_error} cm")

    source_position = _array(
        row["planner_position_scene_units"], (3,), "source Planner position"
    )
    source_rotations, source_basis_sha256 = _source_basis_from_plan_row(
        row, source_position
    )
    raw_faces = raw.get("faces")
    if not isinstance(raw_faces, list) or tuple(
        face.get("face_name") for face in raw_faces if isinstance(face, Mapping)
    ) != FACE_NAMES:
        raise ValueError("raw capture must contain canonical ordered six faces")
    raw_basis_error = 0.0
    for face in raw_faces:
        assert isinstance(face, Mapping)
        face_name = str(face["face_name"])
        if face.get("request_id") != request["request_id"] or face.get("frame_id") != request["frame_id"]:
            raise ValueError(f"raw face {face_name} request/frame mismatch")
        if face.get("capture_timestamp_ns") != actual_timestamp:
            raise ValueError(f"raw face {face_name} capture epoch mismatch")
        transform = _array(
            face.get("T_pioneer_world_from_cam"),
            (4, 4),
            f"raw face {face_name} PIONEER transform",
        )
        basis_error = float(
            np.max(
                np.abs(
                    transform[:3, :3] - source_rotations[face_name]
                )
            )
        )
        raw_basis_error = max(raw_basis_error, basis_error)
        if basis_error > BASIS_TOLERANCE:
            raise ValueError(
                f"raw face {face_name} differs from plan-pinned source basis: "
                f"{basis_error}"
            )
        if float(np.max(np.abs(transform[:3, 3] - source_position))) > POSITION_TOLERANCE_SCENE_UNITS:
            raise ValueError(f"raw face {face_name} source position mismatch")

    canonical = load_bundle(canonical_manifest_path)
    validate_bundle(canonical)
    if canonical.request_id != request["request_id"] or canonical.frame_id != request["frame_id"]:
        raise ValueError("canonical request/frame mismatch")
    if canonical.capture_timestamp_ns != actual_timestamp:
        raise ValueError("canonical capture epoch differs from raw capture")
    _validate_replay_source(canonical.provenance.get("replay_source"), row, metrics_sha)
    pioneer_view = _pioneer_bundle_view(canonical, scene_units_per_meter)
    measured_source_position = np.asarray(pioneer_view.position_world_m, dtype=np.float64)
    source_position_error = float(np.max(np.abs(measured_source_position - source_position)))
    if source_position_error > POSITION_TOLERANCE_SCENE_UNITS:
        raise ValueError(
            f"canonical pose differs from source observation: {source_position_error} scene units"
        )
    canonical_basis_error = 0.0
    for face in pioneer_view.faces:
        expected_rotation = source_rotations[face.face_name]
        actual_rotation = np.asarray(face.T_world_from_cam)[:3, :3]
        error = float(np.max(np.abs(actual_rotation - expected_rotation)))
        canonical_basis_error = max(canonical_basis_error, error)
        if error > BASIS_TOLERANCE:
            raise ValueError(
                f"canonical face {face.face_name} differs from plan-pinned source "
                f"basis: {error}"
            )

    lighting_ablation = raw.get("lighting_ablation")
    if lighting_ablation is not None and not isinstance(lighting_ablation, Mapping):
        raise ValueError("raw capture lacks lighting-ablation provenance")
    pioneer_view, shadow_fill_report = _apply_geometry_shadow_fill(
        pioneer_view,
        None
        if lighting_ablation is None
        else lighting_ablation.get("post_read_shadow_lift"),
    )

    panorama = cubemap_rgb_to_equirectangular(pioneer_view, output_height=erp_height)
    repeated = cubemap_rgb_to_equirectangular(pioneer_view, output_height=erp_height)
    if not np.array_equal(panorama, repeated):
        raise AssertionError("cubemap-to-ERP projection is not byte deterministic")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent))
    )
    try:
        erp_path = temporary / "erp.png"
        Image.fromarray(panorama, mode="RGB").save(
            erp_path, format="PNG", optimize=False, compress_level=9
        )
        canonical_payload = _read_json(canonical_manifest_path)
        face_rgb_hashes = {
            str(item["face_name"]): str(item["assets"]["rgb_uint8"]["sha256"])
            for item in canonical_payload["faces"]
        }
        receipt = {
            "schema_version": "pan29.ue5-replay-receipt.v1",
            "task": plan.get("task", "PAN-29"),
            "result": "PASS",
            "observation_id": observation_id,
            "request_id": request["request_id"],
            "frame_id": request["frame_id"],
            "source_capture_timestamp_ns": row["source_capture_timestamp_ns"],
            "source_capture_timestamp_utc": row["source_capture_timestamp_utc"],
            "ue5_actual_capture_timestamp_ns": actual_timestamp,
            "ue5_actual_capture_timestamp_utc": _actual_timestamp_utc(actual_timestamp),
            "planner_position_scene_units": source_position.tolist(),
            "requested_position_ue_cm": requested_position.tolist(),
            "measured_position_ue_cm": measured_ue_position.tolist(),
            "measured_position_scene_units": measured_source_position.tolist(),
            "ue_position_max_abs_error_cm": ue_position_error,
            "position_max_abs_error_scene_units": source_position_error,
            "face_basis_max_abs_error": max(raw_basis_error, canonical_basis_error),
            "face_names": list(FACE_NAMES),
            "within_pan13_conservative_fly_volume": bool(
                row["within_pan13_conservative_fly_volume"]
            ),
            "planner_input_unchanged": True,
            "ue5_role": "post_run_visualization_only",
            "shadow_fill": shadow_fill_report,
            "source_semantic_validation": {
                "schema_version": SOURCE_SEMANTIC_SCHEMA,
                "result": "PASS",
                "basis_sha256": source_basis_sha256,
                "camera_center_max_abs_error_scene_units": row[
                    "source_semantic_validation"
                ]["camera_center_max_abs_error_scene_units"],
                "face_basis_max_abs_error": row["source_semantic_validation"][
                    "face_basis_max_abs_error"
                ],
            },
            "sources": {
                "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
                "request": {"path": str(request_path), "sha256": _sha256(request_path)},
                "raw_manifest": {
                    "path": str(raw_manifest_path),
                    "sha256": _sha256(raw_manifest_path),
                },
                "canonical_manifest": {
                    "path": str(canonical_manifest_path),
                    "sha256": _sha256(canonical_manifest_path),
                },
                "canonical_rgb_array_sha256": face_rgb_hashes,
            },
            "erp": {
                "path": "erp.png",
                "sha256": _sha256(erp_path),
                "width": int(panorama.shape[1]),
                "height": int(panorama.shape[0]),
                "mode": "RGB",
                "convention": ERP_CONVENTION,
                "byte_deterministic_reprojection": True,
            },
        }
        (temporary / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output_dir)
        temporary = None
        return receipt
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--observation-id", type=int, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--erp-height", type=int, default=512)
    args = parser.parse_args(argv)
    result = process_replay_capture(
        plan_path=args.plan,
        observation_id=args.observation_id,
        request_path=args.request,
        raw_manifest_path=args.raw_manifest,
        canonical_manifest_path=args.canonical_manifest,
        output_dir=args.output,
        erp_height=args.erp_height,
    )
    print(json.dumps({"observation_id": result["observation_id"], "result": "PASS"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
