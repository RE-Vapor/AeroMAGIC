#!/usr/bin/env python3
"""Aggregate five validated PAN-29 replay receipts with fresh source hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


OBSERVATION_IDS = (0, 5, 10, 14, 19)
FACE_NAMES = ("front", "back", "left", "right", "up", "down")
REQUEST_SCHEMA = "pan15.capture-request.v1"
RAW_SCHEMA = "pan15.ue5-raw-rgbd.v1"
CANONICAL_SCHEMA = "pioneer.ue5-observation.v1"
REPLAY_SOURCE_SCHEMA = "pan29.replay-source.v1"
SOURCE_SEMANTIC_SCHEMA = "pan29.source-bundle-semantic-validation.v1"
POSITION_TOLERANCE_SCENE_UNITS = 2.0e-4
SOURCE_SEMANTIC_TOLERANCE = 1.0e-5
UE_POSITION_TOLERANCE_CM = 0.1
BASIS_TOLERANCE = 1.0e-5


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
            raise ValueError(f"invalid or duplicate run manifest key: {line!r}")
        values[key] = value
    return values


def _verify_file_record(
    record: Any, label: str, *, expected_path: Path | None = None
) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} record is missing")
    path = Path(str(record.get("path", ""))).expanduser().resolve()
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(f"{label} does not use the expected capture path")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if _sha256(path) != record.get("sha256"):
        raise ValueError(f"{label} SHA256 changed")
    return path


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must contain {length} finite values")
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain {length} finite values") from error
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain {length} finite values")
    return result


def _matrix4(value: Any, label: str) -> list[list[float]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
    ):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    rows = [_finite_vector(row, 4, label) for row in value]
    if any(
        abs(actual - expected) > 1e-9 for actual, expected in zip(rows[3], (0, 0, 0, 1))
    ):
        raise ValueError(f"{label} must be homogeneous")
    return rows


def _matrix3(value: Any, label: str) -> list[list[float]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    return [_finite_vector(row, 3, label) for row in value]


def _max_abs_difference(actual: Sequence[float], expected: Sequence[float]) -> float:
    return max(abs(float(left) - float(right)) for left, right in zip(actual, expected))


def _expected_replay_source(
    row: Mapping[str, Any], metrics_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": REPLAY_SOURCE_SCHEMA,
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


def _source_semantic_authority(
    row: Mapping[str, Any],
) -> tuple[list[float], dict[str, list[list[float]]], str]:
    observation_id = int(row["observation_id"])
    semantic = row.get("source_semantic_validation")
    if not isinstance(semantic, Mapping):
        raise ValueError(
            f"observation {observation_id} lacks source semantic validation"
        )
    for key, expected in (
        ("schema_version", SOURCE_SEMANTIC_SCHEMA),
        ("result", "PASS"),
        ("bundle_id", observation_id),
        ("source_capture_timestamp_ns", row.get("source_capture_timestamp_ns")),
        ("source_capture_timestamp_utc", row.get("source_capture_timestamp_utc")),
    ):
        if semantic.get(key) != expected:
            raise ValueError(
                f"observation {observation_id} source semantic {key} differs from plan"
            )
    source_position = _finite_vector(
        semantic.get("camera_center_scene_units"),
        3,
        f"observation {observation_id} semantic camera center",
    )
    row_position = _finite_vector(
        row.get("planner_position_scene_units"),
        3,
        f"observation {observation_id} Planner position",
    )
    if _max_abs_difference(source_position, row_position) > SOURCE_SEMANTIC_TOLERANCE:
        raise ValueError(
            f"observation {observation_id} semantic camera center differs from Planner pose"
        )
    for key in (
        "camera_center_max_abs_error_scene_units",
        "face_basis_max_abs_error",
    ):
        try:
            error = float(semantic.get(key, float("inf")))
        except (TypeError, ValueError) as caught:
            raise ValueError(
                f"observation {observation_id} semantic {key} is invalid"
            ) from caught
        if not math.isfinite(error) or error < 0.0 or error > SOURCE_SEMANTIC_TOLERANCE:
            raise ValueError(
                f"observation {observation_id} semantic {key} is too large"
            )
    faces = semantic.get("faces")
    if (
        not isinstance(faces, list)
        or tuple(face.get("face_name") for face in faces if isinstance(face, Mapping))
        != FACE_NAMES
    ):
        raise ValueError(
            f"observation {observation_id} source semantic faces are incomplete"
        )
    rotations: dict[str, list[list[float]]] = {}
    hash_rows = []
    for face in faces:
        assert isinstance(face, Mapping)
        face_name = str(face["face_name"])
        rotations[face_name] = _matrix3(
            face.get("T_world_from_cam_rotation"),
            f"observation {observation_id} semantic face {face_name} rotation",
        )
        rotation = rotations[face_name]
        orthonormal_error = max(
            abs(
                sum(
                    rotation[inner][row] * rotation[inner][column] for inner in range(3)
                )
                - (1.0 if row == column else 0.0)
            )
            for row in range(3)
            for column in range(3)
        )
        determinant = (
            rotation[0][0]
            * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
            - rotation[0][1]
            * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
            + rotation[0][2]
            * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
        )
        if max(orthonormal_error, abs(determinant - 1.0)) > BASIS_TOLERANCE:
            raise ValueError(
                f"observation {observation_id} source basis for face {face_name} "
                "is not right-handed orthonormal"
            )
        face_center = _finite_vector(
            face.get("camera_center_scene_units"),
            3,
            f"observation {observation_id} semantic face {face_name} center",
        )
        if (
            _max_abs_difference(face_center, source_position)
            > SOURCE_SEMANTIC_TOLERANCE
        ):
            raise ValueError(
                f"observation {observation_id} semantic face {face_name} center differs"
            )
        for key in (
            "camera_center_max_abs_error_scene_units",
            "basis_max_abs_error",
        ):
            try:
                error = float(face.get(key, float("inf")))
            except (TypeError, ValueError) as caught:
                raise ValueError(
                    f"observation {observation_id} semantic face {face_name} {key} is invalid"
                ) from caught
            if (
                not math.isfinite(error)
                or error < 0.0
                or error > SOURCE_SEMANTIC_TOLERANCE
            ):
                raise ValueError(
                    f"observation {observation_id} semantic face {face_name} {key} is too large"
                )
        hash_rows.append(
            {
                "face_name": face_name,
                "T_world_from_cam_rotation": rotation,
            }
        )
    basis_sha256 = hashlib.sha256(
        json.dumps(hash_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return source_position, rotations, basis_sha256


def _verify_replay_source(
    actual: Any, row: Mapping[str, Any], metrics_sha256: str, label: str
) -> None:
    if actual != _expected_replay_source(row, metrics_sha256):
        raise ValueError(f"{label} replay_source differs from plan")


def _verify_asset_record(record: Any, root: Path, label: str) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} record is missing")
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or not relative.parts:
        raise ValueError(f"{label} path must be relative to its manifest")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} path escapes its manifest root") from error
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if path.stat().st_size != int(record.get("bytes", -1)) or _sha256(
        path
    ) != record.get("sha256"):
        raise ValueError(f"{label} changed")
    return path


def _asset_records(value: Any, label: str = "root") -> list[tuple[Any, str]]:
    records: list[tuple[Any, str]] = []
    if isinstance(value, Mapping):
        if {"path", "bytes", "sha256"}.issubset(value):
            records.append((value, label))
        else:
            for key, child in value.items():
                records.extend(_asset_records(child, f"{label}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            records.extend(_asset_records(child, f"{label}[{index}]"))
    return records


def _verify_raw_assets(raw: Mapping[str, Any], root: Path) -> None:
    records = _asset_records(raw)
    if len(records) != 42:
        raise ValueError(
            f"raw capture manifest must reference exactly 42 assets, found {len(records)}"
        )
    seen: set[Path] = set()
    for record, location in records:
        path = _verify_asset_record(record, root, f"raw capture asset {location}")
        if path in seen:
            raise ValueError(f"raw capture asset is referenced more than once: {path}")
        seen.add(path)


def _verify_canonical_assets(
    canonical: Mapping[str, Any], root: Path
) -> dict[str, str]:
    faces = canonical.get("faces")
    if not isinstance(faces, list) or len(faces) != 6:
        raise ValueError("canonical manifest must contain six faces")
    rgb_hashes: dict[str, str] = {}
    seen: set[Path] = set()
    for face in faces:
        if not isinstance(face, Mapping):
            raise ValueError("canonical face must be an object")
        face_name = str(face.get("face_name", ""))
        assets = face.get("assets")
        if not isinstance(assets, Mapping) or set(assets) != {
            "rgb_uint8",
            "depth_range_m",
            "valid_mask",
        }:
            raise ValueError(
                f"canonical face {face_name} must reference exactly three assets"
            )
        for field_name, record in assets.items():
            path = _verify_asset_record(
                record, root, f"canonical asset {face_name}.{field_name}"
            )
            if path in seen:
                raise ValueError(
                    f"canonical asset is referenced more than once: {path}"
                )
            seen.add(path)
        rgb_record = assets["rgb_uint8"]
        assert isinstance(rgb_record, Mapping)
        rgb_hashes[face_name] = str(rgb_record["sha256"])
    if len(seen) != 18:
        raise ValueError("canonical manifest must reference exactly 18 unique assets")
    return rgb_hashes


def _rotation_error(
    transform: Sequence[Sequence[float]], expected: Sequence[Sequence[float]]
) -> float:
    return max(
        abs(float(transform[row][column]) - float(expected[row][column]))
        for row in range(3)
        for column in range(3)
    )


def _contract_to_pioneer(
    transform: Sequence[Sequence[float]], scene_units_per_meter: float
) -> tuple[list[list[float]], list[float]]:
    rotation = [
        [float(value) for value in transform[0][:3]],
        [float(value) for value in transform[2][:3]],
        [-float(value) for value in transform[1][:3]],
    ]
    position = [
        float(transform[0][3]) * scene_units_per_meter,
        float(transform[2][3]) * scene_units_per_meter,
        -float(transform[1][3]) * scene_units_per_meter,
    ]
    return rotation, position


def _verify_capture_closure(
    *,
    row: Mapping[str, Any],
    receipt: Mapping[str, Any],
    request_path: Path,
    raw_manifest_path: Path,
    canonical_manifest_path: Path,
    metrics_sha256: str,
    scene_units_per_meter: float,
) -> dict[str, Any]:
    observation_id = int(row["observation_id"])
    expected_request_id = row.get("replay_request_id")
    expected_frame_id = row.get("replay_frame_id")
    (
        expected_source_position,
        source_face_rotations,
        source_basis_sha256,
    ) = _source_semantic_authority(row)
    receipt_semantic = receipt.get("source_semantic_validation")
    plan_semantic = row["source_semantic_validation"]
    assert isinstance(plan_semantic, Mapping)
    if (
        not isinstance(receipt_semantic, Mapping)
        or receipt_semantic.get("schema_version") != SOURCE_SEMANTIC_SCHEMA
        or receipt_semantic.get("result") != "PASS"
        or receipt_semantic.get("basis_sha256") != source_basis_sha256
    ):
        raise ValueError(
            f"observation {observation_id} receipt source basis differs from plan"
        )
    for key in (
        "camera_center_max_abs_error_scene_units",
        "face_basis_max_abs_error",
    ):
        try:
            actual_error = float(receipt_semantic.get(key, float("inf")))
            expected_error = float(plan_semantic.get(key, float("inf")))
        except (TypeError, ValueError) as caught:
            raise ValueError(
                f"observation {observation_id} receipt source basis evidence is invalid"
            ) from caught
        if (
            not math.isfinite(actual_error)
            or not math.isfinite(expected_error)
            or abs(actual_error - expected_error) > 1e-12
        ):
            raise ValueError(
                f"observation {observation_id} receipt source basis evidence differs from plan"
            )
    expected_ue_position = _finite_vector(
        row.get("ue_position_cm"), 3, f"observation {observation_id} UE position"
    )
    request = _read_json(request_path)
    if request.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError(f"observation {observation_id} request schema is invalid")
    for key, expected in (
        ("request_id", expected_request_id),
        ("frame_id", expected_frame_id),
        ("position_ue_cm", row.get("ue_position_cm")),
    ):
        if request.get(key) != expected:
            raise ValueError(
                f"observation {observation_id} request {key} differs from plan"
            )
    if request.get("scenario") != "hkust":
        raise ValueError(f"observation {observation_id} request scenario is not HKUST")
    _verify_replay_source(
        request.get("replay_source"),
        row,
        metrics_sha256,
        f"observation {observation_id} request",
    )

    actual_timestamp = receipt.get("ue5_actual_capture_timestamp_ns")
    raw = _read_json(raw_manifest_path)
    if raw.get("schema_version") != RAW_SCHEMA:
        raise ValueError(f"observation {observation_id} raw capture schema is invalid")
    for key, expected in (
        ("request_id", expected_request_id),
        ("frame_id", expected_frame_id),
        ("capture_timestamp_ns", actual_timestamp),
        ("requested_position_ue_cm", row.get("ue_position_cm")),
    ):
        if raw.get(key) != expected:
            raise ValueError(
                f"observation {observation_id} raw {key} differs from plan/receipt"
            )
    _verify_replay_source(
        raw.get("replay_source"),
        row,
        metrics_sha256,
        f"observation {observation_id} raw capture",
    )
    measured_ue_position = _finite_vector(
        raw.get("shared_optical_center_ue_cm"),
        3,
        f"observation {observation_id} raw UE position",
    )
    ue_position_error = _max_abs_difference(measured_ue_position, expected_ue_position)
    if ue_position_error > UE_POSITION_TOLERANCE_CM:
        raise ValueError(
            f"observation {observation_id} raw UE position differs from request"
        )
    raw_faces = raw.get("faces")
    if (
        not isinstance(raw_faces, list)
        or tuple(
            face.get("face_name") for face in raw_faces if isinstance(face, Mapping)
        )
        != FACE_NAMES
    ):
        raise ValueError(
            f"observation {observation_id} raw capture lacks canonical faces"
        )
    raw_basis_error = 0.0
    raw_position_error = 0.0
    for face in raw_faces:
        if not isinstance(face, Mapping):
            raise ValueError(f"observation {observation_id} raw face is invalid")
        face_name = str(face["face_name"])
        if (
            face.get("request_id") != expected_request_id
            or face.get("frame_id") != expected_frame_id
        ):
            raise ValueError(
                f"observation {observation_id} raw face {face_name} identity differs"
            )
        if face.get("capture_timestamp_ns") != actual_timestamp:
            raise ValueError(
                f"observation {observation_id} raw face {face_name} timestamp differs"
            )
        face_transform = _matrix4(
            face.get("T_pioneer_world_from_cam"),
            f"observation {observation_id} raw face {face_name} transform",
        )
        basis_error = _rotation_error(face_transform, source_face_rotations[face_name])
        position_error = _max_abs_difference(
            [face_transform[index][3] for index in range(3)], expected_source_position
        )
        raw_basis_error = max(raw_basis_error, basis_error)
        raw_position_error = max(raw_position_error, position_error)
        if basis_error > BASIS_TOLERANCE:
            raise ValueError(
                f"raw face {face_name} basis differs from plan source basis"
            )
        if position_error > POSITION_TOLERANCE_SCENE_UNITS:
            raise ValueError(f"raw face {face_name} source position differs from plan")
    _verify_raw_assets(raw, raw_manifest_path.parent)

    canonical = _read_json(canonical_manifest_path)
    if canonical.get("schema_version") != CANONICAL_SCHEMA:
        raise ValueError(f"observation {observation_id} canonical schema is invalid")
    for key, expected in (
        ("request_id", expected_request_id),
        ("frame_id", expected_frame_id),
        ("capture_timestamp_ns", actual_timestamp),
    ):
        if canonical.get(key) != expected:
            raise ValueError(f"observation {observation_id} canonical {key} differs")
    if tuple(canonical.get("face_names") or ()) != FACE_NAMES:
        raise ValueError(f"observation {observation_id} canonical face_names differ")
    provenance = canonical.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(
            f"observation {observation_id} canonical provenance is missing"
        )
    _verify_replay_source(
        provenance.get("replay_source"),
        row,
        metrics_sha256,
        f"observation {observation_id} canonical capture",
    )
    contract_position = _finite_vector(
        canonical.get("position_world_m"),
        3,
        f"observation {observation_id} canonical position",
    )
    canonical_source_position = [
        contract_position[0] * scene_units_per_meter,
        contract_position[2] * scene_units_per_meter,
        -contract_position[1] * scene_units_per_meter,
    ]
    canonical_position_error = _max_abs_difference(
        canonical_source_position, expected_source_position
    )
    if canonical_position_error > POSITION_TOLERANCE_SCENE_UNITS:
        raise ValueError(
            f"observation {observation_id} canonical source position differs"
        )
    canonical_faces = canonical.get("faces")
    assert isinstance(canonical_faces, list)
    if (
        tuple(
            face.get("face_name")
            for face in canonical_faces
            if isinstance(face, Mapping)
        )
        != FACE_NAMES
    ):
        raise ValueError(f"observation {observation_id} canonical faces differ")
    canonical_basis_error = 0.0
    for face in canonical_faces:
        if not isinstance(face, Mapping):
            raise ValueError(f"observation {observation_id} canonical face is invalid")
        face_name = str(face["face_name"])
        if (
            face.get("request_id") != expected_request_id
            or face.get("frame_id") != expected_frame_id
        ):
            raise ValueError(
                f"observation {observation_id} canonical face {face_name} identity differs"
            )
        if face.get("capture_timestamp_ns") != actual_timestamp:
            raise ValueError(
                f"observation {observation_id} canonical face {face_name} timestamp differs"
            )
        face_transform = _matrix4(
            face.get("T_world_from_cam"),
            f"observation {observation_id} canonical face {face_name} transform",
        )
        pioneer_rotation, pioneer_position = _contract_to_pioneer(
            face_transform, scene_units_per_meter
        )
        basis_error = _rotation_error(
            pioneer_rotation, source_face_rotations[face_name]
        )
        position_error = _max_abs_difference(pioneer_position, expected_source_position)
        canonical_basis_error = max(canonical_basis_error, basis_error)
        canonical_position_error = max(canonical_position_error, position_error)
        if basis_error > BASIS_TOLERANCE:
            raise ValueError(
                f"canonical face {face_name} basis differs from plan source basis"
            )
        if position_error > POSITION_TOLERANCE_SCENE_UNITS:
            raise ValueError(
                f"canonical face {face_name} source position differs from plan"
            )
    canonical_rgb_hashes = _verify_canonical_assets(
        canonical, canonical_manifest_path.parent
    )
    if (
        receipt.get("sources", {}).get("canonical_rgb_array_sha256")
        != canonical_rgb_hashes
    ):
        raise ValueError(
            f"observation {observation_id} canonical RGB hashes differ from receipt"
        )

    return {
        "measured_ue_position": measured_ue_position,
        "measured_source_position": canonical_source_position,
        "ue_position_error": ue_position_error,
        "source_position_error": max(raw_position_error, canonical_position_error),
        "basis_error": max(raw_basis_error, canonical_basis_error),
    }


def _verify_source(plan: Mapping[str, Any]) -> None:
    source = plan.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("replay plan lacks source provenance")
    _verify_file_record(source.get("config"), "source config")
    _verify_file_record(source.get("metrics"), "source metrics")
    preview = source.get("original_preview")
    _verify_file_record(preview, "source original preview")
    lmdb = source.get("lmdb")
    if not isinstance(lmdb, Mapping):
        raise ValueError("source LMDB provenance is missing")
    data_file = Path(str(lmdb.get("path", ""))).expanduser().resolve() / "data.mdb"
    if not data_file.is_file() or _sha256(data_file) != lmdb.get("data_sha256"):
        raise ValueError("source LMDB data SHA256 changed")
    capture_root = Path(str(source.get("capture_root", ""))).expanduser().resolve()
    if not capture_root.is_dir():
        raise FileNotFoundError(f"source capture root is missing: {capture_root}")
    rows = plan.get("observations")
    if not isinstance(rows, list):
        raise ValueError("replay plan observations must be a list")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("replay plan observation must be an object")
        artifacts = row.get("source_artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("source observation artifact hashes are missing")
        records = list(artifacts.get("frames") or ()) + list(
            artifacts.get("images") or ()
        )
        marker = artifacts.get("transaction_marker")
        if marker is not None:
            records.append(marker)
        for record in records:
            _verify_asset_record(record, capture_root, "source observation artifact")


def _tree_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{int(row['observation_id']):06d}".encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row["erp"]["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def finalize_replay(
    *, plan_path: Path, processed_root: Path, run_manifest_path: Path
) -> dict[str, Any]:
    plan_path = Path(plan_path).expanduser().resolve()
    processed_root = Path(processed_root).expanduser().resolve()
    run_manifest_path = Path(run_manifest_path).expanduser().resolve()
    plan = _read_json(plan_path)
    if plan.get("schema_version") != "pan29.ue5-postrun-replay.v1":
        raise ValueError("unexpected PAN-29 replay plan schema")
    if tuple(plan.get("selected_bundle_ids") or ()) != OBSERVATION_IDS:
        raise ValueError("PAN-29 replay plan does not select the exact five IDs")
    if plan.get("planner_input_unchanged") is not True:
        raise ValueError("PAN-29 replay plan must keep Planner input unchanged")
    _verify_source(plan)
    plan_sha = _sha256(plan_path)
    source = plan.get("source")
    assert isinstance(source, Mapping)
    metrics_record = source.get("metrics")
    if (
        not isinstance(metrics_record, Mapping)
        or len(str(metrics_record.get("sha256", ""))) != 64
    ):
        raise ValueError("replay plan lacks a valid source metrics SHA256")
    metrics_sha256 = str(metrics_record["sha256"])
    try:
        scene_units_per_meter = float(source.get("scene_units_per_meter", 0.0))
    except (TypeError, ValueError) as error:
        raise ValueError("replay plan scene_units_per_meter is invalid") from error
    if scene_units_per_meter <= 0.0:
        raise ValueError("replay plan scene_units_per_meter must be positive")

    plan_rows = plan.get("observations")
    assert isinstance(plan_rows, list)
    by_id = {
        int(row["observation_id"]): row for row in plan_rows if isinstance(row, Mapping)
    }
    if tuple(by_id) != OBSERVATION_IDS or len(by_id) != len(plan_rows):
        raise ValueError(
            "PAN-29 replay plan observation IDs are duplicate or out of order"
        )

    receipts = []
    actual_timestamps: set[int] = set()
    for observation_id in OBSERVATION_IDS:
        directory = processed_root / f"{observation_id:06d}"
        receipt_path = directory / "receipt.json"
        receipt = _read_json(receipt_path)
        row = by_id[observation_id]
        if (
            receipt.get("schema_version") != "pan29.ue5-replay-receipt.v1"
            or receipt.get("result") != "PASS"
            or receipt.get("observation_id") != observation_id
        ):
            raise ValueError(f"observation {observation_id} receipt is not PASS")
        for key in (
            "request_id",
            "frame_id",
            "source_capture_timestamp_ns",
            "source_capture_timestamp_utc",
            "planner_position_scene_units",
            "requested_position_ue_cm",
            "within_pan13_conservative_fly_volume",
        ):
            expected_key = {
                "request_id": "replay_request_id",
                "frame_id": "replay_frame_id",
                "requested_position_ue_cm": "ue_position_cm",
            }.get(key, key)
            if receipt.get(key) != row.get(expected_key):
                raise ValueError(
                    f"observation {observation_id} {key} differs from plan"
                )
        if (
            receipt.get("planner_input_unchanged") is not True
            or receipt.get("ue5_role") != "post_run_visualization_only"
        ):
            raise ValueError(f"observation {observation_id} has an invalid UE5 role")
        if tuple(receipt.get("face_names") or ()) != FACE_NAMES:
            raise ValueError(f"observation {observation_id} lacks canonical faces")
        if (
            float(receipt.get("position_max_abs_error_scene_units", 1.0))
            > POSITION_TOLERANCE_SCENE_UNITS
        ):
            raise ValueError(
                f"observation {observation_id} position error is too large"
            )
        if (
            float(receipt.get("ue_position_max_abs_error_cm", 1.0))
            > UE_POSITION_TOLERANCE_CM
        ):
            raise ValueError(
                f"observation {observation_id} UE position error is too large"
            )
        if float(receipt.get("face_basis_max_abs_error", 1.0)) > BASIS_TOLERANCE:
            raise ValueError(f"observation {observation_id} basis error is too large")
        actual_timestamp = receipt.get("ue5_actual_capture_timestamp_ns")
        if (
            type(actual_timestamp) is not int
            or actual_timestamp <= 0
            or actual_timestamp in actual_timestamps
        ):
            raise ValueError(
                "UE5 actual capture timestamps must be positive and unique"
            )
        actual_timestamps.add(actual_timestamp)
        sources = receipt.get("sources")
        if not isinstance(sources, Mapping):
            raise ValueError(
                f"observation {observation_id} source receipts are missing"
            )
        plan_record = sources.get("plan")
        if (
            not isinstance(plan_record, Mapping)
            or Path(str(plan_record.get("path", ""))).expanduser().resolve()
            != plan_path
            or plan_record.get("sha256") != plan_sha
        ):
            raise ValueError(
                f"observation {observation_id} is bound to another replay plan"
            )
        capture_directory = processed_root.parent / "captures" / f"{observation_id:06d}"
        request_path = _verify_file_record(
            sources.get("request"),
            f"observation {observation_id} request",
            expected_path=capture_directory / "request.json",
        )
        raw_manifest_path = _verify_file_record(
            sources.get("raw_manifest"),
            f"observation {observation_id} raw_manifest",
            expected_path=capture_directory / "raw_bundle" / "manifest.json",
        )
        canonical_manifest_path = _verify_file_record(
            sources.get("canonical_manifest"),
            f"observation {observation_id} canonical_manifest",
            expected_path=capture_directory / "canonical_bundle" / "manifest.json",
        )
        closure = _verify_capture_closure(
            row=row,
            receipt=receipt,
            request_path=request_path,
            raw_manifest_path=raw_manifest_path,
            canonical_manifest_path=canonical_manifest_path,
            metrics_sha256=metrics_sha256,
            scene_units_per_meter=scene_units_per_meter,
        )
        for receipt_key, closure_key, tolerance in (
            ("measured_position_ue_cm", "measured_ue_position", 1e-9),
            ("measured_position_scene_units", "measured_source_position", 1e-9),
        ):
            actual = _finite_vector(
                receipt.get(receipt_key),
                3,
                f"observation {observation_id} receipt {receipt_key}",
            )
            if _max_abs_difference(actual, closure[closure_key]) > tolerance:
                raise ValueError(
                    f"observation {observation_id} {receipt_key} differs from manifests"
                )
        for receipt_key, closure_key in (
            ("ue_position_max_abs_error_cm", "ue_position_error"),
            ("position_max_abs_error_scene_units", "source_position_error"),
            ("face_basis_max_abs_error", "basis_error"),
        ):
            if abs(float(receipt[receipt_key]) - float(closure[closure_key])) > 1e-9:
                raise ValueError(
                    f"observation {observation_id} {receipt_key} was not recomputed"
                )
        erp = receipt.get("erp")
        if not isinstance(erp, Mapping):
            raise ValueError(f"observation {observation_id} ERP record is missing")
        if erp.get("path") != "erp.png":
            raise ValueError(f"observation {observation_id} ERP path must be erp.png")
        erp_path = directory / str(erp.get("path", ""))
        if (
            not erp_path.is_file()
            or _sha256(erp_path) != erp.get("sha256")
            or int(erp.get("width", -1)) != 2 * int(erp.get("height", -2))
            or int(erp.get("height", -1)) < 2
            or erp.get("mode") != "RGB"
            or erp.get("byte_deterministic_reprojection") is not True
        ):
            raise ValueError(f"observation {observation_id} ERP validation failed")
        receipts.append(
            {
                "observation_id": observation_id,
                "receipt": {
                    "path": str(receipt_path),
                    "sha256": _sha256(receipt_path),
                },
                "erp": {**dict(erp), "path": str(erp_path)},
                "source_capture_timestamp_ns": receipt["source_capture_timestamp_ns"],
                "ue5_actual_capture_timestamp_ns": actual_timestamp,
                "within_pan13_conservative_fly_volume": receipt[
                    "within_pan13_conservative_fly_volume"
                ],
            }
        )

    run_manifest = _read_key_values(run_manifest_path)
    required_run_keys = (
        "git_commit",
        "replay_plan_sha256",
        "ue_project_sha256",
        "ue_default_engine_sha256",
        "ue_level_sha256",
        "capture_config_sha256",
    )
    for key in required_run_keys:
        expected_length = 40 if key == "git_commit" else 64
        if len(run_manifest.get(key, "")) != expected_length:
            raise ValueError(f"run manifest lacks a full {key}")
    if run_manifest["replay_plan_sha256"] != plan_sha:
        raise ValueError("run manifest replay_plan_sha256 differs from the replay plan")

    return {
        "schema_version": "pan29.ue5-postrun-replay-result.v1",
        "task": "PAN-29",
        "result": "PASS",
        "artifact_role": "post_run_visualization_only",
        "planner_input_unchanged": True,
        "observation_ids": list(OBSERVATION_IDS),
        "replay_count": len(receipts),
        "out_of_pan13_fly_policy_ids": [
            row["observation_id"]
            for row in receipts
            if not row["within_pan13_conservative_fly_volume"]
        ],
        "source_scientific_run_commit": source.get("scientific_run_commit"),
        "implementation_commit": run_manifest["git_commit"],
        "run_manifest": {
            "path": str(run_manifest_path),
            "sha256": _sha256(run_manifest_path),
            "ue_project_sha256": run_manifest["ue_project_sha256"],
            "ue_default_engine_sha256": run_manifest["ue_default_engine_sha256"],
            "ue_level_sha256": run_manifest["ue_level_sha256"],
            "capture_config_sha256": run_manifest["capture_config_sha256"],
        },
        "plan": {"path": str(plan_path), "sha256": plan_sha},
        "erp_tree_sha256": _tree_hash(receipts),
        "replays": receipts,
        "claims": {
            "same_observation_id_pose_source_timestamp": True,
            "same_wall_clock_capture_time": False,
            "rgb_pixel_parity": False,
            "depth_parity": False,
            "coverage_parity": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite final replay receipt: {output}")
    result = finalize_replay(
        plan_path=args.plan,
        processed_root=args.processed_root,
        run_manifest_path=args.run_manifest,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(
        json.dumps(
            {"result": "PASS", "replay_count": len(OBSERVATION_IDS)}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
