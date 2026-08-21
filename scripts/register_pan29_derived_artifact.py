#!/usr/bin/env python3
"""Register PAN-29 as a derived visualization receipt, never an experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence


OBSERVATION_IDS = (0, 5, 10, 14, 19)
FACE_NAMES = ("front", "back", "left", "right", "up", "down")
ARTIFACT_ROLE = "post_run_visualization_only"
START_MARKER = "<!-- PIONEER_DERIVED_ARTIFACTS_START -->"
END_MARKER = "<!-- PIONEER_DERIVED_ARTIFACTS_END -->"
DERIVED_SECTION_TITLE = "## Derived visualization artifacts (not scientific experiment records)"
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _file_tree_sha256(paths: Sequence[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _full_hash(value: Any, label: str, *, length: int = 64) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{label} must contain {length} hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    return value.lower()


def _safe_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValueError(f"{label} contains unsupported characters")
    return value


def _resolved_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is missing")
    return Path(value).expanduser().resolve()


def _verify_path(
    path: Path,
    expected_sha256: Any,
    label: str,
    *,
    expected_size: Any | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    expected_hash = _full_hash(expected_sha256, f"{label} SHA256")
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(f"{label} SHA256 changed")
    size = path.stat().st_size
    if expected_size is not None and (type(expected_size) is not int or expected_size != size):
        raise ValueError(f"{label} byte size changed")
    return {"path": str(path), "sha256": actual_hash, "size_bytes": size}


def _verify_file_record(record: Any, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} record is missing")
    path = _resolved_path(record.get("path"), label)
    expected_size = record.get("size_bytes")
    return path, _verify_path(
        path,
        record.get("sha256"),
        label,
        expected_size=expected_size,
    )


def _read_key_values(path: Path, *, label: str = "key-value file") -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise ValueError(f"invalid or duplicate {label} line: {line!r}")
        values[key] = value
    return values


def _utc_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} is not a valid timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{label} must be expressed in UTC")
    return parsed


def _validate_registry_source(
    registry: Mapping[str, Any],
    *,
    source_id: str,
    scientific_commit: str,
    core_artifacts: Sequence[Mapping[str, Any]],
    frame_artifacts: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], str]:
    experiments = registry.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("registry experiments must be a list")
    if registry.get("record_count") != len(experiments):
        raise ValueError("registry record_count differs from experiments length")
    matches = [
        row
        for row in experiments
        if isinstance(row, Mapping) and row.get("experiment_id") == source_id
    ]
    if len(matches) != 1:
        raise ValueError("source experiment must occur exactly once in the registry")
    source = matches[0]
    if source.get("status") != "PASS":
        raise ValueError("source experiment is not a PASS scientific record")
    provenance = source.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("source experiment provenance is missing")
    registered_commit = provenance.get("scientific_run_commit_full") or provenance.get(
        "scientific_run_commit"
    )
    if registered_commit != scientific_commit:
        raise ValueError("source scientific run commit differs from the registry")
    artifact_set_id = source.get("artifacts")
    if not isinstance(artifact_set_id, str) or not artifact_set_id:
        raise ValueError("source experiment does not reference an artifact set")
    artifact_sets = registry.get("artifact_sets")
    if not isinstance(artifact_sets, Mapping):
        raise ValueError("registry artifact_sets must be an object")
    artifact_set = artifact_sets.get(artifact_set_id)
    if not isinstance(artifact_set, Mapping):
        raise ValueError("source experiment artifact set is missing")
    if artifact_set.get("experiment_id") != source_id:
        raise ValueError("source artifact set belongs to another experiment")
    artifacts = artifact_set.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("source artifact set lacks artifacts")
    registered_rows = [row for row in artifacts.values() if isinstance(row, Mapping)]
    missing_core = []
    for expected in core_artifacts:
        if not any(
            row.get("sha256") == expected.get("sha256")
            and row.get("size_bytes") == expected.get("size_bytes")
            for row in registered_rows
        ):
            missing_core.append(str(expected.get("path")))
    if missing_core:
        raise ValueError(
            "source core artifacts are absent from its registered artifact set: "
            + ", ".join(sorted(missing_core))
        )
    if len(frame_artifacts) != 140:
        raise ValueError("source frame Registry evidence must contain exactly 140 PT files")
    registered_frames: dict[Path, list[Mapping[str, Any]]] = {}
    for row in registered_rows:
        path_value = row.get("path")
        if not isinstance(path_value, str) or not Path(path_value).is_absolute():
            continue
        path = Path(path_value).expanduser().resolve()
        registered_frames.setdefault(path, []).append(row)
    missing_frames = []
    for expected in frame_artifacts:
        expected_path = Path(str(expected.get("path", ""))).expanduser().resolve()
        if not any(
            row.get("sha256") == expected.get("sha256")
            and row.get("size_bytes") == expected.get("size_bytes")
            for row in registered_frames.get(expected_path, ())
        ):
            missing_frames.append(str(expected_path))
    if missing_frames:
        raise ValueError(
            "source 140-PT frame evidence differs from its registered artifact set: "
            + ", ".join(missing_frames[:5])
        )
    return source, artifact_set_id


def _verify_source_plan(
    plan: Mapping[str, Any], plan_path: Path
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    if (
        plan.get("schema_version") != "pan29.ue5-postrun-replay.v1"
        or plan.get("task") != "PAN-29"
        or plan.get("artifact_role") != ARTIFACT_ROLE
        or plan.get("planner_input_unchanged") is not True
        or plan.get("ue5_render_is_postrun_visualization_only") is not True
        or tuple(plan.get("selected_bundle_ids") or ()) != OBSERVATION_IDS
    ):
        raise ValueError("PAN-29 replay plan contract is invalid")
    source = plan.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("PAN-29 replay plan lacks source provenance")
    source_id = _safe_id(source.get("registry_experiment_id"), "source experiment ID")
    scientific_commit = _full_hash(
        source.get("scientific_run_commit"),
        "source scientific run commit",
        length=40,
    )
    if (
        source.get("depth_source") != "GT"
        or source.get("scene") != "HKUST"
        or source.get("planner_state_mode") != "position_only"
    ):
        raise ValueError("PAN-29 replay plan is not based on the HKUST GT position-only run")

    source_artifacts: dict[str, Any] = {}
    source_hashes: dict[str, Any] = {}
    for key, label in (
        ("run_manifest", "source run manifest"),
        ("config", "source config"),
        ("metrics", "source metrics"),
    ):
        _, artifact = _verify_file_record(source.get(key), label)
        source_artifacts[key] = artifact
        source_hashes[f"{key}_sha256"] = artifact["sha256"]
    source_run_values = _read_key_values(Path(source_artifacts["run_manifest"]["path"]))
    if (
        source_run_values.get("git_commit") != scientific_commit
        or source_run_values.get("config_sha256")
        != source_artifacts["config"]["sha256"]
        or source_run_values.get("debug_profile") != "pioneer-20"
        or source_run_values.get("expected_observations") != "20"
        or source_run_values.get("expected_real_face_renders") != "120"
    ):
        raise ValueError("source run manifest differs from the frozen GT 20obs run")

    lmdb = source.get("lmdb")
    if not isinstance(lmdb, Mapping):
        raise ValueError("source LMDB record is missing")
    lmdb_root = _resolved_path(lmdb.get("path"), "source LMDB")
    lmdb_artifact = _verify_path(
        lmdb_root / "data.mdb", lmdb.get("data_sha256"), "source LMDB data.mdb"
    )
    source_artifacts["lmdb_data"] = lmdb_artifact
    source_hashes["lmdb_data_sha256"] = lmdb_artifact["sha256"]

    original_preview = source.get("original_preview")
    if not isinstance(original_preview, Mapping):
        raise ValueError("source original preview record is missing")
    preview_artifact = _verify_path(
        _resolved_path(original_preview.get("path"), "source original preview"),
        original_preview.get("sha256"),
        "source original preview",
    )
    preview_sidecar_artifact = _verify_path(
        _resolved_path(
            original_preview.get("sidecar_path"), "source original preview sidecar"
        ),
        original_preview.get("sidecar_sha256"),
        "source original preview sidecar",
    )
    source_artifacts["original_preview"] = preview_artifact
    source_artifacts["original_preview_sidecar"] = preview_sidecar_artifact
    source_hashes["original_preview_sha256"] = preview_artifact["sha256"]
    source_hashes["original_preview_sidecar_sha256"] = preview_sidecar_artifact[
        "sha256"
    ]
    capture_root = _resolved_path(source.get("capture_root"), "source capture root")
    if not capture_root.is_dir():
        raise FileNotFoundError(f"source capture root is missing: {capture_root}")
    frame_root = capture_root / "frames"
    image_root = capture_root / "imgs"
    frame_paths = [
        frame_root / f"{observation_id:06d}" / filename
        for observation_id in range(20)
        for filename in ("bundle.pt", *(f"{face}.pt" for face in FACE_NAMES))
    ]
    image_paths = [
        image_root / f"{observation_id:06d}" / f"{face}.png"
        for observation_id in range(20)
        for face in FACE_NAMES
    ]
    frame_registry_artifacts = []
    for index, path in enumerate(frame_paths):
        if not path.is_file():
            raise FileNotFoundError(f"source frame tree is incomplete at index {index}: {path}")
        frame_registry_artifacts.append(
            {"path": str(path.resolve()), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
        )
    for index, path in enumerate(image_paths):
        if not path.is_file():
            raise FileNotFoundError(f"source image tree is incomplete at index {index}: {path}")
    frames_tree_sha256 = _file_tree_sha256(frame_paths, frame_root)
    images_tree_sha256 = _file_tree_sha256(image_paths, image_root)
    source_hashes.update(
        {
            "frame_file_count": len(frame_paths),
            "frames_tree_sha256": frames_tree_sha256,
            "image_file_count": len(image_paths),
            "images_tree_sha256": images_tree_sha256,
            "frame_registry_evidence": "140 live PT files match exact registered path, SHA256, and size",
            "image_registry_evidence": "120 live PNG tree matches the registered original-preview sidecar aggregate",
        }
    )
    original_sidecar = _read_json(Path(preview_sidecar_artifact["path"]))
    original_sources = original_sidecar.get("sources")
    if not isinstance(original_sources, Mapping):
        raise ValueError("source original preview sidecar lacks source provenance")
    original_metrics = original_sources.get("metrics")
    original_lmdb = original_sources.get("lmdb")
    if (
        original_sources.get("capture_image_count") != len(image_paths)
        or _resolved_path(
            original_sources.get("capture_images_root"),
            "source original preview image root",
        )
        != image_root.resolve()
        or original_sources.get("capture_images_tree_sha256")
        != images_tree_sha256
        or not isinstance(original_metrics, Mapping)
        or _resolved_path(original_metrics.get("path"), "source preview metrics")
        != Path(source_artifacts["metrics"]["path"])
        or original_metrics.get("sha256") != source_artifacts["metrics"]["sha256"]
        or not isinstance(original_lmdb, Mapping)
        or _resolved_path(original_lmdb.get("path"), "source preview LMDB")
        != lmdb_root
        or original_lmdb.get("data_sha256") != lmdb_artifact["sha256"]
    ):
        raise ValueError("source original preview sidecar differs from the source trees")
    rows = plan.get("observations")
    if not isinstance(rows, list) or [
        row.get("observation_id") for row in rows if isinstance(row, Mapping)
    ] != list(OBSERVATION_IDS):
        raise ValueError("PAN-29 replay plan observations are missing or reordered")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("PAN-29 replay plan observation is not an object")
        observation_id = int(row["observation_id"])
        artifacts = row.get("source_artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError(f"source observation {observation_id} artifacts are missing")
        frames = artifacts.get("frames")
        images = artifacts.get("images")
        if not isinstance(frames, list) or len(frames) != 7:
            raise ValueError(f"source observation {observation_id} must bind seven PT files")
        if not isinstance(images, list) or len(images) != 6:
            raise ValueError(f"source observation {observation_id} must bind six PNG files")
        records: list[Any] = [*frames, *images]
        marker = artifacts.get("transaction_marker")
        if marker is not None:
            records.append(marker)
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ValueError(f"source observation {observation_id} artifact is invalid")
            relative = record.get("path")
            if not isinstance(relative, str) or not relative:
                raise ValueError(f"source observation {observation_id} artifact path is missing")
            path = (capture_root / relative).resolve()
            try:
                path.relative_to(capture_root)
            except ValueError as error:
                raise ValueError("source observation artifact escapes capture root") from error
            _verify_path(
                path,
                record.get("sha256"),
                f"source observation {observation_id} artifact {index}",
                expected_size=record.get("bytes"),
            )
    return source, {
        "source_id": source_id,
        "scientific_commit": scientific_commit,
        "source_artifacts": source_artifacts,
        "source_hashes": source_hashes,
        "source_core_registry_artifacts": list(source_artifacts.values()),
        "source_frame_registry_artifacts": frame_registry_artifacts,
        "plan": {
            "path": str(plan_path),
            "sha256": _sha256(plan_path),
            "size_bytes": plan_path.stat().st_size,
        },
        "plan_rows": {int(row["observation_id"]): row for row in rows},
    }


def _verify_replay_result(
    replay_result_path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any], dict[str, Any]]:
    replay_result_path = replay_result_path.expanduser().resolve()
    result = _read_json(replay_result_path)
    expected_claims = {
        "same_observation_id_pose_source_timestamp": True,
        "same_wall_clock_capture_time": False,
        "rgb_pixel_parity": False,
        "depth_parity": False,
        "coverage_parity": False,
    }
    if (
        result.get("schema_version") != "pan29.ue5-postrun-replay-result.v1"
        or result.get("task") != "PAN-29"
        or result.get("result") != "PASS"
        or result.get("artifact_role") != ARTIFACT_ROLE
        or result.get("planner_input_unchanged") is not True
        or tuple(result.get("observation_ids") or ()) != OBSERVATION_IDS
        or result.get("replay_count") != 5
        or tuple(result.get("out_of_pan13_fly_policy_ids") or ()) != (5, 10, 14, 19)
        or result.get("claims") != expected_claims
    ):
        raise ValueError("PAN-29 aggregate replay result contract is invalid")
    result_artifact = {
        "path": str(replay_result_path),
        "sha256": _sha256(replay_result_path),
        "size_bytes": replay_result_path.stat().st_size,
    }
    implementation_commit = _full_hash(
        result.get("implementation_commit"), "PAN-29 implementation commit", length=40
    )
    source_commit = _full_hash(
        result.get("source_scientific_run_commit"),
        "result source scientific run commit",
        length=40,
    )

    plan_path, plan_artifact = _verify_file_record(result.get("plan"), "PAN-29 replay plan")
    plan = _read_json(plan_path)
    source, plan_validation = _verify_source_plan(plan, plan_path)
    if source_commit != plan_validation["scientific_commit"]:
        raise ValueError("result source commit differs from the replay plan")

    run_manifest_path, run_manifest_artifact = _verify_file_record(
        result.get("run_manifest"), "PAN-29 run manifest"
    )
    run_values = _read_key_values(run_manifest_path)
    result_run = result["run_manifest"]
    assert isinstance(result_run, Mapping)
    if run_values.get("git_commit") != implementation_commit:
        raise ValueError("PAN-29 run manifest commit differs from the aggregate result")
    for key in (
        "ue_project_sha256",
        "ue_default_engine_sha256",
        "ue_level_sha256",
        "capture_config_sha256",
    ):
        value = _full_hash(result_run.get(key), f"PAN-29 {key}")
        if run_values.get(key) != value:
            raise ValueError(f"PAN-29 run manifest {key} differs from the result")

    replays = result.get("replays")
    if not isinstance(replays, list) or [
        row.get("observation_id") for row in replays if isinstance(row, Mapping)
    ] != list(OBSERVATION_IDS):
        raise ValueError("PAN-29 aggregate replay rows are missing or reordered")
    derived_artifacts: dict[str, dict[str, Any]] = {
        "replay_result": result_artifact,
        "replay_plan": plan_artifact,
        "run_manifest": run_manifest_artifact,
    }
    receipt_rows = []
    actual_timestamps: set[int] = set()
    tree_digest = hashlib.sha256()
    plan_rows = plan_validation["plan_rows"]
    for replay in replays:
        if not isinstance(replay, Mapping):
            raise ValueError("PAN-29 aggregate replay row is invalid")
        observation_id = int(replay["observation_id"])
        plan_row = plan_rows[observation_id]
        receipt_path, receipt_artifact = _verify_file_record(
            replay.get("receipt"), f"observation {observation_id} receipt"
        )
        receipt = _read_json(receipt_path)
        if (
            receipt.get("schema_version") != "pan29.ue5-replay-receipt.v1"
            or receipt.get("result") != "PASS"
            or receipt.get("observation_id") != observation_id
            or receipt.get("planner_input_unchanged") is not True
            or receipt.get("ue5_role") != ARTIFACT_ROLE
            or tuple(receipt.get("face_names") or ()) != FACE_NAMES
            or float(receipt.get("position_max_abs_error_scene_units", 1.0)) > 2e-4
            or float(receipt.get("face_basis_max_abs_error", 1.0)) > 1e-5
        ):
            raise ValueError(f"observation {observation_id} receipt contract is invalid")
        for receipt_key, plan_key in (
            ("source_capture_timestamp_ns", "source_capture_timestamp_ns"),
            ("planner_position_scene_units", "planner_position_scene_units"),
            ("requested_position_ue_cm", "ue_position_cm"),
            (
                "within_pan13_conservative_fly_volume",
                "within_pan13_conservative_fly_volume",
            ),
        ):
            if receipt.get(receipt_key) != plan_row.get(plan_key):
                raise ValueError(
                    f"observation {observation_id} receipt {receipt_key} differs from plan"
                )
        actual_timestamp = receipt.get("ue5_actual_capture_timestamp_ns")
        if (
            type(actual_timestamp) is not int
            or actual_timestamp <= 0
            or actual_timestamp in actual_timestamps
            or replay.get("ue5_actual_capture_timestamp_ns") != actual_timestamp
        ):
            raise ValueError("UE5 actual capture timestamps must be positive and unique")
        actual_timestamps.add(actual_timestamp)
        if replay.get("source_capture_timestamp_ns") != receipt.get(
            "source_capture_timestamp_ns"
        ):
            raise ValueError("aggregate and receipt source timestamps differ")

        sources = receipt.get("sources")
        if not isinstance(sources, Mapping):
            raise ValueError(f"observation {observation_id} receipt sources are missing")
        receipt_plan_path, _ = _verify_file_record(
            sources.get("plan"), f"observation {observation_id} receipt plan"
        )
        if receipt_plan_path != plan_path:
            raise ValueError(f"observation {observation_id} receipt refers to another plan")
        raw_path, raw_artifact = _verify_file_record(
            sources.get("raw_manifest"), f"observation {observation_id} raw manifest"
        )
        canonical_path, canonical_artifact = _verify_file_record(
            sources.get("canonical_manifest"),
            f"observation {observation_id} canonical manifest",
        )
        receipt_erp = receipt.get("erp")
        replay_erp = replay.get("erp")
        if not isinstance(receipt_erp, Mapping) or not isinstance(replay_erp, Mapping):
            raise ValueError(f"observation {observation_id} ERP record is missing")
        erp_path = _resolved_path(replay_erp.get("path"), f"observation {observation_id} ERP")
        erp_artifact = _verify_path(
            erp_path,
            replay_erp.get("sha256"),
            f"observation {observation_id} ERP",
        )
        receipt_relative_erp = (receipt_path.parent / str(receipt_erp.get("path", ""))).resolve()
        if (
            receipt_relative_erp != erp_path
            or receipt_erp.get("sha256") != erp_artifact["sha256"]
            or replay_erp.get("sha256") != erp_artifact["sha256"]
            or int(receipt_erp.get("width", -1)) != 1024
            or int(receipt_erp.get("height", -1)) != 512
            or receipt_erp.get("byte_deterministic_reprojection") is not True
        ):
            raise ValueError(f"observation {observation_id} ERP contract is invalid")
        for key, artifact in (
            (f"receipt_{observation_id:06d}", receipt_artifact),
            (f"erp_{observation_id:06d}", erp_artifact),
            (f"raw_manifest_{observation_id:06d}", raw_artifact),
            (f"canonical_manifest_{observation_id:06d}", canonical_artifact),
        ):
            derived_artifacts[key] = artifact
        receipt_rows.append(
            {
                "observation_id": observation_id,
                "source_capture_timestamp_ns": receipt["source_capture_timestamp_ns"],
                "ue5_actual_capture_timestamp_ns": actual_timestamp,
                "receipt_sha256": receipt_artifact["sha256"],
                "erp_sha256": erp_artifact["sha256"],
            }
        )
        tree_digest.update(f"{observation_id:06d}".encode("ascii"))
        tree_digest.update(b"\0")
        tree_digest.update(erp_artifact["sha256"].encode("ascii"))
        tree_digest.update(b"\n")
    if tree_digest.hexdigest() != result.get("erp_tree_sha256"):
        raise ValueError("PAN-29 ERP tree SHA256 differs from the aggregate result")
    return result, source, {
        **plan_validation,
        "implementation_commit": implementation_commit,
        "run_manifest": run_manifest_artifact,
        "result": result_artifact,
        "receipt_rows": receipt_rows,
    }, derived_artifacts


def _verify_preview(
    *,
    preview_sidecar_path: Path,
    preview_commit_path: Path,
    replay_result_path: Path,
    result: Mapping[str, Any],
    source: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    preview_sidecar_path = preview_sidecar_path.expanduser().resolve()
    preview_commit_path = preview_commit_path.expanduser().resolve()
    sidecar = _read_json(preview_sidecar_path)
    if sidecar.get("schema_version") != "pan29.preview.v1" or sidecar.get("task") != "PAN-29":
        raise ValueError("PAN-29 preview sidecar schema is invalid")
    preview_path, preview_artifact = _verify_file_record(
        sidecar.get("preview"), "PAN-29 preview"
    )
    preview_record = sidecar["preview"]
    assert isinstance(preview_record, Mapping)
    if (
        int(preview_record.get("width", -1)) <= 0
        or int(preview_record.get("height", -1)) <= 0
        or preview_record.get("mode") != "RGB"
    ):
        raise ValueError("PAN-29 preview dimensions or mode are invalid")
    sources = sidecar.get("sources")
    summary = sidecar.get("summary")
    if not isinstance(sources, Mapping) or not isinstance(summary, Mapping):
        raise ValueError("PAN-29 preview provenance is missing")
    expected_summary = {
        "scene": "HKUST",
        "source_depth": "GT",
        "source_observation_count": 20,
        "displayed_observation_ids": list(OBSERVATION_IDS),
        "planner_face_image_count": 30,
        "ue5_erp_count": 5,
        "planner_input_unchanged": True,
        "ue5_role": ARTIFACT_ROLE,
        "out_of_pan13_fly_policy_ids": [5, 10, 14, 19],
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise ValueError("PAN-29 preview summary differs from the accepted replay")
    if (
        sources.get("all_source_face_count") != 120
        or sources.get("displayed_source_face_count") != 30
        or sources.get("ue5_erp_count") != 5
    ):
        raise ValueError("PAN-29 preview artifact counts are invalid")
    for key in (
        "all_source_face_tree_sha256",
        "displayed_source_face_tree_sha256",
        "ue5_erp_tree_sha256",
    ):
        _full_hash(sources.get(key), f"preview {key}")
    for key, expected_path, expected_hash in (
        (
            "replay_result",
            replay_result_path,
            validation["result"]["sha256"],
        ),
        ("replay_plan", Path(validation["plan"]["path"]), validation["plan"]["sha256"]),
        (
            "metrics",
            Path(validation["source_artifacts"]["metrics"]["path"]),
            validation["source_artifacts"]["metrics"]["sha256"],
        ),
        (
            "original_preview",
            Path(validation["source_artifacts"]["original_preview"]["path"]),
            validation["source_artifacts"]["original_preview"]["sha256"],
        ),
    ):
        actual_path, artifact = _verify_file_record(sources.get(key), f"preview source {key}")
        if actual_path != expected_path.resolve() or artifact["sha256"] != expected_hash:
            raise ValueError(f"preview source {key} differs from PAN-29 provenance")

    commit = _read_json(preview_commit_path)
    if commit.get("schema_version") != "pan29.preview-commit.v1":
        raise ValueError("PAN-29 preview commit schema is invalid")
    if (
        _resolved_path(commit.get("preview_path"), "preview commit preview") != preview_path
        or commit.get("preview_sha256") != preview_artifact["sha256"]
        or _resolved_path(commit.get("sidecar_path"), "preview commit sidecar")
        != preview_sidecar_path
        or commit.get("sidecar_sha256") != _sha256(preview_sidecar_path)
    ):
        raise ValueError("PAN-29 preview commit does not bind the preview and sidecar")
    sidecar_artifact = {
        "path": str(preview_sidecar_path),
        "sha256": _sha256(preview_sidecar_path),
        "size_bytes": preview_sidecar_path.stat().st_size,
    }
    commit_artifact = {
        "path": str(preview_commit_path),
        "sha256": _sha256(preview_commit_path),
        "size_bytes": preview_commit_path.stat().st_size,
    }
    preview_summary = {
        "path": str(preview_path),
        "sha256": preview_artifact["sha256"],
        "sidecar_sha256": sidecar_artifact["sha256"],
        "commit_sha256": commit_artifact["sha256"],
        "width": preview_record["width"],
        "height": preview_record["height"],
        "mode": preview_record["mode"],
    }
    return preview_summary, {
        "preview": preview_artifact,
        "preview_sidecar": sidecar_artifact,
        "preview_commit": commit_artifact,
    }


def _verify_sha256sums(
    *, sha256sums_path: Path, run_root: Path, required_paths: Sequence[Path]
) -> dict[str, Any]:
    if not sha256sums_path.is_file():
        raise FileNotFoundError(f"PAN-29 SHA256SUMS is missing: {sha256sums_path}")
    pattern = re.compile(r"^([0-9a-fA-F]{64}) ([ *])(.+)$")
    listed: dict[Path, str] = {}
    lines = sha256sums_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("PAN-29 SHA256SUMS is empty")
    for line_number, line in enumerate(lines, 1):
        match = pattern.fullmatch(line)
        if match is None:
            raise ValueError(
                f"PAN-29 SHA256SUMS line {line_number} is not canonical sha256sum output"
            )
        expected_hash = match.group(1).lower()
        relative_text = match.group(3)
        relative_path = Path(relative_text)
        if relative_path.is_absolute():
            raise ValueError("PAN-29 SHA256SUMS contains an absolute path")
        path = (run_root / relative_path).resolve()
        try:
            path.relative_to(run_root)
        except ValueError as error:
            raise ValueError("PAN-29 SHA256SUMS path escapes the run directory") from error
        if path == sha256sums_path:
            raise ValueError("PAN-29 SHA256SUMS must not recursively list itself")
        if path in listed:
            raise ValueError(f"PAN-29 SHA256SUMS lists a path more than once: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"PAN-29 SHA256SUMS listed file is missing: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"PAN-29 SHA256SUMS mismatch for {relative_path.as_posix()}"
            )
        listed[path] = actual_hash
    missing = []
    for required_path in required_paths:
        resolved = required_path.expanduser().resolve()
        try:
            resolved.relative_to(run_root)
        except ValueError as error:
            raise ValueError(
                f"required PAN-29 artifact is outside the run directory: {resolved}"
            ) from error
        if resolved not in listed:
            missing.append(resolved.relative_to(run_root).as_posix())
    if missing:
        raise ValueError(
            "PAN-29 SHA256SUMS does not cover required artifacts: "
            + ", ".join(sorted(missing))
        )
    return {
        "path": str(sha256sums_path),
        "sha256": _sha256(sha256sums_path),
        "size_bytes": sha256sums_path.stat().st_size,
        "listed_file_count": len(listed),
    }


def _verify_run_evidence(
    *,
    replay_result_path: Path,
    run_manifest_path: Path,
    derived_files: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    run_root = run_manifest_path.parent.resolve()
    if replay_result_path.parent.resolve() != run_root:
        raise ValueError("PAN-29 replay result is not a sibling of its run manifest")
    status_path = run_root / "status.txt"
    if not status_path.is_file():
        raise FileNotFoundError(f"PAN-29 status.txt is missing: {status_path}")
    status = _read_key_values(status_path, label="PAN-29 status")
    required_status_keys = {
        "started_at_utc",
        "snapshot_integrity_preflight",
        "snapshot_integrity_postflight",
        "finished_at_utc",
        "exit_code",
    }
    if not required_status_keys.issubset(status):
        missing = sorted(required_status_keys - set(status))
        raise ValueError("PAN-29 status lacks required keys: " + ", ".join(missing))
    started = _utc_timestamp(status["started_at_utc"], "PAN-29 started_at_utc")
    finished = _utc_timestamp(status["finished_at_utc"], "PAN-29 finished_at_utc")
    if (
        status["snapshot_integrity_preflight"] != "PASS"
        or status["snapshot_integrity_postflight"] != "PASS"
        or status["exit_code"] != "0"
        or finished < started
    ):
        raise ValueError("PAN-29 status is not a successful finalized run")

    status_artifact = _verify_path(status_path, _sha256(status_path), "PAN-29 status")
    run_log_path = run_root / "run.log"
    preview_log_path = run_root / "preview.log"
    if not run_log_path.is_file():
        raise FileNotFoundError(f"PAN-29 run.log is missing: {run_log_path}")
    if not preview_log_path.is_file():
        raise FileNotFoundError(f"PAN-29 preview.log is missing: {preview_log_path}")
    run_log_artifact = _verify_path(
        run_log_path, _sha256(run_log_path), "PAN-29 run.log"
    )
    preview_log_artifact = _verify_path(
        preview_log_path, _sha256(preview_log_path), "PAN-29 preview.log"
    )

    required_paths = [Path(record["path"]) for record in derived_files.values()]
    required_paths.extend((status_path, run_log_path, preview_log_path))
    sha256sums_artifact = _verify_sha256sums(
        sha256sums_path=run_root / "SHA256SUMS",
        run_root=run_root,
        required_paths=required_paths,
    )
    run_status = {
        "started_at_utc": status["started_at_utc"],
        "snapshot_integrity_preflight": "PASS",
        "snapshot_integrity_postflight": "PASS",
        "finished_at_utc": status["finished_at_utc"],
        "exit_code": 0,
        "status_sha256": status_artifact["sha256"],
        "sha256sums_sha256": sha256sums_artifact["sha256"],
        "sha256sums_listed_file_count": sha256sums_artifact["listed_file_count"],
    }
    sha256sums_registry_artifact = {
        key: value
        for key, value in sha256sums_artifact.items()
        if key != "listed_file_count"
    }
    return run_status, {
        "status": status_artifact,
        "run_log": run_log_artifact,
        "preview_log": preview_log_artifact,
        "sha256sums": sha256sums_registry_artifact,
    }


def _markdown_with_record(markdown: str, record: Mapping[str, Any]) -> str:
    derived_id = record["derived_artifact_id"]
    source_id = record["source_experiment_id"]
    preview_hash = record["preview"]["sha256"]
    row = (
        f"| `{derived_id}` | `{source_id}` | `PAN-29` | `{ARTIFACT_ROLE}` | "
        f"`0, 5, 10, 14, 19` | `{preview_hash}` |"
    )
    prefix = f"| `{derived_id}` |"
    start_count = markdown.count(START_MARKER)
    end_count = markdown.count(END_MARKER)
    if start_count == 0 and end_count == 0:
        separator = "" if not markdown or markdown.endswith("\n\n") else "\n"
        return (
            markdown
            + separator
            + START_MARKER
            + "\n"
            + DERIVED_SECTION_TITLE
            + "\n\n"
            + "| Derived artifact ID | Source experiment | Task | Role | Observation IDs | Preview SHA256 |\n"
            + "|---|---|---|---|---|---|\n"
            + row
            + "\n"
            + END_MARKER
            + "\n"
        )
    if start_count != 1 or end_count != 1 or markdown.index(START_MARKER) > markdown.index(END_MARKER):
        raise ValueError("derived-artifact Markdown markers are malformed or duplicated")
    existing_rows = [line for line in markdown.splitlines() if line.startswith(prefix)]
    if len(existing_rows) > 1:
        raise ValueError("derived artifact ID occurs more than once in Markdown")
    if existing_rows:
        if existing_rows[0] != row:
            raise ValueError("derived artifact ID has different canonical content in Markdown")
        return markdown
    insertion = markdown.index(END_MARKER)
    before = markdown[:insertion]
    if before and not before.endswith("\n"):
        before += "\n"
    return before + row + "\n" + markdown[insertion:]


def _stage_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            os.chmod(temporary, path.stat().st_mode & 0o777)
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _replace_file(source: Path, target: Path) -> None:
    """Small seam for testing multi-file replacement failure and rollback."""

    source.replace(target)


def _atomic_replace_many(updates: Sequence[tuple[Path, bytes]]) -> None:
    targets = [target for target, _ in updates]
    if len(set(targets)) != len(targets):
        raise ValueError("atomic update contains a duplicate target")
    originals: dict[Path, tuple[bool, bytes, int | None]] = {}
    for target in targets:
        exists = target.exists()
        originals[target] = (
            exists,
            target.read_bytes() if exists else b"",
            (target.stat().st_mode & 0o777) if exists else None,
        )
    staged: list[tuple[Path, Path]] = []
    replaced: list[Path] = []
    try:
        for target, payload in updates:
            staged.append((target, _stage_bytes(target, payload)))
        for target, temporary in staged:
            _replace_file(temporary, target)
            replaced.append(target)
    except BaseException as update_error:
        rollback_errors = []
        for target in reversed(replaced):
            existed, original_bytes, original_mode = originals[target]
            try:
                if existed:
                    restoration = _stage_bytes(target, original_bytes)
                    try:
                        assert original_mode is not None
                        os.chmod(restoration, original_mode)
                        _replace_file(restoration, target)
                    finally:
                        restoration.unlink(missing_ok=True)
                else:
                    target.unlink(missing_ok=True)
            except BaseException as rollback_error:
                rollback_errors.append((target, rollback_error))
        if rollback_errors:
            paths = ", ".join(str(target) for target, _ in rollback_errors)
            raise RuntimeError(
                f"atomic Registry update failed and rollback also failed for: {paths}"
            ) from update_error
        raise
    finally:
        for _, temporary in staged:
            temporary.unlink(missing_ok=True)


def register_derived_artifact(
    *,
    registry_json: Path,
    registry_md: Path,
    replay_result_path: Path,
    preview_sidecar_path: Path,
    preview_commit_path: Path,
    derived_artifact_id: str,
) -> dict[str, Any]:
    """Validate the complete chain and append one non-scientific receipt."""

    registry_json = Path(registry_json).expanduser().resolve()
    registry_md = Path(registry_md).expanduser().resolve()
    replay_result_path = Path(replay_result_path).expanduser().resolve()
    preview_sidecar_path = Path(preview_sidecar_path).expanduser().resolve()
    preview_commit_path = Path(preview_commit_path).expanduser().resolve()
    derived_artifact_id = _safe_id(derived_artifact_id, "derived artifact ID")
    if not registry_json.is_file() or not registry_md.is_file():
        raise FileNotFoundError("both Registry JSON and Markdown must already exist")

    registry_value = _read_json(registry_json)
    registry = dict(registry_value)
    protected = {
        key: _canonical_bytes(value)
        for key, value in registry.items()
        if key not in {"derived_artifacts", "derived_artifact_sets"}
    }
    result, source, validation, derived_files = _verify_replay_result(
        replay_result_path
    )
    preview, preview_files = _verify_preview(
        preview_sidecar_path=preview_sidecar_path,
        preview_commit_path=preview_commit_path,
        replay_result_path=replay_result_path,
        result=result,
        source=source,
        validation=validation,
    )
    derived_files.update(preview_files)
    run_status, run_evidence_files = _verify_run_evidence(
        replay_result_path=replay_result_path,
        run_manifest_path=Path(validation["run_manifest"]["path"]),
        derived_files=derived_files,
    )
    derived_files.update(run_evidence_files)
    source_record, source_artifact_set_id = _validate_registry_source(
        registry,
        source_id=validation["source_id"],
        scientific_commit=validation["scientific_commit"],
        core_artifacts=validation["source_core_registry_artifacts"],
        frame_artifacts=validation["source_frame_registry_artifacts"],
    )

    artifact_set_id = f"ART-DERIVED-{derived_artifact_id}"
    source_scientific_run = {
        "commit": validation["scientific_commit"],
        **validation["source_hashes"],
    }
    record = {
        "derived_artifact_id": derived_artifact_id,
        "task": "PAN-29",
        "status": "PASS",
        "artifact_role": ARTIFACT_ROLE,
        "source_experiment_id": validation["source_id"],
        "source_artifact_set_id": source_artifact_set_id,
        "source_scientific_run": source_scientific_run,
        "pan29": {
            "implementation_commit": validation["implementation_commit"],
            "run_dir": str(replay_result_path.parent),
            "replay_result_sha256": validation["result"]["sha256"],
            "replay_plan_sha256": validation["plan"]["sha256"],
            "run_manifest_sha256": validation["run_manifest"]["sha256"],
            "run_status": run_status,
        },
        "preview": preview,
        "observation_ids": list(OBSERVATION_IDS),
        "replay_count": 5,
        "receipts": validation["receipt_rows"],
        "planner_input_unchanged": True,
        "claims": dict(result["claims"]),
        "derived_artifact_set_id": artifact_set_id,
    }
    artifact_set = {
        "derived_artifact_id": derived_artifact_id,
        "task": "PAN-29",
        "artifact_role": ARTIFACT_ROLE,
        "source_experiment_id": validation["source_id"],
        "hash_semantics": "SHA256 of the validated live derived-artifact file bytes",
        "artifacts": dict(sorted(derived_files.items())),
    }

    existing_records = registry.get("derived_artifacts", [])
    existing_sets = registry.get("derived_artifact_sets", {})
    if not isinstance(existing_records, list):
        raise ValueError("registry derived_artifacts must be a list")
    if not isinstance(existing_sets, Mapping):
        raise ValueError("registry derived_artifact_sets must be an object")
    matches = [
        row
        for row in existing_records
        if isinstance(row, Mapping)
        and row.get("derived_artifact_id") == derived_artifact_id
    ]
    if len(matches) > 1:
        raise ValueError("derived artifact ID occurs more than once in Registry JSON")
    existing_set = existing_sets.get(artifact_set_id)
    json_changed = False
    if matches:
        if _canonical_bytes(matches[0]) != _canonical_bytes(record):
            raise ValueError("derived artifact ID has different canonical content")
        if not isinstance(existing_set, Mapping) or _canonical_bytes(existing_set) != _canonical_bytes(artifact_set):
            raise ValueError("derived artifact set has different canonical content")
    else:
        if existing_set is not None:
            raise ValueError("derived artifact set ID is already occupied")
        registry["derived_artifacts"] = [*existing_records, record]
        registry["derived_artifact_sets"] = {**dict(existing_sets), artifact_set_id: artifact_set}
        json_changed = True

    for key, expected in protected.items():
        if _canonical_bytes(registry[key]) != expected:
            raise AssertionError(f"scientific Registry field changed unexpectedly: {key}")
    source_after = next(
        row
        for row in registry["experiments"]
        if isinstance(row, Mapping) and row.get("experiment_id") == validation["source_id"]
    )
    if _canonical_bytes(source_after) != _canonical_bytes(source_record):
        raise AssertionError("source PAN-11 experiment record changed unexpectedly")

    markdown = registry_md.read_text(encoding="utf-8")
    updated_markdown = _markdown_with_record(markdown, record)
    updates: list[tuple[Path, bytes]] = []
    if json_changed:
        updates.append(
            (
                registry_json,
                (json.dumps(registry, ensure_ascii=False, indent=2) + "\n").encode(
                    "utf-8"
                ),
            )
        )
    if updated_markdown != markdown:
        updates.append((registry_md, updated_markdown.encode("utf-8")))
    if updates:
        _atomic_replace_many(updates)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-json", type=Path, required=True)
    parser.add_argument("--registry-md", type=Path, required=True)
    parser.add_argument("--replay-result", type=Path, required=True)
    parser.add_argument("--preview-sidecar", type=Path, required=True)
    parser.add_argument("--preview-commit", type=Path, required=True)
    parser.add_argument("--derived-artifact-id", required=True)
    args = parser.parse_args(argv)
    record = register_derived_artifact(
        registry_json=args.registry_json,
        registry_md=args.registry_md,
        replay_result_path=args.replay_result,
        preview_sidecar_path=args.preview_sidecar,
        preview_commit_path=args.preview_commit,
        derived_artifact_id=args.derived_artifact_id,
    )
    print(
        json.dumps(
            {
                "derived_artifact_id": record["derived_artifact_id"],
                "source_experiment_id": record["source_experiment_id"],
                "status": record["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
