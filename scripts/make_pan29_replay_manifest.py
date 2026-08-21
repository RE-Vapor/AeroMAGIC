#!/usr/bin/env python3
"""Freeze five PAN-11 observations into a PAN-29 UE5 replay plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


FACE_NAMES = ("front", "back", "left", "right", "up", "down")
BUNDLE_IDS = (0, 5, 10, 14, 19)
SCHEMA_VERSION = "pan29.ue5-postrun-replay.v1"
SOURCE_SEMANTIC_SCHEMA_VERSION = "pan29.source-bundle-semantic-validation.v1"
TRANSFORM_VERSION = "pan21-planner-to-ue-cm-v1"
SOURCE_CENTER_TOLERANCE_SCENE_UNITS = 1.0e-5
SOURCE_BASIS_TOLERANCE = 1.0e-5
PLANNER_TO_UE_CM = (
    (500.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 500.0, 0.0),
    (0.0, 500.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
EXPECTED_WORLD_FROM_CAMERA_ROTATIONS = {
    "front": ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "back": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
    "left": ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
    "right": ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)),
    "up": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "down": ((-1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
}


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
        if not line or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key:
            raise ValueError(f"invalid key=value line in {path}: {line!r}")
        if key in values:
            raise ValueError(f"duplicate key in {path}: {key}")
        values[key] = value
    return values


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _finite_xyz(value: Any, label: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must contain three finite values")
    resolved = [float(item) for item in value]
    if len(resolved) != 3 or not all(math.isfinite(item) for item in resolved):
        raise ValueError(f"{label} must contain three finite values")
    return resolved


def _within_aabb(position: Sequence[float], policy: Mapping[str, Any]) -> bool:
    bounds = policy.get("fly_volume_aabb_ue_cm")
    if not isinstance(bounds, Mapping):
        raise ValueError("spatial policy must define fly_volume_aabb_ue_cm")
    minimum = _finite_xyz(bounds.get("min"), "fly-volume minimum")
    maximum = _finite_xyz(bounds.get("max"), "fly-volume maximum")
    if any(low > high for low, high in zip(minimum, maximum)):
        raise ValueError("spatial policy fly-volume bounds are inverted")
    return all(low <= value <= high for value, low, high in zip(position, minimum, maximum))


def _planner_to_ue_cm(position: Sequence[float]) -> list[float]:
    x_value, y_value, z_value = _finite_xyz(position, "planner position")
    return [500.0 * x_value, 500.0 * z_value, 500.0 * y_value]


def _load_torch_mapping(path: Path, label: str) -> Mapping[str, Any]:
    """Load one trusted source-run tensor dictionary without arbitrary pickle code."""

    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one mapping")
    return value


def _finite_tensor(value: Any, shape: tuple[int, ...], label: str) -> torch.Tensor:
    try:
        result = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{label} must be a finite tensor with shape {shape}") from error
    if tuple(result.shape) != shape or not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must be a finite tensor with shape {shape}")
    return result


def _camera_center(value: Any, label: str) -> torch.Tensor:
    try:
        result = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{label} must contain one finite XYZ camera center") from error
    if result.numel() != 3 or not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must contain one finite XYZ camera center")
    return result.reshape(3)


def _maximum_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.max(torch.abs(actual - expected)).item())


def _validate_rotation(rotation: torch.Tensor, label: str) -> None:
    identity_error = _maximum_error(
        rotation.transpose(0, 1) @ rotation,
        torch.eye(3, dtype=torch.float64),
    )
    determinant_error = abs(float(torch.linalg.det(rotation).item()) - 1.0)
    if identity_error > SOURCE_BASIS_TOLERANCE or determinant_error > SOURCE_BASIS_TOLERANCE:
        raise ValueError(
            f"{label} rotation is not right-handed orthonormal: "
            f"orthonormal_error={identity_error}, determinant_error={determinant_error}"
        )


def _validate_source_bundle_semantics(
    *,
    frame_dir: Path,
    bundle_id: int,
    position: Sequence[float],
    timestamp_ns: int,
    timestamp_utc: str,
) -> dict[str, Any]:
    """Prove that the hash-bound source tensors encode the planned identity/pose."""

    expected_center = torch.tensor(position, dtype=torch.float64)
    bundle = _load_torch_mapping(frame_dir / "bundle.pt", f"source bundle {bundle_id}")
    if type(bundle.get("bundle_id")) is not int or bundle["bundle_id"] != bundle_id:
        raise ValueError(f"source bundle {bundle_id} payload has another bundle ID")
    if tuple(bundle.get("face_names") or ()) != FACE_NAMES:
        raise ValueError(f"source bundle {bundle_id} payload has noncanonical face names")
    if bundle.get("capture_timestamp_unix_ns") != timestamp_ns or bundle.get(
        "capture_timestamp_utc"
    ) != timestamp_utc:
        raise ValueError(f"source bundle {bundle_id} payload timestamp differs from metrics")
    if (
        bundle.get("rig_frame") != "world"
        or bundle.get("extrinsics_version") != "pytorch3d-world-axes-v1"
    ):
        raise ValueError(f"source bundle {bundle_id} payload has unexpected extrinsics")
    bundle_center = _camera_center(
        bundle.get("camera_center"), f"source bundle {bundle_id} camera_center"
    )
    center_max_error = _maximum_error(bundle_center, expected_center)
    if center_max_error > SOURCE_CENTER_TOLERANCE_SCENE_UNITS:
        raise ValueError(
            f"source bundle {bundle_id} camera center differs from trajectory: "
            f"max_error={center_max_error}"
        )

    face_reports = []
    basis_max_error = 0.0
    for face_name in FACE_NAMES:
        face = _load_torch_mapping(
            frame_dir / f"{face_name}.pt", f"source bundle {bundle_id} face {face_name}"
        )
        if type(face.get("bundle_id")) is not int or face["bundle_id"] != bundle_id:
            raise ValueError(
                f"source bundle {bundle_id} face {face_name} has another bundle ID"
            )
        if face.get("face_name") != face_name:
            raise ValueError(
                f"source bundle {bundle_id} face {face_name} payload has another face name"
            )
        if face.get("capture_timestamp_unix_ns") != timestamp_ns or face.get(
            "capture_timestamp_utc"
        ) != timestamp_utc:
            raise ValueError(
                f"source bundle {bundle_id} face {face_name} timestamp differs from metrics"
            )
        if (
            face.get("rig_frame") != "world"
            or face.get("extrinsics_version") != "pytorch3d-world-axes-v1"
        ):
            raise ValueError(
                f"source bundle {bundle_id} face {face_name} has unexpected extrinsics"
            )

        face_center = _camera_center(
            face.get("camera_center"),
            f"source bundle {bundle_id} face {face_name} camera_center",
        )
        world_to_camera = _finite_tensor(
            face.get("R_opencv_world_to_camera"),
            (3, 3),
            f"source bundle {bundle_id} face {face_name} R_opencv_world_to_camera",
        )
        translation_world_to_camera = _camera_center(
            face.get("T_opencv_world_to_camera"),
            f"source bundle {bundle_id} face {face_name} T_opencv_world_to_camera",
        )
        _validate_rotation(
            world_to_camera,
            f"source bundle {bundle_id} face {face_name} world-to-camera",
        )
        world_from_camera = world_to_camera.transpose(0, 1)
        expected_basis = torch.tensor(
            EXPECTED_WORLD_FROM_CAMERA_ROTATIONS[face_name], dtype=torch.float64
        )
        basis_error = _maximum_error(world_from_camera, expected_basis)
        if basis_error > SOURCE_BASIS_TOLERANCE:
            raise ValueError(
                f"source bundle {bundle_id} face {face_name} basis differs from "
                f"pytorch3d-world-axes-v1: max_error={basis_error}"
            )
        derived_center = -(world_from_camera @ translation_world_to_camera)
        face_center_error = max(
            _maximum_error(face_center, expected_center),
            _maximum_error(face_center, bundle_center),
            _maximum_error(derived_center, expected_center),
            _maximum_error(derived_center, face_center),
        )
        if face_center_error > SOURCE_CENTER_TOLERANCE_SCENE_UNITS:
            raise ValueError(
                f"source bundle {bundle_id} face {face_name} camera center differs "
                f"from trajectory or extrinsics: max_error={face_center_error}"
            )
        center_max_error = max(center_max_error, face_center_error)
        basis_max_error = max(basis_max_error, basis_error)
        face_reports.append(
            {
                "face_name": face_name,
                "T_world_from_cam_rotation": world_from_camera.tolist(),
                "camera_center_scene_units": derived_center.tolist(),
                "camera_center_max_abs_error_scene_units": face_center_error,
                "basis_max_abs_error": basis_error,
            }
        )

    return {
        "schema_version": SOURCE_SEMANTIC_SCHEMA_VERSION,
        "result": "PASS",
        "bundle_id": bundle_id,
        "source_capture_timestamp_ns": timestamp_ns,
        "source_capture_timestamp_utc": timestamp_utc,
        "camera_center_scene_units": bundle_center.tolist(),
        "camera_center_max_abs_error_scene_units": center_max_error,
        "face_basis_max_abs_error": basis_max_error,
        "basis_convention": (
            "T_world_from_cam rotation in PIONEER world; OpenCV camera axes "
            "x-right, y-down, z-forward"
        ),
        "basis_source": "transpose(R_opencv_world_to_camera)",
        "camera_center_source": "-R_opencv_world_to_camera.T @ T_opencv_world_to_camera",
        "faces": face_reports,
    }


def build_replay_manifest(
    *,
    metrics_path: Path,
    source_config_path: Path,
    source_run_manifest_path: Path,
    capture_root: Path,
    spatial_policy_path: Path,
    source_lmdb_path: Path,
    original_preview_path: Path,
    bundle_ids: Sequence[int],
    source_registry_id: str,
) -> dict[str, Any]:
    """Validate and hash-bind the exact source observations to replay."""

    metrics_path = Path(metrics_path).expanduser().resolve()
    source_config_path = Path(source_config_path).expanduser().resolve()
    source_run_manifest_path = Path(source_run_manifest_path).expanduser().resolve()
    capture_root = Path(capture_root).expanduser().resolve()
    spatial_policy_path = Path(spatial_policy_path).expanduser().resolve()
    source_lmdb_path = Path(source_lmdb_path).expanduser().resolve()
    original_preview_path = Path(original_preview_path).expanduser().resolve()
    selected = tuple(int(value) for value in bundle_ids)
    if selected != BUNDLE_IDS:
        raise ValueError(f"PAN-29 requires exactly bundle IDs {list(BUNDLE_IDS)}")
    if not source_registry_id.strip():
        raise ValueError("source_registry_id must be non-empty")

    metrics = _read_json(metrics_path)
    config = _read_json(source_config_path)
    run_manifest = _read_key_values(source_run_manifest_path)
    spatial_policy = _read_json(spatial_policy_path)
    lmdb_data_path = source_lmdb_path / "data.mdb"
    if not lmdb_data_path.is_file():
        raise FileNotFoundError(f"source LMDB data file is missing: {lmdb_data_path}")
    if not original_preview_path.is_file():
        raise FileNotFoundError(f"source preview is missing: {original_preview_path}")
    original_preview_sidecar = original_preview_path.with_suffix(".json")
    config_sha = _sha256(source_config_path)
    if run_manifest.get("config_sha256") != config_sha:
        raise ValueError("source config SHA256 does not match the run manifest")
    if run_manifest.get("git_commit", "") == "" or len(run_manifest["git_commit"]) != 40:
        raise ValueError("source run manifest must contain a full git_commit")
    if run_manifest.get("debug_profile") != "pioneer-20":
        raise ValueError("source run must use debug_profile=pioneer-20")
    if int(run_manifest.get("expected_observations", -1)) != 20:
        raise ValueError("source run must declare 20 observations")
    if int(run_manifest.get("expected_real_face_renders", -1)) != 120:
        raise ValueError("source run must declare 120 real face renders")

    if metrics.get("planner") != "pioneer" or metrics.get("scene") != "HKUST":
        raise ValueError("PAN-29 source metrics must be the HKUST PIONEER run")
    run = metrics.get("run")
    trajectory = metrics.get("trajectory")
    pioneer = metrics.get("pioneer_observation")
    if not all(isinstance(value, Mapping) for value in (run, trajectory, pioneer)):
        raise ValueError("source metrics lack run/trajectory/pioneer_observation")
    assert isinstance(run, Mapping)
    assert isinstance(trajectory, Mapping)
    assert isinstance(pioneer, Mapping)
    if run.get("planning_observation_mode") != "cubemap6":
        raise ValueError("source metrics must use cubemap6")
    depth_source = str(run.get("depth_source") or config.get("kind_depth_map") or "")
    if depth_source.upper() != "GT" or config.get("use_perfect_depth_map") is not True:
        raise ValueError("the named PAN-29 source must be the GT-depth run")
    if run.get("pioneer_planner_state_mode") != "position_only":
        raise ValueError("source metrics must use position_only planner state")
    if run.get("pioneer_cubemap_rig_frame") != "world":
        raise ValueError("source metrics must use the world cubemap rig")
    if run.get("pioneer_cubemap_extrinsics_version") != "pytorch3d-world-axes-v1":
        raise ValueError("source metrics use an unexpected cubemap extrinsics version")
    if int(trajectory.get("observation_count", -1)) != 20:
        raise ValueError("source trajectory must contain 20 observations")
    positions = trajectory.get("positions")
    bundles = pioneer.get("bundles")
    if not isinstance(positions, list) or len(positions) != 20:
        raise ValueError("source trajectory positions must contain 20 rows")
    if not isinstance(bundles, list) or len(bundles) != 20:
        raise ValueError("source bundle telemetry must contain 20 rows")
    if int(pioneer.get("bundle_count", -1)) != 20:
        raise ValueError("source bundle_count must equal 20")
    if int(pioneer.get("real_face_render_count", -1)) != 120:
        raise ValueError("source real_face_render_count must equal 120")
    expected_capture_dir = (capture_root / "frames").resolve()
    if Path(str(metrics.get("capture_dir", ""))).expanduser().resolve() != expected_capture_dir:
        raise ValueError("capture_root does not match metrics.capture_dir")

    by_id: dict[int, Mapping[str, Any]] = {}
    for row in bundles:
        if not isinstance(row, Mapping):
            raise ValueError("bundle telemetry rows must be objects")
        bundle_id = int(row.get("bundle_id", -1))
        if bundle_id in by_id:
            raise ValueError(f"duplicate source bundle ID {bundle_id}")
        by_id[bundle_id] = row

    replay_rows = []
    for bundle_id in selected:
        row = by_id.get(bundle_id)
        if row is None:
            raise ValueError(f"missing source bundle ID {bundle_id}")
        if int(row.get("face_count", -1)) != 6 or tuple(row.get("face_names") or ()) != FACE_NAMES:
            raise ValueError(f"source bundle {bundle_id} is not canonical cubemap6")
        if row.get("rig_frame") != "world" or row.get("extrinsics_version") != "pytorch3d-world-axes-v1":
            raise ValueError(f"source bundle {bundle_id} has unexpected extrinsics")
        timestamp_ns = row.get("capture_timestamp_unix_ns")
        timestamp_utc = row.get("capture_timestamp_utc")
        if type(timestamp_ns) is not int or timestamp_ns <= 0 or not isinstance(timestamp_utc, str) or not timestamp_utc:
            raise ValueError(f"source bundle {bundle_id} lacks one exact timestamp")
        position = _finite_xyz(positions[bundle_id], f"trajectory position {bundle_id}")
        ue_position = _planner_to_ue_cm(position)
        frame_dir = capture_root / "frames" / f"{bundle_id:06d}"
        image_dir = capture_root / "imgs" / f"{bundle_id:06d}"
        frame_paths = [frame_dir / "bundle.pt", *(frame_dir / f"{face}.pt" for face in FACE_NAMES)]
        image_paths = [image_dir / f"{face}.png" for face in FACE_NAMES]
        marker_path = capture_root / ".pioneer_bundle_commits" / f"{bundle_id:06d}.json"
        frame_artifacts = [_artifact(path, capture_root) for path in frame_paths]
        source_semantic_validation = _validate_source_bundle_semantics(
            frame_dir=frame_dir,
            bundle_id=bundle_id,
            position=position,
            timestamp_ns=timestamp_ns,
            timestamp_utc=timestamp_utc,
        )
        for path, artifact in zip(frame_paths, frame_artifacts):
            if (
                path.stat().st_size != artifact["bytes"]
                or _sha256(path) != artifact["sha256"]
            ):
                raise ValueError(
                    f"source bundle {bundle_id} tensor changed during semantic validation: {path}"
                )
        replay_rows.append(
            {
                "observation_id": bundle_id,
                "display_observation_number": bundle_id + 1,
                "source_capture_timestamp_ns": timestamp_ns,
                "source_capture_timestamp_utc": timestamp_utc,
                "planner_position_scene_units": position,
                "ue_position_cm": ue_position,
                "replay_request_id": f"pan29-hkust-gt20obs-{bundle_id:06d}",
                "replay_frame_id": f"observation-{bundle_id:06d}",
                "source_bundle_transaction_marker_present": marker_path.is_file(),
                "within_pan13_conservative_fly_volume": _within_aabb(
                    ue_position, spatial_policy
                ),
                "source_semantic_validation": source_semantic_validation,
                "source_artifacts": {
                    "frames": frame_artifacts,
                    "images": [_artifact(path, capture_root) for path in image_paths],
                    "transaction_marker": (
                        _artifact(marker_path, capture_root) if marker_path.is_file() else None
                    ),
                },
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "task": "PAN-29",
        "artifact_role": "post_run_visualization_only",
        "planner_input_unchanged": True,
        "ue5_render_is_postrun_visualization_only": True,
        "selected_bundle_ids": list(selected),
        "source": {
            "registry_experiment_id": source_registry_id,
            "scientific_run_commit": run_manifest["git_commit"],
            "run_dir": str(source_run_manifest_path.parent),
            "run_manifest": {
                "path": str(source_run_manifest_path),
                "sha256": _sha256(source_run_manifest_path),
            },
            "config": {"path": str(source_config_path), "sha256": config_sha},
            "metrics": {"path": str(metrics_path), "sha256": _sha256(metrics_path)},
            "capture_root": str(capture_root),
            "lmdb": {
                "path": str(source_lmdb_path),
                "data_sha256": _sha256(lmdb_data_path),
                "key": "HKUST/0",
            },
            "original_preview": {
                "path": str(original_preview_path),
                "sha256": _sha256(original_preview_path),
                "sidecar_path": (
                    str(original_preview_sidecar)
                    if original_preview_sidecar.is_file()
                    else None
                ),
                "sidecar_sha256": (
                    _sha256(original_preview_sidecar)
                    if original_preview_sidecar.is_file()
                    else None
                ),
            },
            "depth_source": "GT",
            "scene": "HKUST",
            "scene_units_per_meter": float(metrics.get("scene_units_per_meter", 0.2)),
            "planner_state_mode": "position_only",
            "cubemap_rig_frame": "world",
            "cubemap_extrinsics_version": "pytorch3d-world-axes-v1",
        },
        "coordinate_transform": {
            "version": TRANSFORM_VERSION,
            "description": "[ue_x,ue_y,ue_z]_cm=[500*x,500*z,500*y]_planner",
            "matrix": [list(row) for row in PLANNER_TO_UE_CM],
            "spatial_policy": {
                "path": str(spatial_policy_path),
                "sha256": _sha256(spatial_policy_path),
                "coordinate_frame": spatial_policy.get("coordinate_frame"),
                "fly_volume_aabb_ue_cm": spatial_policy.get("fly_volume_aabb_ue_cm"),
            },
        },
        "observations": replay_rows,
        "known_limitations": [
            "The source GT run predates per-bundle transaction markers; selected source files are individually SHA256-bound.",
            "PAN13 conservative fly-volume membership is reported but does not clamp post-run visualization poses.",
            "UE5 actual render timestamps must remain separate from source observation timestamps.",
        ],
    }


def _parse_bundle_ids(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("bundle IDs must be comma-separated integers") from error


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite replay manifest: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--source-run-manifest", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--spatial-policy", type=Path, required=True)
    parser.add_argument("--source-lmdb", type=Path, required=True)
    parser.add_argument("--original-preview", type=Path, required=True)
    parser.add_argument("--bundle-ids", type=_parse_bundle_ids, default=BUNDLE_IDS)
    parser.add_argument("--source-registry-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_replay_manifest(
        metrics_path=args.metrics,
        source_config_path=args.source_config,
        source_run_manifest_path=args.source_run_manifest,
        capture_root=args.capture_root,
        spatial_policy_path=args.spatial_policy,
        source_lmdb_path=args.source_lmdb,
        original_preview_path=args.original_preview,
        bundle_ids=args.bundle_ids,
        source_registry_id=args.source_registry_id,
    )
    _atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps({"output": str(args.output), "observation_count": len(BUNDLE_IDS)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
