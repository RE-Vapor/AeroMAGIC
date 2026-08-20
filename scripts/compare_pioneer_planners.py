#!/usr/bin/env python3
"""Compare PAN-11 legacy pose5d and position-only PIONEER runs.

The report is deliberately evidence-only: it validates the controlled-pair
contract before reading planner telemetry and describes the measured wall-time
direction instead of assuming that position-only planning is faster.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]

LEGACY_MODE = "legacy_pose5d"
POSITION_MODE = "position_only"

_FAIR_CONFIG_FIELDS = (
    "test_scenes",
    "params_name",
    "model_name",
    "test_resolution",
    "use_perfect_depth_map",
    "kind_depth_map",
    "compute_collision",
    "random_seed",
    "torch_seed",
    "beam_width",
    "beam_steps",
    "debug_profile",
    "planning_observation_mode",
    "pioneer_face_size",
    "pioneer_face_fov_degrees",
    "pioneer_voxel_size",
)

_ISOLATED_OUTPUT_FIELDS = (
    "results_json_name",
    "lmdb_dir_name",
    "validation_memory_dir_name",
    "experiment_run_id",
    "experiment_run_dir",
    "experiment_metrics_dir",
)

_SEARCH_COUNT_FIELDS = (
    "parent_beam_count",
    "raw_action_proposal_count",
    "translation_action_proposal_count",
    "orientation_action_proposal_count",
    "generated_candidate_count",
    "valid_state_candidate_count",
    "observed_rejected_candidate_count",
    "collision_rejected_candidate_count",
    "rendered_candidate_count",
    "retained_beam_count",
)

_OBSERVATION_COUNT_FIELDS = (
    "bundle_count",
    "real_face_render_count",
    "imagined_bundle_render_count",
    "imagined_face_render_count",
    "imagined_history_bundle_render_count",
    "imagined_history_face_render_count",
    "imagined_candidate_bundle_render_count",
    "imagined_candidate_face_render_count",
)


class ComparisonError(ValueError):
    """Raised when the inputs do not satisfy the PAN-11 comparison contract."""


def _load_json_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ComparisonError(f"{label} does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ComparisonError(f"{label} is not valid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ComparisonError(f"{label} must contain a JSON object: {path}")
    return value


def _require_mapping(value: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise ComparisonError(f"{label}.{key} must be a JSON object")
    return nested


def _require_value(value: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in value:
        raise ComparisonError(f"{label} is missing required field {key!r}")
    return value[key]


def _require_nonnegative_int(value: Mapping[str, Any], key: str, label: str) -> int:
    raw = _require_value(value, key, label)
    if type(raw) is not int or raw < 0:
        raise ComparisonError(f"{label}.{key} must be a non-negative integer")
    return raw


def _require_nonnegative_number(value: Mapping[str, Any], key: str, label: str) -> float:
    raw = _require_value(value, key, label)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw < 0:
        raise ComparisonError(f"{label}.{key} must be a non-negative number")
    return float(raw)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_status_duration(path: Path, label: str) -> float:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ComparisonError(f"{label} does not exist: {path}") from error
    status = {}
    for line in lines:
        if "=" not in line:
            raise ComparisonError(f"{label} contains a malformed line: {line!r}")
        key, value = line.split("=", 1)
        if key in status:
            raise ComparisonError(f"{label} contains duplicate field {key!r}")
        status[key] = value
    if status.get("exit_code") != "0":
        raise ComparisonError(f"{label} must record exit_code=0")
    try:
        started = datetime.fromisoformat(
            status["started_at_utc"].replace("Z", "+00:00")
        )
        finished = datetime.fromisoformat(
            status["finished_at_utc"].replace("Z", "+00:00")
        )
    except (KeyError, ValueError) as error:
        raise ComparisonError(
            f"{label} must contain valid started_at_utc and finished_at_utc timestamps"
        ) from error
    seconds = (finished - started).total_seconds()
    if seconds < 0:
        raise ComparisonError(f"{label} finished before it started")
    return float(seconds)


def _load_kv_file(path: Path, label: str) -> Mapping[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ComparisonError(f"{label} does not exist: {path}") from error
    values = {}
    for line in lines:
        if "=" not in line:
            raise ComparisonError(f"{label} contains a malformed line: {line!r}")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise ComparisonError(f"{label} contains duplicate/empty field {key!r}")
        values[key] = value
    return values


def _validate_execution_evidence(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    metrics_path: Path,
    status_path: Path,
    manifest_path: Path,
    label: str,
    state_mode: str,
    state_dimension: int,
    rig_frame: str,
    extrinsics_version: str,
    fairness: Mapping[str, Any],
) -> Mapping[str, Any]:
    manifest = _load_kv_file(manifest_path, f"{label} manifest")
    run_dir = manifest_path.resolve().parent
    if status_path.resolve().parent != run_dir:
        raise ComparisonError(f"{label} status is not a sibling of its manifest")
    try:
        metrics_path.resolve().relative_to(run_dir)
    except ValueError as error:
        raise ComparisonError(f"{label} metrics are outside the manifest run directory") from error
    expected = {
        "planner": "pioneer",
        "observation_mode": "cubemap6",
        "scene": fairness["scene"],
        "config": config_path.name,
        "config_sha256": _sha256(config_path),
        "debug_profile": fairness["debug_profile"],
        "expected_observations": str(fairness["budget_observations"]),
        "expected_real_face_renders": str(fairness["budget_observations"] * 6),
        "experiment_run_dir": str(run_dir),
        "planner_state_mode": state_mode,
        "planner_state_dimension": str(state_dimension),
        "cubemap_rig_frame": rig_frame,
        "cubemap_extrinsics_version": extrinsics_version,
    }
    for field, expected_value in expected.items():
        actual = manifest.get(field)
        if actual != expected_value:
            raise ComparisonError(
                f"{label} manifest.{field} must be {expected_value!r}, got {actual!r}"
            )
    commit = manifest.get("git_commit", "")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ComparisonError(f"{label} manifest.git_commit must be a full lowercase SHA")
    duration = _load_status_duration(status_path, f"{label} status")
    return {
        "run_dir": str(run_dir),
        "git_commit": commit,
        "wrapper_wall_seconds": duration,
    }


def _debug_budget(config: Mapping[str, Any], profiles_dir: Path, label: str) -> int:
    profile_name = _require_value(config, "debug_profile", label)
    if not isinstance(profile_name, str) or not profile_name:
        raise ComparisonError(f"{label}.debug_profile must be a non-empty string")
    profile_path = profiles_dir / f"{profile_name}.json"
    profile = _load_json_object(profile_path, f"debug profile {profile_name}")
    if profile.get("name") != profile_name:
        raise ComparisonError(
            f"debug profile name mismatch: expected {profile_name!r}, got {profile.get('name')!r}"
        )
    overrides = _require_mapping(profile, "overrides", f"debug profile {profile_name}")
    budget = _require_nonnegative_int(
        overrides,
        "experiment_budget_observations",
        f"debug profile {profile_name}.overrides",
    )
    if budget < 1:
        raise ComparisonError("experiment_budget_observations must be at least one")
    for seed_key in ("random_seed", "torch_seed"):
        config_seed = _require_value(config, seed_key, label)
        profile_seed = _require_value(
            overrides, seed_key, f"debug profile {profile_name}.overrides"
        )
        if config_seed != profile_seed:
            raise ComparisonError(
                f"{label}.{seed_key}={config_seed!r} disagrees with the "
                f"{profile_name} runtime override {profile_seed!r}"
            )
    declared_budget = config.get("experiment_budget_observations")
    if declared_budget is not None and declared_budget != budget:
        raise ComparisonError(
            f"{label}.experiment_budget_observations={declared_budget!r} "
            f"disagrees with the {profile_name} profile budget {budget}"
        )
    return budget


def _validate_config_pair(
    legacy: Mapping[str, Any],
    position: Mapping[str, Any],
    *,
    profiles_dir: Path,
) -> Mapping[str, Any]:
    for field in _FAIR_CONFIG_FIELDS:
        legacy_value = _require_value(legacy, field, "legacy config")
        position_value = _require_value(position, field, "position config")
        if legacy_value != position_value:
            raise ComparisonError(
                f"controlled-pair mismatch for {field}: "
                f"legacy={legacy_value!r}, position={position_value!r}"
            )

    scenes = legacy["test_scenes"]
    if not isinstance(scenes, list) or len(scenes) != 1 or not isinstance(scenes[0], str):
        raise ComparisonError("PAN-11 comparison configs must select exactly one scene")
    if legacy["planning_observation_mode"] != "cubemap6":
        raise ComparisonError("PAN-11 comparison requires planning_observation_mode='cubemap6'")

    legacy_budget = _debug_budget(legacy, profiles_dir, "legacy config")
    position_budget = _debug_budget(position, profiles_dir, "position config")
    if legacy_budget != position_budget:
        raise ComparisonError(
            f"controlled-pair budget mismatch: legacy={legacy_budget}, position={position_budget}"
        )
    selected_profile = _load_json_object(
        profiles_dir / f"{legacy['debug_profile']}.json",
        f"debug profile {legacy['debug_profile']}",
    )
    output_suffix = _require_value(
        selected_profile, "output_suffix", f"debug profile {legacy['debug_profile']}"
    )
    if not isinstance(output_suffix, str) or not output_suffix:
        raise ComparisonError("debug profile output_suffix must be a non-empty string")

    for field in _ISOLATED_OUTPUT_FIELDS:
        legacy_value = _require_value(legacy, field, "legacy config")
        position_value = _require_value(position, field, "position config")
        if not isinstance(legacy_value, str) or not legacy_value:
            raise ComparisonError(f"legacy config.{field} must be a non-empty string")
        if not isinstance(position_value, str) or not position_value:
            raise ComparisonError(f"position config.{field} must be a non-empty string")
        if legacy_value == position_value:
            raise ComparisonError(
                f"isolated output field {field} is shared by both configs: {legacy_value!r}"
            )

    expected_contracts = (
        (
            legacy,
            "legacy config",
            LEGACY_MODE,
            "body",
            "pytorch3d-body-aligned-v1",
        ),
        (
            position,
            "position config",
            POSITION_MODE,
            "world",
            "pytorch3d-world-axes-v1",
        ),
    )
    for config, label, state_mode, rig_frame, extrinsics_version in expected_contracts:
        expected = {
            "pioneer_planner_state_mode": state_mode,
            "pioneer_cubemap_rig_frame": rig_frame,
            "pioneer_cubemap_extrinsics_version": extrinsics_version,
        }
        for field, expected_value in expected.items():
            actual = _require_value(config, field, label)
            if actual != expected_value:
                raise ComparisonError(
                    f"{label}.{field} must be {expected_value!r}, got {actual!r}"
                )

    if legacy.get("pioneer_remove_rotation_only_candidates") is not False:
        raise ComparisonError(
            "legacy config must set pioneer_remove_rotation_only_candidates=false "
            "so the pose5d control retains orientation actions"
        )
    canonical_orientation = position.get("pioneer_canonical_orientation_indices")
    if (
        not isinstance(canonical_orientation, list)
        or len(canonical_orientation) != 2
        or any(type(index) is not int or index < 0 for index in canonical_orientation)
    ):
        raise ComparisonError(
            "position config.pioneer_canonical_orientation_indices must contain "
            "two non-negative integers"
        )
    if canonical_orientation != [2, 0]:
        raise ComparisonError(
            "PAN-11 Eiffel position config requires canonical orientation [2, 0]"
        )

    return {
        "validated": True,
        "scene": scenes[0],
        "random_seed": legacy["random_seed"],
        "torch_seed": legacy["torch_seed"],
        "budget_observations": legacy_budget,
        "debug_profile": legacy["debug_profile"],
        "output_suffix": output_suffix,
        "beam_width": legacy["beam_width"],
        "beam_steps": legacy["beam_steps"],
        "planning_observation_mode": legacy["planning_observation_mode"],
        "face_size": legacy["pioneer_face_size"],
        "face_fov_degrees": legacy["pioneer_face_fov_degrees"],
        "voxel_size": legacy["pioneer_voxel_size"],
        "isolated_output_fields": list(_ISOLATED_OUTPUT_FIELDS),
    }


def _validate_metric_run(
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    label: str,
    state_mode: str,
    state_dimension: int,
    rig_frame: str,
    extrinsics_version: str,
    fairness: Mapping[str, Any],
) -> Mapping[str, Any]:
    if metrics.get("planner") != "pioneer":
        raise ComparisonError(f"{label} metrics.planner must be 'pioneer'")
    if metrics.get("scene") != fairness["scene"]:
        raise ComparisonError(
            f"{label} metrics.scene does not match the controlled scene {fairness['scene']!r}"
        )
    if metrics.get("start_index") != 0:
        raise ComparisonError(f"{label} metrics.start_index must be 0")

    run = _require_mapping(metrics, "run", f"{label} metrics")
    run_expectations = {
        "seed": fairness["random_seed"],
        "torch_seed": fairness["torch_seed"],
        "budget_observations": fairness["budget_observations"],
        "debug_profile": fairness["debug_profile"],
        "planning_observation_mode": "cubemap6",
        "pioneer_planner_state_mode": state_mode,
        "planner_state_dimension": state_dimension,
        "pioneer_cubemap_rig_frame": rig_frame,
        "pioneer_cubemap_extrinsics_version": extrinsics_version,
        "pioneer_canonical_orientation_indices": config.get(
            "pioneer_canonical_orientation_indices"
        ),
    }
    for field, expected in run_expectations.items():
        actual = _require_value(run, field, f"{label} metrics.run")
        if actual != expected:
            raise ComparisonError(
                f"{label} metrics.run.{field} must be {expected!r}, got {actual!r}"
            )
    expected_run_id = f"{config['experiment_run_id']}_{fairness['output_suffix']}"
    if run.get("run_id") != expected_run_id:
        raise ComparisonError(
            f"{label} metrics.run.run_id must be {expected_run_id!r}, "
            f"got {run.get('run_id')!r}"
        )

    trajectory = _require_mapping(metrics, "trajectory", f"{label} metrics")
    observation_count = _require_nonnegative_int(
        trajectory, "observation_count", f"{label} metrics.trajectory"
    )
    if observation_count != fairness["budget_observations"]:
        raise ComparisonError(
            f"{label} observation_count={observation_count} does not match "
            f"the controlled budget {fairness['budget_observations']}"
        )

    search = _require_mapping(metrics, "planner_search", f"{label} metrics")
    search_expectations = {
        "state_mode": state_mode,
        "state_dimension": state_dimension,
        "cubemap_rig_frame": rig_frame,
        "cubemap_extrinsics_version": extrinsics_version,
    }
    for field, expected in search_expectations.items():
        actual = _require_value(search, field, f"{label} metrics.planner_search")
        if actual != expected:
            raise ComparisonError(
                f"{label} metrics.planner_search.{field} must be {expected!r}, "
                f"got {actual!r}"
            )
    totals_value = _require_mapping(search, "totals", f"{label} metrics.planner_search")
    totals = {
        field: _require_nonnegative_int(
            totals_value, field, f"{label} metrics.planner_search.totals"
        )
        for field in _SEARCH_COUNT_FIELDS
    }
    totals["search_seconds"] = _require_nonnegative_number(
        totals_value, "search_seconds", f"{label} metrics.planner_search.totals"
    )

    if state_mode == POSITION_MODE and totals["orientation_action_proposal_count"] != 0:
        raise ComparisonError("position-only metrics must record zero orientation proposals")
    if state_mode == LEGACY_MODE and totals["orientation_action_proposal_count"] == 0:
        raise ComparisonError(
            "legacy pose5d metrics must contain orientation proposals; the control "
            "does not demonstrate rotation-enabled planning"
        )
    if (
        totals["translation_action_proposal_count"]
        + totals["orientation_action_proposal_count"]
        > totals["raw_action_proposal_count"]
    ):
        raise ComparisonError(f"{label} action proposal categories exceed raw proposals")
    if totals["generated_candidate_count"] > totals["raw_action_proposal_count"]:
        raise ComparisonError(f"{label} generated candidates exceed raw proposals")
    if totals["valid_state_candidate_count"] > totals["generated_candidate_count"]:
        raise ComparisonError(f"{label} valid-state candidates exceed generated candidates")
    if totals["observed_rejected_candidate_count"] > totals["generated_candidate_count"]:
        raise ComparisonError(f"{label} observed rejections exceed generated candidates")
    if totals["collision_rejected_candidate_count"] > totals["valid_state_candidate_count"]:
        raise ComparisonError(f"{label} collision rejections exceed valid-state candidates")
    legal_count = (
        totals["valid_state_candidate_count"]
        - totals["collision_rejected_candidate_count"]
    )
    if totals["rendered_candidate_count"] != legal_count:
        raise ComparisonError(
            f"{label} rendered candidates must equal derived legal candidates"
        )
    if totals["retained_beam_count"] > totals["rendered_candidate_count"]:
        raise ComparisonError(f"{label} retained beams exceed rendered candidates")

    observation = _require_mapping(metrics, "pioneer_observation", f"{label} metrics")
    observation_counts = {
        field: _require_nonnegative_int(
            observation, field, f"{label} metrics.pioneer_observation"
        )
        for field in _OBSERVATION_COUNT_FIELDS
    }
    if observation_counts["bundle_count"] != observation_count:
        raise ComparisonError(f"{label} captured bundle count does not match observations")
    if observation_counts["real_face_render_count"] != observation_count * 6:
        raise ComparisonError(f"{label} real face renders must equal six per observation")
    if (
        observation_counts["imagined_candidate_bundle_render_count"]
        != totals["rendered_candidate_count"]
    ):
        raise ComparisonError(
            f"{label} imagined candidate bundles do not match rendered candidates"
        )
    if (
        observation_counts["imagined_candidate_face_render_count"]
        != observation_counts["imagined_candidate_bundle_render_count"] * 6
    ):
        raise ComparisonError(f"{label} imagined candidate faces must equal six per bundle")
    if (
        observation_counts["imagined_history_face_render_count"]
        != observation_counts["imagined_history_bundle_render_count"] * 6
    ):
        raise ComparisonError(f"{label} imagined history faces must equal six per bundle")
    if observation_counts["imagined_bundle_render_count"] != (
        observation_counts["imagined_history_bundle_render_count"]
        + observation_counts["imagined_candidate_bundle_render_count"]
    ):
        raise ComparisonError(f"{label} imagined bundle total does not recombine")
    if observation_counts["imagined_face_render_count"] != (
        observation_counts["imagined_history_face_render_count"]
        + observation_counts["imagined_candidate_face_render_count"]
    ):
        raise ComparisonError(f"{label} imagined face total does not recombine")

    latency = _require_mapping(metrics, "latency", f"{label} metrics")
    trajectory_seconds = _require_nonnegative_number(
        latency, "trajectory_seconds", f"{label} metrics.latency"
    )
    totals["legal_candidate_count"] = legal_count

    return {
        "run_id": run.get("run_id"),
        "state_mode": state_mode,
        "state_dimension": state_dimension,
        "cubemap_rig_frame": rig_frame,
        "cubemap_extrinsics_version": extrinsics_version,
        "canonical_orientation_indices": run.get(
            "pioneer_canonical_orientation_indices"
        ),
        "observation_count": observation_count,
        "planner_search": totals,
        "imagined_renders": observation_counts,
        "trajectory_seconds": trajectory_seconds,
        "config_run_dir": config["experiment_run_dir"],
        "metrics_dir": config["experiment_metrics_dir"],
    }


def _difference(position: Mapping[str, Any], legacy: Mapping[str, Any], key: str) -> float:
    return float(position[key]) - float(legacy[key])


def _duration_interpretation(
    legacy_seconds: float, position_seconds: float, *, metric_label: str
) -> Mapping[str, Any]:
    delta = position_seconds - legacy_seconds
    ratio = legacy_seconds / position_seconds if position_seconds > 0 else None
    if delta < 0:
        statement = (
            f"Measured position-only {metric_label} was {-delta:.6f} s shorter "
            "than the legacy pose5d control for this bounded run."
        )
        direction = "position_only_shorter"
    elif delta > 0:
        statement = (
            f"Measured position-only {metric_label} was {delta:.6f} s longer "
            "than the legacy pose5d control for this bounded run."
        )
        direction = "position_only_longer"
    else:
        statement = f"Measured {metric_label} values were equal for this bounded run."
        direction = "equal"
    return {
        "direction": direction,
        "position_only_minus_legacy_pose5d_seconds": delta,
        "legacy_pose5d_divided_by_position_only_ratio": ratio,
        "statement": statement,
        "scope": "This direction is descriptive for the validated bounded pair only.",
    }


def build_comparison(
    *,
    legacy_config_path: Path,
    position_config_path: Path,
    legacy_metrics_path: Path,
    position_metrics_path: Path,
    legacy_status_path: Path,
    position_status_path: Path,
    legacy_manifest_path: Path,
    position_manifest_path: Path,
    profiles_dir: Path,
) -> Mapping[str, Any]:
    legacy_config = _load_json_object(legacy_config_path, "legacy config")
    position_config = _load_json_object(position_config_path, "position config")
    fairness = _validate_config_pair(
        legacy_config,
        position_config,
        profiles_dir=profiles_dir,
    )
    legacy_metrics = _load_json_object(legacy_metrics_path, "legacy metrics")
    position_metrics = _load_json_object(position_metrics_path, "position metrics")
    legacy = _validate_metric_run(
        legacy_metrics,
        legacy_config,
        label="legacy",
        state_mode=LEGACY_MODE,
        state_dimension=5,
        rig_frame="body",
        extrinsics_version="pytorch3d-body-aligned-v1",
        fairness=fairness,
    )
    position = _validate_metric_run(
        position_metrics,
        position_config,
        label="position",
        state_mode=POSITION_MODE,
        state_dimension=3,
        rig_frame="world",
        extrinsics_version="pytorch3d-world-axes-v1",
        fairness=fairness,
    )
    legacy_evidence = _validate_execution_evidence(
        config_path=legacy_config_path,
        config=legacy_config,
        metrics_path=legacy_metrics_path,
        status_path=legacy_status_path,
        manifest_path=legacy_manifest_path,
        label="legacy",
        state_mode=LEGACY_MODE,
        state_dimension=5,
        rig_frame="body",
        extrinsics_version="pytorch3d-body-aligned-v1",
        fairness=fairness,
    )
    position_evidence = _validate_execution_evidence(
        config_path=position_config_path,
        config=position_config,
        metrics_path=position_metrics_path,
        status_path=position_status_path,
        manifest_path=position_manifest_path,
        label="position",
        state_mode=POSITION_MODE,
        state_dimension=3,
        rig_frame="world",
        extrinsics_version="pytorch3d-world-axes-v1",
        fairness=fairness,
    )
    if legacy_evidence["git_commit"] != position_evidence["git_commit"]:
        raise ComparisonError("controlled-pair manifests must use the same git commit")
    legacy_wrapper_seconds = legacy_evidence["wrapper_wall_seconds"]
    position_wrapper_seconds = position_evidence["wrapper_wall_seconds"]
    legacy["execution_evidence"] = legacy_evidence
    position["execution_evidence"] = position_evidence
    legacy["wrapper_wall_seconds"] = legacy_wrapper_seconds
    position["wrapper_wall_seconds"] = position_wrapper_seconds

    search_delta_fields = (
        "raw_action_proposal_count",
        "translation_action_proposal_count",
        "orientation_action_proposal_count",
        "generated_candidate_count",
        "valid_state_candidate_count",
        "observed_rejected_candidate_count",
        "collision_rejected_candidate_count",
        "legal_candidate_count",
        "rendered_candidate_count",
        "retained_beam_count",
        "parent_beam_count",
        "search_seconds",
    )
    search_deltas = {
        field: _difference(
            position["planner_search"], legacy["planner_search"], field
        )
        for field in search_delta_fields
    }
    imagined_deltas = {
        field: _difference(position["imagined_renders"], legacy["imagined_renders"], field)
        for field in _OBSERVATION_COUNT_FIELDS
        if field.startswith("imagined_")
    }

    return {
        "schema_version": 1,
        "comparison": "PAN-11-pioneer-position-only-vs-legacy-pose5d",
        "fairness": fairness,
        "inputs": {
            "legacy_config": {
                "path": str(legacy_config_path.resolve()),
                "sha256": _sha256(legacy_config_path),
            },
            "position_config": {
                "path": str(position_config_path.resolve()),
                "sha256": _sha256(position_config_path),
            },
            "legacy_metrics": {
                "path": str(legacy_metrics_path.resolve()),
                "sha256": _sha256(legacy_metrics_path),
            },
            "position_metrics": {
                "path": str(position_metrics_path.resolve()),
                "sha256": _sha256(position_metrics_path),
            },
            "legacy_status": {
                "path": str(legacy_status_path.resolve()),
                "sha256": _sha256(legacy_status_path),
            },
            "position_status": {
                "path": str(position_status_path.resolve()),
                "sha256": _sha256(position_status_path),
            },
            "legacy_manifest": {
                "path": str(legacy_manifest_path.resolve()),
                "sha256": _sha256(legacy_manifest_path),
            },
            "position_manifest": {
                "path": str(position_manifest_path.resolve()),
                "sha256": _sha256(position_manifest_path),
            },
        },
        "runs": {
            LEGACY_MODE: legacy,
            POSITION_MODE: position,
        },
        "position_only_minus_legacy_pose5d": {
            "state_dimension": position["state_dimension"] - legacy["state_dimension"],
            "planner_search": search_deltas,
            "imagined_renders": imagined_deltas,
            "trajectory_seconds": position["trajectory_seconds"]
            - legacy["trajectory_seconds"],
            "wrapper_wall_seconds": position_wrapper_seconds
            - legacy_wrapper_seconds,
        },
        "trajectory_time_interpretation": _duration_interpretation(
            legacy["trajectory_seconds"],
            position["trajectory_seconds"],
            metric_label="trajectory compute time",
        ),
        "wall_time_interpretation": _duration_interpretation(
            legacy_wrapper_seconds,
            position_wrapper_seconds,
            metric_label="tmux wrapper wall time",
        ),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    fairness = report["fairness"]
    legacy = report["runs"][LEGACY_MODE]
    position = report["runs"][POSITION_MODE]
    rows: Sequence[tuple[str, Any, Any]] = (
        ("State dimension", legacy["state_dimension"], position["state_dimension"]),
        (
            "Raw action proposals",
            legacy["planner_search"]["raw_action_proposal_count"],
            position["planner_search"]["raw_action_proposal_count"],
        ),
        (
            "Translation proposals",
            legacy["planner_search"]["translation_action_proposal_count"],
            position["planner_search"]["translation_action_proposal_count"],
        ),
        (
            "Orientation proposals",
            legacy["planner_search"]["orientation_action_proposal_count"],
            position["planner_search"]["orientation_action_proposal_count"],
        ),
        (
            "Generated candidates",
            legacy["planner_search"]["generated_candidate_count"],
            position["planner_search"]["generated_candidate_count"],
        ),
        (
            "Valid-state candidates",
            legacy["planner_search"]["valid_state_candidate_count"],
            position["planner_search"]["valid_state_candidate_count"],
        ),
        (
            "Observed-state rejections",
            legacy["planner_search"]["observed_rejected_candidate_count"],
            position["planner_search"]["observed_rejected_candidate_count"],
        ),
        (
            "Collision rejections",
            legacy["planner_search"]["collision_rejected_candidate_count"],
            position["planner_search"]["collision_rejected_candidate_count"],
        ),
        (
            "Legal candidates (derived)",
            legacy["planner_search"]["legal_candidate_count"],
            position["planner_search"]["legal_candidate_count"],
        ),
        (
            "Rendered candidates",
            legacy["planner_search"]["rendered_candidate_count"],
            position["planner_search"]["rendered_candidate_count"],
        ),
        (
            "Retained beams",
            legacy["planner_search"]["retained_beam_count"],
            position["planner_search"]["retained_beam_count"],
        ),
        (
            "Imagined bundles",
            legacy["imagined_renders"]["imagined_bundle_render_count"],
            position["imagined_renders"]["imagined_bundle_render_count"],
        ),
        (
            "Imagined face renders",
            legacy["imagined_renders"]["imagined_face_render_count"],
            position["imagined_renders"]["imagined_face_render_count"],
        ),
        (
            "Trajectory seconds",
            f"{legacy['trajectory_seconds']:.6f}",
            f"{position['trajectory_seconds']:.6f}",
        ),
        (
            "tmux wrapper wall seconds",
            f"{legacy['wrapper_wall_seconds']:.6f}",
            f"{position['wrapper_wall_seconds']:.6f}",
        ),
    )
    table = "\n".join(f"| {name} | {old} | {new} |" for name, old, new in rows)
    trajectory_interpretation = report["trajectory_time_interpretation"]
    wall_interpretation = report["wall_time_interpretation"]
    return (
        "# PAN-11 planner comparison\n\n"
        "## Controlled-pair gate\n\n"
        f"Validated: `{str(fairness['validated']).lower()}`; scene: "
        f"`{fairness['scene']}`; seeds: `{fairness['random_seed']}` / "
        f"`{fairness['torch_seed']}`; profile: `{fairness['debug_profile']}`; "
        f"budget: `{fairness['budget_observations']}` observations; beam: "
        f"`{fairness['beam_width']} x {fairness['beam_steps']}`.\n\n"
        "The configs use separate result, LMDB, capture-memory, run, and metrics "
        "namespaces. Both use the same six-face observation and GT/debug settings.\n\n"
        "## Measured telemetry\n\n"
        "| Metric | legacy_pose5d | position_only |\n"
        "| --- | ---: | ---: |\n"
        f"{table}\n\n"
        "`Legal candidates` is derived as `valid_state_candidate_count - "
        "collision_rejected_candidate_count`, per the telemetry contract.\n\n"
        "## Wall-time reading\n\n"
        f"{trajectory_interpretation['statement']} "
        f"{trajectory_interpretation['scope']}\n\n"
        f"{wall_interpretation['statement']} {wall_interpretation['scope']}\n"
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-config", type=Path, required=True)
    parser.add_argument("--position-config", type=Path, required=True)
    parser.add_argument("--legacy-metrics", type=Path, required=True)
    parser.add_argument("--position-metrics", type=Path, required=True)
    parser.add_argument("--legacy-status", type=Path, required=True)
    parser.add_argument("--position-status", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--position-manifest", type=Path, required=True)
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=ROOT / "configs" / "debug",
        help="Directory containing the selected debug profile JSON.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.output_json.resolve() == args.output_markdown.resolve():
        raise ComparisonError("JSON and Markdown outputs must use different paths")
    report = build_comparison(
        legacy_config_path=args.legacy_config,
        position_config_path=args.position_config,
        legacy_metrics_path=args.legacy_metrics,
        position_metrics_path=args.position_metrics,
        legacy_status_path=args.legacy_status,
        position_status_path=args.position_status,
        legacy_manifest_path=args.legacy_manifest,
        position_manifest_path=args.position_manifest,
        profiles_dir=args.profiles_dir,
    )
    _write_text_atomic(
        args.output_json,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(args.output_markdown, render_markdown(report))
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
