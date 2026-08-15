#!/usr/bin/env python3
"""Validate an N-tile experiment contract and materialize an isolated run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macarons.utility.cross_tile_diagnostics import validate_tile_partition


ALLOWED_CONFIG_CHANGES = {
    "da3_cache_dir",
    "dataset_path",
    "experiment_budget_observations",
    "experiment_metrics_dir",
    "experiment_metrics_enabled",
    "experiment_param_overrides",
    "experiment_planning_range_gate_enabled",
    "experiment_planning_range_gate_min_points",
    "experiment_planning_range_gate_quantile",
    "experiment_run_id",
    "experiment_tile_metrics_enabled",
    "experiment_tile_partition",
    "lmdb_dir_name",
    "numGPU",
    "results_json_name",
    "scone_lmdb_dir_name",
    "validation_max_start_positions",
    "validation_memory_dir_name",
    "validation_n_poses_in_trajectory",
}

OPTIONAL_MANIFEST_FIELDS = {"link_assets", "range_gate"}


def _load(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        {"key": key, "before": before.get(key), "after": after.get(key)}
        for key in sorted(set(before) | set(after))
        if before.get(key) != after.get(key)
    ]


def _require_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _camera_pose(settings: Mapping[str, Any], index: Sequence[int], scale: float) -> list[float]:
    camera = settings["camera"]
    minimum = camera["x_min"]
    maximum = camera["x_max"]
    shape = [camera["pose_l"], camera["pose_w"], camera["pose_h"]]
    xyz = [
        scale
        * (minimum[axis] + (0.5 + index[axis]) * (maximum[axis] - minimum[axis]) / shape[axis])
        for axis in range(3)
    ]
    elevation = -90.0 + 180.0 * (1 + index[3]) / (camera["pose_n_theta"] + 1)
    azimuth = 360.0 * index[4] / camera["pose_n_azim"]
    return [*xyz, elevation, azimuth]


def _validate_start(settings: Mapping[str, Any], index: Sequence[int]) -> None:
    if len(index) != 5 or any(isinstance(value, bool) or not isinstance(value, int) for value in index):
        raise ValueError("start_grid_index must contain five integers")
    camera = settings["camera"]
    shape = [
        camera["pose_l"],
        camera["pose_w"],
        camera["pose_h"],
        camera["pose_n_theta"],
        camera["pose_n_azim"],
    ]
    if any(value < 0 or value >= size for value, size in zip(index, shape)):
        raise ValueError(f"start_grid_index is outside camera lattice {shape}")


def prepare_workflow(manifest_path: Path) -> Path:
    manifest_path = manifest_path.resolve()
    spec = _load(manifest_path)
    required = (
        "issue",
        "scene",
        "source_scene_dir",
        "assembly_manifest",
        "baseline_config",
        "params_config",
        "calibration_report",
        "output_root",
        "python",
        "gpu",
        "run_id",
        "budget_observations",
        "start_grid_index",
        "tile_partition",
        "asset_sha256",
    )
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"workflow manifest is missing required fields: {missing}")
    if spec.get("schema_version") != 1:
        raise ValueError("workflow schema_version must be 1")
    unknown = sorted(set(spec) - set(required) - {"schema_version"} - OPTIONAL_MANIFEST_FIELDS)
    if unknown:
        raise ValueError(f"workflow manifest contains unknown fields: {unknown}")

    scene = str(spec["scene"])
    source_scene = Path(spec["source_scene_dir"]).resolve()
    assembly_path = Path(spec["assembly_manifest"]).resolve()
    baseline_path = Path(spec["baseline_config"]).resolve()
    params_path = Path(spec["params_config"]).resolve()
    calibration_path = Path(spec["calibration_report"]).resolve()
    output_root = Path(spec["output_root"]).resolve()
    for label, path in (
        ("source scene", source_scene),
        ("assembly manifest", assembly_path),
        ("baseline config", baseline_path),
        ("params config", params_path),
        ("calibration report", calibration_path),
    ):
        if not path.exists():
            raise ValueError(f"missing {label}: {path}")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output root: {output_root}")

    partition = validate_tile_partition(spec["tile_partition"])
    assembly = _load(assembly_path)
    members = assembly.get("members")
    if not isinstance(members, list) or len(members) != len(partition["tile_ids"]):
        raise ValueError(
            "assembly members and tile_partition.tile_ids must have the same length"
        )
    member_names = [str(member.get("name", "")) for member in members]
    if member_names != partition["tile_ids"]:
        raise ValueError(
            "assembly member names/order must exactly match tile_partition.tile_ids"
        )
    if assembly.get("scene_name") != scene:
        raise ValueError("assembly scene_name does not match workflow scene")

    params = _load(params_path)
    scene_scale = _require_number(
        params["_data"]["scene_scale_factor"], "scene_scale_factor", positive=True
    )
    zfar = _require_number(params["_depth_module"]["zfar"], "zfar", positive=True)
    bounds = assembly["output"]["bounds"]
    axis = partition["axis"]
    lower = float(bounds["minimum"][axis]) * scene_scale
    upper = float(bounds["maximum"][axis]) * scene_scale
    if any(boundary <= lower or boundary >= upper for boundary in partition["boundaries"]):
        raise ValueError(
            f"tile boundaries must be strictly inside runtime assembly bounds [{lower}, {upper}]"
        )

    calibration = _load(calibration_path)
    if calibration.get("scene") != scene:
        raise ValueError("calibration report scene does not match workflow scene")
    recommended = _require_number(
        calibration["calibration"]["recommended_sensor_range_scene_units"],
        "recommended sensor range",
        positive=True,
    )
    hard_cap = _require_number(
        calibration["calibration"]["hard_cap_scene_units"],
        "calibration hard cap",
        positive=True,
    )
    if recommended > min(hard_cap, zfar):
        raise ValueError("calibrated sensor range exceeds the recorded hard cap or zfar")

    expected_hashes = spec["asset_sha256"]
    if not isinstance(expected_hashes, Mapping) or not expected_hashes:
        raise ValueError("asset_sha256 must be a nonempty object")
    required_hashed_assets = {
        f"{scene}.obj",
        f"{scene}.mtl",
        "settings.json",
        "occupied_pose.pt",
        assembly_path.name,
    }
    missing_hashes = sorted(required_hashed_assets - set(expected_hashes))
    if missing_hashes:
        raise ValueError(f"asset_sha256 is missing required assets: {missing_hashes}")
    verified_hashes = {}
    for name, expected in expected_hashes.items():
        path = assembly_path if name == assembly_path.name else source_scene / name
        if not path.is_file():
            raise ValueError(f"hashed asset is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"asset hash mismatch for {name}: expected {expected}, got {actual}")
        verified_hashes[name] = actual

    settings = _load(source_scene / "settings.json")
    start_index = list(spec["start_grid_index"])
    _validate_start(settings, start_index)
    occupied = _load(source_scene / "occupied_pose.json")
    occupied_lookup = {
        tuple(index): bool(value)
        for index, value in zip(occupied["X_idx"], occupied["occupied"])
    }
    start_occupied = occupied_lookup.get(tuple(start_index[:3]))
    if start_occupied is not False:
        raise ValueError("start_grid_index is not explicitly free in occupied_pose.json")

    budget = spec["budget_observations"]
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 2:
        raise ValueError("budget_observations must be an integer >= 2")
    gpu = spec["gpu"]
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ValueError("gpu must be a non-negative integer")
    gate = dict(spec.get("range_gate", {}))
    quantile = _require_number(gate.get("quantile", 0.9), "range gate quantile")
    min_points = gate.get("min_points", 1)
    if not 0.0 < quantile <= 1.0:
        raise ValueError("range gate quantile must be in (0, 1]")
    if isinstance(min_points, bool) or not isinstance(min_points, int) or min_points < 1:
        raise ValueError("range gate min_points must be an integer >= 1")

    baseline = _load(baseline_path)
    run_id = str(spec["run_id"])
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a nonempty plain name")
    output_root.mkdir(parents=True, exist_ok=True)
    scene_view = output_root / "dataset" / "Macarons++" / scene
    scene_view.mkdir(parents=True)
    link_assets = spec.get(
        "link_assets",
        [
            f"{scene}.obj",
            f"{scene}.mtl",
            "textures",
            "occupied_pose.pt",
            "occupied_pose.json",
            assembly_path.name,
            "magician_loader_report.json",
            "validation_report.json",
        ],
    )
    for name in link_assets:
        source = assembly_path if name == assembly_path.name else source_scene / name
        if source.exists() and name != "settings.json":
            os.symlink(source, scene_view / name, target_is_directory=source.is_dir())
    derived_settings = json.loads(json.dumps(settings))
    derived_settings["camera"]["start_positions"][0] = start_index
    _write_json(scene_view / "settings.json", derived_settings)

    config = json.loads(json.dumps(baseline))
    overrides = dict(config.get("experiment_param_overrides", {}))
    overrides["sensor_range"] = recommended
    config.update(
        {
            "numGPU": gpu,
            "dataset_path": str(output_root / "dataset" / "Macarons++"),
            "validation_n_poses_in_trajectory": budget - 1,
            "validation_max_start_positions": 1,
            "validation_memory_dir_name": run_id,
            "experiment_budget_observations": budget,
            "experiment_run_id": run_id,
            "experiment_metrics_enabled": True,
            "experiment_metrics_dir": str(output_root / "metrics"),
            "da3_cache_dir": str(output_root / "cache"),
            "lmdb_dir_name": run_id + "_lmdb",
            "scone_lmdb_dir_name": run_id + "_lmdb",
            "results_json_name": run_id + "_results.json",
            "experiment_param_overrides": overrides,
            "experiment_planning_range_gate_enabled": True,
            "experiment_planning_range_gate_quantile": quantile,
            "experiment_planning_range_gate_min_points": min_points,
            "experiment_tile_metrics_enabled": True,
            "experiment_tile_partition": partition,
        }
    )
    config_diff = _diff(baseline, config)
    changed = {row["key"] for row in config_diff}
    unexpected = sorted(changed - ALLOWED_CONFIG_CHANGES)
    if unexpected:
        raise ValueError(f"derived config changed non-allowlisted parameters: {unexpected}")
    config_path = output_root / "config.json"
    _write_json(config_path, config)
    _write_json(output_root / "config_diff.json", config_diff)

    gate_id = run_id + "_gate"
    gate_config = json.loads(json.dumps(config))
    gate_config.update(
        {
            "validation_n_poses_in_trajectory": 1,
            "validation_memory_dir_name": gate_id,
            "experiment_budget_observations": 2,
            "experiment_run_id": gate_id,
            "experiment_metrics_dir": str(output_root / "gate" / "metrics"),
            "da3_cache_dir": str(output_root / "gate" / "cache"),
            "lmdb_dir_name": gate_id + "_lmdb",
            "scone_lmdb_dir_name": gate_id + "_lmdb",
            "results_json_name": gate_id + "_results.json",
        }
    )
    gate_config_path = output_root / "gate" / "config.json"
    _write_json(gate_config_path, gate_config)

    runner = str(Path(__file__).resolve().parent / "run_with_da3_overlay.py")
    checker = str(Path(__file__).resolve().parent / "accept_ntile_metrics.py")
    python = str(spec["python"])
    command = [python, runner, "test_magician_planning.py", "-c", str(config_path)]
    gate_command = [python, runner, "test_magician_planning.py", "-c", str(gate_config_path)]
    prepared_manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "issue": str(spec["issue"]),
        "scene": scene,
        "run_id": run_id,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_scene": str(source_scene),
        "assembly_manifest": str(assembly_path),
        "assembly_members": member_names,
        "tile_partition": partition,
        "unit_contract": calibration["unit_contract"],
        "sensor_range_scene_units": recommended,
        "sensor_range_hard_cap_scene_units": hard_cap,
        "verified_asset_sha256": verified_hashes,
        "baseline_config": str(baseline_path),
        "baseline_config_sha256": _sha256(baseline_path),
        "params_config": str(params_path),
        "params_config_sha256": _sha256(params_path),
        "calibration_report": str(calibration_path),
        "calibration_report_sha256": _sha256(calibration_path),
        "dataset_view": str(scene_view),
        "derived_settings_sha256": _sha256(scene_view / "settings.json"),
        "start_grid_index": start_index,
        "start_occupied": start_occupied,
        "start_pose_scene_units_and_degrees": _camera_pose(settings, start_index, scene_scale),
        "budget_observations": budget,
        "config": str(config_path),
        "config_diff": str(output_root / "config_diff.json"),
        "command": command,
        "gate_config": str(gate_config_path),
        "gate_command": gate_command,
        "gate_accept_command": [
            python,
            checker,
            "--manifest",
            str(output_root / "manifest.json"),
            "--metrics",
            str(output_root / "gate" / "metrics"),
            "--mode",
            "gate",
            "--output",
            str(output_root / "gate" / "acceptance.json"),
        ],
        "accept_command": [
            python,
            checker,
            "--manifest",
            str(output_root / "manifest.json"),
            "--metrics",
            str(output_root / "metrics"),
            "--mode",
            "full",
            "--output",
            str(output_root / "acceptance.json"),
        ],
        "paths": {
            "cache": str(output_root / "cache"),
            "metrics": str(output_root / "metrics"),
            "log": str(output_root / "run.log"),
            "gate_cache": str(output_root / "gate" / "cache"),
            "gate_metrics": str(output_root / "gate" / "metrics"),
            "gate_log": str(output_root / "gate" / "run.log"),
        },
    }
    _write_json(output_root / "manifest.json", prepared_manifest)
    return output_root / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    try:
        output = prepare_workflow(Path(args.manifest))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"N-tile workflow validation failed: {error}") from error
    print(output)


if __name__ == "__main__":
    main()
