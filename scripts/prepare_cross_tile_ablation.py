#!/usr/bin/env python3
"""Prepare isolated S/T dataset views and configs for a seam ablation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


STARTS = {
    "S": [11, 9, 5, 1, 3],
    "T": [12, 9, 5, 1, 3],
}


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
    keys = sorted(set(before) | set(after))
    return [
        {"key": key, "before": before.get(key), "after": after.get(key)}
        for key in keys
        if before.get(key) != after.get(key)
    ]


def _pose_from_index(settings: Mapping[str, Any], index: list[int]) -> list[float]:
    camera = settings["camera"]
    minimum = camera["x_min"]
    maximum = camera["x_max"]
    shape = [camera["pose_l"], camera["pose_w"], camera["pose_h"]]
    xyz = [
        10.0 * (minimum[axis] + (0.5 + index[axis]) * (maximum[axis] - minimum[axis]) / shape[axis])
        for axis in range(3)
    ]
    elevation = -90.0 + 180.0 * (1 + index[3]) / (camera["pose_n_theta"] + 1)
    azimuth = 360.0 * index[4] / camera["pose_n_azim"]
    return [*xyz, elevation, azimuth]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--source-scene-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--gpu-s", type=int, required=True)
    parser.add_argument("--gpu-t", type=int, required=True)
    parser.add_argument("--scene", default="12-NW-6C-7_8")
    args = parser.parse_args()

    baseline_path = Path(args.baseline_config).resolve()
    source_scene = Path(args.source_scene_dir).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    source_settings_path = source_scene / "settings.json"
    source_settings = json.loads(source_settings_path.read_text(encoding="utf-8"))
    occupied_path = source_scene / "occupied_pose.json"
    if not occupied_path.exists():
        raise SystemExit(f"missing occupied-pose evidence: {occupied_path}")
    occupied_data = json.loads(occupied_path.read_text(encoding="utf-8"))
    occupied_lookup = {
        tuple(index): bool(occupied)
        for index, occupied in zip(
            occupied_data["X_idx"], occupied_data["occupied"]
        )
    }
    asset_names = [
        args.scene + ".obj",
        args.scene + ".mtl",
        "textures",
        "occupied_pose.pt",
        "occupied_pose.json",
        "assembly-manifest.json",
        "magician_loader_report.json",
        "validation_report.json",
    ]
    source_hashes = {
        name: _sha256(source_scene / name)
        for name in asset_names
        if (source_scene / name).is_file()
    }
    source_hashes["settings.json"] = _sha256(source_settings_path)
    runs = []
    for label, start_index in STARTS.items():
        slug = label.lower()
        run_id = f"myl41_cross_tile_{slug}_obs101_beam10"
        run_root = output_root / slug
        scene_view = run_root / "dataset" / "Macarons++" / args.scene
        scene_view.mkdir(parents=True)
        for name in asset_names:
            source = source_scene / name
            if source.exists():
                os.symlink(source, scene_view / name, target_is_directory=source.is_dir())
        derived_settings = json.loads(json.dumps(source_settings))
        derived_settings["camera"]["start_positions"][0] = start_index
        settings_path = scene_view / "settings.json"
        _write_json(settings_path, derived_settings)

        gpu = args.gpu_s if label == "S" else args.gpu_t
        config = json.loads(json.dumps(baseline))
        config.update(
            {
                "numGPU": gpu,
                "dataset_path": str(run_root / "dataset" / "Macarons++"),
                "validation_n_poses_in_trajectory": 100,
                "validation_max_start_positions": 1,
                "validation_memory_dir_name": run_id,
                "experiment_budget_observations": 101,
                "experiment_run_id": run_id,
                "experiment_metrics_dir": str(run_root / "metrics"),
                "da3_cache_dir": str(run_root / "cache"),
                "lmdb_dir_name": run_id + "_lmdb",
                "scone_lmdb_dir_name": run_id + "_lmdb",
                "results_json_name": run_id + "_results.json",
                "experiment_cross_tile_diagnostics_enabled": True,
                "experiment_cross_tile_seam_x": 75.0,
                "experiment_cross_tile_min_x_index": 12,
                "experiment_cross_tile_gate_from_index": [11, 9, 5, 1, 3],
                "experiment_cross_tile_gate_to_index": [12, 9, 5, 1, 3],
            }
        )
        config_path = run_root / "config.json"
        _write_json(config_path, config)
        config_diff = _diff(baseline, config)
        _write_json(run_root / "config_diff.json", config_diff)
        command = [
            args.python,
            str(Path(__file__).resolve().parent / "run_with_da3_overlay.py"),
            "test_magician_planning.py",
            "-c",
            str(config_path),
        ]
        gate_run_id = run_id + "_gate"
        gate_config = json.loads(json.dumps(config))
        gate_config.update(
            {
                "validation_n_poses_in_trajectory": 1,
                "validation_memory_dir_name": gate_run_id,
                "experiment_budget_observations": 2,
                "experiment_run_id": gate_run_id,
                "experiment_metrics_dir": str(run_root / "gate" / "metrics"),
                "da3_cache_dir": str(run_root / "gate" / "cache"),
                "lmdb_dir_name": gate_run_id + "_lmdb",
                "scone_lmdb_dir_name": gate_run_id + "_lmdb",
                "results_json_name": gate_run_id + "_results.json",
            }
        )
        gate_config_path = run_root / "gate" / "config.json"
        _write_json(gate_config_path, gate_config)
        gate_command = [
            args.python,
            str(Path(__file__).resolve().parent / "run_with_da3_overlay.py"),
            "test_magician_planning.py",
            "-c",
            str(gate_config_path),
        ]
        run_manifest = {
            "schema_version": 1,
            "issue": "MYL-41",
            "label": label,
            "run_id": run_id,
            "scene": args.scene,
            "start_grid_index": start_index,
            "start_occupied": occupied_lookup.get(tuple(start_index[:3])),
            "start_pose_scene_units_and_degrees": _pose_from_index(
                source_settings, start_index
            ),
            "seam_x_scene_units": 75.0,
            "source_scene": str(source_scene),
            "dataset_view": str(scene_view),
            "source_hashes": source_hashes,
            "derived_settings_sha256": _sha256(settings_path),
            "baseline_config": str(baseline_path),
            "baseline_config_sha256": _sha256(baseline_path),
            "config": str(config_path),
            "config_diff": str(run_root / "config_diff.json"),
            "command": command,
            "gate_config": str(gate_config_path),
            "gate_command": gate_command,
            "gpu": gpu,
            "paths": {
                "cache": str(run_root / "cache"),
                "metrics": str(run_root / "metrics"),
                "log": str(run_root / "run.log"),
                "preview": str(run_root / "preview"),
            },
        }
        if run_manifest["start_occupied"] is not False:
            raise SystemExit(
                f"start {label} is not explicitly free in occupied_pose.json"
            )
        _write_json(run_root / "manifest.json", run_manifest)
        runs.append(run_manifest)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "issue": "MYL-41",
        "scene": args.scene,
        "position_only_ablation": True,
        "baseline_config": str(baseline_path),
        "source_scene": str(source_scene),
        "seam_x_scene_units": 75.0,
        "runs": runs,
    }
    _write_json(output_root / "manifest.json", manifest)
    print(output_root / "manifest.json")


if __name__ == "__main__":
    main()
