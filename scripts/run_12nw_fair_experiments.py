#!/usr/bin/env python3
"""Generate and optionally execute the 12-NW-6C-5 Stage 4 matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping

from analyze_planning_diagnostics import analyze_online_metrics


SCENE = "12-NW-6C-5"
CALIBRATED_SCENE_UNITS_PER_METER = 1.0


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    suite: str
    planner: str
    source: str
    budget: int
    max_starts: int
    changes: Mapping[str, Any] = field(default_factory=dict)
    note: str = ""


def _specs(main_budget: int, sensitivity_budget: int) -> list[RunSpec]:
    specs = [
        RunSpec(
            f"main_{planner}_{source}",
            "main",
            planner,
            source,
            main_budget,
            5,
            note="Fixed scene/starts/seeds/budget/beam/mapping/collision fair pair.",
        )
        for planner in ("scone", "magician")
        for source in ("gt", "da3")
    ]
    variants = [
        ("confidence_p25", {"da3_confidence_percentile": 25.0}, "Confidence threshold."),
        ("confidence_p50", {"da3_confidence_percentile": 50.0}, "Confidence threshold."),
        ("confidence_p75", {"da3_confidence_percentile": 75.0}, "Confidence threshold."),
        ("confidence_p90", {"da3_confidence_percentile": 90.0}, "Confidence threshold."),
        (
            "scale_0_9",
            {"da3_scene_units_per_meter": {SCENE: 0.9}},
            "Offline sensitivity around the evidence-backed 1.0 main scale.",
        ),
        (
            "scale_1_1",
            {"da3_scene_units_per_meter": {SCENE: 1.1}},
            "Offline sensitivity around the evidence-backed 1.0 main scale.",
        ),
        ("window_1", {"da3_window_size": 1}, "RGB-only single-frame window."),
        ("window_5", {"da3_window_size": 5}, "Five-frame pose-conditioned window."),
        ("resolution_336", {"da3_process_res": 336}, "Lower process resolution."),
        ("resolution_672", {"da3_process_res": 672}, "Higher process resolution."),
        (
            "mapping_1x",
            {"experiment_param_overrides": {"planning_gathering_factor_multiplier": 1.0}},
            "One-times mapping gathering radius.",
        ),
        (
            "mapping_2x",
            {"experiment_param_overrides": {"planning_gathering_factor_multiplier": 2.0}},
            "Legacy two-times mapping gathering radius used by the main pair.",
        ),
    ]
    specs.extend(
        RunSpec(
            f"sensitivity_magician_da3_{name}",
            "sensitivities",
            "magician",
            "da3",
            sensitivity_budget,
            1,
            changes=changes,
            note=note,
        )
        for name, changes, note in variants
    )
    return specs


def _merge_config(
    base: Mapping[str, Any],
    spec: RunSpec,
    run_root: Path,
    gpu: int,
    collision: bool,
) -> dict[str, Any]:
    config = dict(base)
    config.update(
        {
            "numGPU": gpu,
            "test_scenes": [SCENE],
            "use_perfect_depth_map": spec.source == "gt",
            "kind_depth_map": "DA3",
            "da3_scene_units_per_meter": {SCENE: CALIBRATED_SCENE_UNITS_PER_METER},
            "validation_n_poses_in_trajectory": spec.budget - 1,
            "validation_max_start_positions": spec.max_starts,
            "validation_memory_dir_name": f"myl20_{spec.run_id}",
            "scone_lmdb_dir_name": f"myl20_{spec.run_id}_lmdb",
            "lmdb_dir_name": f"myl20_{spec.run_id}_lmdb",
            "results_json_name": f"myl20_{spec.run_id}_results.json",
            "compute_collision": collision,
            "beam_width": 10,
            "beam_steps": 10,
            "random_seed": 8,
            "torch_seed": 9,
            "validation_n_proxy_points": 800000,
            "validation_n_gt_surface_points": 100000,
            "experiment_metrics_enabled": True,
            "experiment_keep_frames": True,
            "experiment_run_id": spec.run_id,
            "experiment_budget_observations": spec.budget,
            "experiment_metrics_dir": str(run_root / "metrics"),
            "experiment_shared_collision_gate": True,
            "experiment_normalize_coverage_by_visibility": True,
            "experiment_param_overrides": {
                "planning_gathering_factor_multiplier": 2.0
            },
        }
    )
    for key, value in spec.changes.items():
        if key == "experiment_param_overrides":
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="configs/test/test_da3_12-nw-6c-5_real_mesh_config.json",
    )
    parser.add_argument("--output-dir", default="results/12-nw-6c-5_phase4")
    parser.add_argument(
        "--suite", choices=("main", "sensitivities", "all"), default="main"
    )
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--main-budget", type=int, default=101)
    parser.add_argument("--sensitivity-budget", type=int, default=6)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--collision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.main_budget < 2 or args.sensitivity_budget < 2:
        parser.error("budgets must include at least two observations")

    project_root = Path(__file__).resolve().parents[1]
    base_path = Path(args.base_config)
    if not base_path.is_absolute():
        base_path = project_root / base_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    base = json.loads(base_path.read_text(encoding="utf-8"))
    requested = set(args.run_id)
    selected = [
        spec
        for spec in _specs(args.main_budget, args.sensitivity_budget)
        if (args.suite == "all" or spec.suite == args.suite)
        and (not requested or spec.run_id in requested)
    ]
    selected_ids = {spec.run_id for spec in selected}
    if requested - selected_ids:
        parser.error("unknown or out-of-suite run ids: " + ", ".join(sorted(requested - selected_ids)))

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scene": SCENE,
        "scope": "single_scene_only",
        "base_config": str(base_path),
        "invocation": [sys.executable, *sys.argv],
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "hf_home": os.environ.get("HF_HOME"),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
            "da3_append_paths": [
                item
                for item in os.environ.get("MAGICIAN_DA3_APPEND_PATHS", "").split(
                    os.pathsep
                )
                if item
            ],
        },
        "scene_units_per_meter": CALIBRATED_SCENE_UNITS_PER_METER,
        "main_gt_mesh_reference": "identical transformed 12-NW-6C-5 mesh per run",
        "renderer_zbuf_role": "post-run diagnostics only",
        "gt_feedback_to_da3": False,
        "fixed": {
            "start_indices": [0, 1, 2, 3, 4],
            "random_seed": 8,
            "torch_seed": 9,
            "beam_width": 10,
            "beam_steps": 10,
            "mapping_gathering_factor_multiplier": 2.0,
            "compute_collision": args.collision,
            "gt_mesh_pose_validity_prior": False,
            "gt_mesh_segment_collision_prior": args.collision,
            "rade_gs_prior": "MAGICIAN only",
        },
        "runs": [],
        "unselected_sensitivities": [],
    }
    executions_path = output_dir / "executions.json"
    executions = []
    if args.resume and executions_path.exists():
        executions = json.loads(executions_path.read_text(encoding="utf-8"))

    for spec in selected:
        run_root = output_dir / spec.run_id
        config = _merge_config(base, spec, run_root, args.gpu, args.collision)
        config_path = run_root / "config.json"
        _write_json(config_path, config)
        entrypoint = "test_scenes.py" if spec.planner == "scone" else "test_magician_planning.py"
        command = [
            sys.executable,
            str(project_root / "scripts/run_with_da3_overlay.py"),
            entrypoint,
            "-c",
            str(config_path),
        ]
        manifest["runs"].append(
            {
                **asdict(spec),
                "config": str(config_path),
                "command": command,
                "gt_mesh_reference": "transformed mesh sampled by setup_test_scene",
                "renderer_zbuf_feedback": False,
            }
        )
        if args.generate_only:
            continue
        if any(
            execution.get("run_id") == spec.run_id
            and execution.get("returncode") == 0
            for execution in executions
        ):
            continue
        executions = [item for item in executions if item.get("run_id") != spec.run_id]
        log_path = run_root / "run.log"
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=project_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        execution = {
            "run_id": spec.run_id,
            "returncode": result.returncode,
            "elapsed_seconds": time.perf_counter() - started,
            "log": str(log_path),
            "diagnostics": [],
        }
        executions.append(execution)
        if result.returncode:
            _write_json(executions_path, executions)
            if not args.continue_on_error:
                raise SystemExit(f"{spec.run_id} failed; see {log_path}")
            continue
        for online_path in sorted((run_root / "metrics").glob("*.online.json")):
            online = json.loads(online_path.read_text(encoding="utf-8"))
            diagnostic = analyze_online_metrics(online)
            diagnostic_path = online_path.with_name(
                online_path.name.replace(".online.json", ".diagnostic.json")
            )
            _write_json(diagnostic_path, diagnostic)
            execution["diagnostics"].append(str(diagnostic_path))
        _write_json(executions_path, executions)

    sensitivity_ids = {
        spec.run_id
        for spec in _specs(args.main_budget, args.sensitivity_budget)
        if spec.suite == "sensitivities"
    }
    manifest["unselected_sensitivities"] = sorted(sensitivity_ids - selected_ids)
    _write_json(output_dir / "manifest.json", manifest)
    if executions:
        _write_json(executions_path, executions)
    print(output_dir / "manifest.json")


if __name__ == "__main__":
    main()
