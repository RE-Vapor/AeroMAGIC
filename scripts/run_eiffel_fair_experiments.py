#!/usr/bin/env python3
"""Generate and optionally execute the Eiffel Stage 4 experiment matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Optional

from analyze_planning_diagnostics import analyze_online_metrics


HEIGHT_SCALE = 0.2641176363636364
WIDTH_SCALE = 0.23766064


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    suite: str
    planner: str
    source: str
    budget: int
    changes: Mapping[str, Any] = field(default_factory=dict)
    note: str = ""


def _specs(main_budget: int, ablation_budget: int) -> list[RunSpec]:
    specs = []
    for planner in ("scone", "magician"):
        for source in ("gt", "da3"):
            specs.append(
                RunSpec(
                    f"main_{planner}_{source}",
                    "main",
                    planner,
                    source,
                    main_budget,
                    note="Fixed Eiffel/start/seed/budget/collision fair pair.",
                )
            )
    for planner in ("scone", "magician"):
        specs.append(
            RunSpec(
                f"smoke_{planner}_da3_shared_mapping",
                "smoke",
                planner,
                "da3",
                3,
                note="Reproduces the Stage 3 three-view boundary with a shared mapping radius.",
            )
        )

    variants = [
        ("confidence_p25", {"da3_confidence_percentile": 25.0}, "Confidence threshold."),
        ("confidence_p50", {"da3_confidence_percentile": 50.0}, "Confidence threshold."),
        ("confidence_p75", {"da3_confidence_percentile": 75.0}, "Confidence threshold above the confidence=1 mass."),
        ("confidence_p90", {"da3_confidence_percentile": 90.0}, "Effective high-confidence filter after inspecting confidence ties."),
        ("scale_width", {"da3_scene_units_per_meter": {"eiffel": WIDTH_SCALE}}, "Ground-width scale endpoint."),
        (
            "scale_midpoint",
            {"da3_scene_units_per_meter": {"eiffel": (WIDTH_SCALE + HEIGHT_SCALE) / 2.0}},
            "Midpoint of the height/ground-width scale interval.",
        ),
        ("window_1", {"da3_window_size": 1}, "RGB-only single-frame window."),
        ("window_5", {"da3_window_size": 5}, "Five-frame window where the budget permits."),
        ("resolution_336", {"da3_process_res": 336}, "Lower DA3 process resolution."),
        ("resolution_672", {"da3_process_res": 672}, "Higher DA3 process resolution."),
        (
            "mapping_radius_0_5",
            {"experiment_param_overrides": {"planning_gathering_factor_multiplier": 0.5}},
            "Half mapping gathering radius.",
        ),
        (
            "mapping_radius_2_0",
            {"experiment_param_overrides": {"planning_gathering_factor_multiplier": 2.0}},
            "Legacy MAGICIAN two-times mapping gathering radius.",
        ),
        (
            "mapping_carving_5",
            {"experiment_param_overrides": {"carving_tolerance": 5.0}},
            "Lower proxy carving tolerance.",
        ),
        (
            "mapping_carving_20",
            {"experiment_param_overrides": {"carving_tolerance": 20.0}},
            "Higher proxy carving tolerance.",
        ),
    ]
    for name, changes, note in variants:
        specs.append(
            RunSpec(
                f"ablation_magician_da3_{name}",
                "ablations",
                "magician",
                "da3",
                ablation_budget,
                changes=changes,
                note=note,
            )
        )
    return specs


def _merge_config(base: Mapping[str, Any], spec: RunSpec, run_root: Path, gpu: int, collision: bool) -> dict:
    config = dict(base)
    config.update(
        {
            "numGPU": gpu,
            "test_scenes": ["eiffel"],
            "use_perfect_depth_map": spec.source == "gt",
            "kind_depth_map": "DA3",
            "validation_n_poses_in_trajectory": spec.budget - 1,
            "validation_max_start_positions": 1,
            "validation_memory_dir_name": f"myl13_{spec.run_id}",
            "scone_lmdb_dir_name": f"myl13_{spec.run_id}_lmdb",
            "lmdb_dir_name": f"myl13_{spec.run_id}_lmdb",
            "compute_collision": collision,
            "random_seed": 8,
            "torch_seed": 9,
            "experiment_metrics_enabled": True,
            "experiment_keep_frames": True,
            "experiment_run_id": spec.run_id,
            "experiment_budget_observations": spec.budget,
            "experiment_metrics_dir": str(run_root / "metrics"),
            "experiment_shared_collision_gate": True,
            "experiment_normalize_coverage_by_visibility": True,
            "experiment_param_overrides": {"planning_gathering_factor_multiplier": 1.0},
        }
    )
    for key, value in spec.changes.items():
        if key == "experiment_param_overrides":
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config


def _selected(spec: RunSpec, suite: str, run_ids: set[str]) -> bool:
    return (suite == "all" or spec.suite == suite) and (not run_ids or spec.run_id in run_ids)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/test/test_da3_eiffel_real_mesh_config.json")
    parser.add_argument("--output-dir", default="results/eiffel_phase4")
    parser.add_argument("--suite", choices=("main", "smoke", "ablations", "all"), default="main")
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--main-budget", type=int, default=6)
    parser.add_argument("--ablation-budget", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--collision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep successful executions in the output directory and retry missing/failed runs.",
    )
    args = parser.parse_args()
    if args.main_budget < 2 or args.ablation_budget < 2:
        parser.error("budgets must include at least two observations")

    project_root = Path(__file__).resolve().parents[1]
    base_path = Path(args.base_config)
    if not base_path.is_absolute():
        base_path = project_root / base_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    base = json.loads(base_path.read_text(encoding="utf-8"))
    selected = [
        spec
        for spec in _specs(args.main_budget, args.ablation_budget)
        if _selected(spec, args.suite, set(args.run_id))
    ]
    if args.run_id and len(selected) != len(set(args.run_id)):
        known = {spec.run_id for spec in selected}
        missing = sorted(set(args.run_id) - known)
        parser.error("unknown or out-of-suite run ids: " + ", ".join(missing))

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scene": "eiffel",
        "scope": "single_scene_only",
        "base_config": str(base_path),
        "height_scale_scene_units_per_meter": HEIGHT_SCALE,
        "ground_width_scale_scene_units_per_meter": WIDTH_SCALE,
        "main_gt_mesh_reference": "identical sampled Eiffel mesh reference per run",
        "renderer_zbuf_role": "post-run diagnostics only",
        "gt_feedback_to_da3": False,
        "fixed": {
            "start_index": 0,
            "random_seed": 8,
            "torch_seed": 9,
            "compute_collision": args.collision,
            "gt_mesh_pose_validity_prior": True,
            "gt_mesh_segment_collision_prior": args.collision,
            "rade_gs_prior": "MAGICIAN only",
        },
        "runs": [],
        "unselected_ablations": [],
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
        command = [sys.executable, entrypoint, "-c", str(config_path)]
        manifest["runs"].append(
            {
                **asdict(spec),
                "config": str(config_path),
                "command": command,
                "gt_mesh_reference": "Eiffel mesh sampled by setup_test_scene",
                "renderer_zbuf_feedback": False,
            }
        )
        if args.generate_only:
            continue

        completed = next(
            (
                execution
                for execution in executions
                if execution.get("run_id") == spec.run_id
                and execution.get("returncode") == 0
            ),
            None,
        )
        if completed is not None:
            continue

        executions = [
            execution for execution in executions if execution.get("run_id") != spec.run_id
        ]

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

    all_ablation_ids = [
        spec.run_id for spec in _specs(args.main_budget, args.ablation_budget) if spec.suite == "ablations"
    ]
    selected_ids = {spec.run_id for spec in selected}
    manifest["unselected_ablations"] = [run_id for run_id in all_ablation_ids if run_id not in selected_ids]
    _write_json(output_dir / "manifest.json", manifest)
    if executions:
        _write_json(executions_path, executions)
    print(output_dir / "manifest.json")


if __name__ == "__main__":
    main()
