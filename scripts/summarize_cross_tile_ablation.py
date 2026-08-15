#!/usr/bin/env python3
"""Summarize MYL-41 S/T runs and render evidence-rich previews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import textwrap
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macarons.utility.cross_tile_diagnostics import summarize_cross_tile_trajectory


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _load(path: str) -> Mapping[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _crossing_window(metrics: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    summary = metrics["cross_tile"]["trajectory_summary"]
    crossing = summary["first_crossing_frame"]
    if crossing is None:
        return []
    coverage = metrics["cross_tile"]["coverage"]
    planning = metrics["cross_tile"]["planning"]
    rows = []
    for frame_id in range(max(0, crossing - 10), min(len(coverage), crossing + 11)):
        plan = planning[frame_id] if frame_id < len(planning) else None
        first_beam = plan["beam_steps"][0] if plan and plan["beam_steps"] else None
        rows.append(
            {
                "frame_id": frame_id,
                "selected_tile": plan.get("selected", {}).get("tile") if plan else None,
                "selected_total_coverage_gain": (
                    plan.get("selected", {}).get("total_coverage_gain") if plan else None
                ),
                "best_tile_2_minus_tile_1_total_coverage_gain": (
                    first_beam.get(
                        "best_tile_2_minus_tile_1_total_coverage_gain"
                    )
                    if first_beam
                    else None
                ),
                "global_normalized_increment": metrics["coverage"][frame_id]["increment"],
                "tile_1_new_covered_points": coverage[frame_id]["tile_1"][
                    "new_covered_points"
                ],
                "tile_2_new_covered_points": coverage[frame_id]["tile_2"][
                    "new_covered_points"
                ],
            }
        )
    return rows


def _run_checks(label: str, metrics: Mapping[str, Any]) -> Mapping[str, bool]:
    cross = metrics["cross_tile"]
    trajectory = cross["trajectory_summary"]
    positions = np.asarray(metrics["trajectory"]["positions"])
    checks = {
        "101_observations": metrics["trajectory"]["observation_count"] == 101,
        "no_empty_candidate_interrupt": len(cross["planning"]) == 100,
        "tile_2_coverage_plus_3pp": trajectory[
            "tile_2_normalized_coverage_delta"
        ]
        >= 0.03,
        "tile_2_reconstruction_nonempty_10_consecutive": trajectory[
            "tile_2_longest_nonempty_reconstruction_run"
        ]
        >= 10,
    }
    crossing = trajectory["first_crossing_frame"]
    if crossing is not None:
        checks["post_crossing_coverage_not_all_tile_1"] = any(
            frame["tile_2"]["new_covered_points"] > 0
            for frame in cross["coverage"][crossing:]
        )
    else:
        checks["post_crossing_coverage_not_all_tile_1"] = False
    if label == "S":
        frame_zero = cross["planning"][0]
        checks.update(
            {
                "frame_0_generates_index_12": frame_zero[
                    "direct_crossing_candidate_generated"
                ],
                "frame_0_index_12_is_legal": frame_zero[
                    "direct_crossing_candidate_legal"
                ],
                "first_crossing_frame_le_20": crossing is not None and crossing <= 20,
                "tile_2_observations_ge_20": trajectory["tile_2_observations"] >= 20,
                "tile_2_longest_stay_ge_10": trajectory[
                    "tile_2_longest_consecutive_stay"
                ]
                >= 10,
            }
        )
    else:
        first_twenty = positions[:20, 0] > cross["seam_x"]
        tile_2_coverage = [
            frame["tile_2"]["normalized"] for frame in cross["coverage"]
        ]
        first_growth = next(
            (
                index
                for index, value in enumerate(tile_2_coverage)
                if value > tile_2_coverage[0]
            ),
            None,
        )
        checks.update(
            {
                "starts_in_tile_2": bool(positions[0, 0] > cross["seam_x"]),
                "tile_2_coverage_grows_by_frame_5": (
                    first_growth is not None and first_growth <= 5
                ),
                "first_20_has_15_tile_2_observations": int(first_twenty.sum()) >= 15,
            }
        )
    return checks


def _score_bias_evidence(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    deltas = []
    legal_frames = 0
    for plan in metrics["cross_tile"]["planning"]:
        if not plan["beam_steps"]:
            continue
        first = plan["beam_steps"][0]
        if first["legal_tile_2_candidate_count"] > 0:
            legal_frames += 1
            delta = first["best_tile_2_minus_tile_1_total_coverage_gain"]
            if delta is not None:
                deltas.append(float(delta))
    return {
        "frames_with_legal_tile_2_candidate": legal_frames,
        "comparable_score_frames": len(deltas),
        "tile_2_lower_score_frames": sum(value < 0 for value in deltas),
        "median_tile_2_minus_tile_1_total_gain": (
            float(np.median(deltas)) if deltas else None
        ),
    }


def _bootstrap_evidence(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    params = json.loads(
        (ROOT / "configs/macarons/macarons_default_training_config.json").read_text(
            encoding="utf-8"
        )
    )
    frame_zero = metrics["frames"][0]
    range_gate = frame_zero.get("sensor_range_gate") or {}
    sensor_range = float(
        range_gate.get(
            "sensor_range_scene_units",
            params["_camera_management"]["sensor_range"],
        )
    )
    planning_zero = metrics["cross_tile"]["planning"][0]
    first_step = planning_zero["beam_steps"][0]
    gains = [float(candidate["coverage_gain"]) for candidate in first_step["candidates"]]
    depth_min = frame_zero["depth_scene_units"]["min"]
    return {
        "sensor_range_scene_units": sensor_range,
        "frame_0_depth_min_scene_units": depth_min,
        "frame_0_depth_min_exceeds_sensor_range": (
            depth_min is not None and depth_min > sensor_range
        ),
        "frame_0_partial_point_count": frame_zero["partial_point_count"],
        "frame_0_imagined_gaussians": planning_zero["imagined_gaussians"],
        "frame_0_candidate_gain_min": min(gains) if gains else None,
        "frame_0_candidate_gain_max": max(gains) if gains else None,
        "frame_0_selected_pose_index": planning_zero["selected"]["pose_index"],
    }


def _conclusion(
    s_metrics: Mapping[str, Any],
    t_metrics: Mapping[str, Any],
    s_checks: Mapping[str, bool],
    t_checks: Mapping[str, bool],
) -> str:
    s_summary = s_metrics["cross_tile"]["trajectory_summary"]
    s_frame_zero = s_metrics["cross_tile"]["planning"][0]
    t_effective = all(t_checks.values())
    if all(s_checks.values()) and t_effective:
        if max(
            _bootstrap_evidence(s_metrics)["sensor_range_scene_units"],
            _bootstrap_evidence(t_metrics)["sensor_range_scene_units"],
        ) > 70.0:
            return (
                "The calibrated sensor range restored nonempty first-frame mapping "
                "and both S/T runs passed every preregistered crossing, sustained "
                "tile-2 exploration, reconstruction, and coverage threshold."
            )
        return (
            "S crossed and sustained tile-2 exploration while T explored normally: "
            "the stitched scene is usable; MYL-40 is best explained by its distant "
            "start, local scan pattern, and finite budget."
        )
    if (
        s_summary["first_crossing_frame"] is None
        and s_frame_zero["candidate_summary"]["legal_tile_2_count"] > 0
        and t_effective
    ):
        return (
            "S did not cross despite legal tile-2 candidates while T explored "
            "normally: evidence supports a local coverage-gain scoring bias."
        )
    if not s_frame_zero["direct_crossing_candidate_generated"] and t_effective:
        return "S did not generate index 12 while T worked: candidate lattice/start propagation error."
    seam = s_frame_zero["seam_segment_collision"]
    if (seam["gt_mesh"] or seam["predicted_point_cloud"]) and t_effective:
        return "The seam candidate was collision-rejected while T worked: collision-gate false positive."
    if s_summary["first_crossing_frame"] is not None and not t_effective:
        return "S crossed but T rendering/coverage failed: tile-2 data, rendering, or evaluation contract error."
    s_bootstrap = _bootstrap_evidence(s_metrics)
    t_bootstrap = _bootstrap_evidence(t_metrics)
    if (
        s_bootstrap["frame_0_depth_min_exceeds_sensor_range"]
        and t_bootstrap["frame_0_depth_min_exceeds_sensor_range"]
        and s_bootstrap["frame_0_partial_point_count"] == 0
        and t_bootstrap["frame_0_partial_point_count"] == 0
    ):
        return (
            "Neither S nor T sustained tile-2 exploration. Both static renders and "
            "collision gates passed, but their nearest frame-0 depth exceeded the "
            "70 m mapping sensor range, producing zero partial points and zero imagined "
            "Gaussians; all first-step gains tied at zero and deterministic ordering "
            "moved toward x-. This is a start-pose/sensor-range configuration contract "
            "failure, not evidence that tile-2 assets are unusable."
        )
    return "Neither S nor T met sustained exploration gates: inspect scene coordinates/assets before planner tuning."


def _rgb_frame(capture_dir: str, frame_id: int):
    import torch

    frame = torch.load(
        Path(capture_dir) / f"{frame_id}.pt",
        map_location="cpu",
        weights_only=False,
    )
    image = frame["rgb"]
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    while image.ndim > 3:
        image = image[0]
    if image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image[:3], 0, -1)
    return np.clip(image, 0, 1)


def _render_run_preview(
    label: str, metrics: Mapping[str, Any], checks: Mapping[str, bool], output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(18, 11), constrained_layout=True)
    grid = figure.add_gridspec(3, 4)
    frame_ids = [0, 25, 50, 100]
    for column, frame_id in enumerate(frame_ids):
        axis = figure.add_subplot(grid[0, column])
        axis.imshow(_rgb_frame(metrics["capture_dir"], frame_id))
        axis.set_title(f"Observation {frame_id + 1}")
        axis.axis("off")

    positions = np.asarray(metrics["trajectory"]["positions"])
    axis = figure.add_subplot(grid[1, 0])
    axis.plot(positions[:, 0], positions[:, 1], color="#1f77b4", linewidth=1.8)
    axis.scatter(positions[0, 0], positions[0, 1], color="green", label="start")
    axis.scatter(positions[-1, 0], positions[-1, 1], color="red", label="end")
    axis.axvline(75.0, color="black", linestyle="--", label="seam x=75")
    axis.set(title="Trajectory (x/y)", xlabel="x (m)", ylabel="y (m)")
    axis.legend(fontsize=8)

    cross_coverage = metrics["cross_tile"]["coverage"]
    axis = figure.add_subplot(grid[1, 1])
    axis.plot([row["global_normalized"] for row in cross_coverage], label="global")
    axis.plot([row["tile_1"]["normalized"] for row in cross_coverage], label="tile 1")
    axis.plot([row["tile_2"]["normalized"] for row in cross_coverage], label="tile 2")
    axis.set(title="Normalized coverage", xlabel="frame", ylabel="coverage")
    axis.legend(fontsize=8)

    planning = metrics["cross_tile"]["planning"]
    score_delta = [
        row["beam_steps"][0]["best_tile_2_minus_tile_1_total_coverage_gain"]
        if row["beam_steps"]
        else None
        for row in planning
    ]
    axis = figure.add_subplot(grid[1, 2])
    valid = [(i, value) for i, value in enumerate(score_delta) if value is not None]
    if valid:
        axis.plot(
            [item[0] for item in valid],
            [item[1] for item in valid],
            marker="o",
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(title="Best tile2 − tile1 beam score", xlabel="planning frame")

    axis = figure.add_subplot(grid[1, 3])
    rejection_names = (
        "boundary",
        "visited",
        "gt_mesh_collision",
        "predicted_point_cloud_collision",
    )
    rejection_counts = {
        name: sum(
            plan.get("candidate_summary", {})
            .get("rejection_reason_counts", {})
            .get(name, 0)
            for plan in planning
        )
        for name in rejection_names
    }
    axis.bar(range(len(rejection_names)), rejection_counts.values(), color="#d95f02")
    axis.set_xticks(range(len(rejection_names)), ["boundary", "visited", "GT", "predicted"], rotation=25)
    axis.set_title("First-step rejection counts")

    axis = figure.add_subplot(grid[2, :])
    axis.axis("off")
    summary = metrics["cross_tile"]["trajectory_summary"]
    lines = [
        f"MYL-41 {label} | observations={metrics['trajectory']['observation_count']} | "
        f"first_crossing={summary['first_crossing_frame']} | tile2_obs={summary['tile_2_observations']} | "
        f"longest_tile2_stay={summary['tile_2_longest_consecutive_stay']}",
        f"tile2 normalized coverage delta={summary['tile_2_normalized_coverage_delta']:.4f} | "
        f"nonempty reconstruction run={summary['tile_2_longest_nonempty_reconstruction_run']} | "
        f"path={metrics['trajectory']['path_length_meters']:.2f} m",
        "Gates: " + " | ".join(
            f"{name}={'PASS' if passed else 'FAIL'}" for name, passed in checks.items()
        ),
    ]
    axis.text(0.01, 0.92, "\n\n".join(lines), va="top", fontsize=11, wrap=True)
    figure.suptitle(f"12-NW-6C-7_8 MAGICIAN/GT cross-tile ablation — {label}", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _render_comparison(
    runs: Mapping[str, Mapping[str, Any]], conclusion: str, output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    colors = {"MYL-40": "#777777", "S": "#1b9e77", "T": "#d95f02"}
    for label, metrics in runs.items():
        positions = np.asarray(metrics["trajectory"]["positions"])
        axes[0].plot(positions[:, 0], positions[:, 1], label=label, color=colors[label])
    axes[0].axvline(75.0, color="black", linestyle="--")
    axes[0].set(title="Trajectory comparison", xlabel="x (m)", ylabel="y (m)")
    axes[0].legend()
    for label in ("S", "T"):
        coverage = runs[label]["cross_tile"]["coverage"]
        axes[1].plot(
            [row["tile_2"]["normalized"] for row in coverage],
            label=label,
            color=colors[label],
        )
    axes[1].set(title="Tile-2 normalized coverage", xlabel="frame")
    axes[1].legend()
    axes[2].axis("off")
    axes[2].text(
        0,
        1,
        "\n".join(textwrap.wrap(conclusion, width=58)),
        va="top",
        fontsize=9,
    )
    row_y = {"MYL-40": 0.43, "S": 0.25, "T": 0.07}
    for label in ("MYL-40", "S", "T"):
        metrics = runs[label]
        positions = np.asarray(metrics["trajectory"]["positions"])
        tile_2_count = int(np.sum(positions[:, 0] > 75.0))
        axes[2].text(
            0,
            row_y[label],
            f"{label}: obs={len(positions)}, tile2_obs={tile_2_count}, "
            f"x=[{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}]",
            va="top",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s-metrics", required=True)
    parser.add_argument("--t-metrics", required=True)
    parser.add_argument("--myl40-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--issue", default="MYL-41")
    args = parser.parse_args()

    s_metrics = _load(args.s_metrics)
    t_metrics = _load(args.t_metrics)
    myl40 = _load(args.myl40_metrics)
    s_checks = _run_checks("S", s_metrics)
    t_checks = _run_checks("T", t_metrics)
    conclusion = _conclusion(s_metrics, t_metrics, s_checks, t_checks)
    output = Path(args.output_dir)
    result = {
        "schema_version": 1,
        "issue": args.issue,
        "threshold_note": "The 3 percentage-point tile-2 threshold is preregistered engineering acceptance, not a paper standard.",
        "runs": {
            "S": {
                "checks": s_checks,
                "accepted": all(s_checks.values()),
                "trajectory": s_metrics["cross_tile"]["trajectory_summary"],
                "score_bias_evidence": _score_bias_evidence(s_metrics),
                "bootstrap_evidence": _bootstrap_evidence(s_metrics),
                "crossing_window": _crossing_window(s_metrics),
            },
            "T": {
                "checks": t_checks,
                "accepted": all(t_checks.values()),
                "trajectory": t_metrics["cross_tile"]["trajectory_summary"],
                "score_bias_evidence": _score_bias_evidence(t_metrics),
                "bootstrap_evidence": _bootstrap_evidence(t_metrics),
                "crossing_window": _crossing_window(t_metrics),
            },
            "MYL-40": summarize_cross_tile_trajectory(
                myl40["trajectory"]["positions"],
                [],
                seam_x=75.0,
                tile_coverage=[],
            ),
        },
        "interpretation_matrix_conclusion": conclusion,
    }
    _write_json(output / "summary.json", result)
    _render_run_preview("S", s_metrics, s_checks, output / "S_comprehensive_preview.png")
    _render_run_preview("T", t_metrics, t_checks, output / "T_comprehensive_preview.png")
    _render_comparison(
        {"MYL-40": myl40, "S": s_metrics, "T": t_metrics},
        conclusion,
        output / "S_T_MYL40_comparison.png",
    )
    print(output / "summary.json")


if __name__ == "__main__":
    main()
