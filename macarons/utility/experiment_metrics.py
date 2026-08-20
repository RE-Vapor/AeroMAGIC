"""Telemetry for planning experiments.

Single-view runs keep renderer ``zbuf`` and renderer masks in offline
diagnostics.  PIONEER's explicit Python mesh/GT pilot records only aggregate
six-face bundle statistics from geometry already consumed by the planner.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Optional

import numpy as np

from .cross_tile_diagnostics import (
    compute_partitioned_scene_coverage,
    compute_tiled_scene_coverage,
    summarize_cross_tile_trajectory,
    validate_tile_partition,
)


def _as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _scalar(value: Any) -> float:
    if isinstance(value, (tuple, list)) and value:
        return _scalar(value[0])
    array = _as_numpy(value).reshape(-1)
    return float(array[0]) if array.size else math.nan


def _finite_stats(values: Any, mask: Any = None) -> Mapping[str, Optional[float]]:
    array = _as_numpy(values).astype(np.float64, copy=False).reshape(-1)
    if mask is not None:
        selected = _as_numpy(mask).astype(bool, copy=False).reshape(-1)
        if selected.shape != array.shape:
            raise ValueError("Metric values and mask must contain the same number of elements.")
        array = array[selected]
    array = array[np.isfinite(array)]
    if not array.size:
        return {"min": None, "p10": None, "median": None, "p90": None, "max": None}
    return {
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def _count(value: Any) -> int:
    try:
        return int(len(value))
    except TypeError:
        return int(_as_numpy(value).size)


class TrajectoryMetricsRecorder:
    """Collect online-only telemetry for one completed trajectory."""

    schema_version = 1

    def __init__(
        self,
        *,
        planner: str,
        scene: str,
        start_index: int,
        capture_dir: str,
        config: Any,
        device: Any,
        run_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.planner = str(planner)
        self.scene = str(scene)
        self.start_index = int(start_index)
        self.capture_dir = str(Path(capture_dir).resolve())
        self.device = device
        self.run_metadata = dict(run_metadata or {})
        self.scene_units_per_meter = float(
            _config_value(config, "da3_scene_units_per_meter", {}).get(scene, 1.0)
        )
        self.frames = []
        self.observation_bundles = []
        self.imagined_bundle_renders = 0
        self.imagined_face_renders = 0
        self.imagined_history_bundle_renders = 0
        self.imagined_history_face_renders = 0
        self.imagined_candidate_bundle_renders = 0
        self.imagined_candidate_face_renders = 0
        self.planner_search_steps = []
        self.planner_structure_audit = None
        self.coverage = []
        cross_tile_enabled = _config_value(
            config, "experiment_cross_tile_diagnostics_enabled", False
        )
        if type(cross_tile_enabled) is not bool:
            raise ValueError(
                "experiment_cross_tile_diagnostics_enabled must be a boolean."
            )
        self.cross_tile_enabled = cross_tile_enabled
        self.cross_tile_seam_x = float(
            _config_value(config, "experiment_cross_tile_seam_x", 75.0)
        )
        self.cross_tile_min_x_index = int(
            _config_value(config, "experiment_cross_tile_min_x_index", 12)
        )
        self.cross_tile_gate_from_index = list(
            _config_value(
                config, "experiment_cross_tile_gate_from_index", [11, 9, 5, 1, 3]
            )
        )
        self.cross_tile_gate_to_index = list(
            _config_value(
                config, "experiment_cross_tile_gate_to_index", [12, 9, 5, 1, 3]
            )
        )
        self.cross_tile_coverage = []
        tile_metrics_enabled = _config_value(
            config, "experiment_tile_metrics_enabled", False
        )
        if type(tile_metrics_enabled) is not bool:
            raise ValueError("experiment_tile_metrics_enabled must be a boolean.")
        self.tile_metrics_enabled = tile_metrics_enabled
        tile_partition = _config_value(config, "experiment_tile_partition", None)
        if self.tile_metrics_enabled:
            if tile_partition is None:
                raise ValueError(
                    "experiment_tile_partition is required when tile metrics are enabled."
                )
            self.tile_partition = validate_tile_partition(tile_partition)
        else:
            self.tile_partition = None
        self.tile_coverage = []
        self.planning_diagnostics = []
        self.started_at = time.perf_counter()
        self._reset_cuda_peak()

    def _cuda(self):
        try:
            import torch

            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                return torch
        except (ImportError, RuntimeError):
            return None
        return None

    def synchronize(self) -> None:
        torch = self._cuda()
        if torch is not None:
            torch.cuda.synchronize(self.device)

    def _reset_cuda_peak(self) -> None:
        torch = self._cuda()
        if torch is not None:
            torch.cuda.reset_peak_memory_stats(self.device)

    def record_frame(
        self,
        frame_data: Mapping[str, Any],
        *,
        provider_seconds: float,
        geometry_seconds: float,
    ) -> None:
        frame = frame_data["depth_frame"]
        valid_mask = _as_numpy(frame.valid_mask).astype(bool, copy=False)
        error_mask = _as_numpy(frame.error_mask).astype(bool, copy=False)
        planning_mask = _as_numpy(frame_data["planning_mask"]).astype(bool, copy=False)
        pixel_count = int(planning_mask.size)
        metadata = dict(getattr(frame, "cache_metadata", {}) or {})
        camera_metadata = metadata.get("camera", {})
        intrinsics = camera_metadata.get("intrinsics")
        if intrinsics:
            intrinsics = intrinsics[-1]

        confidence = getattr(frame, "confidence", None)
        confidence_stats = None
        if confidence is not None:
            confidence_stats = _finite_stats(confidence, valid_mask)

        signed_distances = frame_data.get("sgn_dists")
        self.frames.append(
            {
                "frame_id": int(frame.frame_id),
                "source": str(frame.source),
                "cache_key": metadata.get("cache_key"),
                "cache_hit": metadata.get("cache_hit"),
                "pose_conditioned": camera_metadata.get("pose_conditioned"),
                "intrinsics": intrinsics,
                "pixel_count": pixel_count,
                "valid_pixels": int(valid_mask.sum()),
                "confidence_accepted_pixels": int(error_mask.sum()),
                "planning_pixels": int(planning_mask.sum()),
                "planning_ratio": float(planning_mask.mean()) if pixel_count else 0.0,
                "depth_scene_units": _finite_stats(frame.depth_z, planning_mask),
                "confidence": confidence_stats,
                "partial_point_count": _count(frame_data["part_pc"]),
                "proxy_points_in_fov": _count(frame_data["fov_proxy_points"]),
                "signed_distance_scene_units": (
                    _finite_stats(signed_distances) if signed_distances is not None else None
                ),
                "provider_seconds": float(provider_seconds),
                "geometry_seconds": float(geometry_seconds),
                "sensor_range_gate": frame_data.get("sensor_range_gate"),
            }
        )

    def record_coverage(self, raw: Any, normalized: float) -> None:
        previous = self.coverage[-1]["normalized"] if self.coverage else 0.0
        self.coverage.append(
            {
                "frame_id": len(self.coverage),
                "raw": _scalar(raw),
                "normalized": float(normalized),
                "increment": float(normalized - previous),
            }
        )

    def record_observation_bundle(
        self,
        bundle_data: Mapping[str, Any],
        *,
        provider_seconds: float,
        geometry_seconds: float,
    ) -> None:
        """Record one PIONEER full-sphere observation as one planner event."""

        face_point_counts = [
            int(value) for value in bundle_data.get("face_point_counts", [])
        ]
        self.observation_bundles.append(
            {
                "bundle_id": int(bundle_data["bundle_id"]),
                "face_count": int(bundle_data.get("face_count", len(face_point_counts))),
                "face_names": list(bundle_data.get("face_names", [])),
                "face_size": int(bundle_data.get("face_size", 0)),
                "rig_frame": bundle_data.get("rig_frame"),
                "extrinsics_version": bundle_data.get("extrinsics_version"),
                "capture_timestamp_utc": bundle_data.get("capture_timestamp_utc"),
                "capture_timestamp_unix_ns": bundle_data.get(
                    "capture_timestamp_unix_ns"
                ),
                "face_point_counts": face_point_counts,
                "raw_point_count": int(bundle_data.get("raw_point_count", 0)),
                "unique_point_count": int(bundle_data.get("unique_point_count", 0)),
                "deduplicated_point_count": int(
                    bundle_data.get("raw_point_count", 0)
                    - bundle_data.get("unique_point_count", 0)
                ),
                "proxy_union_count": int(bundle_data.get("proxy_union_count", 0)),
                "provider_seconds": float(provider_seconds),
                "geometry_seconds": float(geometry_seconds),
            }
        )

    def record_imagined_bundle_render(
        self, *, face_renders: int = 6, kind: str
    ) -> None:
        """Count one candidate/history bundle and its physical face renders."""

        if kind not in {"history", "candidate"}:
            raise ValueError("imagined bundle render kind must be history or candidate.")
        self.imagined_bundle_renders += 1
        self.imagined_face_renders += int(face_renders)
        if kind == "history":
            self.imagined_history_bundle_renders += 1
            self.imagined_history_face_renders += int(face_renders)
        else:
            self.imagined_candidate_bundle_renders += 1
            self.imagined_candidate_face_renders += int(face_renders)

    def record_planner_search_step(self, step: Mapping[str, Any]) -> None:
        """Record one beam-search layer using planner-state, not camera, terms."""

        integer_fields = (
            "planning_iteration",
            "beam_step",
            "parent_beam_count",
            "raw_action_proposal_count",
            "translation_action_proposal_count",
            "orientation_action_proposal_count",
            "generated_candidate_count",
            "valid_state_candidate_count",
            "observed_rejected_candidate_count",
            "occupied_rejected_candidate_count",
            "collision_rejected_candidate_count",
            "rendered_candidate_count",
            "retained_beam_count",
        )
        normalized = {name: int(step.get(name, 0)) for name in integer_fields}
        normalized["search_seconds"] = float(step.get("search_seconds", 0.0))
        if any(normalized[name] < 0 for name in integer_fields):
            raise ValueError("planner search counts must be non-negative")
        if normalized["orientation_action_proposal_count"] > normalized[
            "raw_action_proposal_count"
        ]:
            raise ValueError(
                "orientation action proposals cannot exceed raw action proposals"
            )
        if normalized["generated_candidate_count"] != (
            normalized["valid_state_candidate_count"]
            + normalized["observed_rejected_candidate_count"]
            + normalized["occupied_rejected_candidate_count"]
        ):
            raise ValueError(
                "generated candidates must equal valid, observed-rejected, and "
                "occupied-rejected candidates"
            )
        if normalized["valid_state_candidate_count"] != (
            normalized["collision_rejected_candidate_count"]
            + normalized["rendered_candidate_count"]
        ):
            raise ValueError(
                "valid-state candidates must equal collision-rejected and rendered "
                "candidates"
            )
        self.planner_search_steps.append(normalized)

    def record_planner_structure(self, audit: Mapping[str, Any]) -> None:
        if not isinstance(audit, Mapping):
            raise TypeError("planner structure audit must be a mapping")
        self.planner_structure_audit = dict(audit)

    def record_cross_tile_coverage(
        self,
        *,
        gt_scene: Any,
        covered_scene: Any,
        reconstruction_points: Any,
        surface_epsilon: float,
        normalization: float,
        global_raw: Any,
        global_normalized: float,
    ) -> None:
        if not self.cross_tile_enabled and not self.tile_metrics_enabled:
            return
        global_raw_value = _scalar(global_raw)
        if self.tile_metrics_enabled:
            tiled_frame = dict(
                compute_tiled_scene_coverage(
                    gt_scene,
                    covered_scene,
                    tile_partition=self.tile_partition,
                    surface_epsilon=surface_epsilon,
                    normalization=normalization,
                    reconstruction_points=reconstruction_points,
                )
            )
            tiled_frame["frame_id"] = len(self.tile_coverage)
            tiled_frame["global_raw"] = global_raw_value
            tiled_frame["global_normalized"] = float(global_normalized)
            tiled_frame["combined_raw_error"] = float(
                tiled_frame["combined"]["raw"] - tiled_frame["global_raw"]
            )
            tiled_frame["combined_normalized_error"] = float(
                tiled_frame["combined"]["normalized"]
                - tiled_frame["global_normalized"]
            )
            previous_tiles = (
                self.tile_coverage[-1]["tiles"] if self.tile_coverage else None
            )
            for tile_id, tile in tiled_frame["tiles"].items():
                previous = previous_tiles[tile_id] if previous_tiles else None
                tile["new_covered_points"] = tile["covered_points"] - (
                    previous["covered_points"] if previous else 0
                )
                tile["new_reconstruction_points"] = tile[
                    "reconstruction_points"
                ] - (previous["reconstruction_points"] if previous else 0)
            self.tile_coverage.append(tiled_frame)
        if not self.cross_tile_enabled:
            return
        frame = dict(
            compute_partitioned_scene_coverage(
                gt_scene,
                covered_scene,
                seam_x=self.cross_tile_seam_x,
                surface_epsilon=surface_epsilon,
                normalization=normalization,
                reconstruction_points=reconstruction_points,
            )
        )
        frame["frame_id"] = len(self.cross_tile_coverage)
        frame["global_raw"] = global_raw_value
        frame["global_normalized"] = float(global_normalized)
        frame["combined_raw_error"] = float(
            frame["combined"]["raw"] - frame["global_raw"]
        )
        frame["combined_normalized_error"] = float(
            frame["combined"]["normalized"] - frame["global_normalized"]
        )
        if self.cross_tile_coverage:
            previous = self.cross_tile_coverage[-1]
            for tile in ("tile_1", "tile_2"):
                frame[tile]["new_covered_points"] = (
                    frame[tile]["covered_points"]
                    - previous[tile]["covered_points"]
                )
                frame[tile]["new_reconstruction_points"] = (
                    frame[tile]["reconstruction_points"]
                    - previous[tile]["reconstruction_points"]
                )
        else:
            for tile in ("tile_1", "tile_2"):
                frame[tile]["new_covered_points"] = frame[tile]["covered_points"]
                frame[tile]["new_reconstruction_points"] = frame[tile][
                    "reconstruction_points"
                ]
        self.cross_tile_coverage.append(frame)

    def record_planning_diagnostic(self, diagnostic: Mapping[str, Any]) -> None:
        if self.cross_tile_enabled:
            self.planning_diagnostics.append(dict(diagnostic))

    def finalize(
        self,
        *,
        X_cam_history: Any,
        V_cam_history: Any,
        final_point_count: int,
        pose_index_history: Any = None,
        planner_state_index_history: Any = None,
    ) -> Mapping[str, Any]:
        self.synchronize()
        positions = _as_numpy(X_cam_history).astype(np.float64, copy=False).reshape(-1, 3)
        orientations = _as_numpy(V_cam_history).astype(np.float64, copy=False)
        path_length = (
            float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
            if len(positions) > 1
            else 0.0
        )
        cuda_metrics = None
        torch = self._cuda()
        if torch is not None:
            cuda_metrics = {
                "peak_allocated_mib": float(
                    torch.cuda.max_memory_allocated(self.device) / (1024.0**2)
                ),
                "peak_reserved_mib": float(
                    torch.cuda.max_memory_reserved(self.device) / (1024.0**2)
                ),
            }
        metrics = {
            "schema_version": self.schema_version,
            "online_only": True,
            "renderer_gt_read": (
                self.run_metadata.get("planning_observation_mode") == "cubemap6"
            ),
            "planner": self.planner,
            "scene": self.scene,
            "start_index": self.start_index,
            "capture_dir": self.capture_dir,
            "scene_units_per_meter": self.scene_units_per_meter,
            "run": self.run_metadata,
            "frames": self.frames,
            "coverage": self.coverage,
            "trajectory": {
                "positions": positions.tolist(),
                "orientations": orientations.tolist(),
                "observation_count": int(len(positions)),
                "path_length_scene_units": path_length,
                "path_length_meters": path_length / self.scene_units_per_meter,
                "final_point_count": int(final_point_count),
            },
            "latency": {
                "trajectory_seconds": float(time.perf_counter() - self.started_at),
                "provider_seconds": float(
                    sum(frame["provider_seconds"] for frame in self.frames)
                    + sum(
                        bundle["provider_seconds"]
                        for bundle in self.observation_bundles
                    )
                ),
                "geometry_seconds": float(
                    sum(frame["geometry_seconds"] for frame in self.frames)
                    + sum(
                        bundle["geometry_seconds"]
                        for bundle in self.observation_bundles
                    )
                ),
            },
            "cuda": cuda_metrics,
        }
        if planner_state_index_history is not None:
            planner_states = _as_numpy(planner_state_index_history)
            if planner_states.ndim != 2 or planner_states.shape[0] != len(positions):
                raise ValueError(
                    "planner state history must be a 2D array with one state "
                    "per captured observation"
                )
            expected_dimension = self.run_metadata.get("planner_state_dimension")
            if (
                expected_dimension is not None
                and planner_states.shape[1] != int(expected_dimension)
            ):
                raise ValueError(
                    "planner state history dimension does not match run metadata"
                )
            metrics["trajectory"]["planner_state_indices"] = (
                planner_states.astype(np.int64, copy=False).tolist()
            )
        if self.observation_bundles:
            metrics["pioneer_observation"] = {
                "schema_version": 1,
                "bundle_count": len(self.observation_bundles),
                "real_face_render_count": int(
                    sum(bundle["face_count"] for bundle in self.observation_bundles)
                ),
                "imagined_bundle_render_count": self.imagined_bundle_renders,
                "imagined_face_render_count": self.imagined_face_renders,
                "imagined_history_bundle_render_count": (
                    self.imagined_history_bundle_renders
                ),
                "imagined_history_face_render_count": (
                    self.imagined_history_face_renders
                ),
                "imagined_candidate_bundle_render_count": (
                    self.imagined_candidate_bundle_renders
                ),
                "imagined_candidate_face_render_count": (
                    self.imagined_candidate_face_renders
                ),
                "bundles": self.observation_bundles,
            }
        if self.planner_search_steps:
            count_fields = (
                "parent_beam_count",
                "raw_action_proposal_count",
                "translation_action_proposal_count",
                "orientation_action_proposal_count",
                "generated_candidate_count",
                "valid_state_candidate_count",
                "observed_rejected_candidate_count",
                "occupied_rejected_candidate_count",
                "collision_rejected_candidate_count",
                "rendered_candidate_count",
                "retained_beam_count",
            )
            totals = {
                name: int(sum(step[name] for step in self.planner_search_steps))
                for name in count_fields
            }
            totals["search_seconds"] = float(
                sum(step["search_seconds"] for step in self.planner_search_steps)
            )
            if (
                self.observation_bundles
                and totals["rendered_candidate_count"]
                != self.imagined_candidate_bundle_renders
            ):
                raise ValueError(
                    "planner rendered candidate count does not match imagined "
                    "candidate bundle renders"
                )
            metrics["planner_search"] = {
                "schema_version": 1,
                "state_mode": self.run_metadata.get("pioneer_planner_state_mode"),
                "state_dimension": self.run_metadata.get(
                    "planner_state_dimension"
                ),
                "cubemap_rig_frame": self.run_metadata.get(
                    "pioneer_cubemap_rig_frame"
                ),
                "cubemap_extrinsics_version": self.run_metadata.get(
                    "pioneer_cubemap_extrinsics_version"
                ),
                "canonical_orientation_indices": self.run_metadata.get(
                    "pioneer_canonical_orientation_indices"
                ),
                "steps": self.planner_search_steps,
                "totals": totals,
                "structure": self.planner_structure_audit,
            }
        if self.cross_tile_enabled:
            pose_indices = [] if pose_index_history is None else pose_index_history
            metrics["cross_tile"] = {
                "schema_version": 1,
                "seam_x": self.cross_tile_seam_x,
                "tile_1_definition": "x <= seam_x",
                "tile_2_definition": "x > seam_x",
                "tile_2_min_x_index": self.cross_tile_min_x_index,
                "coverage": self.cross_tile_coverage,
                "planning": self.planning_diagnostics,
                "trajectory_summary": summarize_cross_tile_trajectory(
                    positions,
                    pose_indices,
                    seam_x=self.cross_tile_seam_x,
                    tile_coverage=self.cross_tile_coverage,
                ),
            }
        if self.tile_metrics_enabled:
            metrics["tile_metrics"] = {
                "schema_version": 1,
                "partition": self.tile_partition,
                "coverage": self.tile_coverage,
                "recombination_max_abs_raw_error": max(
                    (
                        abs(frame["combined_raw_error"])
                        for frame in self.tile_coverage
                    ),
                    default=0.0,
                ),
                "recombination_max_abs_normalized_error": max(
                    (
                        abs(frame["combined_normalized_error"])
                        for frame in self.tile_coverage
                    ),
                    default=0.0,
                ),
            }
        return metrics


def _config_value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def experiment_metrics_enabled(config: Any) -> bool:
    value = _config_value(config, "experiment_metrics_enabled", False)
    if type(value) is not bool:
        raise ValueError("experiment_metrics_enabled must be a boolean.")
    return value


def create_trajectory_metrics_recorder(
    config: Any,
    *,
    planner: str,
    scene: str,
    start_index: int,
    capture_dir: str,
    device: Any,
) -> Optional[TrajectoryMetricsRecorder]:
    if not experiment_metrics_enabled(config):
        return None
    run_metadata = {
        "run_id": _config_value(config, "experiment_run_id", None),
        "seed": _config_value(config, "random_seed", None),
        "torch_seed": _config_value(config, "torch_seed", None),
        "budget_observations": _config_value(config, "experiment_budget_observations", None),
        "debug_profile": _config_value(config, "debug_profile", None),
        "debug_only": _config_value(config, "debug_only", False),
        "coverage_comparable": _config_value(config, "coverage_comparable", True),
        "compute_collision": _config_value(config, "compute_collision", None),
        "gt_mesh_reference": True,
        "renderer_zbuf_role": (
            "online_planning_input_gt_mesh"
            if _config_value(config, "planning_observation_mode", "single")
            == "cubemap6"
            else "offline_diagnostic_only"
        ),
        "gt_feedback_to_da3": False,
        "gt_mesh_pose_validity_prior": True,
        "gt_mesh_segment_collision_prior": bool(
            _config_value(config, "compute_collision", False)
        ),
        "rade_gs_prior": planner.lower() in {"magician", "pioneer"},
        "planning_observation_mode": _config_value(
            config, "planning_observation_mode", "single"
        ),
        "pioneer_face_count": (
            6
            if _config_value(config, "planning_observation_mode", "single")
            == "cubemap6"
            else 1
        ),
        "pioneer_face_size": _config_value(config, "pioneer_face_size", None),
        "pioneer_face_fov_degrees": _config_value(
            config, "pioneer_face_fov_degrees", None
        ),
        "pioneer_planner_state_mode": _config_value(
            config, "pioneer_planner_state_mode", "legacy_pose5d"
        ),
        "planner_state_dimension": (
            3
            if _config_value(
                config, "pioneer_planner_state_mode", "legacy_pose5d"
            )
            == "position_only"
            else 5
        ),
        "pioneer_cubemap_rig_frame": _config_value(
            config, "pioneer_cubemap_rig_frame", "body"
        ),
        "pioneer_cubemap_extrinsics_version": _config_value(
            config, "pioneer_cubemap_extrinsics_version", None
        ),
        "pioneer_canonical_orientation_indices": _config_value(
            config, "pioneer_canonical_orientation_indices", None
        ),
        "pioneer_filter_occupied_position_candidates": _config_value(
            config, "pioneer_filter_occupied_position_candidates", False
        ),
        "validation_require_complete_occupied_pose": _config_value(
            config, "validation_require_complete_occupied_pose", False
        ),
    }
    return TrajectoryMetricsRecorder(
        planner=planner,
        scene=scene,
        start_index=start_index,
        capture_dir=capture_dir,
        config=config,
        device=device,
        run_metadata=run_metadata,
    )


def write_online_metrics(config: Any, metrics: Mapping[str, Any]) -> Optional[str]:
    output_dir = _config_value(config, "experiment_metrics_dir", None)
    if output_dir is None:
        return None
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    filename = "{planner}_{scene}_{start}.online.json".format(
        planner=metrics["planner"], scene=metrics["scene"], start=metrics["start_index"]
    )
    path = root / filename
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return str(path)
