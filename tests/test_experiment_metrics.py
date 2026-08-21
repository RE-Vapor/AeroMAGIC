import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from macarons.utility.depth_sources import DepthFrame
from macarons.utility.experiment_metrics import (
    TrajectoryMetricsRecorder,
    create_trajectory_metrics_recorder,
    write_online_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_planning_diagnostics import analyze_online_metrics  # noqa: E402


class ExperimentMetricsTests(unittest.TestCase):
    def _frame_data(self):
        mask = np.array([[[[True], [False]], [[True], [True]]]])
        frame = DepthFrame(
            rgb=np.ones((1, 2, 2, 3), dtype=np.float32),
            depth_z=np.array([[[[2.0], [3.0]], [[4.0], [5.0]]]], dtype=np.float32),
            valid_mask=np.ones_like(mask),
            error_mask=mask,
            R=np.eye(3)[None],
            T=np.zeros((1, 3)),
            confidence=np.array([[[[0.1], [0.2]], [[0.3], [0.4]]]], dtype=np.float32),
            source="DA3",
            frame_id=0,
            cache_metadata={
                "cache_key": "abc",
                "cache_hit": False,
                "camera": {"pose_conditioned": False, "intrinsics": [np.eye(3).tolist()]},
            },
        )
        return {
            "depth_frame": frame,
            "planning_mask": mask,
            "part_pc": np.zeros((3, 3)),
            "fov_proxy_points": np.zeros((2, 3)),
            "sgn_dists": np.array([0.1, -0.2]),
        }

    def test_online_record_contains_no_renderer_gt_and_writes_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = {
                "experiment_metrics_enabled": True,
                "experiment_metrics_dir": temporary,
                "da3_scene_units_per_meter": {"eiffel": 0.25},
            }
            recorder = TrajectoryMetricsRecorder(
                planner="scone",
                scene="eiffel",
                start_index=0,
                capture_dir=temporary,
                config=config,
                device="cpu",
            )
            recorder.record_frame(
                self._frame_data(), provider_seconds=0.1, geometry_seconds=0.2
            )
            recorder.record_coverage((np.array([0.2]), 17), 0.25)
            result = recorder.finalize(
                X_cam_history=np.array([[0, 0, 0], [3, 4, 0]]),
                V_cam_history=np.zeros((2, 2)),
                final_point_count=3,
            )
            path = write_online_metrics(config, result)
            persisted = json.loads(Path(path).read_text())
            self.assertTrue(persisted["online_only"])
            self.assertFalse(persisted["renderer_gt_read"])
            self.assertNotIn("zbuf", json.dumps(persisted).lower())
            self.assertEqual(persisted["trajectory"]["path_length_scene_units"], 5.0)
            self.assertEqual(persisted["trajectory"]["path_length_meters"], 20.0)
            self.assertEqual(persisted["frames"][0]["planning_pixels"], 3)

    def test_recorder_factory_is_opt_in(self):
        self.assertIsNone(
            create_trajectory_metrics_recorder(
                {}, planner="scone", scene="eiffel", start_index=0, capture_dir=".", device="cpu"
            )
        )

    def test_pioneer_metrics_separate_history_and_candidate_face_renders(self):
        recorder = create_trajectory_metrics_recorder(
            {
                "experiment_metrics_enabled": True,
                "planning_observation_mode": "cubemap6",
                "pioneer_face_size": 256,
                "pioneer_face_fov_degrees": 90.0,
            },
            planner="pioneer",
            scene="eiffel",
            start_index=0,
            capture_dir=".",
            device="cpu",
        )
        recorder.record_observation_bundle(
            {
                "bundle_id": 0,
                "face_count": 6,
                "face_names": ["front", "back", "left", "right", "up", "down"],
                "face_size": 256,
                "face_point_counts": [1, 1, 1, 1, 1, 1],
                "raw_point_count": 6,
                "unique_point_count": 6,
                "proxy_union_count": 4,
            },
            provider_seconds=0.1,
            geometry_seconds=0.2,
        )
        recorder.record_imagined_bundle_render(face_renders=6, kind="history")
        recorder.record_imagined_bundle_render(face_renders=6, kind="candidate")
        result = recorder.finalize(
            X_cam_history=np.array([[0, 0, 0]]),
            V_cam_history=np.zeros((1, 2)),
            final_point_count=6,
        )
        self.assertTrue(result["renderer_gt_read"])
        self.assertEqual(
            result["run"]["renderer_zbuf_role"],
            "online_planning_input_gt_mesh",
        )
        counters = result["pioneer_observation"]
        self.assertEqual(counters["imagined_history_face_render_count"], 6)
        self.assertEqual(counters["imagined_candidate_face_render_count"], 6)

    def test_pioneer_da3_metrics_prove_per_face_depth_provenance(self):
        recorder = create_trajectory_metrics_recorder(
            {
                "experiment_metrics_enabled": True,
                "planning_observation_mode": "cubemap6",
                "use_perfect_depth_map": False,
                "kind_depth_map": "DA3",
                "da3_model_id": "depth-anything/DA3NESTED-GIANT-LARGE",
                "da3_model_revision": "model-revision",
                "da3_model_config_sha256": "b" * 64,
                "da3_model_weights_sha256": "c" * 64,
                "da3_source_revision": "source-revision",
                "da3_source_tree_sha256": "a" * 64,
                "da3_window_size": 3,
                "da3_process_res": 504,
                "da3_process_res_method": "upper_bound_resize",
                "da3_output_height": 256,
                "da3_output_width": 256,
                "da3_confidence_percentile": None,
                "da3_cache_enabled": True,
                "da3_cache_dir": "isolated-da3-cache",
                "da3_scene_units_per_meter": {"HKUST": 0.2},
                "compute_collision": False,
                "experiment_shared_collision_gate": False,
            },
            planner="pioneer",
            scene="HKUST",
            start_index=0,
            capture_dir=".",
            device="cpu",
        )
        face_sources = [
            {
                "face_name": name,
                "depth_source": "DA3",
                "cache_key": f"cache-{name}",
                "cache_hit": False,
                "stream_id": f"pioneer/cubemap6/world/{name}",
                "pose_conditioned": False,
                "provider_valid_pixels": 4,
                "provider_error_pixels": 4,
                "planning_pixels": 4,
                "provider_seconds": 0.1,
            }
            for name in ["front", "back", "left", "right", "up", "down"]
        ]
        recorder.record_observation_bundle(
            {
                "bundle_id": 0,
                "face_count": 6,
                "face_names": ["front", "back", "left", "right", "up", "down"],
                "depth_source": "DA3",
                "rgb_source": "gt_mesh",
                "renderer_zbuf_read": False,
                "depth_inference_count": 6,
                "depth_cache_hit_count": 0,
                "depth_faces": face_sources,
                "artifact_transaction_version": "pioneer-bundle-commit-v1",
                "artifact_committed": True,
            },
            provider_seconds=0.6,
            geometry_seconds=0.2,
        )

        result = recorder.finalize(
            X_cam_history=np.array([[0, 0, 0]]),
            V_cam_history=np.zeros((1, 2)),
            final_point_count=1,
        )

        self.assertFalse(result["renderer_gt_read"])
        self.assertEqual(result["run"]["depth_source"], "DA3")
        self.assertEqual(
            result["run"]["renderer_zbuf_role"],
            "rgb_geometry_render_depth_discarded",
        )
        self.assertEqual(result["run"]["da3_window_size"], 3)
        self.assertEqual(result["run"]["da3_source_tree_sha256"], "a" * 64)
        self.assertEqual(result["run"]["da3_model_config_sha256"], "b" * 64)
        self.assertEqual(result["run"]["da3_model_weights_sha256"], "c" * 64)
        self.assertEqual(result["run"]["da3_process_res"], 504)
        self.assertEqual(
            result["run"]["da3_process_res_method"], "upper_bound_resize"
        )
        self.assertIsNone(result["run"]["da3_confidence_percentile"])
        self.assertIs(result["run"]["da3_cache_enabled"], True)
        self.assertEqual(result["run"]["da3_cache_dir"], "isolated-da3-cache")
        self.assertIs(result["run"]["gt_mesh_segment_collision_prior"], True)
        self.assertEqual(
            result["run"]["gt_mesh_segment_collision_prior_reason"],
            "legacy_first_beam_step_fallback",
        )
        bundle = result["pioneer_observation"]["bundles"][0]
        self.assertEqual(bundle["depth_source"], "DA3")
        self.assertEqual(bundle["depth_inference_count"], 6)
        self.assertEqual(bundle["depth_provider_event_count"], 6)
        self.assertTrue(bundle["artifact_committed"])
        self.assertEqual(
            result["pioneer_observation"]["artifact_committed_bundle_count"], 1
        )
        self.assertEqual(len(bundle["depth_faces"]), 6)
        self.assertEqual(
            {face["stream_id"] for face in bundle["depth_faces"]},
            {f"pioneer/cubemap6/world/{name}" for name in bundle["face_names"]},
        )

    def test_position_only_search_audit_matches_candidate_renders(self):
        recorder = create_trajectory_metrics_recorder(
            {
                "experiment_metrics_enabled": True,
                "planning_observation_mode": "cubemap6",
                "pioneer_planner_state_mode": "position_only",
                "pioneer_cubemap_rig_frame": "world",
                "pioneer_cubemap_extrinsics_version": (
                    "pytorch3d-world-axes-v1"
                ),
                "pioneer_canonical_orientation_indices": [2, 0],
            },
            planner="pioneer",
            scene="eiffel",
            start_index=0,
            capture_dir=".",
            device="cpu",
        )
        recorder.record_observation_bundle(
            {
                "bundle_id": 0,
                "face_count": 6,
                "face_names": ["front", "back", "left", "right", "up", "down"],
            },
            provider_seconds=0.0,
            geometry_seconds=0.0,
        )
        recorder.record_planner_structure(
            {
                "legacy_pose_state_count": 21600,
                "position_only_state_count": 432,
                "orientation_state_multiplier": 50,
            }
        )
        for _ in range(5):
            recorder.record_imagined_bundle_render(face_renders=6, kind="candidate")
        recorder.record_planner_search_step(
            {
                "planning_iteration": 0,
                "beam_step": 0,
                "parent_beam_count": 1,
                "raw_action_proposal_count": 6,
                "translation_action_proposal_count": 6,
                "orientation_action_proposal_count": 0,
                "generated_candidate_count": 5,
                "valid_state_candidate_count": 5,
                "observed_rejected_candidate_count": 0,
                "occupied_rejected_candidate_count": 0,
                "collision_rejected_candidate_count": 1,
                "rendered_candidate_count": 4,
                "retained_beam_count": 3,
                "search_seconds": 0.5,
            }
        )
        recorder.record_planner_search_step(
            {
                "planning_iteration": 0,
                "beam_step": 1,
                "parent_beam_count": 3,
                "raw_action_proposal_count": 18,
                "translation_action_proposal_count": 18,
                "orientation_action_proposal_count": 0,
                "generated_candidate_count": 18,
                "valid_state_candidate_count": 12,
                "observed_rejected_candidate_count": 3,
                "occupied_rejected_candidate_count": 1,
                "validation_bound_rejected_candidate_count": 2,
                "collision_rejected_candidate_count": 11,
                "rendered_candidate_count": 1,
                "retained_beam_count": 1,
                "search_seconds": 0.25,
            }
        )
        with self.assertRaisesRegex(
            ValueError,
            "generated candidates must equal valid, observed-rejected, and occupied-rejected",
        ):
            recorder.record_planner_search_step(
                {
                    "generated_candidate_count": 1,
                    "valid_state_candidate_count": 1,
                    "observed_rejected_candidate_count": 0,
                    "occupied_rejected_candidate_count": 1,
                    "collision_rejected_candidate_count": 0,
                    "rendered_candidate_count": 1,
                }
            )
        result = recorder.finalize(
            X_cam_history=np.array([[0, 0, 0]]),
            V_cam_history=np.zeros((1, 2)),
            final_point_count=0,
            planner_state_index_history=np.array([[2, 9, 3]]),
        )
        self.assertEqual(result["run"]["planner_state_dimension"], 3)
        search = result["planner_search"]
        self.assertEqual(search["state_mode"], "position_only")
        self.assertEqual(search["structure"]["position_only_state_count"], 432)
        self.assertEqual(search["totals"]["parent_beam_count"], 4)
        self.assertEqual(search["totals"]["raw_action_proposal_count"], 24)
        self.assertEqual(search["totals"]["orientation_action_proposal_count"], 0)
        self.assertEqual(search["totals"]["occupied_rejected_candidate_count"], 1)
        self.assertEqual(
            search["totals"]["validation_bound_rejected_candidate_count"], 2
        )
        self.assertEqual(search["totals"]["rendered_candidate_count"], 5)
        self.assertEqual(
            result["trajectory"]["planner_state_indices"], [[2, 9, 3]]
        )
        self.assertEqual(
            search["totals"]["rendered_candidate_count"],
            result["pioneer_observation"][
                "imagined_candidate_bundle_render_count"
            ],
        )
        self.assertAlmostEqual(search["totals"]["search_seconds"], 0.75)

    def test_n_tile_metrics_require_manifest_partition_and_recombine(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")

        with self.assertRaisesRegex(ValueError, "experiment_tile_partition"):
            TrajectoryMetricsRecorder(
                planner="magician",
                scene="joined",
                start_index=0,
                capture_dir=".",
                config={"experiment_tile_metrics_enabled": True},
                device="cpu",
            )

        class Cell:
            def __init__(self, points):
                self.cell_pts = torch.tensor(points, dtype=torch.float32)

        class Scene:
            def __init__(self, points):
                self.cells = {"all": Cell(points)}

        recorder = TrajectoryMetricsRecorder(
            planner="magician",
            scene="joined",
            start_index=0,
            capture_dir=".",
            config={
                "experiment_tile_metrics_enabled": True,
                "experiment_tile_partition": {
                    "axis": 0,
                    "boundaries": [10.0, 20.0],
                    "tile_ids": ["west", "center", "east"],
                },
            },
            device="cpu",
        )
        recorder.record_cross_tile_coverage(
            gt_scene=Scene([[-1, 0, 0], [5, 0, 0], [15, 0, 0], [25, 0, 0]]),
            covered_scene=Scene([[-1, 0, 0], [15, 0, 0], [25, 0, 0]]),
            reconstruction_points=[[-1, 0, 0], [15, 0, 0], [25, 0, 0]],
            surface_epsilon=0.1,
            normalization=1.0,
            global_raw=0.75,
            global_normalized=0.75,
        )
        result = recorder.finalize(
            X_cam_history=np.array([[0, 0, 0]]),
            V_cam_history=np.zeros((1, 2)),
            final_point_count=3,
        )
        self.assertNotIn("cross_tile", result)
        self.assertEqual(
            result["tile_metrics"]["coverage"][0]["tiles"]["center"][
                "covered_points"
            ],
            1,
        )
        self.assertAlmostEqual(
            result["tile_metrics"]["recombination_max_abs_raw_error"], 0.0
        )

    def test_post_run_gt_diagnostic_is_zero_and_marked_non_feedback(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            capture_dir = Path(temporary)
            depth = torch.tensor([[[[2.0], [3.0]], [[4.0], [-1.0]]]])
            mask = depth > -1
            torch.save({"zbuf": depth, "mask": mask}, capture_dir / "0.pt")
            online = {
                "online_only": True,
                "renderer_gt_read": False,
                "capture_dir": str(capture_dir),
                "scene_units_per_meter": 0.25,
                "run": {"run_id": "gt"},
                "planner": "scone",
                "scene": "eiffel",
                "start_index": 0,
                "frames": [{"frame_id": 0, "source": "GT", "intrinsics": None}],
            }
            result = analyze_online_metrics(online)
            self.assertTrue(result["diagnostic_only"])
            self.assertFalse(result["feedback_to_online_planner"])
            self.assertEqual(result["aggregate"]["depth_meters"]["rmse"], 0.0)
            self.assertEqual(
                result["aggregate"]["paired_ray_geometry_meters"]["rmse"], 0.0
            )


if __name__ == "__main__":
    unittest.main()
