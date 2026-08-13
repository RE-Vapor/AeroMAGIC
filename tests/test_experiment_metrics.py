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
