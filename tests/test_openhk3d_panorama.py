import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from tools.openhk3d_panorama import PanoramaError, build_render_config, pose_index_to_pose


class OpenHK3DPanoramaTests(unittest.TestCase):
    def camera(self):
        return {
            "x_min": [-8.0, -4.0, -8.0],
            "x_max": [24.0, 10.0, 9.0],
            "pose_l": 24,
            "pose_w": 11,
            "pose_h": 13,
            "pose_n_theta": 5,
            "pose_n_azim": 10,
            "start_positions": [[0, 9, 6, 1, 3]],
        }

    def test_pose_index_matches_magician_grid_semantics(self):
        pose = pose_index_to_pose(self.camera(), [0, 9, 6, 1, 3])
        self.assertAlmostEqual(pose["position_world_xyz"][0], -7.3333333333)
        self.assertAlmostEqual(pose["elevation_degrees"], -30.0)
        self.assertAlmostEqual(pose["azimuth_degrees"], 108.0)

    def test_pose_index_rejects_out_of_bounds_angle(self):
        with self.assertRaises(PanoramaError):
            pose_index_to_pose(self.camera(), [0, 9, 6, 1, 10])

    def test_render_config_fingerprint_is_output_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "scene"
            scene.mkdir()
            (scene / "scene.obj").write_text("v 0 0 0\n", encoding="utf-8")
            (scene / "settings.json").write_text(
                json.dumps({"camera": self.camera()}), encoding="utf-8"
            )
            (scene / "occupied_pose.json").write_text(
                json.dumps({"X_idx": [[0, 9, 6]], "occupied": [False]}),
                encoding="utf-8",
            )
            args = Namespace(
                scene_dir=str(scene),
                output=str(Path(directory) / "output-a"),
                blender="/unused",
                batch_id="test-batch",
                width=1024,
                height=512,
                samples=4,
                compute_device="CPU",
                max_depth=100.0,
                pose_limit=None,
            )
            first = build_render_config(args)
            args.output = str(Path(directory) / "output-b")
            second = build_render_config(args)
            self.assertEqual(first["scope_fingerprint"], second["scope_fingerprint"])


if __name__ == "__main__":
    unittest.main()
