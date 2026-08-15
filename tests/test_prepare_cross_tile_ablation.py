import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PrepareCrossTileAblationTests(unittest.TestCase):
    def test_prepares_independent_views_with_only_first_start_changed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source.mkdir()
            settings = {
                "camera": {
                    "pose_l": 24,
                    "pose_w": 11,
                    "pose_h": 13,
                    "pose_n_theta": 5,
                    "pose_n_azim": 10,
                    "x_min": [-8.0, -3.9913498, -8.0],
                    "x_max": [24.0, 10.151507343, 9.333333333],
                    "start_positions": [[0, 9, 6, 1, 3], [1, 2, 3, 4, 5]],
                }
            }
            (source / "settings.json").write_text(json.dumps(settings))
            for name in (
                "12-NW-6C-7_8.obj",
                "12-NW-6C-7_8.mtl",
                "occupied_pose.pt",
                "assembly-manifest.json",
            ):
                (source / name).write_text(name)
            (source / "occupied_pose.json").write_text(
                json.dumps(
                    {
                        "X_idx": [[11, 9, 5], [12, 9, 5]],
                        "occupied": [False, False],
                    }
                )
            )
            baseline = root / "baseline.json"
            baseline.write_text(
                json.dumps(
                    {
                        "validation_n_poses_in_trajectory": 500,
                        "experiment_budget_observations": 501,
                        "beam_width": 10,
                        "beam_steps": 10,
                    }
                )
            )
            output = root / "output"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/prepare_cross_tile_ablation.py"),
                    "--baseline-config",
                    str(baseline),
                    "--source-scene-dir",
                    str(source),
                    "--output-root",
                    str(output),
                    "--python",
                    sys.executable,
                    "--gpu-s",
                    "4",
                    "--gpu-t",
                    "5",
                    "--issue",
                    "MYL-42",
                    "--run-prefix",
                    "myl42_calibrated",
                    "--sensor-range",
                    "200",
                    "--planning-range-gate",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            s_settings = json.loads(
                (output / "s/dataset/Macarons++/12-NW-6C-7_8/settings.json").read_text()
            )
            t_settings = json.loads(
                (output / "t/dataset/Macarons++/12-NW-6C-7_8/settings.json").read_text()
            )
            self.assertEqual(s_settings["camera"]["start_positions"][0], [11, 9, 5, 1, 3])
            self.assertEqual(t_settings["camera"]["start_positions"][0], [12, 9, 5, 1, 3])
            self.assertEqual(s_settings["camera"]["start_positions"][1], [1, 2, 3, 4, 5])
            self.assertTrue(
                (output / "s/dataset/Macarons++/12-NW-6C-7_8/12-NW-6C-7_8.obj").is_symlink()
            )
            config = json.loads((output / "s/config.json").read_text())
            self.assertEqual(config["validation_n_poses_in_trajectory"], 100)
            self.assertTrue(config["experiment_cross_tile_diagnostics_enabled"])
            self.assertEqual(config["beam_width"], 10)
            self.assertEqual(config["experiment_param_overrides"]["sensor_range"], 200.0)
            self.assertTrue(config["experiment_planning_range_gate_enabled"])
            manifest = json.loads((output / "s/manifest.json").read_text())
            self.assertEqual(manifest["issue"], "MYL-42")
            self.assertEqual(manifest["sensor_range_scene_units"], 200.0)
            self.assertTrue(manifest["planning_range_gate_enabled"])
            self.assertTrue(
                manifest["run_id"].startswith("myl42_calibrated_s_")
            )


if __name__ == "__main__":
    unittest.main()
