import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FairRunnerTests(unittest.TestCase):
    def test_generated_main_matrix_has_shared_fairness_contract(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_12nw_fair_experiments.py"),
                    "--suite",
                    "main",
                    "--generate-only",
                    "--output-dir",
                    str(output),
                    "--gpu",
                    "0",
                    "--collision",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(
                {run["run_id"] for run in manifest["runs"]},
                {
                    "main_scone_gt",
                    "main_scone_da3",
                    "main_magician_gt",
                    "main_magician_da3",
                },
            )
            self.assertEqual(manifest["fixed"]["start_indices"], [0, 1, 2, 3, 4])
            self.assertFalse(manifest["gt_feedback_to_da3"])
            self.assertEqual(manifest["renderer_zbuf_role"], "post-run diagnostics only")

            for run in manifest["runs"]:
                config = json.loads(Path(run["config"]).read_text())
                self.assertEqual(config["validation_n_poses_in_trajectory"], 100)
                self.assertEqual(config["validation_max_start_positions"], 5)
                self.assertEqual(config["random_seed"], 8)
                self.assertEqual(config["torch_seed"], 9)
                self.assertEqual(config["beam_width"], 10)
                self.assertEqual(config["beam_steps"], 10)
                self.assertEqual(config["validation_n_proxy_points"], 800000)
                self.assertEqual(config["validation_n_gt_surface_points"], 100000)
                self.assertEqual(
                    config["experiment_param_overrides"][
                        "planning_gathering_factor_multiplier"
                    ],
                    2.0,
                )
                self.assertTrue(config["compute_collision"])
                self.assertTrue(config["experiment_shared_collision_gate"])
                self.assertEqual(
                    config["use_perfect_depth_map"], run["source"] == "gt"
                )


if __name__ == "__main__":
    unittest.main()
