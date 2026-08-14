import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Summarize12NWResultsTests(unittest.TestCase):
    def test_partial_matrix_is_not_reported_fair_and_conditioning_is_counted(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metrics = root / "sensitivity" / "metrics"
            metrics.mkdir(parents=True)
            online = {
                "scene": "12-NW-6C-7",
                "run": {"run_id": "sensitivity_magician_da3_window_1"},
                "planner": "magician",
                "start_index": 0,
                "trajectory": {
                    "observation_count": 2,
                    "final_point_count": 10,
                    "path_length_meters": 1.0,
                },
                "coverage": [{"normalized": 0.1, "raw": 0.09}],
                "latency": {"provider_seconds": 1.0, "trajectory_seconds": 2.0},
                "cuda": {"peak_allocated_mib": 3.0, "peak_reserved_mib": 4.0},
                "frames": [
                    {
                        "source": "DA3",
                        "planning_ratio": 1.0,
                        "cache_hit": False,
                        "pose_conditioned": False,
                    },
                    {
                        "source": "DA3",
                        "planning_ratio": 1.0,
                        "cache_hit": False,
                        "pose_conditioned": True,
                    },
                ],
                "online_only": True,
                "renderer_gt_read": False,
            }
            (metrics / "magician_12-NW-6C-7_0.online.json").write_text(
                json.dumps(online), encoding="utf-8"
            )
            output = root / "summary.json"

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/summarize_12nw_results.py"),
                    "--scene",
                    "12-NW-6C-7",
                    "--results-root",
                    str(root),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            summary = json.loads(output.read_text())
            row = summary["runs"][0]
            aggregate = summary["aggregates"][row["run_id"]]
            self.assertEqual(row["pose_conditioned_frames"], 1)
            self.assertEqual(row["unconditioned_frames"], 1)
            self.assertEqual(aggregate["pose_conditioned_frame_count"], 1)
            self.assertEqual(aggregate["unconditioned_frame_count"], 1)
            self.assertFalse(
                summary["fairness_checks"]["expected_main_run_ids_present"]
            )
            self.assertFalse(
                summary["fairness_checks"]["each_main_run_has_starts_0_through_4"]
            )


if __name__ == "__main__":
    unittest.main()
