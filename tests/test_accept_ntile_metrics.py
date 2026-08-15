import unittest

from scripts.accept_ntile_metrics import evaluate


class AcceptNTileMetricsTests(unittest.TestCase):
    def test_accepts_full_run_only_when_all_tiles_and_range_gate_pass(self):
        partition = {
            "axis": 0,
            "boundaries": [10.0, 20.0],
            "tile_ids": ["west", "center", "east"],
            "interval_semantics": "(-inf,b0], (b0,b1], ..., (bN,+inf)",
        }
        manifest = {"scene": "joined", "run_id": "run", "tile_partition": partition}
        metrics = {
            "scene": "joined",
            "run": {"run_id": "run"},
            "frames": [
                {
                    "partial_point_count": 12,
                    "sensor_range_gate": {"accepted": True},
                }
            ],
            "trajectory": {"positions": [[0, 0, 0]]},
            "tile_metrics": {
                "partition": partition,
                "recombination_max_abs_raw_error": 0.0,
                "recombination_max_abs_normalized_error": 0.0,
                "coverage": [
                    {
                        "tiles": {
                            "west": {"reference_points": 2, "covered_points": 1},
                            "center": {"reference_points": 2, "covered_points": 1},
                            "east": {"reference_points": 2, "covered_points": 1},
                        }
                    }
                ],
            },
        }
        self.assertTrue(evaluate(manifest, metrics, mode="full")["accepted"])
        metrics["tile_metrics"]["coverage"][0]["tiles"]["center"]["covered_points"] = 0
        result = evaluate(manifest, metrics, mode="full")
        self.assertFalse(result["accepted"])
        self.assertFalse(result["checks"]["all_tiles_have_covered_points"])


if __name__ == "__main__":
    unittest.main()
