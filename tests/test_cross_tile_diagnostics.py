import unittest

import numpy as np

from scripts.summarize_cross_tile_ablation import _bootstrap_evidence

from macarons.utility.cross_tile_diagnostics import (
    audit_neighbor_generation,
    compute_partitioned_scene_coverage,
    compute_tiled_scene_coverage,
    summarize_cross_tile_trajectory,
    validate_tile_partition,
)


class CrossTileDiagnosticsTests(unittest.TestCase):
    def test_bootstrap_prefers_effective_sensor_range_gate_over_training_default(self):
        metrics = {
            "frames": [
                {
                    "sensor_range_gate": {"sensor_range_scene_units": 200.0},
                    "depth_scene_units": {"min": 103.0},
                    "partial_point_count": 12,
                }
            ],
            "cross_tile": {
                "planning": [
                    {
                        "imagined_gaussians": {"tile_1": 1, "tile_2": 2},
                        "selected": {"pose_index": [12, 0, 0, 0, 0]},
                        "beam_steps": [
                            {
                                "candidates": [
                                    {"coverage_gain": 1.0},
                                    {"coverage_gain": 2.0},
                                ]
                            }
                        ],
                    }
                ]
            },
        }
        evidence = _bootstrap_evidence(metrics)
        self.assertEqual(evidence["sensor_range_scene_units"], 200.0)
        self.assertFalse(evidence["frame_0_depth_min_exceeds_sensor_range"])

    def test_neighbor_audit_preserves_crossing_and_boundary_reasons(self):
        audit = audit_neighbor_generation([11, 0, 5, 0, 0], [24, 11, 13, 5, 10])
        crossing = [
            item for item in audit["attempts"] if item["pose_index"] == [12, 0, 5, 0, 0]
        ]
        self.assertEqual(len(crossing), 1)
        self.assertIsNone(crossing[0]["rejection_reason"])
        self.assertEqual(audit["attempted_count"], 10)
        self.assertEqual(audit["boundary_rejected_count"], 2)

    def test_partitioned_coverage_recombines_into_global_reference(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")

        class Cell:
            def __init__(self, points):
                self.cell_pts = torch.tensor(points, dtype=torch.float32)

        class Scene:
            def __init__(self, cells):
                self.cells = {key: Cell(value) for key, value in cells.items()}

        gt = Scene({"a": [[74.0, 0, 0], [76.0, 0, 0]], "b": [[80.0, 0, 0]]})
        recovered = Scene({"a": [[74.1, 0, 0]], "b": [[80.1, 0, 0]]})
        result = compute_partitioned_scene_coverage(
            gt,
            recovered,
            seam_x=75.0,
            surface_epsilon=0.2,
            normalization=0.5,
            reconstruction_points=[[74.1, 0, 0], [80.1, 0, 0]],
        )
        self.assertEqual(result["tile_1"]["covered_points"], 1)
        self.assertEqual(result["tile_2"]["covered_points"], 1)
        self.assertEqual(result["tile_2"]["reference_points"], 2)
        self.assertAlmostEqual(result["combined"]["raw"], 2 / 3)
        self.assertAlmostEqual(result["combined"]["normalized"], 4 / 3)

    def test_three_tile_coverage_recombines_and_partition_is_manifest_driven(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")

        class Cell:
            def __init__(self, points):
                self.cell_pts = torch.tensor(points, dtype=torch.float32)

        class Scene:
            def __init__(self, points):
                self.cells = {"all": Cell(points)}

        gt = Scene([[-1, 0, 0], [5, 0, 0], [15, 0, 0], [25, 0, 0]])
        recovered = Scene([[-1, 0, 0], [15, 0, 0], [25, 0, 0]])
        partition = {
            "axis": 0,
            "boundaries": [10.0, 20.0],
            "tile_ids": ["west", "center", "east"],
        }
        result = compute_tiled_scene_coverage(
            gt,
            recovered,
            tile_partition=partition,
            surface_epsilon=0.1,
            normalization=1.0,
            reconstruction_points=[[-1, 0, 0], [15, 0, 0], [25, 0, 0]],
        )
        self.assertEqual(result["tiles"]["west"]["reference_points"], 2)
        self.assertEqual(result["tiles"]["center"]["reference_points"], 1)
        self.assertEqual(result["tiles"]["east"]["reference_points"], 1)
        self.assertEqual(result["combined"]["reference_points"], 4)
        self.assertEqual(result["combined"]["covered_points"], 3)
        self.assertAlmostEqual(result["combined"]["raw"], 0.75)

    def test_partition_rejects_ambiguous_or_mismatched_schema(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_tile_partition(
                {"axis": 0, "boundaries": [20, 10], "tile_ids": ["a", "b", "c"]}
            )
        with self.assertRaisesRegex(ValueError, r"len\(boundaries\)"):
            validate_tile_partition(
                {"axis": 0, "boundaries": [10], "tile_ids": ["a"]}
            )

    def test_trajectory_summary_requires_sustained_tile_two_activity(self):
        positions = np.asarray(
            [[73.0, 0, 0], [86.0, 0, 0], [86.0, 0, 0], [73.0, 0, 0]]
        )
        indices = np.asarray(
            [[11, 9, 5, 1, 3], [12, 9, 5, 1, 3], [12, 9, 5, 1, 4], [11, 9, 5, 1, 4]]
        )
        coverage = [
            {
                "tile_2": {
                    "raw": value,
                    "normalized": value,
                    "reconstruction_points": count,
                }
            }
            for value, count in ((0.0, 0), (0.01, 10), (0.04, 15), (0.04, 15))
        ]
        summary = summarize_cross_tile_trajectory(
            positions, indices, seam_x=75.0, tile_coverage=coverage
        )
        self.assertEqual(summary["first_crossing_frame"], 1)
        self.assertEqual(summary["tile_2_observations"], 2)
        self.assertEqual(summary["tile_2_longest_consecutive_stay"], 2)
        self.assertEqual(summary["returns_to_tile_1"], 1)
        self.assertEqual(summary["actions"]["positive_x"], 1)
        self.assertEqual(summary["actions"]["negative_x"], 1)
        self.assertEqual(summary["actions"]["in_place_rotation"], 1)
        self.assertEqual(summary["tile_2_longest_nonempty_reconstruction_run"], 3)
        self.assertAlmostEqual(summary["tile_2_raw_coverage_delta"], 0.04)


if __name__ == "__main__":
    unittest.main()
