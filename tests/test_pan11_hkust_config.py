import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "test"
    / "test_pioneer_hkust_pan11_position_only_quick_config.json"
)


class Pan11HkustPositionOnlyConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_pins_position_only_world_cubemap_contract(self):
        self.assertEqual(self.config["test_scenes"], ["HKUST"])
        self.assertEqual(self.config["debug_profile"], "quick")
        self.assertEqual(self.config["planning_observation_mode"], "cubemap6")
        self.assertEqual(
            self.config["pioneer_planner_state_mode"], "position_only"
        )
        self.assertEqual(self.config["pioneer_cubemap_rig_frame"], "world")
        self.assertEqual(
            self.config["pioneer_cubemap_extrinsics_version"],
            "pytorch3d-world-axes-v1",
        )
        self.assertEqual(
            self.config["pioneer_canonical_orientation_indices"], [2, 0]
        )
        self.assertEqual(self.config["scene_texture_atlas_size"], 4)
        self.assertIs(self.config["validation_use_occupied_pose"], True)
        self.assertIs(
            self.config["validation_require_complete_occupied_pose"], True
        )
        self.assertIs(
            self.config["pioneer_filter_occupied_position_candidates"], True
        )
        self.assertEqual(self.config["da3_scene_units_per_meter"], {"HKUST": 0.2})

    def test_uses_identity_transform_for_pretransformed_hkust_mesh(self):
        transform = self.config["scene_mesh_transforms"]["HKUST"]
        self.assertEqual(transform["axis_order"], [0, 1, 2])
        self.assertEqual(transform["axis_signs"], [1, 1, 1])
        self.assertEqual(transform["translation"], [0.0, 0.0, 0.0])
        self.assertEqual(transform["scale"], 1.0)

    def test_isolates_every_mutable_output_from_eiffel_and_other_attempts(self):
        mutable_names = (
            self.config["results_json_name"],
            self.config["lmdb_dir_name"],
            self.config["validation_memory_dir_name"],
            self.config["experiment_run_id"],
            self.config["experiment_run_dir"],
            self.config["experiment_metrics_dir"],
        )
        for value in mutable_names:
            self.assertIn("hkust", value.lower())
            self.assertNotIn("eiffel", value.lower())
        self.assertTrue(self.config["experiment_run_dir"].endswith("attempt-001"))
        self.assertTrue(
            self.config["experiment_metrics_dir"].startswith(
                self.config["experiment_run_dir"] + "/"
            )
        )


if __name__ == "__main__":
    unittest.main()
