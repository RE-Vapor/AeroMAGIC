import copy
import json
import unittest
from pathlib import Path

from macarons.utility.debug_profiles import apply_debug_profile


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "test"
    / "test_pioneer_hkust_pan11_position_only_20obs_a1_config.json"
)
QUICK_CONFIG_PATH = (
    ROOT
    / "configs"
    / "test"
    / "test_pioneer_hkust_pan11_position_only_quick_config.json"
)
PROFILES_DIR = ROOT / "configs" / "debug"


class Pan11HkustTwentyObservationConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_pins_requested_budget_and_position_only_contract(self):
        effective = copy.deepcopy(self.config)
        profile = apply_debug_profile(
            effective,
            cli_profile_name="pioneer-20",
            profiles_dir=str(PROFILES_DIR),
        )

        self.assertEqual(profile["name"], "pioneer-20")
        self.assertEqual(effective["test_scenes"], ["HKUST"])
        self.assertEqual(effective["validation_n_proxy_points"], 100000)
        self.assertEqual(effective["beam_width"], 3)
        self.assertEqual(effective["beam_steps"], 3)
        self.assertEqual(effective["validation_n_interpolation_steps"], 1)
        self.assertEqual(effective["validation_n_poses_in_trajectory"], 19)
        self.assertEqual(effective["experiment_budget_observations"], 20)
        self.assertEqual(effective["validation_max_start_positions"], 1)
        self.assertEqual(effective["pioneer_planner_state_mode"], "position_only")
        self.assertEqual(effective["pioneer_cubemap_rig_frame"], "world")
        self.assertEqual(
            effective["pioneer_cubemap_extrinsics_version"],
            "pytorch3d-world-axes-v1",
        )
        self.assertEqual(effective["pioneer_canonical_orientation_indices"], [2, 0])
        self.assertIs(
            effective["pioneer_filter_occupied_position_candidates"], True
        )
        self.assertIs(
            effective["validation_require_complete_occupied_pose"], True
        )

    def test_isolates_outputs_from_quick_and_future_attempts(self):
        quick = json.loads(QUICK_CONFIG_PATH.read_text(encoding="utf-8"))
        mutable_keys = (
            "results_json_name",
            "lmdb_dir_name",
            "validation_memory_dir_name",
            "experiment_run_id",
            "experiment_run_dir",
            "experiment_metrics_dir",
        )
        for key in mutable_keys:
            value = self.config[key]
            self.assertIn("20obs", value.lower())
            self.assertNotEqual(value, quick[key])
        self.assertTrue(self.config["experiment_run_dir"].endswith("attempt-001"))
        self.assertTrue(
            self.config["experiment_metrics_dir"].startswith(
                self.config["experiment_run_dir"] + "/"
            )
        )


if __name__ == "__main__":
    unittest.main()
