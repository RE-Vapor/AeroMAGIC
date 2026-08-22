import copy
import hashlib
import json
import unittest
from pathlib import Path

from macarons.utility.debug_profiles import apply_debug_profile


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "test"
    / "test_pioneer_hkust_pan30_position_only_low_altitude_50obs_a1_config.json"
)
POLICY_PATH = (
    ROOT
    / "configs"
    / "flight_policies"
    / "hkust_low_altitude_120m_agl_v1.json"
)
PROFILES_DIR = ROOT / "configs" / "debug"


class Pan30LowAltitudeConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_resolves_visual_first_low_altitude_contract(self):
        effective = copy.deepcopy(self.config)
        profile = apply_debug_profile(
            effective,
            cli_profile_name="pioneer-low-altitude-50",
            profiles_dir=str(PROFILES_DIR),
        )

        self.assertEqual(effective["test_scenes"], ["HKUST"])
        self.assertEqual(effective["validation_n_proxy_points"], 100000)
        self.assertEqual(effective["beam_width"], 3)
        self.assertEqual(effective["beam_steps"], 3)
        self.assertEqual(effective["validation_n_interpolation_steps"], 1)
        self.assertEqual(effective["validation_n_poses_in_trajectory"], 49)
        self.assertEqual(effective["experiment_budget_observations"], 50)
        self.assertEqual(effective["validation_max_start_positions"], 1)
        self.assertEqual(effective["validation_start_position_override"], [6, 1, 5, 2, 0])
        self.assertEqual(
            effective["validation_position_policy"]["position_index_min"],
            [0, 1, 0],
        )
        self.assertEqual(
            effective["validation_position_policy"]["position_index_max"],
            [6, 2, 9],
        )
        self.assertEqual(
            effective["validation_position_policy"]["hard_ceiling_agl_m"],
            120.0,
        )
        self.assertTrue(effective["compute_collision"])
        self.assertTrue(effective["experiment_shared_collision_gate"])
        self.assertTrue(effective["pioneer_filter_occupied_position_candidates"])
        self.assertTrue(effective["validation_require_complete_occupied_pose"])
        self.assertFalse(profile["coverage_comparable"])

    def test_policy_hash_and_envelope_are_pinned(self):
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        profile = apply_debug_profile(
            copy.deepcopy(self.config),
            cli_profile_name="pioneer-low-altitude-50",
            profiles_dir=str(PROFILES_DIR),
        )
        digest = hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()

        self.assertEqual(
            digest,
            profile["validation_position_policy"]["source_policy_sha256"],
        )
        self.assertEqual(policy["verified_envelope"]["free_position_count"], 63)
        self.assertEqual(
            policy["verified_envelope"]["connected_component_size_from_start"],
            63,
        )
        self.assertLessEqual(
            policy["verified_envelope"]["maximum_free_position_agl_m"],
            policy["verified_envelope"]["hard_ceiling_agl_m"],
        )
        self.assertEqual(policy["execution_contract"]["observation_count"], 50)
        self.assertEqual(policy["execution_contract"]["planned_move_count"], 49)

    def test_attempt_one_outputs_are_isolated(self):
        mutable_keys = (
            "results_json_name",
            "lmdb_dir_name",
            "validation_memory_dir_name",
            "experiment_run_id",
            "experiment_run_dir",
            "experiment_metrics_dir",
        )
        for key in mutable_keys:
            value = self.config[key].lower()
            self.assertIn("pan30", value)
            self.assertIn("lowalt", value)
        self.assertTrue(self.config["experiment_run_dir"].endswith("attempt-001"))
        self.assertTrue(
            self.config["experiment_metrics_dir"].startswith(
                self.config["experiment_run_dir"] + "/"
            )
        )


if __name__ == "__main__":
    unittest.main()
