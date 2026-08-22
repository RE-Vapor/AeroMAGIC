import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from macarons.utility.debug_profiles import (
    DEBUG_PROFILE_NAMES,
    apply_debug_profile,
    load_debug_profile,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = ROOT / "configs" / "debug"


class DebugProfileTests(unittest.TestCase):
    def test_profile_contract_and_debug_bounds(self):
        expected = {
            "quick": (100000, 3, 3, 3),
            "magician": (200000, 5, 5, 11),
            "large-scene": (200000, 3, 3, 21),
            "pioneer-20": (100000, 3, 3, 20),
            "pioneer-50": (100000, 3, 3, 50),
            "pioneer-low-altitude-50": (100000, 3, 3, 50),
            "pan21-two-observation": (100000, 3, 3, 2),
        }
        for name in DEBUG_PROFILE_NAMES:
            with self.subTest(profile=name):
                profile = load_debug_profile(name, str(PROFILES_DIR))
                overrides = profile["overrides"]
                self.assertTrue(profile["debug_only"])
                self.assertFalse(profile["coverage_comparable"])
                self.assertEqual(
                    (
                        overrides["validation_n_proxy_points"],
                        overrides["beam_width"],
                        overrides["beam_steps"],
                        overrides["experiment_budget_observations"],
                    ),
                    expected[name],
                )
                self.assertEqual(overrides["validation_n_interpolation_steps"], 1)
                self.assertEqual(overrides["validation_max_start_positions"], 1)
                self.assertEqual(overrides["random_seed"], 8)
                self.assertEqual(overrides["torch_seed"], 9)
                if name == "pan21-two-observation":
                    self.assertEqual(
                        profile["validation_start_position_override"],
                        [5, 3, 1, 2, 0],
                    )
                    self.assertEqual(
                        profile["validation_position_policy"]["position_index_min"],
                        [0, 3, 0],
                    )
                if name == "pioneer-low-altitude-50":
                    self.assertEqual(
                        profile["validation_start_position_override"],
                        [6, 1, 5, 2, 0],
                    )
                    self.assertEqual(
                        profile["validation_position_policy"]["position_index_min"],
                        [0, 1, 0],
                    )
                    self.assertEqual(
                        profile["validation_position_policy"]["position_index_max"],
                        [6, 2, 9],
                    )
                    self.assertEqual(
                        profile["validation_position_policy"]["hard_ceiling_agl_m"],
                        120.0,
                    )

    def test_apply_preserves_base_scene_and_isolates_outputs(self):
        config = SimpleNamespace(
            test_scenes=["eiffel"],
            beam_width=10,
            beam_steps=10,
            lmdb_dir_name="formal_lmdb",
            scone_lmdb_dir_name="formal_scone_lmdb",
            validation_memory_dir_name="formal_memory",
            results_json_name="formal.json",
            experiment_run_id="formal_run",
            experiment_metrics_dir="results/formal/metrics",
        )
        profile = apply_debug_profile(
            config,
            cli_profile_name="quick",
            profiles_dir=str(PROFILES_DIR),
        )
        self.assertEqual(profile["name"], "quick")
        self.assertEqual(config.test_scenes, ["eiffel"])
        self.assertEqual(config.validation_n_proxy_points, 100000)
        self.assertEqual(config.beam_width, 3)
        self.assertEqual(config.lmdb_dir_name, "formal_lmdb_debug_quick")
        self.assertEqual(config.scone_lmdb_dir_name, "formal_scone_lmdb_debug_quick")
        self.assertEqual(config.validation_memory_dir_name, "formal_memory_debug_quick")
        self.assertEqual(config.results_json_name, "formal_debug_quick.json")
        self.assertEqual(config.experiment_run_id, "formal_run_debug_quick")
        self.assertEqual(
            config.experiment_metrics_dir,
            "results/formal/metrics_debug_quick",
        )
        self.assertTrue(config.debug_only)
        self.assertFalse(config.coverage_comparable)

    def test_config_selector_matches_cli_selector(self):
        config = {"debug_profile": "large-scene"}
        profile = apply_debug_profile(
            config,
            cli_profile_name="large-scene",
            profiles_dir=str(PROFILES_DIR),
        )
        self.assertEqual(profile["name"], "large-scene")
        self.assertEqual(config["experiment_budget_observations"], 21)

        with self.assertRaisesRegex(ValueError, "must match"):
            apply_debug_profile(
                {"debug_profile": "quick"},
                cli_profile_name="magician",
                profiles_dir=str(PROFILES_DIR),
            )

    def test_profiles_are_valid_json(self):
        for name in DEBUG_PROFILE_NAMES:
            with (PROFILES_DIR / f"{name}.json").open(encoding="utf-8") as stream:
                self.assertIsInstance(json.load(stream), dict)

    def test_apply_pan21_start_override_is_explicit_and_isolated(self):
        config = {
            "lmdb_dir_name": "base",
            "scone_lmdb_dir_name": "base_scone",
            "validation_memory_dir_name": "base_memory",
            "results_json_name": "base.json",
        }
        apply_debug_profile(
            config,
            cli_profile_name="pan21-two-observation",
            profiles_dir=str(PROFILES_DIR),
        )
        self.assertEqual(config["validation_start_position_override"], [5, 3, 1, 2, 0])
        self.assertEqual(config["experiment_budget_observations"], 2)
        self.assertEqual(
            config["validation_position_policy"]["position_index_max"],
            [11, 3, 9],
        )

    def test_pioneer_50_config_resolves_exact_budget_and_isolated_outputs(self):
        config_path = (
            ROOT / "configs" / "test" / "test_pioneer_eiffel_50step_a1_config.json"
        )
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
        profile = apply_debug_profile(
            config,
            cli_profile_name="pioneer-50",
            profiles_dir=str(PROFILES_DIR),
        )
        self.assertEqual(profile["name"], "pioneer-50")
        self.assertEqual(config["validation_n_poses_in_trajectory"], 49)
        self.assertEqual(config["validation_n_interpolation_steps"], 1)
        self.assertEqual(config["experiment_budget_observations"], 50)
        self.assertEqual(config["validation_n_proxy_points"], 100000)
        self.assertEqual(config["beam_width"], 3)
        self.assertEqual(config["beam_steps"], 3)
        self.assertTrue(config["lmdb_dir_name"].endswith("_debug_pioneer50"))
        self.assertTrue(
            config["validation_memory_dir_name"].endswith("_debug_pioneer50")
        )
        self.assertTrue(
            config["experiment_metrics_dir"].endswith("metrics_debug_pioneer50")
        )

    def test_pioneer_20_profile_resolves_exact_requested_budget(self):
        config_path = (
            ROOT
            / "configs"
            / "test"
            / "test_pioneer_hkust_pan11_position_only_20obs_a1_config.json"
        )
        with config_path.open(encoding="utf-8") as stream:
            config = json.load(stream)
        profile = apply_debug_profile(
            config,
            cli_profile_name="pioneer-20",
            profiles_dir=str(PROFILES_DIR),
        )
        self.assertEqual(profile["name"], "pioneer-20")
        self.assertEqual(config["validation_n_poses_in_trajectory"], 19)
        self.assertEqual(config["validation_n_interpolation_steps"], 1)
        self.assertEqual(config["experiment_budget_observations"], 20)
        self.assertEqual(config["validation_n_proxy_points"], 100000)
        self.assertEqual(config["beam_width"], 3)
        self.assertEqual(config["beam_steps"], 3)
        self.assertEqual(config["validation_max_start_positions"], 1)
        self.assertTrue(config["lmdb_dir_name"].endswith("_debug_pioneer20"))
        self.assertTrue(
            config["validation_memory_dir_name"].endswith("_debug_pioneer20")
        )
        self.assertTrue(
            config["experiment_metrics_dir"].endswith("metrics_debug_pioneer20")
        )


if __name__ == "__main__":
    unittest.main()
