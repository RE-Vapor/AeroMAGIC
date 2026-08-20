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
GT_20OBS_CONFIG_PATH = (
    ROOT
    / "configs"
    / "test"
    / "test_pioneer_hkust_pan11_position_only_20obs_a1_config.json"
)
DA3_20OBS_CONFIG_PATH = (
    ROOT
    / "configs"
    / "test"
    / "test_pioneer_hkust_pan11_position_only_da3_20obs_a1_config.json"
)
PIONEER_20_PROFILE_PATH = ROOT / "configs" / "debug" / "pioneer-20.json"
CALIBRATIONS_PATH = ROOT / "configs" / "test" / "scene_metric_calibrations.json"


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

    def test_da3_20obs_changes_only_depth_contract_and_isolated_outputs(self):
        gt = json.loads(GT_20OBS_CONFIG_PATH.read_text(encoding="utf-8"))
        da3 = json.loads(DA3_20OBS_CONFIG_PATH.read_text(encoding="utf-8"))
        profile = json.loads(PIONEER_20_PROFILE_PATH.read_text(encoding="utf-8"))
        output_keys = {
            "results_json_name",
            "lmdb_dir_name",
            "validation_memory_dir_name",
            "experiment_run_id",
            "experiment_run_dir",
            "experiment_metrics_dir",
        }
        da3_only_keys = {
            "da3_model_id",
            "da3_model_revision",
            "da3_model_config_sha256",
            "da3_model_weights_sha256",
            "da3_source_revision",
            "da3_source_tree_sha256",
            "da3_window_size",
            "da3_process_res",
            "da3_process_res_method",
            "da3_output_height",
            "da3_output_width",
            "da3_confidence_percentile",
            "da3_cache_enabled",
            "da3_cache_dir",
            "scene_texture_tree_sha256",
            "macarons_params_sha256",
        }
        shared_keys = set(gt) - output_keys - {"use_perfect_depth_map", "kind_depth_map"}
        for key in shared_keys:
            self.assertEqual(gt[key], da3[key], key)
        self.assertEqual(
            set(da3) - set(gt), da3_only_keys
        )
        self.assertIs(da3["use_perfect_depth_map"], False)
        self.assertEqual(da3["kind_depth_map"], "DA3")
        self.assertEqual(
            da3["da3_source_tree_sha256"],
            "65fa2f4829a831512492964fcda6dcdf9193b5cbefa1862c4773c054b4b77633",
        )
        self.assertEqual(
            da3["da3_model_config_sha256"],
            "09adf89474017e717bc05aa86fd3a378708ba8914b036d61874eced328069468",
        )
        self.assertEqual(
            da3["da3_model_weights_sha256"],
            "8899faf998dedbc230261ab736fa57015280727399429122d44d4f9e7aac2ddd",
        )
        self.assertEqual(
            da3["scene_texture_tree_sha256"],
            "7d01172e4b7d24a1d7e998e5ef8dbdae2f4c0aa003d486ec4bc8bf627c5618d9",
        )
        self.assertEqual(
            da3["macarons_params_sha256"],
            "e3ccb7b7083090c73fbed75f7f4e50bb01685ed1c527afe47b4bf204ad5156c9",
        )
        self.assertEqual(da3["da3_output_height"], da3["pioneer_face_size"])
        self.assertEqual(da3["da3_output_width"], da3["pioneer_face_size"])
        self.assertEqual(da3["da3_scene_units_per_meter"], {"HKUST": 0.2})
        for key in output_keys | {"da3_cache_dir"}:
            self.assertIn("da3", da3[key].lower())
            self.assertNotEqual(da3[key], gt.get(key))
        overrides = profile["overrides"]
        self.assertEqual(overrides["validation_n_proxy_points"], 100000)
        self.assertEqual(overrides["beam_width"], 3)
        self.assertEqual(overrides["beam_steps"], 3)
        self.assertEqual(overrides["validation_n_interpolation_steps"], 1)
        self.assertEqual(overrides["validation_n_poses_in_trajectory"], 19)
        self.assertEqual(overrides["experiment_budget_observations"], 20)
        self.assertEqual(overrides["validation_max_start_positions"], 1)

    def test_hkust_da3_scale_has_hash_pinned_asset_provenance(self):
        calibration = json.loads(
            CALIBRATIONS_PATH.read_text(encoding="utf-8")
        )["calibrations"]["HKUST"]
        self.assertEqual(calibration["scene_units_per_meter"], 0.2)
        self.assertEqual(calibration["materialized_preprocess_scale"], 0.02)
        self.assertEqual(calibration["scene_scale_factor"], 10.0)
        self.assertEqual(
            calibration["adaptation_manifest"]["sha256"],
            "64992e97d4632c71473410decfccd9a63e5763867d5c0ed0b0df8b1d5e14065c",
        )
        self.assertEqual(
            calibration["mesh"]["sha256"],
            "cd7b18645cc574c43e55d5f927f8440c810b32f2f8e859700018582535ffa635",
        )


if __name__ == "__main__":
    unittest.main()
