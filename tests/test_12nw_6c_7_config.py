import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCENE = "12-NW-6C-7"


class Scene12NW6C7ConfigTests(unittest.TestCase):
    def test_preprocessed_mesh_uses_identity_transform_and_metric_scale(self):
        config = json.loads(
            (
                ROOT
                / "configs/test/test_da3_12-nw-6c-7_real_mesh_config.json"
            ).read_text()
        )

        self.assertEqual(config["test_scenes"], [SCENE])
        self.assertEqual(config["da3_scene_units_per_meter"], {SCENE: 1.0})
        self.assertEqual(
            config["scene_mesh_transforms"][SCENE],
            {
                "axis_order": [0, 1, 2],
                "axis_signs": [1, 1, 1],
                "translation": [0.0, 0.0, 0.0],
                "scale": 1.0,
            },
        )

    def test_smoke_config_is_isolated_and_matches_fairness_defaults(self):
        config = json.loads(
            (
                ROOT
                / "configs/test/test_da3_12-nw-6c-7_real_mesh_config.json"
            ).read_text()
        )

        self.assertFalse(config["validation_use_occupied_pose"])
        self.assertTrue(config["compute_collision"])
        self.assertTrue(config["experiment_shared_collision_gate"])
        self.assertEqual(config["random_seed"], 8)
        self.assertEqual(config["torch_seed"], 9)
        self.assertEqual(
            config["experiment_param_overrides"][
                "planning_gathering_factor_multiplier"
            ],
            2.0,
        )
        for name in (
            config["validation_memory_dir_name"],
            config["lmdb_dir_name"],
            config["scone_lmdb_dir_name"],
        ):
            self.assertIn("myl21", name)

    def test_calibration_matches_materialized_preprocessing_and_settings(self):
        manifest = json.loads(
            (ROOT / "configs/test/scene_metric_calibrations.json").read_text()
        )
        calibration = manifest["calibrations"][SCENE]
        measurement = calibration["mesh_measurement"]
        cross_check = calibration["cross_check"]
        expected = (
            measurement["materialized_preprocess_scale"]
            * measurement["scene_scale_factor"]
        )

        self.assertAlmostEqual(calibration["scene_units_per_meter"], expected)
        self.assertTrue(measurement["preprocessing_materialized_in_obj"])
        self.assertEqual(measurement["planning_transform"], "identity")
        self.assertEqual(
            cross_check["source_to_preprocessed_extents"],
            cross_check["current_preprocessed_extents"],
        )
        self.assertTrue(
            all(
                mesh_extent <= settings_extent
                for mesh_extent, settings_extent in zip(
                    cross_check["current_preprocessed_extents"],
                    cross_check["settings_pre_runtime_extents"],
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
