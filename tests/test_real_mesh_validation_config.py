import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RealMeshValidationConfigTests(unittest.TestCase):
    def test_eiffel_scale_matches_evidence_manifest_and_formula(self):
        manifest = json.loads(
            (ROOT / "configs/test/scene_metric_calibrations.json").read_text()
        )
        config = json.loads(
            (ROOT / "configs/test/test_da3_eiffel_real_mesh_config.json").read_text()
        )
        calibration = manifest["calibrations"]["eiffel"]
        measurement = calibration["mesh_measurement"]
        expected = (
            (measurement["top_y_obj_units"] - measurement["base_y_obj_units"])
            * measurement["scene_scale_factor"]
            / calibration["physical_reference"]["meters"]
        )

        self.assertAlmostEqual(calibration["scene_units_per_meter"], expected, places=12)
        self.assertAlmostEqual(
            config["da3_scene_units_per_meter"]["eiffel"], expected, places=12
        )
        self.assertEqual(config["test_scenes"], ["eiffel"])
        self.assertFalse(config["use_perfect_depth_map"])
        self.assertEqual(config["kind_depth_map"], "DA3")
        self.assertEqual(config["validation_n_poses_in_trajectory"], 2)
        self.assertEqual(config["validation_max_start_positions"], 1)
        self.assertEqual(
            config["validation_memory_dir_name"],
            "test_memory_da3_eiffel_real_mesh",
        )

    def test_12_nw_scale_transform_and_data_contract_match_evidence(self):
        manifest = json.loads(
            (ROOT / "configs/test/scene_metric_calibrations.json").read_text()
        )
        config = json.loads(
            (
                ROOT
                / "configs/test/test_da3_12-nw-6c-5_real_mesh_config.json"
            ).read_text()
        )
        scene = "12-NW-6C-5"
        calibration = manifest["calibrations"][scene]
        measurement = calibration["mesh_measurement"]
        expected = measurement["preprocess_scale"] * measurement["scene_scale_factor"]

        self.assertEqual(config["test_scenes"], [scene])
        self.assertAlmostEqual(calibration["scene_units_per_meter"], expected)
        self.assertAlmostEqual(config["da3_scene_units_per_meter"][scene], expected)
        raw_extents = measurement["raw_extents"]
        transformed_extents = [
            raw_extents[index] * measurement["preprocess_scale"]
            for index in config["scene_mesh_transforms"][scene]["axis_order"]
        ]
        for actual, recorded in zip(
            transformed_extents,
            calibration["cross_check"]["transformed_pre_runtime_extents"],
        ):
            self.assertAlmostEqual(actual, recorded, places=9)
        self.assertEqual(
            config["scene_mesh_transforms"][scene]["axis_order"], [0, 2, 1]
        )
        self.assertEqual(
            config["scene_mesh_transforms"][scene]["axis_signs"], [1, 1, -1]
        )
        self.assertFalse(config["validation_use_occupied_pose"])
        self.assertTrue(config["compute_collision"])
        self.assertTrue(config["experiment_shared_collision_gate"])
        self.assertFalse(config["use_perfect_depth_map"])
        self.assertEqual(config["kind_depth_map"], "DA3")


if __name__ == "__main__":
    unittest.main()
