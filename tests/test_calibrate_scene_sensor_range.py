import unittest

from scripts.calibrate_scene_sensor_range import derive_sensor_range


class CalibrateSceneSensorRangeTests(unittest.TestCase):
    def test_derives_bounded_explicit_range_from_geometry_and_depth_evidence(self):
        assembly = {
            "output": {
                "bounds": {
                    "minimum": [-7.5, -7.5, -7.5],
                    "maximum": [22.5, 3.5, 7.5],
                }
            },
            "shared_transform": {"scale": 0.1},
        }
        settings = {
            "camera": {
                "x_min": [-8.0, -4.0, -8.0],
                "x_max": [24.0, 10.0, 9.0],
            }
        }
        params = {
            "_data": {"scene_scale_factor": 10.0},
            "_camera_management": {"sensor_range": 70.0},
            "_depth_module": {"znear": 0.5, "zfar": 750.0},
        }
        experiment = {"da3_scene_units_per_meter": {"joined": 1.0}}
        evidence = [
            {
                "label": "S",
                "depth": {
                    "valid_pixels": 100,
                    "scene_units": {
                        "min": 103.5,
                        "median": 138.4,
                        "p90": 177.984,
                        "max": 215.1,
                    },
                },
            },
            {
                "label": "T",
                "depth": {
                    "valid_pixels": 90,
                    "scene_units": {
                        "min": 106.3,
                        "median": 144.3,
                        "p90": 175.296,
                        "max": 203.7,
                    },
                },
            },
        ]
        result = derive_sensor_range(
            scene="joined",
            assembly=assembly,
            settings=settings,
            params=params,
            experiment=experiment,
            gate_evidence=evidence,
        )
        self.assertAlmostEqual(
            result["calibration"]["unrounded_recommendation_scene_units"],
            195.7824,
        )
        self.assertEqual(
            result["calibration"]["recommended_sensor_range_scene_units"], 200.0
        )
        self.assertGreater(result["calibration"]["hard_cap_scene_units"], 200.0)
        self.assertEqual(
            result["unit_contract"]["assembly_source_to_runtime_scale"], 1.0
        )

    def test_rejects_recommendation_above_geometry_or_renderer_cap(self):
        with self.assertRaisesRegex(ValueError, "hard cap"):
            derive_sensor_range(
                scene="joined",
                assembly={
                    "output": {
                        "bounds": {"minimum": [0, 0, 0], "maximum": [1, 1, 1]}
                    },
                    "shared_transform": {"scale": 1.0},
                },
                settings={
                    "camera": {"x_min": [0, 0, 0], "x_max": [1, 1, 1]}
                },
                params={
                    "_data": {"scene_scale_factor": 1.0},
                    "_camera_management": {"sensor_range": 0.5},
                    "_depth_module": {"znear": 0.1, "zfar": 2.0},
                },
                experiment={"da3_scene_units_per_meter": {"joined": 1.0}},
                gate_evidence=[
                    {
                        "label": "S",
                        "depth": {
                            "valid_pixels": 1,
                            "scene_units": {
                                "min": 2.0,
                                "median": 2.0,
                                "p90": 2.0,
                                "max": 2.0,
                            },
                        },
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
