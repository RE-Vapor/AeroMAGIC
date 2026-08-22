import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PAN34SkyIndirectConfigTests(unittest.TestCase):
    def test_replay_keeps_the_frozen_low_altitude_observations(self):
        replay = json.loads(
            (
                ROOT
                / "configs/replay/pan34_hkust_low_altitude_50obs_sky_indirect.json"
            ).read_text()
        )
        self.assertEqual(replay["task"], "PAN-34")
        self.assertEqual(replay["observation_ids"], [0, 12, 24, 36, 49])
        self.assertEqual(
            replay["capture_config_repo_relative"],
            "unreal/PAN34/Config/pan34_hkust_sky_indirect.json",
        )
        self.assertEqual(replay["lighting_variant_id"], "sky_diffuse_indirect_v1")

    def test_treatment_uses_ue5_sky_and_lumen_without_shadow_postprocessing(self):
        config = json.loads(
            (
                ROOT / "unreal/PAN34/Config/pan34_hkust_sky_indirect.json"
            ).read_text()
        )
        lighting = config["lighting_ablation"]
        self.assertEqual(lighting["variant_id"], "sky_diffuse_indirect_v1")
        self.assertEqual(lighting["post_read_exposure_transform_ev"], 0.5)
        self.assertNotIn("post_read_shadow_lift", lighting)
        self.assertEqual(
            lighting["directional_light"],
            {
                "intensity": 7.0,
                "source_angle_degrees": 3.0,
                "indirect_lighting_intensity": 2.0,
            },
        )
        self.assertEqual(
            lighting["sky_light"],
            {
                "intensity_scale": 3.0,
                "recapture_scene": True,
                "indirect_lighting_intensity": 2.0,
                "lower_hemisphere_is_solid_color": True,
                "lower_hemisphere_color_linear": [0.05, 0.07, 0.1],
            },
        )
        self.assertEqual(
            lighting["global_illumination"],
            {
                "method": "lumen",
                "indirect_lighting_intensity": 1.25,
                "indirect_lighting_color_linear": [0.9, 0.95, 1.0],
                "lumen_final_gather_quality": 1.0,
                "lumen_skylight_leaking": 0.15,
                "scene_capture_warmup_count": 3,
            },
        )


if __name__ == "__main__":
    unittest.main()
