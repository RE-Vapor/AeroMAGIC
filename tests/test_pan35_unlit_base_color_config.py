import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PAN35UnlitBaseColorConfigTests(unittest.TestCase):
    def test_public_replay_contract_uses_same_five_observations_and_base_color(self):
        replay = json.loads(
            (ROOT / "configs/replay/pan35_hkust_low_altitude_50obs_unlit_base_color.json").read_text()
        )
        capture = json.loads(
            (ROOT / "unreal/PAN35/Config/pan35_hkust_unlit_base_color.json").read_text()
        )
        self.assertEqual(replay["task"], "PAN-35")
        self.assertEqual(replay["observation_ids"], [0, 12, 24, 36, 49])
        self.assertEqual(
            capture["rgb_render_mode"],
            {
                "mode": "material_base_color",
                "capture_source": "SCS_BASE_COLOR",
                "lighting_dependency": "none_unlit_diagnostic",
            },
        )
        self.assertNotIn("lighting_ablation", capture)


if __name__ == "__main__":
    unittest.main()
