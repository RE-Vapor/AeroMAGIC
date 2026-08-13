import tempfile
import unittest
from pathlib import Path

from macarons.utility.huge_3dgs_adapter import Huge3DGSConfig


class Huge3DGSConfigTest(unittest.TestCase):
    def test_disabled_config_needs_no_assets(self):
        parsed = Huge3DGSConfig.from_config({"huge_3dgs_rgb_enabled": False})
        self.assertFalse(parsed.enabled)

    def test_enabled_config_validates_assets_and_numeric_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ply = root / "scene.ply"
            manifest = root / "manifest.json"
            ply.write_bytes(b"ply")
            manifest.write_text("{}", encoding="utf-8")
            parsed = Huge3DGSConfig.from_config(
                {
                    "huge_3dgs_rgb_enabled": True,
                    "huge_3dgs_ply_path": str(ply),
                    "huge_3dgs_manifest_path": str(manifest),
                    "huge_3dgs_render_device": "cuda:4",
                    "huge_3dgs_alpha_threshold": 0.2,
                }
            )
            self.assertTrue(parsed.enabled)
            self.assertEqual(parsed.render_device, "cuda:4")
            self.assertEqual(parsed.alpha_threshold, 0.2)

    def test_invalid_alpha_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ply = root / "scene.ply"
            manifest = root / "manifest.json"
            ply.write_bytes(b"ply")
            manifest.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "alpha_threshold"):
                Huge3DGSConfig.from_config(
                    {
                        "huge_3dgs_rgb_enabled": True,
                        "huge_3dgs_ply_path": str(ply),
                        "huge_3dgs_manifest_path": str(manifest),
                        "huge_3dgs_alpha_threshold": 1.1,
                    }
                )


if __name__ == "__main__":
    unittest.main()
