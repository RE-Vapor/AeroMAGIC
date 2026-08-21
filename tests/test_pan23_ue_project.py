import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "unreal" / "PAN23"


class PAN23UEProjectTests(unittest.TestCase):
    def setUp(self):
        self.project = json.loads(
            (PROJECT / "PioneerPAN23.uproject").read_text(encoding="utf-8")
        )
        self.capture = json.loads(
            (PROJECT / "Config" / "pan23_capture.json").read_text(encoding="utf-8")
        )
        self.scene = json.loads(
            (PROJECT / "Config" / "pan23_analytic_scene.json").read_text(
                encoding="utf-8"
            )
        )

    def test_project_is_content_only_and_pinned_to_ue54(self):
        self.assertEqual(self.project["EngineAssociation"], "5.4")
        self.assertNotIn("Modules", self.project)
        plugins = {item["Name"]: item["Enabled"] for item in self.project["Plugins"]}
        self.assertTrue(plugins["PythonScriptPlugin"])
        self.assertTrue(plugins["EditorScriptingUtilities"])
        self.assertFalse(plugins["AndroidFileServer"])

    def test_capture_contract_is_fixed_and_six_face(self):
        self.assertEqual(
            self.capture["face_names"],
            ["front", "back", "left", "right", "up", "down"],
        )
        self.assertEqual([item["face_name"] for item in self.capture["faces"]], self.capture["face_names"])
        self.assertEqual(self.capture["face_resolution"], 256)
        self.assertEqual(self.capture["fov_degrees"], 90.0)
        self.assertEqual(self.capture["world_to_meters"], 100.0)
        self.assertEqual(self.capture["gpu_index"], 1)
        self.assertEqual(self.capture["rhi"], "Vulkan")

    def test_analytic_scene_has_plane_markers_seam_and_open_sky(self):
        labels = {item["label"] for item in self.scene["actors"]}
        self.assertIn("PAN23_Analytic_FrontPlane", labels)
        self.assertIn("PAN23_Seam_Box_FrontRight", labels)
        for suffix in ("PosX", "NegX", "PosY", "NegY", "PosZ", "NegZ"):
            self.assertIn(f"PAN23_Marker_{suffix}", labels)
        self.assertTrue(self.scene["required_open_sky"])
        self.assertEqual(self.scene["analytic_front_plane_camera_z_m"], 5.0)

    def test_paper_stage_visual_preset_is_bounded(self):
        lighting = self.capture["lighting_preset"]
        self.assertTrue(lighting["sky_atmosphere"])
        self.assertTrue(lighting["sky_light"])
        self.assertFalse(lighting["volumetric_cloud"])
        self.assertEqual(self.capture["console_variables"]["r.DefaultFeature.AutoExposure"], 0)
        self.assertEqual(self.capture["console_variables"]["r.DefaultFeature.MotionBlur"], 0)


if __name__ == "__main__":
    unittest.main()
