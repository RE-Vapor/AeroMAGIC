import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PAN15CaptureWorkflowTests(unittest.TestCase):
    def test_ue_script_preserves_both_raw_depth_sources_and_mask(self):
        source = (
            ROOT / "unreal" / "PAN23" / "Scripts" / "capture_six_face_rgbd.py"
        ).read_text(encoding="utf-8")
        self.assertIn("SCS_SCENE_DEPTH", source)
        self.assertIn("SCS_DEVICE_DEPTH", source)
        self.assertIn("python_readback_candidate_mask_uint8.bin", source)
        self.assertIn("read_render_target_raw", source)
        self.assertIn("read_render_target_raw_pixel_area", source)
        self.assertIn('"raw_exr"', source)
        self.assertIn("create_render_target2d", source)
        self.assertIn("RTF_RGBA16F", source)
        self.assertIn("T_world_from_cam", source)

    def test_shell_workflow_has_atomic_publish_and_two_scenarios(self):
        source = (ROOT / "scripts" / "run_pan15_capture.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("mktemp -d", source)
        self.assertIn('mv "$temporary" "$output_dir"', source)
        self.assertIn("analytic)", source)
        self.assertIn("hkust)", source)
        self.assertIn("PAN13_Legal_Start_Camera", source)
        self.assertIn("adapt_pan15_ue_capture.py", source)


if __name__ == "__main__":
    unittest.main()
