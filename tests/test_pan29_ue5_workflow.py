import json
from pathlib import Path
import tempfile
import unittest


from scripts.make_pan29_capture_request import build_capture_request


ROOT = Path(__file__).resolve().parents[1]
FACE_NAMES = ("front", "back", "left", "right", "up", "down")
EXPECTED_ROTATIONS = {
    "front": [0.0, 90.0, 0.0],
    "back": [0.0, -90.0, 0.0],
    "left": [0.0, 0.0, 0.0],
    "right": [0.0, 180.0, 0.0],
    "up": [90.0, 0.0, -90.0],
    "down": [-90.0, 0.0, 90.0],
}
EXPECTED_PIONEER_ROTATIONS = {
    "front": [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
    "back": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
    "left": [[0, 0, 1], [0, -1, 0], [1, 0, 0]],
    "right": [[0, 0, -1], [0, -1, 0], [-1, 0, 0]],
    "up": [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
    "down": [[-1, 0, 0], [0, 0, -1], [0, -1, 0]],
}


class PAN29UE5WorkflowTests(unittest.TestCase):
    def test_replay_rig_is_name_aligned_with_pioneer_world_axes(self):
        config = json.loads(
            (ROOT / "unreal/PAN29/Config/pan29_hkust_replay.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["schema_version"], "pan29.ue5-replay-rig.v1")
        self.assertEqual(tuple(config["face_names"]), FACE_NAMES)
        self.assertEqual(config["level_path"], "/Game/PAN13_Derived/HKUST_ZUp_QA")
        self.assertEqual(config["scene_units_per_meter"], 0.2)
        by_name = {row["face_name"]: row for row in config["faces"]}
        self.assertEqual(tuple(by_name), FACE_NAMES)
        for face_name in FACE_NAMES:
            self.assertEqual(
                by_name[face_name]["rotation_degrees"],
                EXPECTED_ROTATIONS[face_name],
            )
            self.assertEqual(
                by_name[face_name]["expected_T_pioneer_world_from_cam_rotation"],
                EXPECTED_PIONEER_ROTATIONS[face_name],
            )

    def test_request_preserves_source_identity_without_faking_capture_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan_path = Path(temporary) / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "schema_version": "pan29.ue5-postrun-replay.v1",
                        "source": {"metrics": {"sha256": "a" * 64}},
                        "observations": [
                            {
                                "observation_id": 5,
                                "replay_request_id": "pan29-hkust-gt20obs-000005",
                                "replay_frame_id": "observation-000005",
                                "source_capture_timestamp_ns": 1787242568309971778,
                                "source_capture_timestamp_utc": "2026-08-20T16:16:08.309972Z",
                                "planner_position_scene_units": [-20.0, 58.5, 6.5],
                                "ue_position_cm": [-10000.0, 3250.0, 29250.0],
                                "within_pan13_conservative_fly_volume": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            request = build_capture_request(plan_path=plan_path, observation_id=5)
            self.assertEqual(request["schema_version"], "pan15.capture-request.v1")
            self.assertEqual(request["frame_id"], "observation-000005")
            self.assertEqual(request["position_ue_cm"], [-10000.0, 3250.0, 29250.0])
            replay = request["replay_source"]
            self.assertEqual(replay["observation_id"], 5)
            self.assertEqual(replay["source_capture_timestamp_ns"], 1787242568309971778)
            self.assertFalse(replay["within_pan13_conservative_fly_volume"])
            self.assertNotIn("capture_timestamp_ns", request)
            self.assertNotIn("ue5_actual_capture_timestamp_ns", replay)

    def test_capture_script_fail_closes_on_full_basis_and_propagates_replay_source(self):
        source = (
            ROOT / "unreal/PAN23/Scripts/capture_six_face_rgbd.py"
        ).read_text(encoding="utf-8")
        self.assertIn("expected_T_pioneer_world_from_cam_rotation", source)
        self.assertIn("T_pioneer_world_from_cam", source)
        self.assertIn('"replay_source": request.get("replay_source")', source)

    def test_runner_uses_tmux_and_refuses_output_reuse(self):
        source = (ROOT / "scripts/run_pan29_ue5_replay.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("tmux new-session", source)
        self.assertIn("refusing to reuse", source)
        self.assertIn("make_pan29_replay_manifest.py", source)
        self.assertIn("process_pan29_replay_capture.py", source)
        self.assertIn("replay_plan_sha256", source)
        self.assertIn("replay_result.pending.json", source)
        self.assertIn("make_pan29_preview.py", source)
        self.assertIn('SOURCE_PREVIEW_SIDECAR="${SOURCE_PREVIEW%.png}.json"', source)
        self.assertNotIn('"$SOURCE_PREVIEW.json"', source)
        self.assertIn("process_exit", source)
        self.assertIn("-process-failed-", source)
        self.assertIn("( set -euo pipefail; run_science )", source)
        preview_invocation = source.index(
            '"$python_bin" "$repo_root/scripts/make_pan29_preview.py"'
        )
        self.assertLess(
            source.index('snapshot_integrity_postflight=%s'),
            preview_invocation,
        )
        self.assertLess(
            preview_invocation,
            source.rindex('finished_at_utc=%s'),
        )


if __name__ == "__main__":
    unittest.main()
