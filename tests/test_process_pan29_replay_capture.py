import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from macarons.utility.ue5_observation_contract import (
    CanonicalFace,
    CanonicalObservationBundle,
    SCHEMA_VERSION,
    write_bundle,
)
from scripts.process_pan29_replay_capture import process_replay_capture


FACE_NAMES = ("front", "back", "left", "right", "up", "down")
CONTRACT_TO_PIONEER = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
)
PIONEER_FACE_ROTATIONS = {
    "front": ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "back": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
    "left": ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
    "right": ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)),
    "up": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "down": ((-1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
}


class ProcessPAN29ReplayCaptureTests(unittest.TestCase):
    def _fixture(self, root: Path):
        observation_id = 5
        source_timestamp = 1_787_242_568_309_971_778
        actual_timestamp = source_timestamp + 123456
        source_position = np.asarray([1.0, 2.0, 3.0])
        ue_position = [500.0, 1500.0, 1000.0]
        replay_source = {
            "schema_version": "pan29.replay-source.v1",
            "observation_id": observation_id,
            "source_bundle_id": observation_id,
            "source_capture_timestamp_ns": source_timestamp,
            "source_capture_timestamp_utc": "2026-08-20T16:16:08.309972Z",
            "planner_position_scene_units": source_position.tolist(),
            "requested_position_ue_cm": ue_position,
            "source_metrics_sha256": "a" * 64,
            "within_pan13_conservative_fly_volume": False,
            "planner_input_unchanged": True,
            "ue5_role": "post_run_visualization_only",
        }
        request = {
            "schema_version": "pan15.capture-request.v1",
            "scenario": "hkust",
            "level_path": "/Game/PAN13_Derived/HKUST_ZUp_QA",
            "position_actor_label": None,
            "position_ue_cm": ue_position,
            "request_id": "pan29-test-000005",
            "frame_id": "observation-000005",
            "replay_source": replay_source,
        }
        request_path = root / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        plan = {
            "schema_version": "pan29.ue5-postrun-replay.v1",
            "selected_bundle_ids": [0, 5, 10, 14, 19],
            "source": {
                "metrics": {"sha256": "a" * 64},
                "scene_units_per_meter": 0.2,
            },
            "observations": [
                {
                    "observation_id": observation_id,
                    "replay_request_id": request["request_id"],
                    "replay_frame_id": request["frame_id"],
                    "source_capture_timestamp_ns": source_timestamp,
                    "source_capture_timestamp_utc": replay_source[
                        "source_capture_timestamp_utc"
                    ],
                    "planner_position_scene_units": source_position.tolist(),
                    "ue_position_cm": ue_position,
                    "within_pan13_conservative_fly_volume": False,
                    "source_semantic_validation": {
                        "schema_version": "pan29.source-bundle-semantic-validation.v1",
                        "result": "PASS",
                        "bundle_id": observation_id,
                        "source_capture_timestamp_ns": source_timestamp,
                        "source_capture_timestamp_utc": replay_source[
                            "source_capture_timestamp_utc"
                        ],
                        "camera_center_scene_units": source_position.tolist(),
                        "camera_center_max_abs_error_scene_units": 0.0,
                        "face_basis_max_abs_error": 0.0,
                        "faces": [
                            {
                                "face_name": face_name,
                                "bundle_id": observation_id,
                                "camera_center_scene_units": source_position.tolist(),
                                "camera_center_max_abs_error_scene_units": 0.0,
                                "T_world_from_cam_rotation": [
                                    list(values)
                                    for values in PIONEER_FACE_ROTATIONS[face_name]
                                ],
                                "basis_max_abs_error": 0.0,
                            }
                            for face_name in FACE_NAMES
                        ],
                    },
                }
            ],
        }
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")

        contract_position = CONTRACT_TO_PIONEER.T @ (source_position / 0.2)
        faces = []
        raw_faces = []
        intrinsic = np.asarray([[16.0, 0, 15.5], [0, 16.0, 15.5], [0, 0, 1]])
        for face_index, face_name in enumerate(FACE_NAMES):
            pioneer_rotation = np.asarray(PIONEER_FACE_ROTATIONS[face_name])
            contract_rotation = CONTRACT_TO_PIONEER.T @ pioneer_rotation
            transform = np.eye(4)
            transform[:3, :3] = contract_rotation
            transform[:3, 3] = contract_position
            rgb = np.zeros((32, 32, 3), dtype=np.uint8)
            rgb[..., face_index % 3] = 60 + 25 * face_index
            faces.append(
                CanonicalFace(
                    face_name=face_name,
                    request_id=request["request_id"],
                    frame_id=request["frame_id"],
                    capture_timestamp_ns=actual_timestamp,
                    rgb_uint8=rgb,
                    depth_range_m=np.ones((32, 32), dtype=np.float32),
                    valid_mask=np.ones((32, 32), dtype=np.bool_),
                    K_pixel=intrinsic,
                    T_world_from_cam=transform,
                    image_size=(32, 32),
                )
            )
            pioneer_transform = np.eye(4)
            pioneer_transform[:3, :3] = pioneer_rotation
            pioneer_transform[:3, 3] = source_position
            raw_faces.append(
                {
                    "face_name": face_name,
                    "request_id": request["request_id"],
                    "frame_id": request["frame_id"],
                    "capture_timestamp_ns": actual_timestamp,
                    "location_ue_cm": ue_position,
                    "T_pioneer_world_from_cam": pioneer_transform.tolist(),
                    "pioneer_basis_max_abs_error": 0.0,
                }
            )
        canonical = CanonicalObservationBundle(
            schema_version=SCHEMA_VERSION,
            request_id=request["request_id"],
            frame_id=request["frame_id"],
            capture_timestamp_ns=actual_timestamp,
            position_world_m=tuple(contract_position),
            faces=tuple(faces),
            provenance={"task": "PAN-15", "replay_source": replay_source},
        )
        canonical_manifest = write_bundle(canonical, root / "canonical_bundle")
        raw = {
            "schema_version": "pan15.ue5-raw-rgbd.v1",
            "scenario": "hkust",
            "request_id": request["request_id"],
            "frame_id": request["frame_id"],
            "capture_timestamp_ns": actual_timestamp,
            "requested_position_ue_cm": ue_position,
            "shared_optical_center_ue_cm": ue_position,
            "replay_source": replay_source,
            "faces": raw_faces,
        }
        raw_manifest = root / "raw_manifest.json"
        raw_manifest.write_text(json.dumps(raw), encoding="utf-8")
        return plan_path, request_path, raw_manifest, canonical_manifest

    def test_validates_identity_pose_basis_and_writes_deterministic_erp(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, request, raw, canonical = self._fixture(root)
            output = root / "processed"
            receipt = process_replay_capture(
                plan_path=plan,
                observation_id=5,
                request_path=request,
                raw_manifest_path=raw,
                canonical_manifest_path=canonical,
                output_dir=output,
                erp_height=32,
            )

            self.assertEqual(receipt["result"], "PASS")
            self.assertEqual(receipt["observation_id"], 5)
            self.assertEqual(
                receipt["source_capture_timestamp_ns"], 1_787_242_568_309_971_778
            )
            self.assertEqual(
                receipt["ue5_actual_capture_timestamp_ns"],
                1_787_242_568_310_095_234,
            )
            self.assertLessEqual(receipt["position_max_abs_error_scene_units"], 1e-9)
            self.assertEqual(receipt["face_basis_max_abs_error"], 0.0)
            self.assertTrue(receipt["planner_input_unchanged"])
            self.assertEqual(receipt["ue5_role"], "post_run_visualization_only")
            with Image.open(output / "erp.png") as image:
                self.assertEqual(image.size, (64, 32))
                self.assertEqual(image.mode, "RGB")
            persisted = json.loads((output / "receipt.json").read_text())
            self.assertEqual(persisted, receipt)
            self.assertEqual(len(receipt["erp"]["sha256"]), 64)
            self.assertTrue(receipt["erp"]["byte_deterministic_reprojection"])
            self.assertEqual(
                receipt["source_semantic_validation"]["schema_version"],
                "pan29.source-bundle-semantic-validation.v1",
            )
            self.assertEqual(
                len(receipt["source_semantic_validation"]["basis_sha256"]), 64
            )

    def test_rejects_polar_roll_mismatch_without_publishing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, request, raw, canonical = self._fixture(root)
            payload = json.loads(raw.read_text())
            payload["faces"][4]["T_pioneer_world_from_cam"][0][0] = 0.0
            raw.write_text(json.dumps(payload))
            output = root / "invalid"
            with self.assertRaisesRegex(ValueError, "basis"):
                process_replay_capture(
                    plan_path=plan,
                    observation_id=5,
                    request_path=request,
                    raw_manifest_path=raw,
                    canonical_manifest_path=canonical,
                    output_dir=output,
                    erp_height=32,
                )
            self.assertFalse(output.exists())

    def test_rejects_capture_that_matches_constants_but_not_plan_pinned_source_basis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, request, raw, canonical = self._fixture(root)
            payload = json.loads(plan.read_text())
            semantic = payload["observations"][0]["source_semantic_validation"]
            semantic["faces"][4]["T_world_from_cam_rotation"] = [
                [0.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
            plan.write_text(json.dumps(payload))
            output = root / "source-basis-mismatch"
            with self.assertRaisesRegex(ValueError, "source basis"):
                process_replay_capture(
                    plan_path=plan,
                    observation_id=5,
                    request_path=request,
                    raw_manifest_path=raw,
                    canonical_manifest_path=canonical,
                    output_dir=output,
                    erp_height=32,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
