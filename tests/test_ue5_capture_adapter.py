import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2

from macarons.utility.ue5_capture_adapter import adapt_pan15_raw_manifest
from macarons.utility.ue5_observation_contract import FACE_NAMES, validate_bundle


class UE5CaptureAdapterTests(unittest.TestCase):
    @staticmethod
    def _asset(root, relative, array):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        array.tofile(path)
        return {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shape": list(array.shape),
        }

    @staticmethod
    def _exr_asset(root, relative, red):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        bgra = np.zeros((*red.shape, 4), dtype=np.float32)
        bgra[..., 2] = red
        if not cv2.imwrite(str(path), bgra):
            raise RuntimeError("could not write test EXR")
        return {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def test_raw_scene_depth_candidate_becomes_valid_contract_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            faces = []
            for index, name in enumerate(FACE_NAMES):
                rgb = np.full((3, 3, 3), index, dtype=np.uint8)
                depth = np.full((3, 3), 500.0, dtype="<f4")
                mask = np.ones((3, 3), dtype=np.uint8)
                pose = np.eye(4)
                pose[:3, 3] = (1.0, 2.0, 3.0)
                face_root = f"faces/{name}"
                scene = self._exr_asset(root, f"{face_root}/scene.exr", depth)
                device = self._exr_asset(root, f"{face_root}/device.exr", depth)
                faces.append(
                    {
                        "face_name": name,
                        "request_id": "request",
                        "frame_id": "frame",
                        "capture_timestamp_ns": 123,
                        "image_size": [3, 3],
                        "fov_degrees": 90.0,
                        "K_pixel": [[1, 0, 1], [0, 1, 1], [0, 0, 1]],
                        "T_world_from_cam": pose.tolist(),
                        "rgb_uint8": self._asset(root, f"{face_root}/rgb.bin", rgb),
                        "scene_depth_r": {"raw_exr": scene},
                        "device_depth_r": {"raw_exr": device},
                        "python_readback_candidate_mask": self._asset(
                            root, f"{face_root}/mask.bin", mask
                        ),
                    }
                )
            manifest = {
                "schema_version": "pan15.ue5-raw-rgbd.v1",
                "scenario": "analytic",
                "request_id": "request",
                "frame_id": "frame",
                "capture_timestamp_ns": 123,
                "world_to_meters": 100.0,
                "near_m": 0.1,
                "far_m": 1000.0,
                "engine_version": "5.4.4-test",
                "project_commit": "deadbeef",
                "capture_config_sha256": "config",
                "faces": faces,
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            bundle = adapt_pan15_raw_manifest(manifest_path)
            summary = validate_bundle(bundle)
            self.assertEqual(summary.face_count, 6)
            self.assertTrue(np.allclose(bundle.faces[0].depth_range_m, 5.0))
            self.assertEqual(
                bundle.provenance["geometry_status"],
                "candidate_only_pending_PAN-20",
            )

    def test_checksum_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "pan15.ue5-raw-rgbd.v1",
                        "world_to_meters": 100,
                        "faces": [
                            {
                                "rgb_uint8": {
                                    "path": "missing.bin",
                                    "sha256": "0" * 64,
                                    "bytes": 1,
                                    "shape": [1],
                                }
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises((FileNotFoundError, ValueError)):
                adapt_pan15_raw_manifest(path)


if __name__ == "__main__":
    unittest.main()
