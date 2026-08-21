import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from macarons.utility.ue5_observation_contract import (
    BundleValidationError,
    FACE_NAMES,
    adapt_pan10_observation_bundle,
    camera_z_to_ray_range,
    load_bundle,
    make_synthetic_plane_bundle,
    validate_bundle,
    write_bundle,
)


class UE5ObservationContractTests(unittest.TestCase):
    def test_camera_z_plane_has_analytic_center_edge_corner_ranges(self):
        size = 5
        K = np.asarray([[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]])
        depth_z = np.full((size, size), 2.0, dtype=np.float32)
        ranges = camera_z_to_ray_range(depth_z, K, np.ones_like(depth_z, bool))
        self.assertAlmostEqual(float(ranges[2, 2]), 2.0, places=6)
        self.assertAlmostEqual(float(ranges[2, 0]), 2.0 * np.sqrt(2.0), places=6)
        self.assertAlmostEqual(float(ranges[0, 0]), 2.0 * np.sqrt(3.0), places=6)

    def test_synthetic_six_face_bundle_is_valid_and_shares_one_center(self):
        bundle = make_synthetic_plane_bundle()
        summary = validate_bundle(bundle)
        self.assertEqual(tuple(face.face_name for face in bundle.faces), FACE_NAMES)
        self.assertEqual(summary.face_count, 6)
        for face in bundle.faces:
            np.testing.assert_allclose(
                face.T_world_from_cam[:3, 3], bundle.position_world_m
            )

    def test_missing_face_is_rejected(self):
        bundle = make_synthetic_plane_bundle()
        incomplete = replace(bundle, faces=bundle.faces[:-1])
        with self.assertRaisesRegex(BundleValidationError, "exactly ordered"):
            validate_bundle(incomplete)

    def test_mismatched_epoch_is_rejected(self):
        bundle = make_synthetic_plane_bundle()
        changed = replace(
            bundle.faces[0], capture_timestamp_ns=bundle.capture_timestamp_ns + 1
        )
        invalid = replace(bundle, faces=(changed,) + bundle.faces[1:])
        with self.assertRaisesRegex(BundleValidationError, "bundle epoch"):
            validate_bundle(invalid)

    def test_no_hit_requires_nan(self):
        bundle = make_synthetic_plane_bundle()
        face = bundle.faces[0]
        mask = face.valid_mask.copy()
        mask[0, 0] = False
        invalid = replace(bundle, faces=(replace(face, valid_mask=mask),) + bundle.faces[1:])
        with self.assertRaisesRegex(BundleValidationError, "represented by NaN"):
            validate_bundle(invalid)

    def test_manifest_round_trip_and_checksum_rejection(self):
        bundle = make_synthetic_plane_bundle()
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "bundle"
            manifest = write_bundle(bundle, destination)
            loaded = load_bundle(manifest)
            self.assertEqual(validate_bundle(loaded).face_count, 6)
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            rgb_path = destination / manifest_payload["faces"][0]["assets"]["rgb_uint8"]["path"]
            rgb_path.write_bytes(rgb_path.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(BundleValidationError, "checksum mismatch"):
                load_bundle(manifest)

    def test_pan10_adapter_converts_camera_z_without_reimplementing_fusion(self):
        bundle = make_synthetic_plane_bundle()
        fake_faces = []
        for face in bundle.faces:
            world_from_camera = face.T_world_from_cam
            camera_from_world = np.linalg.inv(world_from_camera)
            fake_faces.append(
                SimpleNamespace(
                    name=face.face_name,
                    rgb=face.rgb_uint8.astype(np.float32)[None] / 255.0,
                    depth_z=np.full((1, 5, 5, 1), 2.0, dtype=np.float32),
                    valid_mask=np.ones((1, 5, 5, 1), dtype=bool),
                    fov_degrees=90.0,
                    metadata={
                        "K_pixel": face.K_pixel[None],
                        "R_opencv_world_to_camera": camera_from_world[:3, :3],
                        "T_opencv_world_to_camera": camera_from_world[:3, 3:4],
                        "depth_source": "GT",
                        "rig_frame": "world",
                    },
                )
            )
        pan10 = SimpleNamespace(
            bundle_id=7,
            center=np.asarray(bundle.position_world_m)[None],
            faces=tuple(fake_faces),
            metadata={
                "observation_mode": "cubemap6",
                "capture_timestamp_unix_ns": bundle.capture_timestamp_ns,
            },
        )
        adapted = adapt_pan10_observation_bundle(pan10)
        self.assertEqual(validate_bundle(adapted).face_count, 6)
        self.assertAlmostEqual(
            float(adapted.faces[0].depth_range_m[0, 0]),
            2.0 * np.sqrt(3.0),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
