import unittest

import numpy as np

from macarons.utility.ue5_geometry_conformance import (
    backproject_face,
    direction_gram,
    forward_directions,
    seam_summary,
)
from macarons.utility.ue5_observation_contract import (
    FACE_NAMES,
    SCHEMA_VERSION,
    CanonicalFace,
    CanonicalObservationBundle,
)


ROTATIONS = {
    "front": ((0, 0, 1), (-1, 0, 0), (0, -1, 0)),
    "back": ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
    "left": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "right": ((-1, 0, 0), (0, 0, -1), (0, -1, 0)),
    "up": ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    "down": ((0, -1, 0), (-1, 0, 0), (0, 0, -1)),
}


def make_face(name):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(ROTATIONS[name], dtype=np.float64)
    return CanonicalFace(
        face_name=name,
        request_id="request",
        frame_id="frame",
        capture_timestamp_ns=1,
        rgb_uint8=np.zeros((3, 3, 3), dtype=np.uint8),
        depth_range_m=np.full((3, 3), 2.0, dtype=np.float32),
        valid_mask=np.ones((3, 3), dtype=np.bool_),
        K_pixel=np.asarray(((1, 0, 1), (0, 1, 1), (0, 0, 1)), dtype=np.float64),
        T_world_from_cam=transform,
        image_size=(3, 3),
    )


class UE5GeometryConformanceTests(unittest.TestCase):
    def setUp(self):
        self.bundle = CanonicalObservationBundle(
            schema_version=SCHEMA_VERSION,
            request_id="request",
            frame_id="frame",
            capture_timestamp_ns=1,
            position_world_m=(0.0, 0.0, 0.0),
            faces=tuple(make_face(name) for name in FACE_NAMES),
            provenance={"test": True},
        )

    def test_backprojection_uses_euclidean_range_and_world_pose(self):
        points, colors = backproject_face(self.bundle.faces[0])
        self.assertEqual(points.shape, (9, 3))
        self.assertEqual(colors.shape, (9, 3))
        self.assertTrue(np.allclose(points[4], (2.0, 0.0, 0.0)))
        self.assertAlmostEqual(float(np.linalg.norm(points[0])), 2.0)

    def test_direction_gram_matches_signed_cardinal_basis(self):
        directions = forward_directions(self.bundle)
        gram = direction_gram(directions)
        self.assertTrue(np.allclose(np.diag(gram), 1.0))
        self.assertEqual(float(gram[0, 1]), -1.0)
        self.assertEqual(float(gram[2, 3]), -1.0)
        self.assertEqual(float(gram[4, 5]), -1.0)

    def test_adjacent_synthetic_faces_share_boundary_points(self):
        summary = seam_summary(self.bundle)
        self.assertEqual(summary["seam_count"], 12)
        self.assertEqual(summary["min_validity_agreement_fraction"], 1.0)
        self.assertAlmostEqual(summary["max_world_point_distance_p95_m"], 0.0)
        self.assertAlmostEqual(summary["max_world_point_distance_median_m"], 0.0)
        self.assertAlmostEqual(summary["max_relative_world_point_distance_p95"], 0.0)


if __name__ == "__main__":
    unittest.main()
