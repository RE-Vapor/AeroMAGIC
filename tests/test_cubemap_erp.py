import unittest
from dataclasses import replace

import numpy as np

from macarons.utility.cubemap_erp import (
    CubemapProjectionError,
    cubemap_rgb_to_equirectangular,
)
from macarons.utility.ue5_observation_contract import make_synthetic_plane_bundle


def _direction_rgb_faces(size=65):
    """Return a canonical bundle whose RGB values encode world-ray direction."""

    bundle = make_synthetic_plane_bundle(image_size=size)
    rows, columns = np.indices((size, size), dtype=np.float64)
    encoded_faces = []
    for index, face in enumerate(bundle.faces):
        pixels = np.stack((columns, rows, np.ones_like(columns)), axis=-1)
        camera_rays = pixels @ np.linalg.inv(face.K_pixel).T
        camera_rays /= np.linalg.norm(camera_rays, axis=-1, keepdims=True)
        world_rays = camera_rays @ face.T_world_from_cam[:3, :3].T
        rgb = np.rint((world_rays + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        # Deliberately destroy both conventional names and conventional ordering.
        encoded_faces.append(
            replace(face, face_name=f"orientation-{index}", rgb_uint8=rgb)
        )
    return replace(
        bundle,
        faces=tuple(encoded_faces[index] for index in (3, 5, 1, 4, 0, 2)),
    )


class CubemapERPTests(unittest.TestCase):
    def setUp(self):
        self.bundle = _direction_rgb_faces()

    def assertColorNear(self, actual, expected, tolerance=9):
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.int16),
            np.asarray(expected, dtype=np.int16),
            atol=tolerance,
            rtol=0.0,
        )

    def test_world_ray_projection_hits_six_cardinal_directions(self):
        height = 33
        erp = cubemap_rgb_to_equirectangular(self.bundle, output_height=height)

        self.assertEqual(erp.shape, (height, 2 * height, 3))
        self.assertEqual(erp.dtype, np.uint8)
        equator = height // 2
        # PIONEER world convention: +Y is up and +Z/front is the image centre.
        # Moving right reaches -X/right; the horizontal seam is -Z/back.
        self.assertColorNear(erp[equator, 0], (128, 128, 0))
        self.assertColorNear(erp[equator, height // 2], (255, 128, 128))
        self.assertColorNear(erp[equator, height], (128, 128, 255))
        self.assertColorNear(erp[equator, 3 * height // 2], (0, 128, 128))
        self.assertColorNear(erp[0, height], (128, 255, 128))
        self.assertColorNear(erp[-1, height], (128, 0, 128))

    def test_projection_is_byte_deterministic_at_seams_and_under_permutation(self):
        expected = cubemap_rgb_to_equirectangular(self.bundle, output_height=33)
        reordered = replace(self.bundle, faces=tuple(reversed(self.bundle.faces)))
        repeated = cubemap_rgb_to_equirectangular(reordered, output_height=33)

        np.testing.assert_array_equal(repeated, expected)
        # Both sides of the wrap-around seam sample the same -Z/back neighbourhood.
        np.testing.assert_allclose(
            expected[:, 0].astype(np.int16),
            expected[:, -1].astype(np.int16),
            # Pixel-centre ERP rays straddle the seam by one longitude step.
            atol=16,
            rtol=0.0,
        )

    def test_incomplete_bundle_is_rejected(self):
        incomplete = replace(self.bundle, faces=self.bundle.faces[:-1])
        with self.assertRaisesRegex(CubemapProjectionError, "exactly six"):
            cubemap_rgb_to_equirectangular(incomplete, output_height=33)

    def test_bad_face_size_is_rejected(self):
        first = self.bundle.faces[0]
        rectangular = replace(
            first,
            rgb_uint8=first.rgb_uint8[:-1],
            image_size=(first.image_size[0] - 1, first.image_size[1]),
        )
        invalid = replace(self.bundle, faces=(rectangular,) + self.bundle.faces[1:])
        with self.assertRaisesRegex(CubemapProjectionError, "square"):
            cubemap_rgb_to_equirectangular(invalid, output_height=33)


if __name__ == "__main__":
    unittest.main()
