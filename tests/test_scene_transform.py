from types import SimpleNamespace
import unittest

import numpy as np

from macarons.utility.scene_transform import (
    resolve_scene_mesh_transform,
    transform_scene_vertices,
)


SCENE = "12-NW-6C-5"
TRANSFORM = {
    "axis_order": [0, 2, 1],
    "axis_signs": [1, 1, -1],
    "translation": [-45675.0, 0.0, 22125.0],
    "scale": 0.1,
}


class SceneTransformTests(unittest.TestCase):
    def test_absent_scene_transform_is_identity(self):
        vertices = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        transform = resolve_scene_mesh_transform({}, "eiffel")
        np.testing.assert_array_equal(
            transform_scene_vertices(vertices, transform),
            vertices,
        )

    def test_transform_recenters_rotates_and_scales_tile_coordinates(self):
        config = SimpleNamespace(scene_mesh_transforms={SCENE: TRANSFORM})
        transform = resolve_scene_mesh_transform(config, SCENE)
        rotation = np.eye(3)[list(transform["axis_order"])]
        rotation *= np.asarray(transform["axis_signs"])[:, None]
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)
        raw = np.array(
            [
                [45600.0, 22050.0, 0.0],
                [45750.0, 22200.0, 10.0],
            ],
            dtype=np.float64,
        )
        expected_preprocessed = np.array(
            [[-7.5, 0.0, 7.5], [7.5, 1.0, -7.5]],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            transform_scene_vertices(raw, transform),
            expected_preprocessed,
        )
        np.testing.assert_allclose(
            transform_scene_vertices(raw, transform, scene_scale_factor=10.0),
            expected_preprocessed * 10.0,
        )

    def test_transform_is_scene_scoped(self):
        config = {"scene_mesh_transforms": {SCENE: TRANSFORM}}
        self.assertEqual(
            resolve_scene_mesh_transform(config, "eiffel")["axis_order"],
            (0, 1, 2),
        )

    def test_rejects_reflection_prone_or_ambiguous_contracts(self):
        with self.assertRaisesRegex(ValueError, "axis_order"):
            resolve_scene_mesh_transform(
                {"scene_mesh_transforms": {SCENE: {"axis_order": [0, 0, 2]}}},
                SCENE,
            )
        with self.assertRaisesRegex(ValueError, "axis_signs"):
            resolve_scene_mesh_transform(
                {"scene_mesh_transforms": {SCENE: {"axis_signs": [1, 0, -1]}}},
                SCENE,
            )
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            resolve_scene_mesh_transform(
                {"scene_mesh_transforms": {SCENE: {"offset": [0, 0, 0]}}},
                SCENE,
            )


if __name__ == "__main__":
    unittest.main()
