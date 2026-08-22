import unittest

import numpy as np

from scripts.make_pan30_sky_cloud_composite import (
    composite_background,
    procedural_sky_cloud_erp,
)


class PAN30SkyCloudCompositeTests(unittest.TestCase):
    def test_composite_changes_only_no_hit_background(self):
        foreground = np.full((4, 8, 3), 20, dtype=np.uint8)
        sky = np.full((4, 8, 3), 180, dtype=np.uint8)
        valid = np.zeros((4, 8), dtype=np.bool_)
        valid[1:3, 2:6] = True

        result = composite_background(foreground, valid, sky)

        np.testing.assert_array_equal(result[valid], foreground[valid])
        np.testing.assert_array_equal(result[~valid], sky[~valid])

    def test_world_fixed_sky_is_deterministic_nonblack_and_cloudy(self):
        first = procedural_sky_cloud_erp(height=64)
        second = procedural_sky_cloud_erp(height=64)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (64, 128, 3))
        self.assertEqual(first.dtype, np.uint8)
        self.assertGreater(int(first.min()), 0)
        self.assertGreater(len(np.unique(first.reshape(-1, 3), axis=0)), 100)


if __name__ == "__main__":
    unittest.main()
