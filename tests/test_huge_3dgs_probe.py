import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT = Path(__file__).parents[1] / "tools" / "probe_huge_3dgs_alignment.py"
SPEC = importlib.util.spec_from_file_location("huge_3dgs_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROBE
SPEC.loader.exec_module(PROBE)


class Huge3dgsProbeTest(unittest.TestCase):
    def test_binary_ply_header_accepts_only_the_official_schema(self):
        with tempfile.TemporaryDirectory() as temp_text:
            path = Path(temp_text) / "point_cloud_utm50.ply"
            dtype = np.dtype([(name, "<f4") for name in PROBE.REQUIRED_PLY_PROPERTIES])
            values = np.zeros(2, dtype=dtype)
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                "element vertex 2\n"
                + "".join(
                    f"property float {name}\n" for name in PROBE.REQUIRED_PLY_PROPERTIES
                )
                + "end_header\n"
            ).encode("ascii")
            path.write_bytes(header + values.tobytes())
            parsed = PROBE.parse_binary_ply_header(path)
            self.assertEqual(parsed.vertex_count, 2)
            self.assertEqual(parsed.dtype.itemsize, 72)
            self.assertEqual(parsed.properties, PROBE.REQUIRED_PLY_PROPERTIES)

    def test_camera_frame_rotation_preserves_view_coordinates(self):
        q = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ]
        )
        r_magic = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        points_huge = np.array([[4.0, 5.0, 6.0], [-1.0, 2.0, 3.0]])
        points_magic = points_huge @ q.T
        r_huge = q.T @ r_magic
        np.testing.assert_allclose(points_magic @ r_magic, points_huge @ r_huge)

    def test_prefilter_returns_empty_buffers_for_an_empty_camera_view(self):
        torch = PROBE.torch
        buffers = PROBE.GaussianBuffers(
            xyz=torch.tensor([[0.0, 0.0, -1.0], [1.0, 1.0, -2.0]]),
            features_dc=torch.zeros((2, 1, 3)),
            opacity=torch.ones((2, 1)),
            scales=torch.ones((2, 3)),
            rotations=torch.zeros((2, 4)),
        )
        camera = SimpleNamespace(
            FoVx=1.0,
            FoVy=1.0,
            world_view_transform=torch.eye(4),
            full_proj_transform=torch.eye(4),
        )

        compacted, metadata = PROBE.prefilter_gaussians_for_view(
            buffers,
            camera,
            chunk_size=2,
            center_margin=0.15,
        )

        self.assertEqual(compacted.xyz.shape[0], 0)
        self.assertEqual(metadata["retained_gaussians"], 0)
        self.assertEqual(metadata["retained_fraction"], 0.0)

    def test_render_returns_black_outputs_for_empty_gaussian_buffers(self):
        torch = PROBE.torch
        buffers = PROBE.GaussianBuffers(
            xyz=torch.empty((0, 3)),
            features_dc=torch.empty((0, 1, 3)),
            opacity=torch.empty((0, 1)),
            scales=torch.empty((0, 3)),
            rotations=torch.empty((0, 4)),
        )
        camera = SimpleNamespace(
            FoVx=1.0,
            FoVy=1.0,
            world_view_transform=torch.eye(4),
            full_proj_transform=torch.eye(4),
            camera_center=torch.zeros(3),
        )

        rendered = PROBE.render_gaussians(
            buffers,
            camera,
            image_height=4,
            image_width=6,
            kernel_size=0.0,
        )

        self.assertEqual(tuple(rendered["rgb"].shape), (3, 4, 6))
        self.assertEqual(tuple(rendered["alpha"].shape), (1, 4, 6))
        self.assertEqual(tuple(rendered["expected_depth"].shape), (1, 4, 6))
        self.assertEqual(tuple(rendered["median_depth"].shape), (1, 4, 6))
        self.assertEqual(rendered["radii"].numel(), 0)
        self.assertEqual(int(torch.count_nonzero(rendered["rgb"])), 0)
        self.assertEqual(int(torch.count_nonzero(rendered["alpha"])), 0)

    def test_mask_and_depth_statistics_are_raw_not_fitted(self):
        mesh_mask = np.array([[True, True], [False, False]])
        gaussian_mask = np.array([[True, False], [True, False]])
        mask = PROBE.mask_metrics(mesh_mask, gaussian_mask)
        self.assertEqual(mask["intersection_pixels"], 1)
        self.assertEqual(mask["union_pixels"], 3)
        self.assertAlmostEqual(mask["iou"], 1.0 / 3.0)

        mesh_depth = np.array([[10.0, 20.0], [0.0, 0.0]])
        gaussian_depth = np.array([[11.0, 16.0], [0.0, 0.0]])
        depth = PROBE.depth_metrics(
            mesh_depth, gaussian_depth, mesh_mask & np.isfinite(gaussian_depth)
        )
        self.assertEqual(depth["overlap_pixels"], 2)
        self.assertAlmostEqual(depth["absolute_error_m"]["median"], 2.5)
        self.assertAlmostEqual(depth["relative_error"]["median"], 0.15)


if __name__ == "__main__":
    unittest.main()
