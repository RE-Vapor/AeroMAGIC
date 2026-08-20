import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from macarons.utility.planning_observations import (
    CUBEMAP_FACE_NAMES,
    CUBEMAP_RIG_FRAME_WORLD,
    CUBEMAP_WORLD_EXTRINSICS_VERSION,
    ObservationBundle,
    PerspectiveFaceObservation,
    build_cubemap_cameras,
    capture_cubemap_observation,
    cubemap_pixel_intrinsics,
    process_cubemap_observation,
    visible_union_from_depth_maps,
)
from macarons.utility.planning_depth import update_proxy_state


class PlanningObservationTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.center = torch.zeros(1, 3)
        self.cameras = build_cubemap_cameras(
            self.center,
            znear=0.1,
            zfar=10.0,
            device=self.device,
        )

    def _capture_camera(self, *, bundle_id=0):
        return SimpleNamespace(
            fov_camera=self.cameras["front"],
            device=self.device,
            zfar=10.0,
            contrast_factor=1.0,
            n_frames_captured=bundle_id,
            save_dir_path=None,
            last_observation_bundle=None,
        )

    @staticmethod
    def _fake_renderer(*, face_size=2, fail_on_call=None):
        class Renderer:
            def __init__(self):
                self.calls = 0

            def __call__(self, mesh, cameras):
                self.calls += 1
                if self.calls == fail_on_call:
                    raise RuntimeError("synthetic face render failure")
                rgba = torch.ones(1, face_size, face_size, 4)
                depth = torch.ones(1, face_size, face_size, 1)
                return rgba, SimpleNamespace(zbuf=depth)

        return Renderer()

    def test_six_faces_share_center_and_cover_cardinal_directions(self):
        self.assertEqual(tuple(self.cameras), CUBEMAP_FACE_NAMES)
        expected_directions = {
            "front": (0.0, 0.0, 1.0),
            "back": (0.0, 0.0, -1.0),
            "left": (1.0, 0.0, 0.0),
            "right": (-1.0, 0.0, 0.0),
            "up": (0.0, 1.0, 0.0),
            "down": (0.0, -1.0, 0.0),
        }
        for name, values in expected_directions.items():
            face = self.cameras[name]
            self.assertTrue(torch.allclose(face.get_camera_center(), self.center))
            target = torch.tensor(values).reshape(1, 3)
            view = face.get_world_to_view_transform().transform_points(target)
            self.assertGreater(float(view[0, 2]), 0.0)
            self.assertAlmostEqual(float(view[0, 0]), 0.0, places=5)
            self.assertAlmostEqual(float(view[0, 1]), 0.0, places=5)

    def test_visible_union_returns_one_boolean_per_world_point(self):
        points = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        depth_maps = {
            name: torch.full((5, 5), 2.0) for name in CUBEMAP_FACE_NAMES
        }
        visible = visible_union_from_depth_maps(
            points,
            self.cameras,
            depth_maps,
            depth_tolerance=0.0,
        )
        self.assertEqual(tuple(visible.shape), (6,))
        self.assertTrue(bool(visible.all()))

    def test_world_rig_ignores_reference_yaw_for_extrinsics_and_visibility(self):
        world_from_front = build_cubemap_cameras(
            self.center,
            znear=0.1,
            zfar=10.0,
            device=self.device,
            reference_camera=self.cameras["front"],
            rig_frame=CUBEMAP_RIG_FRAME_WORLD,
        )
        world_from_left = build_cubemap_cameras(
            self.center,
            znear=0.1,
            zfar=10.0,
            device=self.device,
            reference_camera=self.cameras["left"],
            rig_frame=CUBEMAP_RIG_FRAME_WORLD,
        )
        for name in CUBEMAP_FACE_NAMES:
            self.assertTrue(
                torch.equal(world_from_front[name].R, world_from_left[name].R)
            )
            self.assertTrue(
                torch.equal(world_from_front[name].T, world_from_left[name].T)
            )

        # The default remains the legacy body-aligned behavior.
        body_from_front = build_cubemap_cameras(
            self.center,
            znear=0.1,
            zfar=10.0,
            device=self.device,
            reference_camera=self.cameras["front"],
        )
        body_from_left = build_cubemap_cameras(
            self.center,
            znear=0.1,
            zfar=10.0,
            device=self.device,
            reference_camera=self.cameras["left"],
        )
        self.assertFalse(
            torch.equal(body_from_front["front"].R, body_from_left["front"].R)
        )

        points = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        depth_maps = {
            name: torch.full((5, 5), 2.0) for name in CUBEMAP_FACE_NAMES
        }
        visible_from_front = visible_union_from_depth_maps(
            points, world_from_front, depth_maps, depth_tolerance=0.0
        )
        visible_from_left = visible_union_from_depth_maps(
            points, world_from_left, depth_maps, depth_tolerance=0.0
        )
        self.assertTrue(torch.equal(visible_from_front, visible_from_left))
        self.assertTrue(bool(visible_from_front.all()))

    def test_world_capture_persists_rig_metadata_and_face_geometry(self):
        camera = self._capture_camera()
        renderer = self._fake_renderer()
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "macarons.utility.planning_observations._build_square_renderer",
            return_value=renderer,
        ):
            bundle = capture_cubemap_observation(
                camera,
                mesh=object(),
                face_size=2,
                save_png=False,
                dir_path=temporary,
                rig_frame=CUBEMAP_RIG_FRAME_WORLD,
            )

            self.assertEqual(bundle.metadata["rig_frame"], CUBEMAP_RIG_FRAME_WORLD)
            self.assertEqual(
                bundle.metadata["extrinsics_version"],
                CUBEMAP_WORLD_EXTRINSICS_VERSION,
            )
            self.assertEqual(camera.n_frames_captured, 1)
            self.assertIs(camera.last_observation_bundle, bundle)
            for face in bundle.faces:
                payload = face.frame_dict(cpu=True)
                self.assertEqual(payload["rig_frame"], CUBEMAP_RIG_FRAME_WORLD)
                self.assertEqual(
                    payload["extrinsics_version"],
                    CUBEMAP_WORLD_EXTRINSICS_VERSION,
                )
                self.assertIn("R", payload)
                self.assertIn("T", payload)
                self.assertIn("K", payload)

            bundle_dir = Path(temporary) / "000000"
            persisted_face = torch.load(bundle_dir / "front.pt")
            persisted_bundle = torch.load(bundle_dir / "bundle.pt")
            self.assertEqual(
                persisted_face["extrinsics_version"],
                CUBEMAP_WORLD_EXTRINSICS_VERSION,
            )
            self.assertEqual(
                persisted_bundle["rig_frame"], CUBEMAP_RIG_FRAME_WORLD
            )
            self.assertTrue(all(key in persisted_face for key in ("R", "T", "K")))

    def test_failed_sixth_face_does_not_commit_bundle_counter(self):
        camera = self._capture_camera(bundle_id=7)
        sentinel = object()
        camera.last_observation_bundle = sentinel
        renderer = self._fake_renderer(fail_on_call=6)
        with mock.patch(
            "macarons.utility.planning_observations._build_square_renderer",
            return_value=renderer,
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic face render failure"):
                capture_cubemap_observation(
                    camera,
                    mesh=object(),
                    face_size=2,
                    save_frame=False,
                    save_png=False,
                    rig_frame=CUBEMAP_RIG_FRAME_WORLD,
                )
        self.assertEqual(renderer.calls, 6)
        self.assertEqual(camera.n_frames_captured, 7)
        self.assertIs(camera.last_observation_bundle, sentinel)

    def test_pixel_intrinsics_and_opencv_extrinsics_match_pytorch3d_rays(self):
        image_size = 5
        K = cubemap_pixel_intrinsics(image_size, image_size)[0]
        camera = self.cameras["front"]
        world_point = torch.tensor([[0.2, -0.1, 1.0]])
        projection = camera.get_full_projection_transform().transform_points(
            world_point
        )[0]
        expected_pixel = torch.tensor(
            [
                (1.0 - projection[0]) * (image_size - 1.0) / 2.0,
                (1.0 - projection[1]) * (image_size - 1.0) / 2.0,
            ]
        )
        axis_conversion = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
        R_opencv = (camera.R[0] @ axis_conversion).transpose(0, 1)
        T_opencv = (camera.T[0] @ axis_conversion).reshape(3, 1)
        point_opencv = R_opencv @ world_point[0].reshape(3, 1) + T_opencv
        pixel_h = K @ point_opencv.reshape(3)
        actual_pixel = pixel_h[:2] / pixel_h[2]
        self.assertTrue(torch.allclose(actual_pixel, expected_pixel, atol=1e-5))

    def test_bundle_processing_fuses_points_and_updates_proxy_once(self):
        K = cubemap_pixel_intrinsics(3, 3)
        faces = []
        for name in CUBEMAP_FACE_NAMES:
            face_camera = self.cameras[name]
            faces.append(
                PerspectiveFaceObservation(
                    name=name,
                    camera=face_camera,
                    rgb=torch.ones(1, 3, 3, 3),
                    depth_z=torch.ones(1, 3, 3, 1),
                    valid_mask=torch.ones(1, 3, 3, 1, dtype=torch.bool),
                    R=face_camera.R,
                    T=face_camera.T,
                    K=K,
                    camera_center=self.center,
                    metadata={"zfar": 10.0},
                )
            )
        bundle = ObservationBundle(
            bundle_id=0,
            center=self.center,
            faces=tuple(faces),
            capture_seconds=0.0,
        )
        proxy_points = torch.tensor(
            [
                [0.0, 0.0, 0.5],
                [0.0, 0.0, -0.5],
                [0.5, 0.0, 0.0],
                [-0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, -0.5, 0.0],
            ]
        )
        result = process_cubemap_observation(
            bundle=bundle,
            proxy_points=proxy_points,
            gathering_factor=1.0,
            sensor_range=2.0,
            voxel_size=1e-3,
            device=self.device,
        )
        stats = result["bundle_stats"]
        self.assertEqual(stats["face_count"], 6)
        self.assertEqual(stats["raw_point_count"], 54)
        self.assertLess(stats["unique_point_count"], stats["raw_point_count"])
        self.assertEqual(int(result["fov_proxy_mask"].sum()), 6)
        self.assertEqual(tuple(result["sgn_dists"].shape), (6, 1))
        self.assertEqual(tuple(result["proxy_face_owner"].shape), (6,))

        class FakeProxyScene:
            def __init__(self):
                self.calls = []

            def get_proxy_indices_from_mask(self, mask):
                self.calls.append("indices")
                return torch.where(mask)[0]

            def fill_cells(self, points, features):
                self.calls.append("fill")

            def update_proxy_view_states(self, *args, **kwargs):
                self.calls.append("view")

            def update_proxy_supervision_occ(self, *args, **kwargs):
                self.calls.append("supervision")

            def update_proxy_out_of_field(self, *args, **kwargs):
                self.calls.append("out_of_field")

        proxy_scene = FakeProxyScene()
        self.assertTrue(
            update_proxy_state(
                camera=object(),
                proxy_scene=proxy_scene,
                frame_data=result,
                carving_tolerance=0.1,
            )
        )
        self.assertEqual(
            proxy_scene.calls,
            ["indices", "fill", "view", "supervision", "out_of_field"],
        )

        radial_result = process_cubemap_observation(
            bundle=bundle,
            proxy_points=proxy_points,
            gathering_factor=1.0,
            sensor_range=1.1,
            voxel_size=1e-3,
            device=self.device,
        )
        self.assertEqual(radial_result["bundle_stats"]["raw_point_count"], 6)

    def test_proxy_owner_prefers_valid_depth_at_a_face_seam(self):
        K = cubemap_pixel_intrinsics(3, 3)
        faces = []
        for name in CUBEMAP_FACE_NAMES:
            face_camera = self.cameras[name]
            valid = torch.zeros(1, 3, 3, 1, dtype=torch.bool)
            if name == "left":
                valid[:] = True
            faces.append(
                PerspectiveFaceObservation(
                    name=name,
                    camera=face_camera,
                    rgb=torch.ones(1, 3, 3, 3),
                    depth_z=torch.ones(1, 3, 3, 1),
                    valid_mask=valid,
                    R=face_camera.R,
                    T=face_camera.T,
                    K=K,
                    camera_center=self.center,
                    metadata={"zfar": 10.0},
                )
            )
        result = process_cubemap_observation(
            bundle=ObservationBundle(
                bundle_id=0,
                center=self.center,
                faces=tuple(faces),
                capture_seconds=0.0,
            ),
            proxy_points=torch.tensor([[0.5, 0.0, 0.5]]),
            gathering_factor=1.0,
            sensor_range=2.0,
            voxel_size=1e-3,
            device=self.device,
        )
        self.assertEqual(int(result["proxy_face_owner"][0]), 2)


if __name__ == "__main__":
    unittest.main()
