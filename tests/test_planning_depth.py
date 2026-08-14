from types import SimpleNamespace
import unittest

import numpy as np

from macarons.utility.depth_sources import DepthFrame, GTDepthProvider
from macarons.utility.planning_depth import (
    apply_planning_validation_limits,
    compute_planning_coverage,
    create_scene_depth_providers,
    path_is_blocked,
    process_planning_depth_frame,
    set_planning_seeds,
    update_proxy_state,
    validation_uses_occupied_pose,
)


class PlanningValidationLimitTests(unittest.TestCase):
    def test_applies_short_run_limits_without_touching_other_params(self):
        params = SimpleNamespace(
            n_poses_in_trajectory=100,
            n_gt_surface_points=100000,
            n_proxy_points=800000,
            untouched="value",
        )
        max_starts = apply_planning_validation_limits(
            params,
            {
                "validation_n_poses_in_trajectory": 2,
                "validation_n_gt_surface_points": 50000,
                "validation_n_proxy_points": 200000,
                "validation_max_start_positions": 1,
                "validation_memory_dir_name": "real_mesh_validation",
            },
        )
        self.assertEqual(params.n_poses_in_trajectory, 2)
        self.assertEqual(params.n_gt_surface_points, 50000)
        self.assertEqual(params.n_proxy_points, 200000)
        self.assertEqual(params.memory_dir_name, "real_mesh_validation")
        self.assertEqual(params.untouched, "value")
        self.assertEqual(max_starts, 1)

    def test_absent_limits_preserve_production_values(self):
        params = SimpleNamespace(n_poses_in_trajectory=100)
        self.assertIsNone(apply_planning_validation_limits(params, {}))
        self.assertEqual(params.n_poses_in_trajectory, 100)
        self.assertFalse(params.planning_shared_collision_gate)
        self.assertFalse(params.planning_normalize_coverage_by_visibility)
        self.assertFalse(hasattr(params, "planning_gathering_factor_multiplier"))

    def test_rejects_invalid_limits(self):
        with self.assertRaisesRegex(ValueError, "validation_n_poses_in_trajectory"):
            apply_planning_validation_limits(
                SimpleNamespace(), {"validation_n_poses_in_trajectory": -1}
            )
        with self.assertRaisesRegex(ValueError, "validation_max_start_positions"):
            apply_planning_validation_limits(
                SimpleNamespace(), {"validation_max_start_positions": True}
            )
        with self.assertRaisesRegex(ValueError, "validation_memory_dir_name"):
            apply_planning_validation_limits(
                SimpleNamespace(), {"validation_memory_dir_name": "../outside"}
            )

    def test_occupied_pose_use_is_opt_out_and_strict(self):
        self.assertTrue(validation_uses_occupied_pose({}))
        self.assertFalse(
            validation_uses_occupied_pose({"validation_use_occupied_pose": False})
        )
        with self.assertRaisesRegex(ValueError, "validation_use_occupied_pose"):
            validation_uses_occupied_pose({"validation_use_occupied_pose": 0})

    def test_applies_only_allowlisted_experiment_mapping_overrides(self):
        params = SimpleNamespace(gathering_factor=0.05, carving_tolerance=10.0)
        apply_planning_validation_limits(
            params,
            {
                "experiment_param_overrides": {
                    "planning_gathering_factor_multiplier": 2.0,
                    "carving_tolerance": 5.0,
                },
                "experiment_shared_collision_gate": True,
                "experiment_normalize_coverage_by_visibility": True,
            },
        )
        self.assertEqual(params.planning_gathering_factor_multiplier, 2.0)
        self.assertEqual(params.carving_tolerance, 5.0)
        self.assertTrue(params.planning_shared_collision_gate)
        self.assertTrue(params.planning_normalize_coverage_by_visibility)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            apply_planning_validation_limits(
                SimpleNamespace(), {"experiment_param_overrides": {"n_poses_in_trajectory": 1}}
            )

    def test_fixed_seed_helper_and_shared_collision_gate(self):
        self.assertEqual(
            set_planning_seeds({"random_seed": 8, "torch_seed": 9}),
            {"random_seed": 8, "torch_seed": 9},
        )
        calls = []

        def intersects(start, end, mesh):
            calls.append((start, end, mesh))
            return True

        self.assertFalse(
            path_is_blocked(1, 2, 3, compute_collision=False, intersection_fn=intersects)
        )
        self.assertEqual(calls, [])
        self.assertTrue(
            path_is_blocked(1, 2, 3, compute_collision=True, intersection_fn=intersects)
        )
        self.assertEqual(calls, [(1, 2, 3)])


class SceneProviderTests(unittest.TestCase):
    def test_gt_does_not_require_or_read_da3_calibrations(self):
        providers = create_scene_depth_providers(
            {"da3_scene_units_per_meter": object()},
            scene_names=["a", "b"],
            use_perfect_depth_map=True,
            kind_depth_map="DA3",
            scene_scale_factor=10.0,
            znear=0.5,
            zfar=750.0,
            device="cpu",
        )
        self.assertIsInstance(providers["a"], GTDepthProvider)
        self.assertIs(providers["a"], providers["b"])

    def test_da3_requires_explicit_calibration_for_every_scene(self):
        common = dict(
            scene_names=["a", "b"],
            use_perfect_depth_map=False,
            kind_depth_map="DA3",
            scene_scale_factor=10.0,
            znear=0.5,
            zfar=750.0,
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "da3_scene_units_per_meter"):
            create_scene_depth_providers({}, **common)
        with self.assertRaisesRegex(ValueError, "b"):
            create_scene_depth_providers(
                {"da3_scene_units_per_meter": {"a": 2.0}}, **common
            )

        providers = create_scene_depth_providers(
            {"da3_scene_units_per_meter": {"a": 2.0, "b": 3.0}}, **common
        )
        self.assertEqual(providers["a"].source, "DA3")
        self.assertEqual(providers["a"].scene_units_per_meter, 2.0)
        self.assertEqual(providers["b"].scene_units_per_meter, 3.0)
        self.assertIsNot(providers["a"], providers["b"])


class _Provider:
    source = "DA3"

    def __init__(self, frame):
        self.frame = frame
        self.calls = []

    def get_frame(self, observation):
        self.calls.append(observation)
        return self.frame


class _FovCamera:
    def get_camera_center(self):
        return np.array([[1.0, 2.0, 3.0]], dtype=np.float32)


class _Camera:
    def __init__(self):
        self.fov = _FovCamera()
        self.point_cloud_kwargs = None
        self.fov_kwargs = None
        self.signed_kwargs = None

    def get_fov_camera_from_RT(self, **kwargs):
        self.pose_kwargs = kwargs
        return self.fov

    def compute_partial_point_cloud(self, **kwargs):
        self.point_cloud_kwargs = kwargs
        return np.array([[0.0, 0.0, 2.0]]), np.array([[1.0, 0.0, 0.0]])

    def get_points_in_fov(self, points, **kwargs):
        self.fov_kwargs = kwargs
        return points[:2], np.array([True, True, False])

    def get_signed_distance_to_depth_maps(self, **kwargs):
        self.signed_kwargs = kwargs
        return np.array([0.1, -0.2])


class _ProxyScene:
    def __init__(self):
        self.proxy_points = np.arange(9, dtype=np.float32).reshape(3, 3)
        self.calls = []

    def get_proxy_indices_from_mask(self, mask):
        self.calls.append(("indices", mask.copy()))
        return np.array([4, 5])

    def fill_cells(self, points, features):
        self.calls.append(("fill", points.copy(), features.copy()))

    def update_proxy_view_states(self, *args, **kwargs):
        self.calls.append(("view", args, kwargs))

    def update_proxy_supervision_occ(self, *args, **kwargs):
        self.calls.append(("occ", args, kwargs))

    def update_proxy_out_of_field(self, mask):
        self.calls.append(("out", mask.copy()))


class _GTScene:
    def __init__(self):
        self.calls = []

    def scene_coverage(self, covered_scene, surface_epsilon):
        self.calls.append((covered_scene, surface_epsilon))
        return np.array([0.75], dtype=np.float32)


class PlanningPipelineTests(unittest.TestCase):
    def test_provider_depth_point_cloud_signed_distance_proxy_and_coverage_pipeline(self):
        valid = np.array([[[[True], [True]], [[False], [True]]]])
        confidence = np.array([[[[True], [False]], [[True], [True]]]])
        frame = DepthFrame(
            rgb=np.ones((1, 2, 2, 3), dtype=np.float32),
            depth_z=np.full((1, 2, 2, 1), 2.0, dtype=np.float32),
            valid_mask=valid,
            error_mask=confidence,
            R=np.eye(3, dtype=np.float32)[None],
            T=np.zeros((1, 3), dtype=np.float32),
            source="DA3",
        )
        provider = _Provider(frame)
        camera = _Camera()
        proxy = _ProxyScene()

        frame_data = process_planning_depth_frame(
            camera=camera,
            depth_provider=provider,
            proxy_scene=proxy,
            device="cuda:1",
            gathering_factor=2.0,
            sensor_range=50.0,
        )
        expected_mask = valid & confidence
        self.assertEqual(provider.calls[0].device, "cuda:1")
        self.assertEqual(frame_data["depth_frame"].source, "DA3")
        np.testing.assert_array_equal(camera.point_cloud_kwargs["depth"], frame.depth_z)
        np.testing.assert_array_equal(camera.point_cloud_kwargs["mask"], expected_mask)
        np.testing.assert_array_equal(camera.signed_kwargs["mask"], expected_mask)
        self.assertIs(camera.point_cloud_kwargs["fov_cameras"], camera.fov)
        self.assertIs(camera.signed_kwargs["fov_camera"], camera.fov)
        np.testing.assert_array_equal(frame_data["part_pc"], [[0.0, 0.0, 2.0]])
        np.testing.assert_array_equal(frame_data["sgn_dists"], [0.1, -0.2])

        self.assertTrue(
            update_proxy_state(
                camera=camera,
                proxy_scene=proxy,
                frame_data=frame_data,
                carving_tolerance=0.05,
            )
        )
        self.assertEqual([call[0] for call in proxy.calls], ["indices", "fill", "view", "occ", "out"])
        self.assertEqual(proxy.calls[3][2]["tol"], 0.05)

        gt_scene = _GTScene()
        covered_scene = SimpleNamespace()
        raw, normalized = compute_planning_coverage(
            gt_scene=gt_scene,
            covered_scene=covered_scene,
            surface_epsilon=0.1,
            normalization=0.5,
        )
        np.testing.assert_array_equal(raw, [0.75])
        self.assertEqual(normalized, 1.5)
        self.assertEqual(gt_scene.calls, [(covered_scene, 0.1)])


if __name__ == "__main__":
    unittest.main()
