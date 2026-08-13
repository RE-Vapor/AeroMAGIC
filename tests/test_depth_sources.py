import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from macarons.utility.depth_sources import (
    DepthFrame,
    DepthObservation,
    DepthProvider,
    GTDepthProvider,
    create_depth_provider,
    register_depth_provider,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class _SentinelProvider(DepthProvider):
    def __init__(self, source):
        self.source = source

    def get_frame(self, observation):
        raise AssertionError("selector tests must not request a frame")


class _FactorySpy:
    def __init__(self, source):
        self.source = source
        self.calls = []

    def __call__(self, *, config, device):
        self.calls.append((config, device))
        return _SentinelProvider(self.source)


class DepthProviderSelectorTests(unittest.TestCase):
    def setUp(self):
        self.gt_factory = _FactorySpy("GT")
        self.da3_factory = _FactorySpy("DA3")
        self.registry = {"GT": self.gt_factory, "DA3": self.da3_factory}

    def test_true_always_selects_gt_without_constructing_da3(self):
        for config in (
            {"use_perfect_depth_map": True},
            {"use_perfect_depth_map": True, "kind_depth_map": "DA3"},
            {"use_perfect_depth_map": True, "kind_depth_map": object()},
        ):
            provider = create_depth_provider(config, device="cuda:1", registry=self.registry)
            self.assertEqual(provider.source, "GT")

        self.assertEqual(len(self.gt_factory.calls), 3)
        self.assertEqual(self.da3_factory.calls, [])

    def test_false_normalizes_and_constructs_only_registered_backend(self):
        for kind in ("da3", "DA3", " DA3 "):
            provider = create_depth_provider(
                {"use_perfect_depth_map": False, "kind_depth_map": kind},
                device="cuda:3",
                registry=self.registry,
            )
            self.assertEqual(provider.source, "DA3")

        self.assertEqual(len(self.da3_factory.calls), 3)
        self.assertEqual(self.gt_factory.calls, [])

    def test_false_rejects_missing_empty_unregistered_and_gt_without_fallback(self):
        invalid_configs = (
            {"use_perfect_depth_map": False},
            {"use_perfect_depth_map": False, "kind_depth_map": None},
            {"use_perfect_depth_map": False, "kind_depth_map": ""},
            {"use_perfect_depth_map": False, "kind_depth_map": "GT"},
            {"use_perfect_depth_map": False, "kind_depth_map": "NONE"},
            {"use_perfect_depth_map": False, "kind_depth_map": "DA2"},
        )

        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "kind_depth_map"):
                create_depth_provider(config, registry=self.registry)

        self.assertEqual(self.gt_factory.calls, [])
        self.assertEqual(self.da3_factory.calls, [])

    def test_boolean_field_is_required_and_strict(self):
        with self.assertRaisesRegex(ValueError, "Missing required config field"):
            create_depth_provider({"kind_depth_map": "DA3"}, registry=self.registry)
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            create_depth_provider(
                {"use_perfect_depth_map": 1, "kind_depth_map": "DA3"},
                registry=self.registry,
            )

    def test_registry_normalizes_names_and_rejects_duplicates(self):
        registry = {}
        register_depth_provider(" da3 ", self.da3_factory, registry=registry)
        self.assertIn("DA3", registry)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_depth_provider("DA3", self.da3_factory, registry=registry)


class _Camera:
    def __init__(self, save_dir_path):
        self.save_dir_path = save_dir_path
        self.n_frames_captured = 1


def _legacy_perfect_depth(frame):
    mask = frame["mask"].astype(bool)
    return np.clip(frame["zbuf"], 0.5, 750.0), mask, mask, frame["R"], frame["T"]


def _fake_point_cloud_and_coverage(rgb, depth, valid_mask):
    valid = np.logical_and(valid_mask, valid_mask)[0, ..., 0]
    ys, xs = np.nonzero(valid)
    z = depth[0, ..., 0][valid]
    points = np.stack((xs.astype(np.float32), ys.astype(np.float32), z), axis=1)
    colors = rgb[0][valid]
    coverage = len({tuple(point) for point in points.tolist()})
    return points, colors, coverage


class GTDepthProviderParityTests(unittest.TestCase):
    def _frame(self, offset=0.0):
        return {
            "rgb": np.array(
                [[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                  [[0.0, 0.0, 1.0], [0.5, 0.5, 0.5]]]],
                dtype=np.float32,
            ),
            "zbuf": np.array([[[[0.1 + offset], [2.0 + offset]],
                                  [[800.0], [4.0 + offset]]]], dtype=np.float32),
            "mask": np.array([[[[1], [0]], [[1], [1]]]], dtype=np.int64),
            "R": np.eye(3, dtype=np.float32)[None],
            "T": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        }

    def _provider(self, frames):
        def load_frame(path, device):
            return frames[int(Path(path).stem)]

        return GTDepthProvider(
            device="cpu",
            frame_loader=load_frame,
            clamp_depth=lambda value, minimum, maximum: np.clip(value, minimum, maximum),
            mask_to_bool=lambda value: value.astype(bool),
        )

    def test_depth_mask_pose_and_point_cloud_match_legacy_helpers(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "0.pt").touch()
            frame = self._frame()
            camera = _Camera(directory)
            result = self._provider({0: frame}).get_frame(DepthObservation(camera, "cpu"))
            legacy_depth, legacy_mask, legacy_error, legacy_R, legacy_T = _legacy_perfect_depth(frame)

            self.assertIsInstance(result, DepthFrame)
            np.testing.assert_array_equal(result.depth_z, legacy_depth)
            np.testing.assert_array_equal(result.valid_mask, legacy_mask)
            np.testing.assert_array_equal(result.error_mask, legacy_error)
            np.testing.assert_array_equal(result.R, legacy_R)
            np.testing.assert_array_equal(result.T, legacy_T)
            self.assertEqual(result.source, "GT")
            self.assertEqual(result.frame_id, 0)

            provider_pc = _fake_point_cloud_and_coverage(
                result.rgb, result.depth_z, result.valid_mask & result.error_mask
            )
            legacy_pc = _fake_point_cloud_and_coverage(
                frame["rgb"], legacy_depth, legacy_mask & legacy_error
            )
            np.testing.assert_array_equal(provider_pc[0], legacy_pc[0])
            np.testing.assert_array_equal(provider_pc[1], legacy_pc[1])
            self.assertEqual(provider_pc[2], legacy_pc[2])

    def test_three_frame_short_trajectory_coverage_matches_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = {index: self._frame(offset=index) for index in range(3)}
            for index in frames:
                Path(directory, f"{index}.pt").touch()

            camera = _Camera(directory)
            provider = self._provider(frames)
            provider_coverages = []
            legacy_coverages = []
            provider_points = []
            legacy_points = []
            for index in range(3):
                camera.n_frames_captured = index + 1
                result = provider.get_frame(DepthObservation(camera, "cpu"))
                provider_pc = _fake_point_cloud_and_coverage(
                    result.rgb, result.depth_z, result.valid_mask & result.error_mask
                )
                legacy_depth, legacy_mask, legacy_error, _, _ = _legacy_perfect_depth(frames[index])
                legacy_pc = _fake_point_cloud_and_coverage(
                    frames[index]["rgb"], legacy_depth, legacy_mask & legacy_error
                )
                provider_points.extend(provider_pc[0].tolist())
                legacy_points.extend(legacy_pc[0].tolist())
                provider_coverages.append(len({tuple(point) for point in provider_points}))
                legacy_coverages.append(len({tuple(point) for point in legacy_points}))

            self.assertEqual(provider_coverages, legacy_coverages)
            self.assertEqual(provider_coverages, [3, 5, 7])


class PlanningEntryPointRegressionTests(unittest.TestCase):
    def test_both_planning_testers_consume_the_shared_provider(self):
        for relative_path in (
            "macarons/testers/magician_planning.py",
            "macarons/testers/scene.py",
        ):
            source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
            with self.subTest(relative_path=relative_path):
                self.assertIn("create_depth_provider(", source)
                self.assertIn("depth_provider.get_frame(DepthObservation(", source)
                self.assertNotIn("load_current_frame_perfect_depth", source)
                self.assertNotIn("apply_perfect_depth_simple", source)

    def test_default_test_config_remains_gt(self):
        config = json.loads(
            (REPO_ROOT / "configs/test/test_in_default_scenes_config.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIs(config["use_perfect_depth_map"], True)
        self.assertEqual(config["kind_depth_map"], "DA3")
        self.assertIsInstance(create_depth_provider(config, device="cpu"), GTDepthProvider)

    def test_default_registry_has_no_implicit_da3_or_gt_fallback(self):
        with self.assertRaisesRegex(ValueError, r"supported: \(none registered\)"):
            create_depth_provider(
                {"use_perfect_depth_map": False, "kind_depth_map": "DA3"},
                device="cpu",
            )

    def test_training_depth_switch_is_not_rewired_to_planning_provider(self):
        source = (REPO_ROOT / "macarons/trainers/train_macarons.py").read_text(encoding="utf-8")
        self.assertIn("params.use_perfect_depth", source)
        self.assertNotIn("depth_sources", source)


if __name__ == "__main__":
    unittest.main()
