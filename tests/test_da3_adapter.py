import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from macarons.utility.da3_adapter import (
    DA3_ADAPTER_VERSION,
    DA3_DEFAULT_MODEL_REVISION,
    DA3_SOURCE_REVISION,
    DA3DepthProvider,
    _MODEL_INSTANCES,
    _has_translation_baseline,
    camera_intrinsics,
    pytorch3d_to_opencv_extrinsics,
)
from macarons.utility.depth_sources import DepthFrame, DepthObservation


REPO_ROOT = Path(__file__).resolve().parents[1]


def _numpy_tensor(value, device, dtype):
    return np.asarray(value, dtype=bool if dtype == "bool" else np.float32).copy()


class _FovCamera:
    fov = np.array([60.0], dtype=np.float32)
    aspect_ratio = np.array([1.0], dtype=np.float32)


class _Camera:
    def __init__(self, save_dir_path, n_frames_captured):
        self.save_dir_path = save_dir_path
        self.n_frames_captured = n_frames_captured
        self.fov_camera = _FovCamera()


class _FakeModel:
    def __init__(self, prediction_factory=None):
        self.calls = []
        self.prediction_factory = prediction_factory or self._default_prediction

    @staticmethod
    def _default_prediction(count):
        return SimpleNamespace(
            depth=np.full((count, 2, 3), 2.0, dtype=np.float32),
            conf=np.broadcast_to(
                np.linspace(0.0, 1.0, 6, dtype=np.float32).reshape(2, 3),
                (count, 2, 3),
            ).copy(),
        )

    def inference(self, **kwargs):
        self.calls.append(kwargs)
        return self.prediction_factory(len(kwargs["image"]))


class GeometryException(RuntimeError):
    pass


GeometryException.__module__ = "evo.core.geometry"


class _DegeneratePoseAlignmentModel(_FakeModel):
    def inference(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["extrinsics"] is not None:
            raise GeometryException(
                "Degenerate covariance rank, Umeyama alignment is not possible"
            )
        return self.prediction_factory(len(kwargs["image"]))


class _ModelLoader:
    def __init__(self, model):
        self.model = model
        self.calls = []

    def __call__(self, model_id, revision, device):
        self.calls.append((model_id, revision, device))
        return self.model


def _frame(index, *, include_gt=False):
    rgb = np.zeros((1, 2, 3, 3), dtype=np.float32)
    rgb[..., 0] = index / 10.0
    frame = {
        "rgb": rgb,
        "R": np.eye(3, dtype=np.float32)[None],
        "T": np.array(
            [[float(index), 2.0 + (1.0 if index == 2 else 0.0), 3.0]],
            dtype=np.float32,
        ),
    }
    if include_gt:
        frame["zbuf"] = np.full((1, 2, 3, 1), index + 100.0, dtype=np.float32)
        frame["mask"] = np.zeros((1, 2, 3, 1), dtype=bool)
    return frame


class DA3CoordinateTests(unittest.TestCase):
    def test_submillimeter_rotation_only_jitter_is_not_a_translation_baseline(self):
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
        centers = np.array(
            [
                [-73.333336, 35.0, 6.6666718],
                [-73.333336, 35.000004, 6.6666713],
                [-73.333336, 35.000004, 6.6666727],
            ],
            dtype=np.float32,
        )
        extrinsics[:, :3, 3] = -centers
        self.assertFalse(_has_translation_baseline(extrinsics))

        extrinsics[1, :3, 3] = -np.array([-72.0, 35.0, 6.0])
        extrinsics[2, :3, 3] = -np.array([-73.0, 36.0, 6.0])
        self.assertTrue(_has_translation_baseline(extrinsics))

    def test_dependency_and_model_revisions_are_immutable_and_in_sync(self):
        environment = (REPO_ROOT / "environment.yml").read_text(encoding="utf-8")
        config = json.loads(
            (REPO_ROOT / "configs/test/test_in_default_scenes_config.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn(f"Depth-Anything-3.git@{DA3_SOURCE_REVISION}", environment)
        self.assertEqual(config["da3_source_revision"], DA3_SOURCE_REVISION)
        self.assertEqual(config["da3_model_revision"], DA3_DEFAULT_MODEL_REVISION)

    def test_pytorch3d_extrinsics_convert_to_opencv_without_changing_camera_center(self):
        R = np.eye(3, dtype=np.float32)
        T = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        extrinsic = pytorch3d_to_opencv_extrinsics(R, T)

        np.testing.assert_array_equal(
            extrinsic[:3, :3], np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
        )
        np.testing.assert_array_equal(extrinsic[:3, 3], [-1.0, -2.0, 3.0])
        p3d_center = -T @ R.T
        opencv_center = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
        np.testing.assert_allclose(opencv_center, p3d_center)

    def test_fov_intrinsics_match_pytorch3d_non_square_ndc_scaling(self):
        camera = _Camera("unused", 1)
        K = camera_intrinsics(camera, 256, 456)
        expected_focal = 128.0 / math.tan(math.radians(30.0))
        np.testing.assert_allclose(K[0, 0], expected_focal, rtol=1e-6)
        np.testing.assert_allclose(K[1, 1], expected_focal, rtol=1e-6)
        np.testing.assert_array_equal(K[:2, 2], [228.0, 128.0])


class DA3ProviderTests(unittest.TestCase):
    def setUp(self):
        _MODEL_INSTANCES.clear()

    def tearDown(self):
        _MODEL_INSTANCES.clear()

    def _provider(self, directory, frames, model, **overrides):
        loader = _ModelLoader(model)
        config = {
            "da3_cache_dir": str(Path(directory, "cache")),
            "da3_window_size": 2,
            "scene_name": "synthetic",
            "scene_units_per_meter": 10.0,
            "znear": 0.5,
            "zfar": 50.0,
        }
        config.update(overrides)

        def load_frame(path, device):
            return frames[int(Path(path).stem)]

        provider = DA3DepthProvider(
            config=config,
            device="cuda:0",
            model_loader=loader,
            frame_loader=load_frame,
            tensor_factory=_numpy_tensor,
        )
        return provider, loader

    @staticmethod
    def _touch_frames(directory, count):
        for index in range(count):
            Path(directory, f"{index}.pt").touch()

    def test_rgb_only_sliding_window_pose_conditioning_and_depth_frame_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            self._touch_frames(directory, 3)
            frames = {index: _frame(index) for index in range(3)}
            model = _FakeModel()
            provider, loader = self._provider(
                directory, frames, model, da3_window_size=3
            )

            result = provider.get_frame(DepthObservation(_Camera(directory, 3), "cuda:0"))

            self.assertIsInstance(result, DepthFrame)
            self.assertEqual(result.source, "DA3")
            self.assertEqual(result.frame_id, 2)
            self.assertEqual(result.rgb.shape, (1, 256, 456, 3))
            self.assertEqual(result.depth_z.shape, (1, 256, 456, 1))
            self.assertEqual(result.valid_mask.shape, (1, 256, 456, 1))
            self.assertEqual(result.error_mask.shape, (1, 256, 456, 1))
            self.assertEqual(result.confidence.shape, (1, 256, 456, 1))
            np.testing.assert_allclose(result.depth_z, 20.0)
            np.testing.assert_array_equal(result.R, frames[2]["R"])
            np.testing.assert_array_equal(result.T, frames[2]["T"])

            self.assertEqual(len(loader.calls), 1)
            self.assertEqual(len(model.calls), 1)
            call = model.calls[0]
            self.assertEqual(len(call["image"]), 3)
            np.testing.assert_array_equal(
                call["image"][0], np.rint(frames[0]["rgb"][0] * 255).astype(np.uint8)
            )
            self.assertEqual(call["extrinsics"].shape, (3, 4, 4))
            self.assertEqual(call["intrinsics"].shape, (3, 3, 3))
            self.assertIs(call["align_to_input_ext_scale"], True)
            expected_model_extrinsics = np.asarray(
                result.cache_metadata["camera"]["extrinsics"]
            )
            expected_model_extrinsics[:, :3, 3] /= 10.0
            np.testing.assert_allclose(
                call["extrinsics"][:, :3, 3],
                expected_model_extrinsics[:, :3, 3],
            )
            self.assertEqual(call["process_res"], 504)
            self.assertEqual(call["process_res_method"], "upper_bound_resize")

            metadata = result.cache_metadata
            self.assertEqual(metadata["adapter_version"], DA3_ADAPTER_VERSION)
            for name in ("source", "model", "preprocess", "camera", "scale"):
                self.assertIn(name, metadata)

    def test_two_or_collinear_frames_disable_degenerate_pose_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            self._touch_frames(directory, 3)
            frames = {index: _frame(index) for index in range(3)}
            model = _FakeModel()
            two_frame, _ = self._provider(directory, frames, model)
            two_frame.get_frame(DepthObservation(_Camera(directory, 2)))
            self.assertIsNone(model.calls[-1]["extrinsics"])
            self.assertIs(model.calls[-1]["align_to_input_ext_scale"], False)

            frames[2]["T"] = np.array([[2.0, 2.0, 3.0]], dtype=np.float32)
            collinear, _ = self._provider(
                directory,
                frames,
                model,
                da3_window_size=3,
                da3_cache_dir=str(Path(directory, "collinear-cache")),
            )
            collinear.get_frame(DepthObservation(_Camera(directory, 3)))
            self.assertIsNone(model.calls[-1]["extrinsics"])
            self.assertIs(model.calls[-1]["align_to_input_ext_scale"], False)

    def test_degenerate_model_pose_alignment_retries_without_pose_conditioning(self):
        with tempfile.TemporaryDirectory() as directory:
            self._touch_frames(directory, 3)
            frames = {index: _frame(index) for index in range(3)}
            model = _DegeneratePoseAlignmentModel()
            provider, _ = self._provider(
                directory,
                frames,
                model,
                da3_window_size=3,
                da3_cache_enabled=False,
            )

            result = provider.get_frame(DepthObservation(_Camera(directory, 3)))

            self.assertEqual(len(model.calls), 2)
            self.assertIsNotNone(model.calls[0]["extrinsics"])
            self.assertIsNone(model.calls[1]["extrinsics"])
            self.assertIs(model.calls[1]["align_to_input_ext_scale"], False)
            self.assertIs(result.cache_metadata["camera"]["pose_conditioned"], False)
            self.assertEqual(
                result.cache_metadata["camera"]["pose_conditioning_fallback"],
                "degenerate_model_pose_alignment",
            )

    def test_metric_scale_validity_and_confidence_masks(self):
        def prediction(count):
            depth = np.array(
                [[0.01, 1.0, 100.0], [np.nan, 2.0, 3.0]], dtype=np.float32
            )
            confidence = np.array(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
            )
            return SimpleNamespace(
                depth=np.broadcast_to(depth, (count, 2, 3)).copy(),
                conf=np.broadcast_to(confidence, (count, 2, 3)).copy(),
            )

        with tempfile.TemporaryDirectory() as directory:
            self._touch_frames(directory, 1)
            frames = {0: _frame(0)}
            model = _FakeModel(prediction)
            provider, _ = self._provider(
                directory,
                frames,
                model,
                da3_output_height=2,
                da3_output_width=3,
                da3_confidence_percentile=50.0,
            )
            result = provider.get_frame(DepthObservation(_Camera(directory, 1)))

            self.assertIsNone(model.calls[0]["extrinsics"])
            self.assertIsNone(model.calls[0]["intrinsics"])

            expected_valid = np.array(
                [[False, True, False], [False, True, True]], dtype=bool
            )[None, ..., None]
            expected_confidence = np.array(
                [[False, False, False], [False, True, True]], dtype=bool
            )[None, ..., None]
            np.testing.assert_array_equal(result.valid_mask, expected_valid)
            np.testing.assert_array_equal(result.error_mask, expected_confidence)
            np.testing.assert_allclose(
                result.depth_z[0, ..., 0],
                [[0.5, 10.0, 50.0], [50.0, 20.0, 30.0]],
            )

    def test_gt_zbuf_and_renderer_mask_do_not_affect_output_or_cache_key(self):
        with tempfile.TemporaryDirectory() as directory:
            self._touch_frames(directory, 1)
            frames = {0: _frame(0, include_gt=True)}
            model = _FakeModel()
            provider, _ = self._provider(
                directory, frames, model, da3_output_height=2, da3_output_width=3
            )
            camera = _Camera(directory, 1)
            first = provider.get_frame(DepthObservation(camera))

            frames[0]["zbuf"][:] = -9999.0
            frames[0]["mask"][:] = True
            second = provider.get_frame(DepthObservation(camera))

            self.assertEqual(first.cache_metadata["cache_key"], second.cache_metadata["cache_key"])
            self.assertIs(second.cache_metadata["cache_hit"], True)
            self.assertEqual(len(model.calls), 1)
            np.testing.assert_array_equal(first.depth_z, second.depth_z)
            metadata_text = json.dumps(second.cache_metadata, sort_keys=True)
            self.assertNotIn("zbuf", metadata_text)
            self.assertNotIn("mask", metadata_text)

    def test_cache_metadata_mismatch_is_a_strict_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            self._touch_frames(directory, 1)
            frames = {0: _frame(0)}
            model = _FakeModel()
            provider, _ = self._provider(
                directory, frames, model, da3_output_height=2, da3_output_width=3
            )
            camera = _Camera(directory, 1)
            provider.get_frame(DepthObservation(camera))

            cache_path = next(Path(directory, "cache").glob("*.npz"))
            with np.load(str(cache_path), allow_pickle=False) as cached:
                arrays = {name: cached[name].copy() for name in cached.files}
            stored = json.loads(str(arrays["metadata_json"].item()))
            stored["model"]["revision"] = "tampered"
            arrays["metadata_json"] = np.asarray(
                json.dumps(stored, sort_keys=True, separators=(",", ":"))
            )
            np.savez_compressed(str(cache_path), **arrays)

            result = provider.get_frame(DepthObservation(camera))
            self.assertIs(result.cache_metadata["cache_hit"], False)
            self.assertEqual(len(model.calls), 2)

    def test_model_is_lazy_and_shared_across_providers(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            self._touch_frames(first_dir, 1)
            self._touch_frames(second_dir, 1)
            model = _FakeModel()
            shared_loader = _ModelLoader(model)

            def make_provider(directory, rgb_value):
                frames = {0: _frame(rgb_value)}

                def load_frame(path, device):
                    return frames[0]

                return DA3DepthProvider(
                    config={
                        "da3_cache_dir": str(Path(directory, "cache")),
                        "da3_output_height": 2,
                        "da3_output_width": 3,
                        "scene_units_per_meter": 1.0,
                    },
                    device="cuda:0",
                    model_loader=shared_loader,
                    frame_loader=load_frame,
                    tensor_factory=_numpy_tensor,
                )

            first = make_provider(first_dir, 1)
            second = make_provider(second_dir, 2)
            self.assertEqual(shared_loader.calls, [])

            first.get_frame(DepthObservation(_Camera(first_dir, 1)))
            second.get_frame(DepthObservation(_Camera(second_dir, 1)))
            self.assertEqual(len(shared_loader.calls), 1)
            self.assertEqual(len(model.calls), 2)

    def test_cache_key_isolates_model_preprocess_camera_and_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            self._touch_frames(directory, 1)
            frames = {0: _frame(0)}
            model = _FakeModel()

            def key(**config):
                provider, _ = self._provider(
                    directory,
                    frames,
                    model,
                    da3_output_height=2,
                    da3_output_width=3,
                    **config,
                )
                return provider.get_frame(
                    DepthObservation(_Camera(directory, 1))
                ).cache_metadata["cache_key"]

            baseline = key()
            self.assertNotEqual(baseline, key(da3_model_revision="other-revision"))
            self.assertNotEqual(baseline, key(da3_process_res=392))
            self.assertNotEqual(baseline, key(scene_units_per_meter=5.0))
            frames[0]["T"] = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
            self.assertNotEqual(baseline, key())

            frames[0]["T"] = np.array([[0.0, 2.0, 3.0]], dtype=np.float32)
            changed_camera = _Camera(directory, 1)
            changed_camera.fov_camera = SimpleNamespace(
                focal_length=np.array([[2.0, 2.0]], dtype=np.float32),
                principal_point=np.zeros((1, 2), dtype=np.float32),
            )
            provider, _ = self._provider(
                directory,
                frames,
                model,
                da3_output_height=2,
                da3_output_width=3,
            )
            self.assertNotEqual(
                baseline,
                provider.get_frame(DepthObservation(changed_camera)).cache_metadata["cache_key"],
            )

    def test_cache_on_off_outputs_are_equivalent_and_off_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            self._touch_frames(directory, 1)
            frames = {0: _frame(0)}
            model = _FakeModel()
            camera = _Camera(directory, 1)
            cached, _ = self._provider(
                directory,
                frames,
                model,
                da3_output_height=2,
                da3_output_width=3,
                da3_cache_enabled=True,
            )
            uncached, _ = self._provider(
                directory,
                frames,
                model,
                da3_output_height=2,
                da3_output_width=3,
                da3_cache_enabled=False,
            )

            cached_miss = cached.get_frame(DepthObservation(camera))
            cached_hit = cached.get_frame(DepthObservation(camera))
            cache_files_before = tuple(Path(directory, "cache").glob("*.npz"))
            uncached_result = uncached.get_frame(DepthObservation(camera))
            cache_files_after = tuple(Path(directory, "cache").glob("*.npz"))

            self.assertIs(cached_miss.cache_metadata["cache_hit"], False)
            self.assertIs(cached_hit.cache_metadata["cache_hit"], True)
            self.assertIs(uncached_result.cache_metadata["cache_enabled"], False)
            self.assertIs(uncached_result.cache_metadata["cache_hit"], False)
            self.assertEqual(cache_files_before, cache_files_after)
            for field in ("rgb", "depth_z", "valid_mask", "error_mask", "confidence"):
                np.testing.assert_array_equal(
                    getattr(cached_miss, field), getattr(uncached_result, field)
                )

    def test_explicit_scene_scale_and_cache_switch_are_strict(self):
        with self.assertRaisesRegex(ValueError, "scene_units_per_meter"):
            DA3DepthProvider(config={}, device="cpu")
        with self.assertRaisesRegex(ValueError, "da3_cache_enabled"):
            DA3DepthProvider(
                config={"scene_units_per_meter": 1.0, "da3_cache_enabled": 1},
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
