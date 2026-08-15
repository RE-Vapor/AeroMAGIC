"""Depth Anything 3 planning adapter.

The module intentionally keeps PyTorch and Depth Anything 3 imports inside
runtime helpers.  Selecting GT depth therefore never imports or constructs the
DA3 model, while all DA3 providers in one process share a single model instance
per model/revision/device tuple.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .depth_sources import DepthFrame, DepthObservation, DepthProvider


DA3_ADAPTER_VERSION = "3"
DA3_SOURCE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DA3_DEFAULT_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE"
DA3_DEFAULT_MODEL_REVISION = "8615eefb62f2db4f8d6ebaa59160086981672829"

_MODEL_INSTANCES = {}
_MODEL_LOCK = threading.Lock()


def _config_value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number.")
    return float(value)


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean.")
    return value


def _as_numpy(value: Any, *, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _single_rotation(value: Any) -> np.ndarray:
    rotation = _as_numpy(value, dtype=np.float32)
    if rotation.shape == (1, 3, 3):
        rotation = rotation[0]
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected R with shape (3, 3) or (1, 3, 3), got {rotation.shape}.")
    return rotation


def _single_translation(value: Any) -> np.ndarray:
    translation = _as_numpy(value, dtype=np.float32)
    if translation.shape == (1, 3):
        translation = translation[0]
    if translation.shape == (3, 1):
        translation = translation[:, 0]
    if translation.shape != (3,):
        raise ValueError(f"Expected T with shape (3,), (1, 3), or (3, 1), got {translation.shape}.")
    return translation


def pytorch3d_to_opencv_extrinsics(R: Any, T: Any) -> np.ndarray:
    """Convert PyTorch3D row-vector ``R/T`` to an OpenCV 4x4 world-to-camera pose."""

    rotation = _single_rotation(R).copy()
    translation = _single_translation(T).copy()
    rotation[:, :2] *= -1.0
    translation[:2] *= -1.0

    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = rotation.T
    extrinsic[:3, 3] = translation
    return extrinsic


def _first_scalar(value: Any, name: str) -> float:
    array = _as_numpy(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError(f"Camera {name} is empty.")
    return float(array[0])


def camera_intrinsics(camera: Any, height: int, width: int) -> np.ndarray:
    """Return pixel-space OpenCV intrinsics for the camera's PyTorch3D projection."""

    fov_camera = getattr(camera, "fov_camera", None)
    if fov_camera is None:
        raise ValueError("DA3 requires camera.fov_camera to compute intrinsics.")

    scale = min(height, width) / 2.0
    center_x = width / 2.0
    center_y = height / 2.0

    focal = getattr(fov_camera, "focal_length", None)
    principal = getattr(fov_camera, "principal_point", None)
    if focal is not None and principal is not None:
        focal_array = _as_numpy(focal, dtype=np.float32).reshape(-1, 2)[0]
        principal_array = _as_numpy(principal, dtype=np.float32).reshape(-1, 2)[0]
        fx = float(focal_array[0] * scale)
        fy = float(focal_array[1] * scale)
        center_x -= float(principal_array[0] * scale)
        center_y -= float(principal_array[1] * scale)
    else:
        fov_degrees = _first_scalar(getattr(fov_camera, "fov", 60.0), "fov")
        aspect_ratio = _first_scalar(
            getattr(fov_camera, "aspect_ratio", 1.0), "aspect_ratio"
        )
        if not 0.0 < fov_degrees < 180.0 or aspect_ratio <= 0.0:
            raise ValueError("Camera fov must be in (0, 180) and aspect_ratio must be positive.")
        fy_ndc = 1.0 / math.tan(math.radians(fov_degrees) / 2.0)
        fx = scale * fy_ndc / aspect_ratio
        fy = scale * fy_ndc

    return np.array(
        [[fx, 0.0, center_x], [0.0, fy, center_y], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _rgb_hwc(value: Any) -> np.ndarray:
    rgb = _as_numpy(value)
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
        rgb = np.moveaxis(rgb, 0, -1)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB with shape (H, W, 3), got {rgb.shape}.")
    return rgb


def _rgb_uint8(value: Any) -> np.ndarray:
    rgb = _rgb_hwc(value)
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
        if rgb.size and float(np.max(rgb)) <= 1.0 + 1e-6:
            rgb = rgb * 255.0
    return np.clip(np.rint(rgb), 0.0, 255.0).astype(np.uint8)


def _resize_bilinear(array: np.ndarray, output_height: int, output_width: int) -> np.ndarray:
    """Small NumPy bilinear resize for depth, confidence, and RGB arrays."""

    source = np.asarray(array, dtype=np.float32)
    if source.ndim not in (2, 3):
        raise ValueError(f"Bilinear resize expects a 2D or 3D array, got {source.shape}.")
    input_height, input_width = source.shape[:2]
    if (input_height, input_width) == (output_height, output_width):
        return source.copy()

    ys = np.linspace(0.0, max(input_height - 1, 0), output_height, dtype=np.float32)
    xs = np.linspace(0.0, max(input_width - 1, 0), output_width, dtype=np.float32)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, input_height - 1)
    x1 = np.minimum(x0 + 1, input_width - 1)
    wy = ys - y0
    wx = xs - x0

    top_left = source[y0[:, None], x0[None, :]]
    top_right = source[y0[:, None], x1[None, :]]
    bottom_left = source[y1[:, None], x0[None, :]]
    bottom_right = source[y1[:, None], x1[None, :]]
    if source.ndim == 3:
        wy = wy[:, None, None]
        wx = wx[None, :, None]
    else:
        wy = wy[:, None]
        wx = wx[None, :]
    top = top_left * (1.0 - wx) + top_right * wx
    bottom = bottom_left * (1.0 - wx) + bottom_right * wx
    return top * (1.0 - wy) + bottom * wy


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _cache_key(metadata: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(metadata).encode("utf-8")).hexdigest()


def _has_translation_baseline(extrinsics: np.ndarray, epsilon: float = 1e-4) -> bool:
    """Whether DA3's 3D Umeyama alignment has sufficient camera-center rank."""

    if len(extrinsics) < 3:
        return False
    camera_to_world = np.linalg.inv(extrinsics)
    centers = camera_to_world[:, :3, 3]
    centered = centers - np.mean(centers, axis=0, keepdims=True)
    if np.max(np.linalg.norm(centers - centers[0], axis=1)) <= epsilon:
        return False
    return bool(np.linalg.matrix_rank(centered, tol=epsilon) >= 2)


def _is_degenerate_pose_alignment_error(error: Exception) -> bool:
    """Recognize DA3/evo's recoverable estimated-pose alignment failure."""

    error_type = type(error)
    return (
        error_type.__name__ == "GeometryException"
        and error_type.__module__ == "evo.core.geometry"
        and "Degenerate covariance rank" in str(error)
    )


def _default_frame_loader(path: str, device: Any) -> Mapping[str, Any]:
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before weights_only was introduced.
        return torch.load(path, map_location=device)


def _default_tensor_factory(value: np.ndarray, device: Any, dtype: str) -> Any:
    import torch

    torch_dtype = torch.bool if dtype == "bool" else torch.float32
    return torch.as_tensor(value, dtype=torch_dtype, device=device)


def _default_model_loader(model_id: str, revision: str, device: Any) -> Any:
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(model_id, revision=revision)
    model = model.to(device)
    model.eval()
    return model


def _shared_model(
    model_id: str,
    revision: str,
    device: Any,
    loader: Callable[[str, str, Any], Any],
) -> Any:
    key = (model_id, revision, str(device), id(loader))
    with _MODEL_LOCK:
        if key not in _MODEL_INSTANCES:
            _MODEL_INSTANCES[key] = loader(model_id, revision, device)
        return _MODEL_INSTANCES[key]


def _prediction_value(prediction: Any, name: str) -> Any:
    if isinstance(prediction, Mapping):
        return prediction.get(name)
    return getattr(prediction, name, None)


def _last_map(value: Any, name: str) -> np.ndarray:
    array = _as_numpy(value, dtype=np.float32)
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return array[-1]
    if array.ndim == 4 and array.shape[-1] == 1:
        return array[-1, ..., 0]
    raise ValueError(f"DA3 {name} must have shape (N, H, W), got {array.shape}.")


class DA3DepthProvider(DepthProvider):
    """Pose-conditioned, RGB-only Depth Anything 3 provider for planning."""

    source = "DA3"

    def __init__(
        self,
        *,
        config: Any = None,
        device: Any = None,
        model_loader: Callable[[str, str, Any], Any] = _default_model_loader,
        frame_loader: Callable[[str, Any], Mapping[str, Any]] = _default_frame_loader,
        tensor_factory: Callable[[np.ndarray, Any, str], Any] = _default_tensor_factory,
    ):
        self.device = device
        self.model_id = str(_config_value(config, "da3_model_id", DA3_DEFAULT_MODEL))
        self.model_revision = str(
            _config_value(config, "da3_model_revision", DA3_DEFAULT_MODEL_REVISION)
        )
        self.source_revision = str(
            _config_value(config, "da3_source_revision", DA3_SOURCE_REVISION)
        )
        if self.source_revision != DA3_SOURCE_REVISION:
            raise ValueError(
                "da3_source_revision must match the dependency pinned in environment.yml: "
                f"{DA3_SOURCE_REVISION}."
            )
        self.window_size = _positive_int(
            _config_value(config, "da3_window_size", 8), "da3_window_size"
        )
        self.process_res = _positive_int(
            _config_value(config, "da3_process_res", 504), "da3_process_res"
        )
        self.process_res_method = str(
            _config_value(config, "da3_process_res_method", "upper_bound_resize")
        )
        self.output_height = _positive_int(
            _config_value(config, "da3_output_height", 256), "da3_output_height"
        )
        self.output_width = _positive_int(
            _config_value(config, "da3_output_width", 456), "da3_output_width"
        )
        missing = object()
        raw_scene_units_per_meter = _config_value(
            config, "scene_units_per_meter", missing
        )
        if raw_scene_units_per_meter is missing:
            raise ValueError(
                "DA3 requires an explicit positive scene_units_per_meter calibration; "
                "scene_scale_factor is not a physical-unit conversion."
            )
        self.scene_units_per_meter = _positive_float(
            raw_scene_units_per_meter, "scene_units_per_meter"
        )
        self.scene_name = str(_config_value(config, "scene_name", "<unspecified>"))
        self.znear = _positive_float(_config_value(config, "znear", 0.5), "znear")
        self.zfar = _positive_float(_config_value(config, "zfar", 750.0), "zfar")
        if self.zfar <= self.znear:
            raise ValueError("zfar must be greater than znear.")
        raw_confidence_percentile = _config_value(
            config, "da3_confidence_percentile", None
        )
        self.confidence_percentile = (
            None
            if raw_confidence_percentile is None
            else float(raw_confidence_percentile)
        )
        if self.confidence_percentile is not None and not (
            0.0 <= self.confidence_percentile <= 100.0
        ):
            raise ValueError("da3_confidence_percentile must be in [0, 100].")
        self.cache_enabled = _strict_bool(
            _config_value(config, "da3_cache_enabled", True), "da3_cache_enabled"
        )

        raw_cache_dir = _config_value(config, "da3_cache_dir", None)
        self.cache_dir = Path(raw_cache_dir) if raw_cache_dir else None
        self._model_loader = model_loader
        self._frame_loader = frame_loader
        self._tensor_factory = tensor_factory

    def _frame_path(self, camera: Any, frame_id: int) -> Path:
        save_dir = getattr(camera, "save_dir_path", None)
        if not save_dir:
            raise ValueError("DA3 requires camera.save_dir_path for captured RGB frames.")
        return Path(save_dir) / f"{frame_id}.pt"

    def _cache_root(self, camera: Any) -> Path:
        return self.cache_dir or (Path(camera.save_dir_path) / ".da3_cache")

    def _load_window(
        self, camera: Any, frame_id: int
    ) -> Tuple[Sequence[int], Sequence[Mapping[str, Any]]]:
        first_id = max(0, frame_id - self.window_size + 1)
        frame_ids = list(range(first_id, frame_id + 1))
        frames = []
        for index in frame_ids:
            path = self._frame_path(camera, index)
            if not path.exists():
                raise FileNotFoundError(f"Captured RGB frame does not exist: {path}")
            frame = self._frame_loader(str(path), "cpu")
            missing = [name for name in ("rgb", "R", "T") if name not in frame]
            if missing:
                raise KeyError(f"RGB frame {path} is missing keys: {', '.join(missing)}")
            frames.append(frame)
        return frame_ids, frames

    def _metadata(
        self,
        frame_ids: Sequence[int],
        rgbs: Sequence[np.ndarray],
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
    ) -> Mapping[str, Any]:
        return {
            "adapter_version": DA3_ADAPTER_VERSION,
            "source": self.source,
            "source_revision": self.source_revision,
            "model": {"id": self.model_id, "revision": self.model_revision},
            "preprocess": {
                "window_size": self.window_size,
                "process_res": self.process_res,
                "process_res_method": self.process_res_method,
                "output_size": [self.output_height, self.output_width],
                "confidence_percentile": self.confidence_percentile,
            },
            "camera": {
                "extrinsics": extrinsics.astype(np.float32).tolist(),
                "intrinsics": intrinsics.astype(np.float32).tolist(),
            },
            "scale": {
                "scene": self.scene_name,
                "scene_units_per_meter": self.scene_units_per_meter,
                "znear": self.znear,
                "zfar": self.zfar,
            },
            "frames": [
                {
                    "frame_id": int(index),
                    "size": [int(rgb.shape[0]), int(rgb.shape[1])],
                    "rgb_sha256": _array_digest(rgb),
                }
                for index, rgb in zip(frame_ids, rgbs)
            ],
        }

    def _load_cache(
        self, cache_path: Path, expected_metadata: Mapping[str, Any]
    ) -> Optional[Mapping[str, np.ndarray]]:
        try:
            with np.load(str(cache_path), allow_pickle=False) as cached:
                stored_metadata = json.loads(str(cached["metadata_json"].item()))
                if stored_metadata != expected_metadata:
                    return None
                return {
                    "rgb": cached["rgb"].copy(),
                    "depth_z": cached["depth_z"].copy(),
                    "valid_mask": cached["valid_mask"].copy(),
                    "error_mask": cached["error_mask"].copy(),
                    "confidence": cached["confidence"].copy(),
                }
        except (FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def _save_cache(
        self,
        cache_path: Path,
        metadata: Mapping[str, Any],
        arrays: Mapping[str, np.ndarray],
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".npz.tmp", dir=str(cache_path.parent), delete=False
            ) as handle:
                temporary_path = handle.name
                np.savez_compressed(
                    handle,
                    metadata_json=np.asarray(_canonical_json(metadata)),
                    **arrays,
                )
            os.replace(temporary_path, cache_path)
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _postprocess(
        self,
        current_rgb: np.ndarray,
        depth_metric: np.ndarray,
        confidence: Optional[np.ndarray],
    ) -> Mapping[str, np.ndarray]:
        resized_depth = _resize_bilinear(
            depth_metric, self.output_height, self.output_width
        )
        depth_scene = resized_depth * self.scene_units_per_meter
        valid = np.isfinite(depth_scene) & (depth_scene >= self.znear) & (depth_scene <= self.zfar)
        depth_z = np.nan_to_num(
            depth_scene, nan=self.zfar, posinf=self.zfar, neginf=self.znear
        )
        depth_z = np.clip(depth_z, self.znear, self.zfar).astype(np.float32)

        if confidence is None:
            confidence_map = np.ones_like(depth_z, dtype=np.float32)
            confidence_mask = valid.copy()
        else:
            confidence_map = _resize_bilinear(
                confidence, self.output_height, self.output_width
            )
            finite_confidence = np.isfinite(confidence_map)
            if self.confidence_percentile is None:
                confidence_mask = valid & finite_confidence
            else:
                threshold_values = confidence_map[valid & finite_confidence]
                threshold = (
                    float(np.percentile(threshold_values, self.confidence_percentile))
                    if threshold_values.size
                    else math.inf
                )
                confidence_mask = valid & finite_confidence & (confidence_map >= threshold)
            confidence_map = np.nan_to_num(
                confidence_map, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)

        rgb = _resize_bilinear(
            current_rgb.astype(np.float32) / 255.0,
            self.output_height,
            self.output_width,
        ).astype(np.float32)
        return {
            "rgb": rgb[None],
            "depth_z": depth_z[None, ..., None],
            "valid_mask": valid.astype(bool)[None, ..., None],
            "error_mask": confidence_mask.astype(bool)[None, ..., None],
            "confidence": confidence_map[None, ..., None],
        }

    def get_frame(self, observation: DepthObservation) -> DepthFrame:
        camera = observation.camera
        frame_id = int(getattr(camera, "n_frames_captured", 0)) - 1
        if frame_id < 0:
            raise ValueError("DA3 requires at least one captured RGB frame.")

        frame_ids, frames = self._load_window(camera, frame_id)
        rgbs = [_rgb_uint8(frame["rgb"]) for frame in frames]
        extrinsics = np.stack(
            [pytorch3d_to_opencv_extrinsics(frame["R"], frame["T"]) for frame in frames]
        )
        intrinsics = np.stack(
            [camera_intrinsics(camera, rgb.shape[0], rgb.shape[1]) for rgb in rgbs]
        )
        pose_conditioned = _has_translation_baseline(extrinsics)
        metadata = self._metadata(frame_ids, rgbs, extrinsics, intrinsics)
        metadata["camera"]["pose_conditioned"] = pose_conditioned
        model_extrinsics = extrinsics.copy()
        model_extrinsics[:, :3, 3] /= self.scene_units_per_meter
        metadata["camera"]["model_extrinsics_meters"] = model_extrinsics.tolist()
        key = _cache_key(metadata)
        cache_path = self._cache_root(camera) / f"{key}.npz"

        arrays = self._load_cache(cache_path, metadata) if self.cache_enabled else None
        cache_hit = arrays is not None
        if arrays is None:
            device = observation.device if observation.device is not None else self.device
            model = _shared_model(
                self.model_id, self.model_revision, device, self._model_loader
            )
            try:
                prediction = model.inference(
                    image=list(rgbs),
                    extrinsics=model_extrinsics if pose_conditioned else None,
                    intrinsics=intrinsics if pose_conditioned else None,
                    align_to_input_ext_scale=pose_conditioned,
                    process_res=self.process_res,
                    process_res_method=self.process_res_method,
                )
            except Exception as error:
                if (
                    not pose_conditioned
                    or not _is_degenerate_pose_alignment_error(error)
                ):
                    raise
                pose_conditioned = False
                metadata["camera"]["pose_conditioned"] = False
                metadata["camera"]["pose_conditioning_fallback"] = (
                    "degenerate_model_pose_alignment"
                )
                key = _cache_key(metadata)
                cache_path = self._cache_root(camera) / f"{key}.npz"
                prediction = model.inference(
                    image=list(rgbs),
                    extrinsics=None,
                    intrinsics=None,
                    align_to_input_ext_scale=False,
                    process_res=self.process_res,
                    process_res_method=self.process_res_method,
                )
            raw_depth = _prediction_value(prediction, "depth")
            if raw_depth is None:
                raise ValueError("DA3 prediction is missing depth.")
            raw_confidence = _prediction_value(prediction, "conf")
            arrays = self._postprocess(
                rgbs[-1],
                _last_map(raw_depth, "depth"),
                _last_map(raw_confidence, "confidence")
                if raw_confidence is not None
                else None,
            )
            if self.cache_enabled:
                self._save_cache(cache_path, metadata, arrays)

        device = observation.device if observation.device is not None else self.device
        current_R = _single_rotation(frames[-1]["R"])[None]
        current_T = _single_translation(frames[-1]["T"])[None]
        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "cache_key": key,
                "cache_enabled": self.cache_enabled,
                "cache_hit": cache_hit,
            }
        )
        return DepthFrame(
            rgb=self._tensor_factory(arrays["rgb"], device, "float32"),
            depth_z=self._tensor_factory(arrays["depth_z"], device, "float32"),
            valid_mask=self._tensor_factory(arrays["valid_mask"], device, "bool"),
            error_mask=self._tensor_factory(arrays["error_mask"], device, "bool"),
            R=self._tensor_factory(current_R, device, "float32"),
            T=self._tensor_factory(current_T, device, "float32"),
            confidence=self._tensor_factory(arrays["confidence"], device, "float32"),
            source=self.source,
            frame_id=frame_id,
            cache_metadata=output_metadata,
        )
