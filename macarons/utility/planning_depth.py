"""Shared planning-depth wiring used by both evaluation entry points.

The helpers in this module keep provider selection, confidence masking, proxy
updates, and coverage accounting identical between the SCONE and MAGICIAN
planners.  They intentionally avoid importing PyTorch so configuration errors
can fail before model, dataset, or renderer setup.
"""

import random
import time
from typing import Any, Callable, Mapping, Optional, Sequence

from .depth_sources import DepthObservation, create_depth_provider


DEFAULT_SCENE_TEXTURE_ATLAS_SIZE = 32


def _config_dict(config: Any) -> dict:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    try:
        return dict(vars(config))
    except TypeError as error:
        raise TypeError("Depth config must be a mapping or an object with attributes.") from error


def _config_value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def apply_planning_validation_limits(params: Any, config: Any) -> Optional[int]:
    """Apply explicit short-run limits shared by both planning entry points.

    Production configs omit these keys and retain the training configuration.
    The dedicated real-mesh validation config uses them to exercise the same
    entry points with one three-view trajectory instead of a full benchmark.
    """

    overrides = {
        "validation_n_poses_in_trajectory": ("n_poses_in_trajectory", 0),
        "validation_n_gt_surface_points": ("n_gt_surface_points", 1),
        "validation_n_proxy_points": ("n_proxy_points", 1),
    }
    for config_name, (param_name, minimum) in overrides.items():
        value = _config_value(config, config_name, None)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{config_name} must be an integer >= {minimum}.")
        setattr(params, param_name, value)

    max_start_positions = _config_value(config, "validation_max_start_positions", None)
    if max_start_positions is not None:
        if (
            isinstance(max_start_positions, bool)
            or not isinstance(max_start_positions, int)
            or max_start_positions < 1
        ):
            raise ValueError("validation_max_start_positions must be an integer >= 1.")

    memory_dir_name = _config_value(config, "validation_memory_dir_name", None)
    if memory_dir_name is not None:
        if (
            not isinstance(memory_dir_name, str)
            or not memory_dir_name
            or memory_dir_name in {".", ".."}
            or "/" in memory_dir_name
            or "\\" in memory_dir_name
        ):
            raise ValueError("validation_memory_dir_name must be a non-empty directory name.")
        params.memory_dir_name = memory_dir_name

    experiment_overrides = _config_value(config, "experiment_param_overrides", {})
    if not isinstance(experiment_overrides, Mapping):
        raise ValueError("experiment_param_overrides must be an object.")
    allowed_overrides = {
        "carving_tolerance",
        "gathering_factor",
        "planning_gathering_factor_multiplier",
        "proxy_cell_resolution",
        "score_threshold",
    }
    unknown = sorted(set(experiment_overrides) - allowed_overrides)
    if unknown:
        raise ValueError("Unsupported experiment_param_overrides: " + ", ".join(unknown))
    for name, value in experiment_overrides.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"experiment_param_overrides.{name} must be positive.")
        setattr(params, name, float(value))

    for config_name, param_name in (
        ("experiment_shared_collision_gate", "planning_shared_collision_gate"),
        (
            "experiment_normalize_coverage_by_visibility",
            "planning_normalize_coverage_by_visibility",
        ),
    ):
        value = _config_value(config, config_name, False)
        if type(value) is not bool:
            raise ValueError(f"{config_name} must be a boolean.")
        setattr(params, param_name, value)
    return max_start_positions


def validation_uses_occupied_pose(config: Any) -> bool:
    """Return the explicit dataset occupancy policy, defaulting to legacy use."""

    value = _config_value(config, "validation_use_occupied_pose", True)
    if type(value) is not bool:
        raise ValueError("validation_use_occupied_pose must be a boolean.")
    return value


def scene_texture_atlas_size(config: Any) -> int:
    """Return the validated per-face texture atlas resolution for scene loading.

    The legacy value remains the default. Large textured meshes can opt into a
    smaller atlas without changing geometry, camera, collision, or planning
    parameters, and every planning entry point shares this validation.
    """

    value = _config_value(
        config, "scene_texture_atlas_size", DEFAULT_SCENE_TEXTURE_ATLAS_SIZE
    )
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("scene_texture_atlas_size must be an integer >= 1.")
    return value


def set_planning_seeds(config: Any) -> Mapping[str, int]:
    """Set all stochastic sources used by the two planning entry points."""

    numpy_seed = _config_value(config, "random_seed", 8)
    torch_seed = _config_value(config, "torch_seed", 9)
    for name, value in (("random_seed", numpy_seed), ("torch_seed", torch_seed)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")

    import numpy as np

    random.seed(numpy_seed)
    np.random.seed(numpy_seed)
    try:
        import torch

        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)
    except ImportError:
        pass
    return {"random_seed": numpy_seed, "torch_seed": torch_seed}


def path_is_blocked(
    start_point: Any,
    end_point: Any,
    intersector: Any,
    *,
    compute_collision: bool,
    intersection_fn: Callable[[Any, Any, Any], bool],
) -> bool:
    """Apply the same optional GT-mesh segment collision gate to both planners."""

    if type(compute_collision) is not bool:
        raise ValueError("compute_collision must be a boolean.")
    return bool(
        compute_collision and intersection_fn(start_point, end_point, intersector)
    )


def create_scene_depth_providers(
    config: Any,
    *,
    scene_names: Sequence[str],
    use_perfect_depth_map: bool,
    kind_depth_map: Any,
    scene_scale_factor: float,
    znear: float,
    zfar: float,
    device: Any,
) -> Mapping[str, Any]:
    """Create lazy providers and validate every requested scene up front.

    DA3 metric depth may only be converted to scene coordinates from an
    explicit per-scene calibration.  ``scene_scale_factor`` is still forwarded
    for other backends, but is never used as a DA3 unit conversion fallback.
    """

    names = tuple(scene_names)
    base = _config_dict(config)
    base.update(
        {
            "use_perfect_depth_map": use_perfect_depth_map,
            "kind_depth_map": kind_depth_map,
            "scene_scale_factor": scene_scale_factor,
            "znear": znear,
            "zfar": zfar,
        }
    )

    if use_perfect_depth_map:
        provider = create_depth_provider(base, device=device)
        return {name: provider for name in names}

    normalized_kind = kind_depth_map.strip().upper() if isinstance(kind_depth_map, str) else None
    if normalized_kind != "DA3":
        provider = create_depth_provider(base, device=device)
        return {name: provider for name in names}

    missing = object()
    calibrations = _config_value(config, "da3_scene_units_per_meter", missing)
    if calibrations is missing or not isinstance(calibrations, Mapping):
        raise ValueError(
            "DA3 requires 'da3_scene_units_per_meter' to map every requested "
            "scene name to an explicit positive calibration."
        )
    absent = [name for name in names if name not in calibrations]
    if absent:
        raise ValueError(
            "DA3 is missing scene_units_per_meter calibration for: " + ", ".join(absent)
        )

    providers = {}
    for name in names:
        scene_config = dict(base)
        scene_config.update(
            {
                "scene_name": name,
                "scene_units_per_meter": calibrations[name],
            }
        )
        providers[name] = create_depth_provider(scene_config, device=device)
    return providers


def _boolean_mask(value: Any) -> Any:
    if hasattr(value, "bool"):
        return value.bool()
    if hasattr(value, "astype"):
        return value.astype(bool)
    return value


def process_planning_depth_frame(
    *,
    camera: Any,
    depth_provider: Any,
    proxy_scene: Any,
    device: Any,
    gathering_factor: float,
    sensor_range: float,
    metrics_recorder: Any = None,
) -> Mapping[str, Any]:
    """Acquire one provider frame and derive geometry from its trusted mask."""

    if metrics_recorder is not None:
        metrics_recorder.synchronize()
    provider_started = time.perf_counter()
    depth_frame = depth_provider.get_frame(DepthObservation(camera=camera, device=device))
    if metrics_recorder is not None:
        metrics_recorder.synchronize()
    provider_seconds = time.perf_counter() - provider_started

    geometry_started = time.perf_counter()
    planning_mask = _boolean_mask(depth_frame.valid_mask & depth_frame.error_mask)
    fov_camera = camera.get_fov_camera_from_RT(R_cam=depth_frame.R, T_cam=depth_frame.T)
    camera_center = fov_camera.get_camera_center()

    part_pc, part_pc_features = camera.compute_partial_point_cloud(
        depth=depth_frame.depth_z,
        mask=planning_mask,
        images=depth_frame.rgb,
        fov_cameras=fov_camera,
        gathering_factor=gathering_factor,
        fov_range=sensor_range,
    )
    fov_proxy_points, fov_proxy_mask = camera.get_points_in_fov(
        proxy_scene.proxy_points,
        return_mask=True,
        fov_camera=fov_camera,
        fov_range=sensor_range,
    )
    signed_distances = None
    if fov_proxy_mask.any():
        signed_distances = camera.get_signed_distance_to_depth_maps(
            pts=fov_proxy_points,
            depth_maps=depth_frame.depth_z,
            mask=planning_mask,
            fov_camera=fov_camera,
        )

    result = {
        "part_pc": part_pc,
        "part_pc_features": part_pc_features,
        "fov_proxy_points": fov_proxy_points,
        "fov_proxy_mask": fov_proxy_mask,
        "sgn_dists": signed_distances,
        "X_cam": camera_center,
        "planning_mask": planning_mask,
        "depth_frame": depth_frame,
    }
    if metrics_recorder is not None:
        metrics_recorder.synchronize()
        metrics_recorder.record_frame(
            result,
            provider_seconds=provider_seconds,
            geometry_seconds=time.perf_counter() - geometry_started,
        )
    return result


def update_proxy_state(
    *,
    camera: Any,
    proxy_scene: Any,
    frame_data: Mapping[str, Any],
    carving_tolerance: float,
) -> bool:
    """Apply signed-distance evidence to proxy state; return whether it ran."""

    if not frame_data["fov_proxy_mask"].any():
        return False
    proxy_indices = proxy_scene.get_proxy_indices_from_mask(frame_data["fov_proxy_mask"])
    features = proxy_indices.reshape(-1, 1)
    proxy_scene.fill_cells(frame_data["fov_proxy_points"], features=features)
    proxy_scene.update_proxy_view_states(
        camera,
        frame_data["fov_proxy_mask"],
        signed_distances=frame_data["sgn_dists"],
        distance_to_surface=None,
        X_cam=frame_data["X_cam"],
    )
    proxy_scene.update_proxy_supervision_occ(
        frame_data["fov_proxy_mask"],
        frame_data["sgn_dists"],
        tol=carving_tolerance,
    )
    proxy_scene.update_proxy_out_of_field(frame_data["fov_proxy_mask"])
    return True


def compute_planning_coverage(
    *,
    gt_scene: Any,
    covered_scene: Any,
    surface_epsilon: float,
    normalization: float = 1.0,
):
    """Return the native coverage result and its scalar normalized value."""

    if normalization <= 0:
        raise ValueError("Coverage normalization must be positive.")
    raw = gt_scene.scene_coverage(covered_scene, surface_epsilon=surface_epsilon)
    first = raw[0]
    value = first.item() if hasattr(first, "item") else float(first)
    return raw, value / normalization
