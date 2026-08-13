"""Shared planning-depth wiring used by both evaluation entry points.

The helpers in this module keep provider selection, confidence masking, proxy
updates, and coverage accounting identical between the SCONE and MAGICIAN
planners.  They intentionally avoid importing PyTorch so configuration errors
can fail before model, dataset, or renderer setup.
"""

from typing import Any, Mapping, Sequence

from .depth_sources import DepthObservation, create_depth_provider


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
) -> Mapping[str, Any]:
    """Acquire one provider frame and derive geometry from its trusted mask."""

    depth_frame = depth_provider.get_frame(DepthObservation(camera=camera, device=device))
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

    return {
        "part_pc": part_pc,
        "part_pc_features": part_pc_features,
        "fov_proxy_points": fov_proxy_points,
        "fov_proxy_mask": fov_proxy_mask,
        "sgn_dists": signed_distances,
        "X_cam": camera_center,
        "planning_mask": planning_mask,
        "depth_frame": depth_frame,
    }


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
