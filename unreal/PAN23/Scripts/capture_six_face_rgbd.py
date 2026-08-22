"""Capture one atomic raw RGB + UE depth six-face artifact for PAN-15."""

from __future__ import annotations

import array
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import unreal


BUNDLE_DIR = Path(os.environ["PAN15_RAW_BUNDLE_DIR"]).resolve()
CONFIG_PATH = Path(os.environ["PAN15_CAPTURE_CONFIG"]).resolve()
REQUEST_PATH = Path(os.environ["PAN15_REQUEST_PATH"]).resolve()
PROJECT_COMMIT = os.environ["PAN15_PROJECT_COMMIT"]
GPU_INDEX = int(os.environ["PAN15_GPU_INDEX"])
FACE_NAMES = ("front", "back", "left", "right", "up", "down")
PAN29_REPLAY_RIG_SCHEMA = "pan29.ue5-replay-rig.v1"
PAN31_LIGHTING_SCHEMA = "pan31.ue5-lighting-ablation.v1"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vec(value):
    return [float(value.x), float(value.y), float(value.z)]


def rot(value):
    return [float(value.pitch), float(value.yaw), float(value.roll)]


def ue_rotator(pitch_yaw_roll):
    pitch, yaw, roll = pitch_yaw_roll
    return unreal.Rotator(roll, pitch, yaw)


def contract_pose(capture, world_to_meters):
    """Map left-handed UE world to a right-handed world by flipping UE Y."""

    def transform_world_axis(value):
        return [value.x, -value.y, value.z]
    camera_right = transform_world_axis(
        unreal.MathLibrary.get_right_vector(capture.get_actor_rotation())
    )
    camera_up = transform_world_axis(
        unreal.MathLibrary.get_up_vector(capture.get_actor_rotation())
    )
    camera_forward = transform_world_axis(
        unreal.MathLibrary.get_forward_vector(capture.get_actor_rotation())
    )
    location = capture.get_actor_location()
    position_m = [
        location.x / world_to_meters,
        -location.y / world_to_meters,
        location.z / world_to_meters,
    ]
    return [
        [camera_right[0], -camera_up[0], camera_forward[0], position_m[0]],
        [camera_right[1], -camera_up[1], camera_forward[1], position_m[1]],
        [camera_right[2], -camera_up[2], camera_forward[2], position_m[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def pioneer_pose(contract_transform, scene_units_per_meter):
    """Represent the UE contract pose in the PIONEER XYZ world frame."""

    # This is the same UE_CONTRACT_TO_MAGICIAN transform used by the PAN-21
    # adapter: pioneer [x,y,z] = [contract_x,contract_z,-contract_y].
    rows = contract_transform
    rotation = [rows[0][:3], rows[2][:3], [-value for value in rows[1][:3]]]
    translation = [
        rows[0][3] * scene_units_per_meter,
        rows[2][3] * scene_units_per_meter,
        -rows[1][3] * scene_units_per_meter,
    ]
    return [
        [*rotation[0], translation[0]],
        [*rotation[1], translation[1]],
        [*rotation[2], translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def maximum_matrix_error(actual, expected):
    return max(
        abs(float(actual[row][column]) - float(expected[row][column]))
        for row in range(len(expected))
        for column in range(len(expected[row]))
    )


def validate_replay_source(request):
    replay = request.get("replay_source")
    if not isinstance(replay, dict):
        raise RuntimeError("PAN-29 request must contain replay_source")
    if replay.get("schema_version") != "pan29.replay-source.v1":
        raise RuntimeError("unexpected PAN-29 replay_source schema")
    if type(replay.get("observation_id")) is not int or replay["observation_id"] < 0:
        raise RuntimeError("PAN-29 replay observation_id must be non-negative")
    if replay.get("source_bundle_id") != replay["observation_id"]:
        raise RuntimeError("PAN-29 source bundle and observation IDs differ")
    if type(replay.get("source_capture_timestamp_ns")) is not int or replay[
        "source_capture_timestamp_ns"
    ] <= 0:
        raise RuntimeError("PAN-29 replay source timestamp must be positive")
    if replay.get("planner_input_unchanged") is not True:
        raise RuntimeError("PAN-29 replay must not become Planner input")
    if replay.get("ue5_role") != "post_run_visualization_only":
        raise RuntimeError("PAN-29 UE5 role must be post-run visualization only")
    if replay.get("requested_position_ue_cm") != request.get("position_ue_cm"):
        raise RuntimeError("PAN-29 replay position differs from capture request")
    if "ue5_actual_capture_timestamp_ns" in replay:
        raise RuntimeError("actual UE5 capture time cannot be supplied by replay metadata")
    return replay


def write_float32(path, values):
    payload = array.array("f", (float(value) for value in values))
    if payload.itemsize != 4:
        raise RuntimeError("platform float32 array itemsize is not 4")
    with path.open("wb") as stream:
        payload.tofile(stream)


def _srgb_to_linear(value):
    srgb = float(value) / 255.0
    return srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4


def _linear_to_srgb_byte(value):
    linear = min(1.0, max(0.0, float(value)))
    encoded = 12.92 * linear if linear <= 0.0031308 else 1.055 * (linear ** (1.0 / 2.4)) - 0.055
    return min(255, max(0, int(round(255.0 * encoded))))


def write_rgb(path, colors, exposure_ev=0.0, shadow_lift=None, valid_mask=None):
    if valid_mask is not None and len(valid_mask) != len(colors):
        raise RuntimeError("RGB and geometry mask lengths differ")
    payload = bytearray()
    changed_channels = 0
    exposure_changed_channels = 0
    shadow_changed_channels = 0
    shadow_lifted_pixels = 0
    shadow_no_hit_changed_pixels = 0
    shadow_bright_changed_pixels = 0
    geometry_valid_pixels = (
        sum(1 for value in valid_mask if bool(value)) if valid_mask is not None else 0
    )
    exposure_multiplier = 2.0 ** exposure_ev
    max_lift = float(shadow_lift["max_linear_lift"]) if shadow_lift else 0.0
    cutoff = float(shadow_lift["cutoff_linear_luma"]) if shadow_lift else 1.0
    power = float(shadow_lift["rolloff_power"]) if shadow_lift else 1.0
    for index, color in enumerate(colors):
        source = (int(color.r), int(color.g), int(color.b))
        exposed_linear = tuple(
            min(1.0, max(0.0, _srgb_to_linear(value) * exposure_multiplier))
            for value in source
        )
        exposure_only = tuple(_linear_to_srgb_byte(value) for value in exposed_linear)
        exposure_changed_channels += sum(a != b for a, b in zip(source, exposure_only))
        transformed = exposure_only
        if shadow_lift and valid_mask is not None and bool(valid_mask[index]):
            luma = (
                0.2126 * exposed_linear[0]
                + 0.7152 * exposed_linear[1]
                + 0.0722 * exposed_linear[2]
            )
            if luma < cutoff:
                weight = (1.0 - luma / cutoff) ** power
                lifted_linear = tuple(min(1.0, value + max_lift * weight) for value in exposed_linear)
                transformed = tuple(_linear_to_srgb_byte(value) for value in lifted_linear)
                shadow_delta = sum(a != b for a, b in zip(exposure_only, transformed))
                shadow_changed_channels += shadow_delta
                shadow_lifted_pixels += int(shadow_delta > 0)
            elif transformed != exposure_only:
                shadow_bright_changed_pixels += 1
        elif shadow_lift and transformed != exposure_only:
            shadow_no_hit_changed_pixels += 1
        changed_channels += sum(a != b for a, b in zip(source, transformed))
        payload.extend(transformed)
    path.write_bytes(payload)
    return {
        "method": "deterministic_srgb_exposure_plus_geometry_masked_linear_toe_lift_v2",
        "exposure_ev": float(exposure_ev),
        "linear_multiplier": float(exposure_multiplier),
        "changed_channel_count": int(changed_channels),
        "exposure_changed_channel_count": int(exposure_changed_channels),
        "shadow_lift_changed_channel_count": int(shadow_changed_channels),
        "shadow_lifted_pixel_count": int(shadow_lifted_pixels),
        "shadow_lift_geometry_valid_pixel_count": int(geometry_valid_pixels),
        "shadow_lift_no_hit_pixel_count": int(len(colors) - geometry_valid_pixels),
        "shadow_lift_no_hit_changed_pixel_count": int(shadow_no_hit_changed_pixels),
        "shadow_lift_bright_region_changed_pixel_count": int(shadow_bright_changed_pixels),
        "shadow_lift": shadow_lift,
        "channel_count": int(len(colors) * 3),
        "canonical_rgb_authority": "rgb_uint8.bin",
        "exported_png_role": "untransformed_ue5_diagnostic",
    }


def depth_stats(values):
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "sample_count": len(values),
        "finite_count": len(finite),
        "nonfinite_count": len(values) - len(finite),
        "finite_min": min(finite) if finite else None,
        "finite_max": max(finite) if finite else None,
    }


def asset(path, relative_to):
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _single_actor_by_label(actors, label):
    matches = [actor for actor in actors if actor.get_actor_label() == label]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label} actor, found {len(matches)}")
    return matches[0]


def _component_float(component, property_name):
    return float(component.get_editor_property(property_name))


def _lighting_snapshot(directional, skylight):
    directional_component = directional.get_component_by_class(
        unreal.DirectionalLightComponent
    )
    skylight_component = skylight.get_component_by_class(unreal.SkyLightComponent)
    if directional_component is None or skylight_component is None:
        raise RuntimeError("PAN-31 could not resolve the level light components")
    return {
        "directional_light": {
            "actor_label": directional.get_actor_label(),
            "rotation_degrees": rot(directional.get_actor_rotation()),
            "intensity": _component_float(directional_component, "intensity"),
            "source_angle_degrees": _component_float(
                directional_component, "light_source_angle"
            ),
            "cast_shadows": bool(
                directional_component.get_editor_property("cast_shadows")
            ),
        },
        "sky_light": {
            "actor_label": skylight.get_actor_label(),
            "intensity_scale": _component_float(
                skylight_component, "intensity"
            ),
            "real_time_capture": bool(
                skylight_component.get_editor_property("real_time_capture")
            ),
        },
    }


def _exposure_cvars():
    return {
        "r.DefaultFeature.AutoExposure": int(
            unreal.SystemLibrary.get_console_variable_int_value(
                "r.DefaultFeature.AutoExposure"
            )
        ),
        "r.EyeAdaptationQuality": int(
            unreal.SystemLibrary.get_console_variable_int_value(
                "r.EyeAdaptationQuality"
            )
        ),
    }


def apply_lighting_ablation(world, actors, config):
    """Apply one transient PAN-31 lighting variant without saving the level."""

    spec = config.get("lighting_ablation")
    if spec is None:
        return {
            "schema_version": PAN31_LIGHTING_SCHEMA,
            "variant_id": "level_baseline",
            "applied": False,
            "capture_exposure_compensation_ev": 0.0,
            "post_read_exposure_transform_ev": 0.0,
            "post_read_shadow_lift": None,
            "before": None,
            "after": None,
        }
    directional = _single_actor_by_label(actors, "PAN13_DirectionalLight")
    skylight = _single_actor_by_label(actors, "PAN13_SkyLight")
    before = _lighting_snapshot(directional, skylight)
    if not isinstance(spec, dict) or spec.get("schema_version") != PAN31_LIGHTING_SCHEMA:
        raise RuntimeError("invalid PAN-31 lighting_ablation schema")
    variant_id = spec.get("variant_id")
    if not isinstance(variant_id, str) or not variant_id:
        raise RuntimeError("PAN-31 lighting variant_id must be non-empty")
    exposure = float(spec.get("capture_exposure_compensation_ev", 0.0))
    if not math.isfinite(exposure) or not -5.0 <= exposure <= 5.0:
        raise RuntimeError("PAN-31 exposure compensation must be finite in [-5,5]")
    post_read_exposure = float(spec.get("post_read_exposure_transform_ev", 0.0))
    if not math.isfinite(post_read_exposure) or not -5.0 <= post_read_exposure <= 5.0:
        raise RuntimeError("PAN-31 post-read exposure must be finite in [-5,5]")
    shadow_lift_spec = spec.get("post_read_shadow_lift")
    shadow_lift = None
    if shadow_lift_spec is not None:
        if not isinstance(shadow_lift_spec, dict):
            raise RuntimeError("PAN-33 post-read shadow lift must be an object")
        if shadow_lift_spec.get("method") != "geometry_masked_linear_toe_lift_v1":
            raise RuntimeError("PAN-33 shadow lift method is unsupported")
        max_lift = float(shadow_lift_spec.get("max_linear_lift", -1.0))
        cutoff = float(shadow_lift_spec.get("cutoff_linear_luma", -1.0))
        power = float(shadow_lift_spec.get("rolloff_power", -1.0))
        if not math.isfinite(max_lift) or not 0.0 < max_lift <= 0.1:
            raise RuntimeError("PAN-33 max linear shadow lift must be in (0,0.1]")
        if not math.isfinite(cutoff) or not 0.0 < cutoff < 1.0:
            raise RuntimeError("PAN-33 shadow cutoff must be in (0,1)")
        if not math.isfinite(power) or not 1.0 <= power <= 8.0:
            raise RuntimeError("PAN-33 shadow rolloff power must be in [1,8]")
        shadow_lift = {
            "method": "geometry_masked_linear_toe_lift_v1",
            "max_linear_lift": max_lift,
            "cutoff_linear_luma": cutoff,
            "rolloff_power": power,
        }
    exposure_cvars_before = _exposure_cvars()
    enable_manual_pipeline = bool(spec.get("enable_manual_exposure_pipeline", False))
    if enable_manual_pipeline:
        unreal.SystemLibrary.execute_console_command(
            world, "r.DefaultFeature.AutoExposure 1"
        )
        unreal.SystemLibrary.execute_console_command(world, "r.EyeAdaptationQuality 2")
    exposure_cvars_after = _exposure_cvars()
    if enable_manual_pipeline and exposure_cvars_after != {
        "r.DefaultFeature.AutoExposure": 1,
        "r.EyeAdaptationQuality": 2,
    }:
        raise RuntimeError("PAN-31 could not enable the manual exposure pipeline")

    directional_override = spec.get("directional_light")
    if directional_override is not None:
        if not isinstance(directional_override, dict):
            raise RuntimeError("PAN-31 directional_light override must be an object")
        directional_component = directional.get_component_by_class(
            unreal.DirectionalLightComponent
        )
        if "intensity" in directional_override:
            intensity = float(directional_override["intensity"])
            if not math.isfinite(intensity) or intensity < 0.0:
                raise RuntimeError("PAN-31 directional intensity must be non-negative")
            directional_component.set_editor_property("intensity", intensity)
        if "source_angle_degrees" in directional_override:
            source_angle = float(directional_override["source_angle_degrees"])
            if not math.isfinite(source_angle) or not 0.0 <= source_angle <= 10.0:
                raise RuntimeError("PAN-31 source angle must be finite in [0,10]")
            directional_component.set_editor_property("light_source_angle", source_angle)

    skylight_override = spec.get("sky_light")
    if skylight_override is not None:
        if not isinstance(skylight_override, dict):
            raise RuntimeError("PAN-31 sky_light override must be an object")
        skylight_component = skylight.get_component_by_class(unreal.SkyLightComponent)
        if "intensity_scale" in skylight_override:
            intensity_scale = float(skylight_override["intensity_scale"])
            if not math.isfinite(intensity_scale) or intensity_scale < 0.0:
                raise RuntimeError("PAN-31 sky intensity must be non-negative")
            skylight_component.set_editor_property("intensity", intensity_scale)
        if bool(skylight_override.get("recapture_scene", False)):
            skylight_component.set_mobility(unreal.ComponentMobility.MOVABLE)
            skylight_component.set_editor_property("real_time_capture", True)
            skylight_component.recapture_sky()

    return {
        "schema_version": PAN31_LIGHTING_SCHEMA,
        "variant_id": variant_id,
        "applied": True,
        "capture_exposure_compensation_ev": exposure,
        "post_read_exposure_transform_ev": post_read_exposure,
        "post_read_shadow_lift": shadow_lift,
        "manual_exposure_pipeline_enabled": enable_manual_pipeline,
        "exposure_cvars_before": exposure_cvars_before,
        "exposure_cvars_after": exposure_cvars_after,
        "before": before,
        "after": _lighting_snapshot(directional, skylight),
    }


def apply_capture_exposure(component, lighting_report):
    compensation = float(lighting_report["capture_exposure_compensation_ev"])
    if lighting_report["applied"] is not True or compensation == 0.0:
        return {
            "manual_auto_exposure_disabled_by_project": True,
            "auto_exposure_bias_override": False,
            "exposure_compensation_ev": 0.0,
            "post_process_blend_weight": 0.0,
        }
    settings = component.get_editor_property("post_process_settings")
    settings.set_editor_property("override_auto_exposure_method", True)
    settings.set_editor_property(
        "auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL
    )
    settings.set_editor_property(
        "override_auto_exposure_apply_physical_camera_exposure", True
    )
    settings.set_editor_property(
        "auto_exposure_apply_physical_camera_exposure", False
    )
    settings.set_editor_property("override_auto_exposure_bias", True)
    settings.set_editor_property("auto_exposure_bias", compensation)
    component.set_editor_property("post_process_settings", settings)
    component.set_editor_property("post_process_blend_weight", 1.0)
    return {
        "manual_auto_exposure_disabled_by_project": True,
        "auto_exposure_method_override": "AEM_MANUAL",
        "physical_camera_exposure_enabled": False,
        "auto_exposure_bias_override": True,
        "exposure_compensation_ev": compensation,
        "post_process_blend_weight": 1.0,
    }


def create_transient_rig(actor_subsystem, location, config):
    root = actor_subsystem.spawn_actor_from_class(
        unreal.TargetPoint, location, unreal.Rotator(), transient=True
    )
    root.set_actor_label("PAN15_TransientSixFaceRigRoot")
    captures = {}
    for definition in config["faces"]:
        capture = actor_subsystem.spawn_actor_from_class(
            unreal.SceneCapture2D,
            location,
            ue_rotator(definition["rotation_degrees"]),
            transient=True,
        )
        capture.set_actor_label(f"PAN15_Capture_{definition['face_name']}")
        if not capture.attach_to_actor(
            root,
            "",
            unreal.AttachmentRule.KEEP_WORLD,
            unreal.AttachmentRule.KEEP_WORLD,
            unreal.AttachmentRule.KEEP_WORLD,
            False,
        ):
            raise RuntimeError(f"could not attach {definition['face_name']} to root")
        captures[definition["face_name"]] = capture
    return root, captures, "transient_from_pan23_config"


def resolve_rig(actor_subsystem, actors, request, config):
    position_actor_label = request.get("position_actor_label")
    position_ue_cm = request.get("position_ue_cm")
    actor_by_label = {actor.get_actor_label(): actor for actor in actors}
    if position_actor_label and position_ue_cm is not None:
        raise RuntimeError("capture request cannot select actor and explicit position")
    if position_ue_cm is not None:
        if (
            not isinstance(position_ue_cm, list)
            or len(position_ue_cm) != 3
            or not all(math.isfinite(float(value)) for value in position_ue_cm)
        ):
            raise RuntimeError("position_ue_cm must contain three finite values")
        return create_transient_rig(
            actor_subsystem,
            unreal.Vector(*[float(value) for value in position_ue_cm]),
            config,
        )
    if position_actor_label:
        position_actor = actor_by_label.get(position_actor_label)
        if position_actor is None:
            raise RuntimeError(f"missing position actor {position_actor_label}")
        return create_transient_rig(
            actor_subsystem, position_actor.get_actor_location(), config
        )
    roots = [
        actor for actor in actors if actor.get_actor_label() == "PAN23_SixFaceRigRoot"
    ]
    if len(roots) != 1:
        raise RuntimeError(f"expected one persistent PAN23 rig root, found {len(roots)}")
    root = roots[0]
    captures = {}
    for name in FACE_NAMES:
        capture = actor_by_label.get(f"PAN23_Capture_{name}")
        if capture is None or capture.get_attach_parent_actor() != root:
            raise RuntimeError(f"persistent capture {name} is missing its shared root")
        captures[name] = capture
    return root, captures, "persistent_pan23_level_rig"


def main():
    if BUNDLE_DIR.exists():
        raise FileExistsError(f"refusing to overwrite raw bundle {BUNDLE_DIR}")
    BUNDLE_DIR.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{BUNDLE_DIR.name}.tmp-", dir=str(BUNDLE_DIR.parent))
    )
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        request = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
        if tuple(config["face_names"]) != FACE_NAMES:
            raise RuntimeError("capture config does not contain canonical six faces")
        if GPU_INDEX != int(config["gpu_index"]):
            raise RuntimeError("runtime GPU does not match pinned capture config")
        if request["level_path"] != config.get("level_path") and request["scenario"] == "analytic":
            raise RuntimeError("analytic request level does not match capture config")
        is_pan29_replay = config.get("schema_version") == PAN29_REPLAY_RIG_SCHEMA
        if is_pan29_replay:
            validate_replay_source(request)
        if is_pan29_replay and request["level_path"] != config.get("level_path"):
            raise RuntimeError("PAN-29 request level does not match replay config")

        level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        if not level_subsystem.load_level(request["level_path"]):
            raise RuntimeError(f"could not load {request['level_path']}")
        world = unreal.EditorLevelLibrary.get_editor_world()
        world_to_meters = float(
            world.get_world_settings().get_editor_property("world_to_meters")
        )
        if abs(world_to_meters - float(config["world_to_meters"])) > 1e-6:
            raise RuntimeError("WorldToMeters does not match capture config")
        actors = actor_subsystem.get_all_level_actors()
        lighting_report = apply_lighting_ablation(world, actors, config)
        root, captures, rig_source = resolve_rig(
            actor_subsystem, actors, request, config
        )
        root_location = vec(root.get_actor_location())
        resolution = int(config["face_resolution"])
        # UE raster pixels sample at half-pixel centres. For a 90 degree FOV,
        # fx=fy=N/2 while the array-index principal point is (N-1)/2.
        focal = 0.5 * resolution
        principal = 0.5 * (resolution - 1.0)
        K_pixel = [
            [focal, 0.0, principal],
            [0.0, focal, principal],
            [0.0, 0.0, 1.0],
        ]
        started = time.perf_counter()
        face_reports = []
        raw_root = temporary / "faces"
        try:
            unreal.AutomationLibrary.finish_loading_before_screenshot()
        except Exception as exc:
            unreal.log_warning(f"PAN15 finish_loading_before_screenshot: {exc}")
        # This is the shared bundle epoch immediately before the first face
        # capture.  It is deliberately distinct from PAN-29's source timestamp.
        timestamp_ns = time.time_ns()

        for definition in config["faces"]:
            name = definition["face_name"]
            capture = captures[name]
            if max(
                abs(a - b)
                for a, b in zip(vec(capture.get_actor_location()), root_location)
            ) > 1e-6:
                raise RuntimeError(f"capture {name} does not share root location")
            forward = vec(
                unreal.MathLibrary.get_forward_vector(capture.get_actor_rotation())
            )
            expected_forward = definition.get("expected_forward_ue")
            if expected_forward is not None and max(
                abs(float(actual) - float(expected))
                for actual, expected in zip(forward, expected_forward)
            ) > 1e-5:
                raise RuntimeError(f"capture {name} forward axis does not match config")
            contract_transform = contract_pose(capture, world_to_meters)
            pioneer_transform = (
                pioneer_pose(
                    contract_transform, float(config["scene_units_per_meter"])
                )
                if is_pan29_replay
                else None
            )
            expected_rotation = definition.get(
                "expected_T_pioneer_world_from_cam_rotation"
            )
            if is_pan29_replay:
                if expected_rotation is None:
                    raise RuntimeError(f"PAN-29 face {name} lacks an expected basis")
                basis_error = maximum_matrix_error(
                    [row[:3] for row in pioneer_transform[:3]], expected_rotation
                )
                if basis_error > 1e-5:
                    raise RuntimeError(
                        f"PAN-29 face {name} basis mismatch: max_error={basis_error}"
                    )
            component = capture.get_component_by_class(unreal.SceneCaptureComponent2D)
            component.set_editor_property("fov_angle", config["fov_degrees"])
            component.set_editor_property("capture_every_frame", False)
            component.set_editor_property("capture_on_movement", False)
            exposure_report = apply_capture_exposure(component, lighting_report)
            face_dir = raw_root / name
            face_dir.mkdir(parents=True, exist_ok=True)

            rgb_target = unreal.TextureRenderTarget2D()
            rgb_target.set_editor_property("size_x", resolution)
            rgb_target.set_editor_property("size_y", resolution)
            rgb_target.set_editor_property(
                "render_target_format", unreal.TextureRenderTargetFormat.RTF_RGBA8_SRGB
            )
            rgb_target.set_editor_property(
                "clear_color", unreal.LinearColor(0.02, 0.02, 0.02, 1.0)
            )
            component.set_editor_property("texture_target", rgb_target)
            component.set_editor_property(
                "capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR
            )
            capture_started = time.perf_counter()
            component.capture_scene()
            rgb_capture_seconds = time.perf_counter() - capture_started
            rgb_read_started = time.perf_counter()
            rgb_samples = unreal.RenderingLibrary.read_render_target(
                world, rgb_target, normalize=True
            )
            rgb_readback_seconds = time.perf_counter() - rgb_read_started
            if rgb_samples is None or len(rgb_samples) != resolution * resolution:
                raise RuntimeError(f"RGB readback failed for {name}")
            rgb_raw = face_dir / "rgb_uint8.bin"
            export_started = time.perf_counter()
            unreal.RenderingLibrary.export_render_target(
                world, rgb_target, str(face_dir), "rgb"
            )
            rgb_candidates = sorted(
                path for path in face_dir.glob("rgb*") if path != rgb_raw and path.is_file()
            )
            if len(rgb_candidates) != 1:
                raise RuntimeError(f"RGB export for {name} produced {len(rgb_candidates)} files")
            rgb_png = rgb_candidates[0]
            if rgb_png.suffix == "":
                target = rgb_png.with_suffix(".png")
                rgb_png.replace(target)
                rgb_png = target
            rgb_export_seconds = time.perf_counter() - export_started

            depth_payloads = {}
            for label, capture_source in (
                ("scene_depth_r", unreal.SceneCaptureSource.SCS_SCENE_DEPTH),
                ("device_depth_r", unreal.SceneCaptureSource.SCS_DEVICE_DEPTH),
            ):
                # RenderingLibrary creates and initialises the GPU resource.
                # UE 5.4 ReadLinearColorPixels supports PF_FloatRGBA (the
                # RTF_RGBA16F backing format) on this Vulkan path; R32F and
                # RGBA32F produced no Python samples in retained attempts.
                target = unreal.RenderingLibrary.create_render_target2d(
                    world,
                    resolution,
                    resolution,
                    unreal.TextureRenderTargetFormat.RTF_RGBA16F,
                    unreal.LinearColor(0.0, 0.0, 0.0, 0.0),
                )
                if target is None:
                    raise RuntimeError(f"could not create {label} render target")
                component.set_editor_property("texture_target", target)
                component.set_editor_property("capture_source", capture_source)
                depth_capture_started = time.perf_counter()
                component.capture_scene()
                depth_capture_seconds = time.perf_counter() - depth_capture_started
                readback_started = time.perf_counter()
                samples = unreal.RenderingLibrary.read_render_target_raw(
                    world, target, normalize=False
                )
                readback_method = "read_render_target_raw"
                if samples is None:
                    # The UE 5.4 full-target binding can return None even for
                    # an initialised float resource. Its pixel-area sibling
                    # invokes the same raw float read without that optional
                    # output-array binding failure.
                    samples = unreal.RenderingLibrary.read_render_target_raw_pixel_area(
                        world,
                        target,
                        0,
                        0,
                        resolution,
                        resolution,
                        normalize=False,
                    )
                    readback_method = "read_render_target_raw_pixel_area"
                readback_seconds = time.perf_counter() - readback_started
                if samples is None or len(samples) != resolution * resolution:
                    sample_count = None if samples is None else len(samples)
                    raise RuntimeError(
                        f"{label} readback failed for {name}: sample_count={sample_count}"
                    )
                export_started = time.perf_counter()
                export_stem = f"{label}_raw"
                unreal.RenderingLibrary.export_render_target(
                    world, target, str(face_dir), export_stem
                )
                export_candidates = sorted(face_dir.glob(f"{export_stem}*"))
                if len(export_candidates) != 1:
                    raise RuntimeError(
                        f"{label} EXR export for {name} produced "
                        f"{len(export_candidates)} files"
                    )
                exr_path = export_candidates[0]
                if exr_path.suffix == "":
                    renamed = exr_path.with_suffix(".exr")
                    exr_path.replace(renamed)
                    exr_path = renamed
                export_seconds = time.perf_counter() - export_started
                values = [float(sample.r) for sample in samples]
                path = face_dir / f"{label}_python_readback_float32.bin"
                write_started = time.perf_counter()
                write_float32(path, values)
                write_seconds = time.perf_counter() - write_started
                depth_payloads[label] = {
                    "capture_source": str(capture_source),
                    "render_target_format": "RTF_RGBA16F",
                    "raw_exr": asset(exr_path, temporary),
                    "raw_exr_channel": "R",
                    "raw_exr_storage_precision": "IEEE-754 binary16",
                    "python_readback": {
                        **asset(path, temporary),
                        "dtype": "<f4",
                        "shape": [resolution, resolution],
                        "method": readback_method,
                        "stats": depth_stats(values),
                        "status": "diagnostic_only_not_geometry_authority",
                    },
                    "timings_seconds": {
                        "capture": depth_capture_seconds,
                        "readback": readback_seconds,
                        "export_exr": export_seconds,
                        "serialization_write": write_seconds,
                    },
                    "_python_values": values,
                }

            near_world = float(config["near_m"]) * world_to_meters
            far_world = float(config["far_m"]) * world_to_meters
            scene_values = depth_payloads["scene_depth_r"].pop("_python_values")
            depth_payloads["device_depth_r"].pop("_python_values")
            mask_values = bytearray(
                1 if math.isfinite(value) and near_world < value < far_world else 0
                for value in scene_values
            )
            mask_path = face_dir / "python_readback_candidate_mask_uint8.bin"
            mask_write_started = time.perf_counter()
            mask_path.write_bytes(mask_values)
            mask_write_seconds = time.perf_counter() - mask_write_started
            serialization_started = time.perf_counter()
            rgb_transform_report = write_rgb(
                rgb_raw,
                rgb_samples,
                float(lighting_report["post_read_exposure_transform_ev"]),
                lighting_report.get("post_read_shadow_lift"),
                mask_values,
            )
            rgb_serialization_seconds = time.perf_counter() - serialization_started
            face_reports.append(
                {
                    "face_name": name,
                    "request_id": request["request_id"],
                    "frame_id": request["frame_id"],
                    "capture_timestamp_ns": timestamp_ns,
                    "image_size": [resolution, resolution],
                    "fov_degrees": float(config["fov_degrees"]),
                    "location_ue_cm": vec(capture.get_actor_location()),
                    "rotation_degrees": rot(capture.get_actor_rotation()),
                    "forward_ue": forward,
                    "K_pixel": K_pixel,
                    "T_world_from_cam": contract_transform,
                    "T_pioneer_world_from_cam": pioneer_transform,
                    "pioneer_basis_max_abs_error": (
                        maximum_matrix_error(
                            [row[:3] for row in pioneer_transform[:3]],
                            expected_rotation,
                        )
                        if pioneer_transform is not None and expected_rotation is not None
                        else None
                    ),
                    "capture_exposure": exposure_report,
                    "rgb_post_read_exposure_transform": rgb_transform_report,
                    "rgb_uint8": {
                        **asset(rgb_raw, temporary),
                        "dtype": "uint8",
                        "shape": [resolution, resolution, 3],
                    },
                    "rgb_png": asset(rgb_png, temporary),
                    "scene_depth_r": depth_payloads["scene_depth_r"],
                    "device_depth_r": depth_payloads["device_depth_r"],
                    "python_readback_candidate_mask": {
                        **asset(mask_path, temporary),
                        "dtype": "uint8",
                        "shape": [resolution, resolution],
                        "valid_count": int(sum(mask_values)),
                        "no_hit_count": int(len(mask_values) - sum(mask_values)),
                        "rule": "diagnostic Python readback only; canonical adapter recomputes from raw EXR",
                        "write_seconds": mask_write_seconds,
                    },
                    "timings_seconds": {
                        "rgb_capture": rgb_capture_seconds,
                        "rgb_readback": rgb_readback_seconds,
                        "rgb_serialization": rgb_serialization_seconds,
                        "rgb_export": rgb_export_seconds,
                    },
                }
            )

        if tuple(face["face_name"] for face in face_reports) != FACE_NAMES:
            raise RuntimeError("raw bundle face set is incomplete or misordered")
        positions = [face["location_ue_cm"] for face in face_reports]
        if any(
            max(abs(a - b) for a, b in zip(position, positions[0])) > 1e-6
            for position in positions[1:]
        ):
            raise RuntimeError("raw bundle faces do not share one optical centre")
        manifest = {
            "schema_version": "pan15.ue5-raw-rgbd.v1",
            "task": "PAN-15",
            "scenario": request["scenario"],
            "request_id": request["request_id"],
            "frame_id": request["frame_id"],
            "capture_timestamp_ns": timestamp_ns,
            "level_path": request["level_path"],
            "position_actor_label": request.get("position_actor_label"),
            "requested_position_ue_cm": request.get("position_ue_cm"),
            "replay_source": request.get("replay_source"),
            "rig_source": rig_source,
            "shared_optical_center_ue_cm": positions[0],
            "world_to_meters": world_to_meters,
            "engine_version": unreal.SystemLibrary.get_engine_version(),
            "project_commit": PROJECT_COMMIT,
            "capture_config_path": str(CONFIG_PATH),
            "capture_config_sha256": sha256(CONFIG_PATH),
            "gpu_index": GPU_INDEX,
            "rhi": config["rhi"],
            "near_m": config["near_m"],
            "far_m": config["far_m"],
            "source_depth_provenance": {
                "scene_depth_r": "UE SceneCaptureSource.SCS_SCENE_DEPTH raw R from RGBA16F",
                "device_depth_r": "UE SceneCaptureSource.SCS_DEVICE_DEPTH raw R from RGBA16F",
                "readback_format_limit": "binary16 EXR source; 65504 world units is treated as invalid overflow/no-hit",
                "geometry_authority": "raw_exr R channel; Python readback is diagnostic only",
                "pixel_intrinsics": "UE pixel-centre convention fx=fy=N/(2*tan(FOV/2)), cx=cy=(N-1)/2",
                "final_encoding_decision": "deferred to PAN-20",
            },
            "lighting_preset": config["lighting_preset"],
            "lighting_ablation": lighting_report,
            "console_variables": config["console_variables"],
            "faces": face_reports,
            "timings_seconds": {
                "total_capture_readback_serialization_write": time.perf_counter()
                - started
            },
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(str(temporary), str(BUNDLE_DIR))
        temporary = None
        unreal.log(
            f"PAN15_CAPTURE result=PASS scenario={request['scenario']} faces=6"
        )
    finally:
        if temporary is not None:
            failed = BUNDLE_DIR.parent / "attempts" / temporary.name
            failed.parent.mkdir(parents=True, exist_ok=True)
            if temporary.exists():
                shutil.move(str(temporary), str(failed))


try:
    main()
except Exception:
    BUNDLE_DIR.parent.mkdir(parents=True, exist_ok=True)
    failure_path = BUNDLE_DIR.parent / f"{BUNDLE_DIR.name}.failure.json"
    failure_path.write_text(
        json.dumps({"task": "PAN-15", "error": traceback.format_exc()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    unreal.log_error(f"PAN15_CAPTURE failed; see {failure_path}")
    raise
