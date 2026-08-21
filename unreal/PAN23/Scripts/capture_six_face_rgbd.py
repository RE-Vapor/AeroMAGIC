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

    transform_world_axis = lambda value: [value.x, -value.y, value.z]
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


def write_float32(path, values):
    payload = array.array("f", (float(value) for value in values))
    if payload.itemsize != 4:
        raise RuntimeError("platform float32 array itemsize is not 4")
    with path.open("wb") as stream:
        payload.tofile(stream)


def write_rgb(path, colors):
    payload = bytearray()
    for color in colors:
        payload.extend((int(color.r), int(color.g), int(color.b)))
    path.write_bytes(payload)


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
    actor_by_label = {actor.get_actor_label(): actor for actor in actors}
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
        timestamp_ns = time.time_ns()
        started = time.perf_counter()
        face_reports = []
        raw_root = temporary / "faces"
        try:
            unreal.AutomationLibrary.finish_loading_before_screenshot()
        except Exception as exc:
            unreal.log_warning(f"PAN15 finish_loading_before_screenshot: {exc}")

        for definition in config["faces"]:
            name = definition["face_name"]
            capture = captures[name]
            if max(
                abs(a - b)
                for a, b in zip(vec(capture.get_actor_location()), root_location)
            ) > 1e-6:
                raise RuntimeError(f"capture {name} does not share root location")
            component = capture.get_component_by_class(unreal.SceneCaptureComponent2D)
            component.set_editor_property("fov_angle", config["fov_degrees"])
            component.set_editor_property("capture_every_frame", False)
            component.set_editor_property("capture_on_movement", False)
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
            serialization_started = time.perf_counter()
            write_rgb(rgb_raw, rgb_samples)
            rgb_serialization_seconds = time.perf_counter() - serialization_started
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
                    "forward_ue": vec(
                        unreal.MathLibrary.get_forward_vector(capture.get_actor_rotation())
                    ),
                    "K_pixel": K_pixel,
                    "T_world_from_cam": contract_pose(capture, world_to_meters),
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
