"""Capture one deterministic offscreen RGB image for every PAN-23 rig face."""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import unreal


OUTPUT_DIR = Path(os.environ["PAN23_OUTPUT_DIR"]).resolve()
CONFIG_PATH = Path(os.environ["PAN23_CAPTURE_CONFIG"]).resolve()
PROJECT_COMMIT = os.environ.get("PAN23_PROJECT_COMMIT", "unknown")
GPU_INDEX = int(os.environ["PAN23_GPU_INDEX"])


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vec(value):
    return [float(value.x), float(value.y), float(value.z)]


def rot(value):
    return [float(value.pitch), float(value.yaw), float(value.roll)]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if GPU_INDEX != int(config["gpu_index"]):
        raise RuntimeError(
            f"GPU index {GPU_INDEX} does not match pinned config {config['gpu_index']}"
        )
    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    if not level_subsystem.load_level(config["level_path"]):
        raise RuntimeError(f"could not load {config['level_path']}")
    actors = actor_subsystem.get_all_level_actors()
    roots = [actor for actor in actors if actor.get_actor_label() == "PAN23_SixFaceRigRoot"]
    if len(roots) != 1:
        raise RuntimeError(f"expected one rig root, found {len(roots)}")
    root = roots[0]
    world = unreal.EditorLevelLibrary.get_editor_world()
    world_to_meters = float(world.get_world_settings().get_editor_property("world_to_meters"))
    if abs(world_to_meters - config["world_to_meters"]) > 1e-6:
        raise RuntimeError("WorldToMeters does not match capture config")

    actor_by_label = {actor.get_actor_label(): actor for actor in actors}
    capture_timestamp_ns = time.time_ns()
    started = time.perf_counter()
    face_reports = []
    shared_locations = []
    for face in config["faces"]:
        name = face["face_name"]
        capture = actor_by_label.get(f"PAN23_Capture_{name}")
        if capture is None:
            raise RuntimeError(f"missing capture actor {name}")
        if capture.get_attach_parent_actor() != root:
            raise RuntimeError(f"capture {name} lost the shared rig root")
        component = capture.get_component_by_class(unreal.SceneCaptureComponent2D)
        if abs(float(component.get_editor_property("fov_angle")) - config["fov_degrees"]) > 1e-6:
            raise RuntimeError(f"capture {name} FOV mismatch")
        render_target = unreal.TextureRenderTarget2D()
        render_target.set_editor_property("size_x", config["face_resolution"])
        render_target.set_editor_property("size_y", config["face_resolution"])
        render_target.set_editor_property(
            "render_target_format", unreal.TextureRenderTargetFormat.RTF_RGBA8_SRGB
        )
        render_target.set_editor_property(
            "clear_color", unreal.LinearColor(0.02, 0.02, 0.02, 1.0)
        )
        component.set_editor_property("texture_target", render_target)
        component.set_editor_property(
            "capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR
        )
        face_started = time.perf_counter()
        component.capture_scene()
        unreal.RenderingLibrary.export_render_target(
            world, render_target, str(OUTPUT_DIR), name
        )
        candidates = sorted(
            path for path in OUTPUT_DIR.glob(f"{name}*") if path.is_file()
        )
        if len(candidates) != 1:
            raise RuntimeError(f"capture {name} produced {len(candidates)} files")
        path = candidates[0]
        if path.suffix == "":
            normalized_path = path.with_suffix(".png")
            path.replace(normalized_path)
            path = normalized_path
        location = vec(capture.get_actor_location())
        shared_locations.append(location)
        face_reports.append(
            {
                "face_name": name,
                "capture_timestamp_ns": capture_timestamp_ns,
                "location_ue_cm": location,
                "rotation_degrees": rot(capture.get_actor_rotation()),
                "forward_ue": vec(unreal.MathLibrary.get_forward_vector(capture.get_actor_rotation())),
                "fov_degrees": float(component.get_editor_property("fov_angle")),
                "resolution": [config["face_resolution"], config["face_resolution"]],
                "rgb_path": str(path),
                "rgb_bytes": path.stat().st_size,
                "rgb_sha256": sha256(path),
                "capture_seconds": time.perf_counter() - face_started,
            }
        )
    reference = shared_locations[0]
    if any(max(abs(a - b) for a, b in zip(location, reference)) > 1e-6 for location in shared_locations[1:]):
        raise RuntimeError("six faces do not share one optical centre")
    report = {
        "schema_version": "pan23.rgb-smoke.v1",
        "task": "PAN-23",
        "result": "PASS",
        "engine_version": unreal.SystemLibrary.get_engine_version(),
        "project_commit": PROJECT_COMMIT,
        "level_path": config["level_path"],
        "capture_timestamp_ns": capture_timestamp_ns,
        "gpu_index": GPU_INDEX,
        "rhi": config["rhi"],
        "world_to_meters": world_to_meters,
        "near_m": config["near_m"],
        "far_m": config["far_m"],
        "lighting_preset": config["lighting_preset"],
        "console_variables": config["console_variables"],
        "shared_optical_center_ue_cm": reference,
        "face_count": len(face_reports),
        "faces": face_reports,
        "capture_seconds": time.perf_counter() - started,
    }
    (OUTPUT_DIR / "capture_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    unreal.log("PAN23_CAPTURE result=PASS faces=6 shared_root=PASS")


try:
    main()
except Exception:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "capture_failure.json").write_text(
        json.dumps({"task": "PAN-23", "error": traceback.format_exc()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    unreal.log_error(f"PAN23_CAPTURE failed; see {OUTPUT_DIR / 'capture_failure.json'}")
    raise
