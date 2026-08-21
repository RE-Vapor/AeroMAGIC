"""Build the PAN-23 analytic level and persistent six-face rig."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import unreal


OUTPUT_DIR = Path(os.environ["PAN23_OUTPUT_DIR"]).resolve()
CONFIG_PATH = Path(os.environ["PAN23_CAPTURE_CONFIG"]).resolve()
SCENE_PATH = Path(os.environ["PAN23_SCENE_CONFIG"]).resolve()


def vec(value):
    return [float(value.x), float(value.y), float(value.z)]


def rotation(value):
    return [float(value.pitch), float(value.yaw), float(value.roll)]


def ue_rotator(pitch_yaw_roll):
    """Translate config [pitch, yaw, roll] to Unreal Python constructor order."""

    pitch, yaw, roll = pitch_yaw_roll
    return unreal.Rotator(roll, pitch, yaw)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    scene = json.loads(SCENE_PATH.read_text(encoding="utf-8"))
    level_path = config["level_path"]
    if unreal.EditorAssetLibrary.does_asset_exist(level_path):
        raise RuntimeError(
            f"refusing to overwrite existing analytic level {level_path}"
        )

    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    if not level_subsystem.new_level(level_path):
        raise RuntimeError(f"failed to create {level_path}")
    world = unreal.EditorLevelLibrary.get_editor_world()
    world_settings = world.get_world_settings()
    world_settings.set_editor_property("world_to_meters", config["world_to_meters"])

    actor_records = []
    for definition in scene["actors"]:
        mesh = unreal.EditorAssetLibrary.load_asset(definition["mesh"])
        if not mesh:
            raise RuntimeError(f"could not load {definition['mesh']}")
        actor = actor_subsystem.spawn_actor_from_class(
            unreal.StaticMeshActor,
            unreal.Vector(*definition["location_ue_cm"]),
            unreal.Rotator(),
            transient=False,
        )
        if not actor:
            raise RuntimeError(f"could not spawn {definition['label']}")
        actor.set_actor_label(definition["label"])
        actor.set_actor_scale3d(unreal.Vector(*definition["scale"]))
        actor.get_component_by_class(unreal.StaticMeshComponent).set_static_mesh(mesh)
        actor_records.append(
            {
                "label": actor.get_actor_label(),
                "location_ue_cm": vec(actor.get_actor_location()),
                "scale": vec(actor.get_actor_scale3d()),
                "mesh": definition["mesh"],
            }
        )

    directional = actor_subsystem.spawn_actor_from_class(
        unreal.DirectionalLight,
        unreal.Vector(0.0, 0.0, 2000.0),
        ue_rotator(config["lighting_preset"]["directional_light_rotation_degrees"]),
        transient=False,
    )
    if not directional:
        raise RuntimeError("could not spawn DirectionalLight")
    directional.set_actor_label("PAN23_DirectionalLight")
    directional_component = directional.get_component_by_class(
        unreal.DirectionalLightComponent
    )
    directional_component.set_editor_property(
        "intensity", config["lighting_preset"]["directional_light_intensity_lux"]
    )
    directional_component.set_editor_property("cast_shadows", True)

    skylight = actor_subsystem.spawn_actor_from_class(
        unreal.SkyLight, unreal.Vector(), unreal.Rotator(), transient=False
    )
    if not skylight:
        raise RuntimeError("could not spawn SkyLight")
    skylight.set_actor_label("PAN23_SkyLight")
    sky = actor_subsystem.spawn_actor_from_class(
        unreal.SkyAtmosphere, unreal.Vector(), unreal.Rotator(), transient=False
    )
    if not sky:
        raise RuntimeError("could not spawn SkyAtmosphere")
    sky.set_actor_label("PAN23_SkyAtmosphere")

    root_location = unreal.Vector(*config["rig_root_location_ue_cm"])
    root = actor_subsystem.spawn_actor_from_class(
        unreal.TargetPoint, root_location, unreal.Rotator(), transient=False
    )
    if not root:
        raise RuntimeError("could not spawn six-face rig root")
    root.set_actor_label("PAN23_SixFaceRigRoot")
    face_records = []
    for face in config["faces"]:
        face_rotation = ue_rotator(face["rotation_degrees"])
        capture = actor_subsystem.spawn_actor_from_class(
            unreal.SceneCapture2D, root_location, face_rotation, transient=False
        )
        if not capture:
            raise RuntimeError(f"could not spawn capture {face['face_name']}")
        capture.set_actor_label(f"PAN23_Capture_{face['face_name']}")
        attached = capture.attach_to_actor(
            root,
            "",
            unreal.AttachmentRule.KEEP_WORLD,
            unreal.AttachmentRule.KEEP_WORLD,
            unreal.AttachmentRule.KEEP_WORLD,
            False,
        )
        if not attached or capture.get_attach_parent_actor() != root:
            raise RuntimeError(f"capture {face['face_name']} is not attached to rig root")
        component = capture.get_component_by_class(unreal.SceneCaptureComponent2D)
        component.set_editor_property("fov_angle", config["fov_degrees"])
        component.set_editor_property("capture_every_frame", False)
        component.set_editor_property("capture_on_movement", False)
        actual_forward = vec(unreal.MathLibrary.get_forward_vector(capture.get_actor_rotation()))
        expected_forward = face["expected_forward_ue"]
        if max(abs(a - b) for a, b in zip(actual_forward, expected_forward)) > 1e-5:
            raise RuntimeError(
                f"{face['face_name']} forward mismatch: {actual_forward} != {expected_forward}"
            )
        face_records.append(
            {
                "face_name": face["face_name"],
                "actor_label": capture.get_actor_label(),
                "parent_label": capture.get_attach_parent_actor().get_actor_label(),
                "location_ue_cm": vec(capture.get_actor_location()),
                "rotation_degrees": rotation(capture.get_actor_rotation()),
                "forward_ue": actual_forward,
                "fov_degrees": float(component.get_editor_property("fov_angle")),
            }
        )

    if not level_subsystem.save_current_level():
        raise RuntimeError(f"failed to save {level_path}")
    report = {
        "schema_version": "pan23.build-report.v1",
        "task": "PAN-23",
        "result": "PASS",
        "engine_version": unreal.SystemLibrary.get_engine_version(),
        "level_path": level_path,
        "world_to_meters": float(world_settings.get_editor_property("world_to_meters")),
        "analytic_scene": scene,
        "actors": actor_records,
        "lighting": {
            "sky_atmosphere": sky.get_actor_label(),
            "directional_light": directional.get_actor_label(),
            "sky_light": skylight.get_actor_label(),
            "cast_shadows": bool(directional_component.get_editor_property("cast_shadows")),
        },
        "rig": {
            "root_label": root.get_actor_label(),
            "root_location_ue_cm": vec(root.get_actor_location()),
            "faces": face_records,
        },
    }
    (OUTPUT_DIR / "build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    unreal.log("PAN23_BUILD result=PASS faces=6 shared_root=PASS")


try:
    main()
except Exception:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "build_failure.json").write_text(
        json.dumps({"task": "PAN-23", "error": traceback.format_exc()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    unreal.log_error(f"PAN23_BUILD failed; see {OUTPUT_DIR / 'build_failure.json'}")
    raise
