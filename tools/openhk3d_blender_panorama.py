#!/usr/bin/env python3
"""Blender backend for GT equirectangular RGB and range-depth rendering."""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import bpy
import numpy as np
import OpenImageIO as oiio
from mathutils import Matrix, Vector


REQUIRED_OBJ_IMPORT_PROPERTIES = {
    "filepath",
    "forward_axis",
    "up_axis",
    "use_split_objects",
    "use_split_groups",
    "validate_meshes",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def blender_arguments() -> List[str]:
    try:
        separator = sys.argv.index("--")
    except ValueError:
        return []
    return sys.argv[separator + 1 :]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args(blender_arguments() if argv is None else argv)


def require_runtime_capabilities() -> Dict[str, object]:
    obj_import = getattr(bpy.ops.wm, "obj_import", None)
    if obj_import is None:
        raise RuntimeError("Blender does not expose bpy.ops.wm.obj_import")
    available = {prop.identifier for prop in obj_import.get_rna_type().properties}
    missing = sorted(REQUIRED_OBJ_IMPORT_PROPERTIES - available)
    if missing:
        raise RuntimeError(
            "Blender OBJ importer is missing required properties: {}".format(
                ", ".join(missing)
            )
        )
    camera = bpy.data.cameras.new("capability-probe")
    try:
        camera.type = "PANO"
        panorama_api = set_equirectangular(camera)
    finally:
        bpy.data.cameras.remove(camera)
    build_hash = bpy.app.build_hash
    if isinstance(build_hash, bytes):
        build_hash = build_hash.decode("ascii", errors="replace")
    return {
        "blender_version": bpy.app.version_string,
        "blender_version_tuple": list(bpy.app.version),
        "build_hash": build_hash,
        "obj_import_properties": sorted(REQUIRED_OBJ_IMPORT_PROPERTIES),
        "equirectangular_cycles_camera": True,
        "panorama_api": panorama_api,
        "depth_pass": hasattr(bpy.context.view_layer, "use_pass_z"),
    }


def set_equirectangular(camera: bpy.types.Camera) -> str:
    """Configure panorama API used by Blender 5.2+ or earlier Cycles builds."""
    if hasattr(camera, "panorama_type"):
        camera.panorama_type = "EQUIRECTANGULAR"
        return "Camera.panorama_type"
    cycles = getattr(camera, "cycles", None)
    if cycles is None or not hasattr(cycles, "panorama_type"):
        raise RuntimeError("Blender camera exposes no panoramic projection API")
    cycles.panorama_type = "EQUIRECTANGULAR"
    return "Camera.cycles.panorama_type"


def configure_cycles(scene: bpy.types.Scene, requested: str) -> Dict[str, object]:
    if requested == "CPU":
        scene.cycles.device = "CPU"
        return {"requested": requested, "selected": "CPU", "devices": []}
    addon = bpy.context.preferences.addons.get("cycles")
    if addon is None:
        raise RuntimeError("Cycles add-on preferences are unavailable")
    preferences = addon.preferences
    try:
        preferences.compute_device_type = requested
        preferences.get_devices()
    except Exception as exc:
        raise RuntimeError("Cannot initialize Cycles {}: {}".format(requested, exc)) from exc
    candidates = [device for device in preferences.devices if device.type == requested]
    if not candidates:
        raise RuntimeError("Cycles reports no {} device".format(requested))
    selected = candidates[0]
    for device in preferences.devices:
        device.use = device == selected
    scene.cycles.device = "GPU"
    return {
        "requested": requested,
        "selected": selected.type,
        "devices": [{"name": selected.name, "type": selected.type}],
    }


def clear_and_import(obj_path: Path) -> Dict[str, object]:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    result = bpy.ops.wm.obj_import(
        filepath=str(obj_path),
        forward_axis="NEGATIVE_Z",
        up_axis="Y",
        use_split_objects=True,
        use_split_groups=False,
        validate_meshes=True,
    )
    if "FINISHED" not in result:
        raise RuntimeError("Blender OBJ importer did not finish: {}".format(result))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("Blender imported no mesh objects")
    images = []
    missing = []
    for image in bpy.data.images:
        if image.source != "FILE":
            continue
        image_path = Path(bpy.path.abspath(image.filepath)).resolve()
        images.append(str(image_path))
        if not image_path.is_file():
            missing.append(str(image_path))
    if missing:
        raise RuntimeError("Blender reports missing texture images: {}".format(missing))
    return {
        "obj_axis_convention": {"forward_axis": "NEGATIVE_Z", "up_axis": "Y"},
        "mesh_objects": len(meshes),
        "vertices": sum(len(obj.data.vertices) for obj in meshes),
        "polygons": sum(len(obj.data.polygons) for obj in meshes),
        "materials": len(bpy.data.materials),
        "file_images": len(images),
        "missing_images": [],
    }


def configure_world(scene: bpy.types.Scene) -> None:
    world = scene.world or bpy.data.worlds.new("PanoramaWorld")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is None:
        raise RuntimeError("Blender world has no Background node")
    background.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    background.inputs["Strength"].default_value = 0.8


def configure_camera(scene: bpy.types.Scene, max_depth: float) -> bpy.types.Object:
    data = bpy.data.cameras.new("OpenHK3D_Equirectangular")
    data.type = "PANO"
    set_equirectangular(data)
    data.clip_start = min(0.05, max_depth / 1000.0)
    data.clip_end = max_depth
    camera = bpy.data.objects.new("OpenHK3D_Equirectangular", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    return camera


def configure_compositor(scene: bpy.types.Scene) -> bpy.types.CompositorNodeOutputFile:
    if hasattr(scene, "node_tree"):
        scene.use_nodes = True
        node_tree = scene.node_tree
    else:
        node_tree = bpy.data.node_groups.new(
            "OpenHK3D_Panorama_Compositor", "CompositorNodeTree"
        )
        scene.compositing_node_group = node_tree
    nodes = node_tree.nodes
    nodes.clear()
    render_layers = nodes.new("CompositorNodeRLayers")
    depth_output = nodes.new("CompositorNodeOutputFile")
    depth_output.name = "GT_RANGE_DEPTH"
    if hasattr(depth_output, "file_output_items"):
        item = depth_output.file_output_items.new("FLOAT", "Depth")
        item.override_node_format = True
        item.format.file_format = "OPEN_EXR"
        item.format.color_depth = "32"
        item.format.exr_codec = "ZIP"
        item.save_as_render = False
        depth_output.file_name = "depth"
        node_tree.links.new(render_layers.outputs["Depth"], depth_output.inputs["Depth"])
    else:
        depth_output.format.file_format = "OPEN_EXR"
        depth_output.format.color_mode = "BW"
        depth_output.format.color_depth = "32"
        depth_output.format.exr_codec = "ZIP"
        depth_output.file_slots[0].path = "depth"
        node_tree.links.new(render_layers.outputs["Depth"], depth_output.inputs[0])
    return depth_output


def set_pose(camera: bpy.types.Object, pose: Dict[str, object]) -> Dict[str, object]:
    camera.location = tuple(float(value) for value in pose["position_world_xyz"])
    elevation = math.radians(float(pose["elevation_degrees"]))
    azimuth = math.radians(float(pose["azimuth_degrees"]))
    source_perspective_forward = Vector(
        (
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
            math.cos(elevation) * math.cos(azimuth),
        )
    )
    forward = Vector((math.sin(azimuth), 0.0, math.cos(azimuth)))
    world_up = Vector((0.0, 1.0, 0.0))
    camera_z = -forward
    camera_x = world_up.cross(camera_z).normalized()
    rotation = Matrix((camera_x, world_up, camera_z)).transposed()
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = rotation.to_quaternion()
    matrix = [[float(value) for value in row] for row in camera.matrix_world]
    return {
        "center_ray_world_xyz": [float(value) for value in forward],
        "source_perspective_center_ray_world_xyz": [
            float(value) for value in source_perspective_forward
        ],
        "camera_to_world_matrix": matrix,
        "orientation_policy": "world_up_yaw_only",
        "source_perspective_elevation_ignored_for_panorama_rotation": True,
        "world_up_axis": "+Y",
        "camera_forward_axis": "-Z",
    }


def inspect_depth_exr(
    scene: bpy.types.Scene, path: Path, width: int, height: int
) -> Dict[str, object]:
    image_input = oiio.ImageInput.open(str(path))
    if image_input is None:
        raise RuntimeError("OpenImageIO cannot open depth EXR: {}".format(path))
    try:
        spec = image_input.spec()
        image_width, image_height = int(spec.width), int(spec.height)
        if [image_width, image_height] != [width, height]:
            raise RuntimeError(
                "Depth EXR resolution {}x{} does not match {}x{}".format(
                    image_width, image_height, width, height
                )
            )
        channels = int(spec.nchannels)
        values = np.asarray(image_input.read_image(format=oiio.FLOAT), dtype=np.float32)
        if values.size != width * height * channels:
            raise RuntimeError("Depth EXR has an invalid pixel buffer")
        depth = values.reshape(-1, channels)[:, 0]
    finally:
        image_input.close()
    valid = np.isfinite(depth) & (depth >= scene.camera.data.clip_start) & (
        depth < scene.camera.data.clip_end
    )
    valid_depth = depth[valid]
    if valid_depth.size == 0:
        raise RuntimeError("Rendered panorama contains no valid depth pixels")
    return {
        "pixels": int(depth.size),
        "loaded_channels": int(channels),
        "channel_names": list(spec.channelnames),
        "valid_pixels": int(valid_depth.size),
        "valid_ratio": float(valid_depth.size / depth.size),
        "minimum": float(valid_depth.min()),
        "median": float(np.median(valid_depth)),
        "p95": float(np.percentile(valid_depth, 95)),
        "maximum": float(valid_depth.max()),
    }


def render_pose(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    depth_output: bpy.types.CompositorNodeOutputFile,
    output_root: Path,
    pose_number: int,
    pose: Dict[str, object],
    width: int,
    height: int,
) -> Dict[str, object]:
    pose_root = output_root / "poses" / "{:03d}".format(pose_number)
    pose_root.mkdir(parents=True, exist_ok=False)
    extrinsics = set_pose(camera, pose)
    rgb_path = pose_root / "rgb.png"
    scene.render.filepath = str(rgb_path)
    if hasattr(depth_output, "directory"):
        depth_output.directory = str(pose_root)
    else:
        depth_output.base_path = str(pose_root)
    scene.frame_set(1)
    bpy.ops.render.render(write_still=True)
    candidates = sorted(pose_root.glob("depth*.exr"))
    if len(candidates) != 1:
        raise RuntimeError("Expected one depth EXR; found {}".format(len(candidates)))
    depth_path = pose_root / "depth.exr"
    os.replace(str(candidates[0]), str(depth_path))
    if not rgb_path.is_file() or not depth_path.is_file():
        raise RuntimeError("Panorama render did not create RGB and depth outputs")
    depth_stats = inspect_depth_exr(scene, depth_path, width, height)
    metadata = {
        "pose_number": pose_number,
        "pose_index": pose["pose_index"],
        "position_world_xyz": pose["position_world_xyz"],
        "elevation_degrees": pose["elevation_degrees"],
        "azimuth_degrees": pose["azimuth_degrees"],
        "extrinsics": extrinsics,
        "camera_model": {
            "projection": "equirectangular",
            "resolution": [width, height],
            "longitude_range_radians": [-math.pi, math.pi],
            "latitude_range_radians": [-math.pi / 2.0, math.pi / 2.0],
            "orientation_policy": "world_up_yaw_only",
            "rgb": "rgb.png",
            "rgb_alpha_semantics": "zero_for_unobserved_background",
            "depth": "depth.exr",
            "depth_channel": "V (loaded through Blender as the first pixel channel)",
            "depth_representation": "radial_distance_from_camera_center",
            "depth_units": "MAGICIAN_world_units",
        },
        "depth_statistics": depth_stats,
        "files": {
            "rgb.png": {"bytes": rgb_path.stat().st_size, "sha256": _sha256(rgb_path)},
            "depth.exr": {
                "bytes": depth_path.stat().st_size,
                "sha256": _sha256(depth_path),
            },
        },
    }
    _write_json(pose_root / "pose.json", metadata)
    return metadata


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    output_root = Path(config["output_dir"]).resolve()
    obj_path = Path(config["obj_path"]).resolve()
    if config_path.parent != output_root or not obj_path.is_file():
        raise RuntimeError("Render config or source OBJ is outside the expected location")

    capabilities = require_runtime_capabilities()
    imported = clear_and_import(obj_path)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = int(config["width"])
    scene.render.resolution_y = int(config["height"])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15
    scene.render.film_transparent = True
    scene.cycles.samples = int(config["samples"])
    scene.cycles.use_denoising = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    bpy.context.view_layer.use_pass_z = True
    compute_device = configure_cycles(scene, config["compute_device"])
    configure_world(scene)
    camera = configure_camera(scene, float(config["max_depth"]))
    depth_output = configure_compositor(scene)

    rendered = []
    for pose_number, pose in enumerate(config["poses"]):
        rendered.append(
            render_pose(
                scene,
                camera,
                depth_output,
                output_root,
                pose_number,
                pose,
                int(config["width"]),
                int(config["height"]),
            )
        )
    manifest = {
        "schema_version": config["schema_version"],
        "status": "passed",
        "batch_id": config["batch_id"],
        "scope_fingerprint_algorithm": config["scope_fingerprint_algorithm"],
        "scope_fingerprint": config["scope_fingerprint"],
        "scope": config["scope"],
        "runtime_capabilities": capabilities,
        "compute_device": compute_device,
        "source_import": imported,
        "rendered_poses": rendered,
        "quality_gates": {
            "configured_start_poses_rendered": len(rendered) == len(config["poses"]),
            "all_rgb_and_depth_files_present": True,
            "all_depth_maps_nonempty": all(
                pose["depth_statistics"]["valid_pixels"] > 0 for pose in rendered
            ),
        },
        "limitations": [
            "This is a GT-mesh panorama data pilot, not panoramic MAGICIAN planning.",
            "The source assembly seam remains diagnostic until a continuity tolerance is approved.",
            "RGBA alpha marks unobserved background; depth EXR background values are invalid.",
        ],
    }
    _write_json(output_root / "panorama-manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
