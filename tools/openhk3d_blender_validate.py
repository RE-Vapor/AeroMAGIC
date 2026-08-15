#!/usr/bin/env python3
"""Blender 4.2 headless backend for OpenHK3D scene validation.

Run through ``openhk3d_assemble.py``.  This module intentionally depends only
on Blender's bundled Python modules.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--occupancy-clearance", required=True, type=float)
    return parser.parse_args(argv)


def blender_arguments() -> List[str]:
    try:
        separator = sys.argv.index("--")
    except ValueError:
        return []
    return sys.argv[separator + 1 :]


def parse_obj_geometry(path: Path) -> Tuple[List[Vector], List[Tuple[int, int, int]], int]:
    vertices: List[Vector] = []
    triangles: List[Tuple[int, int, int]] = []
    source_faces = 0
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            parts = raw_line.split()
            if not parts or parts[0].startswith("#"):
                continue
            keyword = parts[0].lower()
            if keyword == "v":
                if len(parts) < 4:
                    raise RuntimeError("{}:{} invalid vertex".format(path, line_number))
                vertices.append(Vector(tuple(float(value) for value in parts[1:4])))
            elif keyword == "f":
                if len(parts) < 4:
                    raise RuntimeError("{}:{} invalid face".format(path, line_number))
                face = []
                for element in parts[1:]:
                    raw_index = int(element.split("/", 1)[0])
                    if raw_index == 0:
                        raise RuntimeError("{}:{} zero OBJ index".format(path, line_number))
                    index = raw_index - 1 if raw_index > 0 else len(vertices) + raw_index
                    if index < 0 or index >= len(vertices):
                        raise RuntimeError("{}:{} out-of-range OBJ index".format(path, line_number))
                    face.append(index)
                source_faces += 1
                for offset in range(1, len(face) - 1):
                    triangles.append((face[0], face[offset], face[offset + 1]))
    if not vertices or not triangles:
        raise RuntimeError("OBJ has no geometry")
    return vertices, triangles, source_faces


def bbox(vertices: Sequence[Vector]) -> Dict[str, List[float]]:
    return {
        "minimum": [min(vertex[axis] for vertex in vertices) for axis in range(3)],
        "maximum": [max(vertex[axis] for vertex in vertices) for axis in range(3)],
    }


def normalized(value: Tuple[float, float, float]) -> Vector:
    result = Vector(value)
    result.normalize()
    return result


RAY_DIRECTIONS = [
    normalized((1.0, 0.173, 0.079)),
    normalized((0.137, 1.0, 0.211)),
    normalized((0.193, 0.113, 1.0)),
]


def ray_intersections(tree: BVHTree, point: Vector, direction: Vector) -> int:
    count = 0
    origin = point.copy()
    epsilon = 1e-6
    for _ in range(10000):
        location, _normal, _face_index, _distance = tree.ray_cast(origin, direction)
        if location is None:
            break
        count += 1
        origin = location + direction * epsilon
    else:
        raise RuntimeError("BVH ray exceeded the intersection safety limit")
    return count


def point_inside(tree: BVHTree, point: Vector) -> bool:
    votes = sum(1 for direction in RAY_DIRECTIONS if ray_intersections(tree, point, direction) % 2 == 1)
    return votes >= 2


def camera_positions(settings: Dict[str, object]):
    camera = settings["camera"]
    minimum = [float(value) for value in camera["x_min"]]
    maximum = [float(value) for value in camera["x_max"]]
    counts = [int(camera[key]) for key in ("pose_l", "pose_w", "pose_h")]
    if any(count <= 0 for count in counts):
        raise RuntimeError("Camera pose dimensions must be positive")
    steps = [(maximum[axis] - minimum[axis]) / counts[axis] for axis in range(3)]
    for x in range(counts[0]):
        for y in range(counts[1]):
            for z in range(counts[2]):
                index = [x, y, z]
                point = Vector(
                    tuple(minimum[axis] + (index[axis] + 0.5) * steps[axis] for axis in range(3))
                )
                yield index, point


def classify_occupancy(
    tree: BVHTree,
    settings: Dict[str, object],
    clearance: float,
) -> Tuple[Dict[str, object], Dict[Tuple[int, int, int], bool]]:
    indices = []
    flags = []
    lookup: Dict[Tuple[int, int, int], bool] = {}
    near_surface_count = 0
    inside_count = 0
    for index, point in camera_positions(settings):
        _nearest, _normal, _face_index, distance = tree.find_nearest(point)
        near_surface = distance is not None and distance <= clearance
        inside = point_inside(tree, point)
        occupied = bool(near_surface or inside)
        near_surface_count += int(near_surface)
        inside_count += int(inside)
        indices.append(index)
        flags.append(occupied)
        lookup[tuple(index)] = occupied
    return (
        {
            "schema_version": "magician-occupied-pose-json-v1",
            "classification": {
                "occupied_when": "within_clearance_or_majority_ray_parity_inside",
                "clearance": clearance,
                "ray_directions": [list(direction) for direction in RAY_DIRECTIONS],
            },
            "X_idx": indices,
            "occupied": flags,
            "summary": {
                "poses": len(indices),
                "occupied": sum(1 for value in flags if value),
                "near_surface": near_surface_count,
                "inside_by_ray_parity": inside_count,
            },
        },
        lookup,
    )


def smoke_import(path: Path) -> Dict[str, object]:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    result = bpy.ops.wm.obj_import(
        filepath=str(path),
        forward_axis="NEGATIVE_Z",
        up_axis="Y",
        use_split_objects=True,
        use_split_groups=False,
        validate_meshes=True,
    )
    if "FINISHED" not in result:
        raise RuntimeError("Blender OBJ importer did not finish: {}".format(result))
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("Blender imported no mesh objects")
    images = []
    missing_images = []
    for image in bpy.data.images:
        if image.source != "FILE":
            continue
        image_path = Path(bpy.path.abspath(image.filepath)).resolve()
        images.append(str(image_path))
        if not image_path.is_file():
            missing_images.append(str(image_path))
    if missing_images:
        raise RuntimeError("Blender reports missing texture images: {}".format(missing_images))
    return {
        "status": "passed",
        "blender_version": bpy.app.version_string,
        "mesh_objects": len(mesh_objects),
        "vertices_after_import": sum(len(obj.data.vertices) for obj in mesh_objects),
        "polygons_after_import": sum(len(obj.data.polygons) for obj in mesh_objects),
        "materials": len(bpy.data.materials),
        "file_images": len(images),
        "missing_images": [],
    }


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(blender_arguments() if argv is None else argv)
    if bpy.app.version[:2] != (4, 2):
        raise RuntimeError(
            "This validation contract requires Blender >=4.2,<4.3; found {}"
            .format(bpy.app.version_string)
        )
    scene_root = Path(args.scene_dir).resolve()
    obj_path = (scene_root / args.obj).resolve()
    if obj_path.parent != scene_root or not obj_path.is_file():
        raise RuntimeError("OBJ must be a direct child of --scene-dir")
    if args.occupancy_clearance < 0:
        raise RuntimeError("Occupancy clearance must be non-negative")
    settings_path = scene_root / "settings.json"
    with settings_path.open("r", encoding="utf-8") as stream:
        settings = json.load(stream)

    vertices, triangles, source_faces = parse_obj_geometry(obj_path)
    tree = BVHTree.FromPolygons(vertices, triangles, all_triangles=True, epsilon=0.0)
    if tree is None:
        raise RuntimeError("Blender could not construct a BVH")
    occupancy, occupancy_lookup = classify_occupancy(tree, settings, args.occupancy_clearance)
    import_report = smoke_import(obj_path)

    invalid_starts = []
    for start in settings["camera"]["start_positions"]:
        spatial = tuple(int(value) for value in start[:3])
        if occupancy_lookup.get(spatial, True):
            invalid_starts.append(start)
    report = {
        "status": "passed" if not invalid_starts else "failed",
        "source_geometry": {
            "vertices": len(vertices),
            "faces": source_faces,
            "triangles_for_bvh": len(triangles),
            "bounds": bbox(vertices),
        },
        "obj_import_smoke": import_report,
        "occupancy": occupancy["summary"],
        "start_positions": {
            "checked": len(settings["camera"]["start_positions"]),
            "occupied": invalid_starts,
        },
        "limitations": [
            "Ray parity is diagnostic for open/non-manifold meshes.",
            "This validates Blender import and camera occupancy, not the full MAGICIAN loader.",
        ],
    }
    write_json(scene_root / "occupied_pose.json", occupancy)
    write_json(scene_root / "validation_report.json", report)
    if invalid_starts:
        raise RuntimeError("One or more configured start positions are occupied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
