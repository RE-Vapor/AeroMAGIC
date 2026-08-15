#!/usr/bin/env python3
"""Assemble adjacent OpenHK3D tiles into an incrementally extensible scene.

The Python side deliberately performs the OBJ/MTL merge itself.  This keeps
vertex, face, UV, normal, and material assignments deterministic.  Blender is
then used headlessly for an independent import smoke test and for camera-grid
occupancy classification.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


MANIFEST_NAME = "assembly-manifest.json"
MANIFEST_SCHEMA = "openhk3d-assembly-v1"
TILE_OBJ_RE = re.compile(r"^Tile_([+-]\d+)_([+-]\d+)\.obj$", re.IGNORECASE)
TEXTURE_DIRECTIVES = {
    "map_ka",
    "map_kd",
    "map_ks",
    "map_ns",
    "map_d",
    "bump",
    "disp",
    "decal",
    "refl",
    "norm",
}


class AssemblyError(RuntimeError):
    """Raised when an input would make the assembly ambiguous or unsafe."""


@dataclass(frozen=True, order=True)
class GridCoord:
    x: int
    y: int

    def manhattan_distance(self, other: "GridCoord") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def as_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y}


@dataclass(frozen=True)
class Bounds:
    minimum: Tuple[float, float, float]
    maximum: Tuple[float, float, float]

    @classmethod
    def empty(cls) -> "Bounds":
        inf = float("inf")
        return cls((inf, inf, inf), (-inf, -inf, -inf))

    def include(self, point: Sequence[float]) -> "Bounds":
        return Bounds(
            tuple(min(self.minimum[i], float(point[i])) for i in range(3)),
            tuple(max(self.maximum[i], float(point[i])) for i in range(3)),
        )

    def union(self, other: "Bounds") -> "Bounds":
        return Bounds(
            tuple(min(self.minimum[i], other.minimum[i]) for i in range(3)),
            tuple(max(self.maximum[i], other.maximum[i]) for i in range(3)),
        )

    def center(self) -> Tuple[float, float, float]:
        return tuple((self.minimum[i] + self.maximum[i]) / 2.0 for i in range(3))

    def span(self) -> Tuple[float, float, float]:
        return tuple(self.maximum[i] - self.minimum[i] for i in range(3))

    def padded(self, padding: float) -> "Bounds":
        return Bounds(
            tuple(value - padding for value in self.minimum),
            tuple(value + padding for value in self.maximum),
        )

    def as_dict(self) -> Dict[str, List[float]]:
        return {
            "minimum": [round(value, 9) for value in self.minimum],
            "maximum": [round(value, 9) for value in self.maximum],
        }


@dataclass(frozen=True)
class SharedTransform:
    scale: float
    origin: Tuple[float, float, float]

    def point(self, value: Sequence[float]) -> Tuple[float, float, float]:
        # OpenHK3D source: X/Y horizontal and Z vertical.
        # MAGICIAN convention used by the existing tile-7 asset: Y vertical.
        x, y, z = value
        ox, oy, oz = self.origin
        return (
            self.scale * (x - ox),
            self.scale * (z - oz),
            -self.scale * (y - oy),
        )

    def normal(self, value: Sequence[float]) -> Tuple[float, float, float]:
        x, y, z = value
        return (x, z, -y)

    def bounds(self, source: Bounds) -> Bounds:
        result = Bounds.empty()
        for x in (source.minimum[0], source.maximum[0]):
            for y in (source.minimum[1], source.maximum[1]):
                for z in (source.minimum[2], source.maximum[2]):
                    result = result.include(self.point((x, y, z)))
        return result

    def as_dict(self) -> Dict[str, object]:
        return {
            "scale": self.scale,
            "origin_source_xyz": list(self.origin),
            "mapping": [
                "magic_x = scale * (source_x - origin_x)",
                "magic_y = scale * (source_z - origin_z)",
                "magic_z = -scale * (source_y - origin_y)",
            ],
        }


@dataclass
class ObjStats:
    bounds: Bounds
    vertices: int
    texture_vertices: int
    normals: int
    faces: int
    material_libraries: List[str]
    used_materials: List[str]

    def as_dict(self) -> Dict[str, object]:
        return {
            "bounds": self.bounds.as_dict(),
            "vertices": self.vertices,
            "texture_vertices": self.texture_vertices,
            "normals": self.normals,
            "faces": self.faces,
            "material_libraries": self.material_libraries,
            "used_materials": self.used_materials,
        }


@dataclass
class Asset:
    name: str
    root: Path
    obj_path: Path
    stats: ObjStats
    members: List[Tuple[str, GridCoord]]
    already_transformed: bool


def _tokens(value: str) -> List[str]:
    try:
        return shlex.split(value, comments=False, posix=True)
    except ValueError as exc:
        raise AssemblyError("Cannot parse quoted path: {!r}".format(value)) from exc


def _safe_relative(root: Path, raw_path: str, purpose: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise AssemblyError("{} uses an absolute path: {}".format(purpose, raw_path))
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise AssemblyError("{} escapes its asset directory: {}".format(purpose, raw_path)) from exc
    if not resolved.is_file():
        raise AssemblyError("{} does not exist: {}".format(purpose, resolved))
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not clean:
        raise AssemblyError("Name has no filesystem-safe characters: {!r}".format(value))
    return clean


def parse_grid_from_obj_name(name: str) -> GridCoord:
    match = TILE_OBJ_RE.match(name)
    if not match:
        raise AssemblyError("Not an OpenHK3D tile OBJ name: {}".format(name))
    return GridCoord(int(match.group(1)), int(match.group(2)))


def _find_tile_obj(root: Path) -> Path:
    candidates = sorted(path for path in root.glob("*.obj") if TILE_OBJ_RE.match(path.name))
    if len(candidates) != 1:
        raise AssemblyError(
            "Expected exactly one Tile_<x>_<y>.obj in {}; found {}".format(root, len(candidates))
        )
    return candidates[0]


def _parse_indexed_path(line: str, keyword: str) -> List[str]:
    remainder = line[len(keyword) :].strip()
    values = _tokens(remainder)
    if not values:
        raise AssemblyError("{} directive has no path".format(keyword))
    return values


def scan_obj(path: Path) -> ObjStats:
    bounds = Bounds.empty()
    counts = {"v": 0, "vt": 0, "vn": 0, "f": 0}
    libraries: List[str] = []
    materials = set()
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            keyword = parts[0].lower()
            if keyword == "v":
                if len(parts) < 4:
                    raise AssemblyError("{}:{} has an invalid vertex".format(path, line_number))
                try:
                    point = tuple(float(value) for value in parts[1:4])
                except ValueError as exc:
                    raise AssemblyError("{}:{} has a non-numeric vertex".format(path, line_number)) from exc
                bounds = bounds.include(point)
                counts["v"] += 1
            elif keyword in counts:
                counts[keyword] += 1
            elif keyword == "mtllib":
                libraries.extend(_parse_indexed_path(stripped, parts[0]))
            elif keyword == "usemtl":
                material = stripped[len(parts[0]) :].strip()
                if material:
                    materials.add(material)
    if counts["v"] == 0 or counts["f"] == 0:
        raise AssemblyError("OBJ has no usable mesh: {}".format(path))
    if not libraries:
        raise AssemblyError("OBJ does not declare an MTL library: {}".format(path))
    return ObjStats(
        bounds=bounds,
        vertices=counts["v"],
        texture_vertices=counts["vt"],
        normals=counts["vn"],
        faces=counts["f"],
        material_libraries=libraries,
        used_materials=sorted(materials),
    )


def _load_raw_tile(root: Path) -> Asset:
    obj = _find_tile_obj(root)
    coord = parse_grid_from_obj_name(obj.name)
    return Asset(
        name=root.name,
        root=root,
        obj_path=obj,
        stats=scan_obj(obj),
        members=[(root.name, coord)],
        already_transformed=False,
    )


def _load_assembled_scene(root: Path) -> Tuple[Asset, SharedTransform, Dict[str, object]]:
    manifest_path = root / MANIFEST_NAME
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise AssemblyError("Unsupported assembly manifest: {}".format(manifest.get("schema_version")))
    obj_name = manifest.get("output", {}).get("obj")
    if not isinstance(obj_name, str):
        raise AssemblyError("Assembly manifest has no output OBJ")
    obj = _safe_relative(root, obj_name, "assembly OBJ")
    transform_data = manifest.get("shared_transform", {})
    try:
        transform = SharedTransform(
            scale=float(transform_data["scale"]),
            origin=tuple(float(v) for v in transform_data["origin_source_xyz"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AssemblyError("Assembly manifest has an invalid shared transform") from exc
    members = []
    for item in manifest.get("members", []):
        try:
            members.append(
                (
                    str(item["name"]),
                    GridCoord(int(item["grid"]["x"]), int(item["grid"]["y"])),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AssemblyError("Assembly manifest has an invalid member entry") from exc
    if not members:
        raise AssemblyError("Assembly manifest has no tile members")
    return (
        Asset(
            name=root.name,
            root=root,
            obj_path=obj,
            stats=scan_obj(obj),
            members=members,
            already_transformed=True,
        ),
        transform,
        manifest,
    )


def load_base(root: Path) -> Tuple[Asset, Optional[SharedTransform], Optional[Dict[str, object]]]:
    if (root / MANIFEST_NAME).is_file():
        return _load_assembled_scene(root)
    return _load_raw_tile(root), None, None


def resolve_asset_path(dataset_root: Path, selector: str) -> Path:
    candidate = Path(selector).expanduser()
    if not candidate.is_absolute():
        candidate = dataset_root / candidate
    candidate = candidate.resolve()
    if not candidate.is_dir():
        raise AssemblyError("Asset directory does not exist: {}".format(candidate))
    return candidate


def discover_tiles(dataset_root: Path) -> Dict[GridCoord, Tuple[str, Path]]:
    result: Dict[GridCoord, Tuple[str, Path]] = {}
    for directory in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        matches = [path for path in directory.glob("*.obj") if TILE_OBJ_RE.match(path.name)]
        if len(matches) != 1:
            continue
        coord = parse_grid_from_obj_name(matches[0].name)
        if coord in result:
            raise AssemblyError("Duplicate grid coordinate {} in dataset".format(coord))
        result[coord] = (directory.name, directory)
    return result


def adjacent_tiles(dataset_root: Path, base_selector: str) -> Dict[str, object]:
    base_root = resolve_asset_path(dataset_root, base_selector)
    base, _, _ = load_base(base_root)
    catalog = discover_tiles(dataset_root)
    member_coords = {coord for _, coord in base.members}
    neighbors = []
    for coord, (name, path) in sorted(catalog.items()):
        adjacent_to = [member_name for member_name, member in base.members if coord.manhattan_distance(member) == 1]
        if adjacent_to and coord not in member_coords:
            neighbors.append(
                {
                    "name": name,
                    "path": str(path),
                    "grid": coord.as_dict(),
                    "adjacent_to": adjacent_to,
                }
            )
    return {
        "base": base.name,
        "members": [{"name": name, "grid": coord.as_dict()} for name, coord in base.members],
        "neighbors": neighbors,
    }


def _material_libraries(asset: Asset) -> List[Path]:
    return [
        _safe_relative(asset.root, library, "OBJ material library")
        for library in asset.stats.material_libraries
    ]


def _resolve_texture(mtl_root: Path, arguments: str, directive: str) -> Tuple[str, Path]:
    tokens = _tokens(arguments)
    if not tokens:
        raise AssemblyError("{} has no texture operand".format(directive))
    # Prefer the longest existing suffix. This supports filenames containing
    # spaces without trying to reinterpret every vendor-specific MTL option.
    for start in range(len(tokens)):
        raw_path = " ".join(tokens[start:])
        try:
            resolved = _safe_relative(mtl_root, raw_path, "MTL texture")
        except AssemblyError:
            continue
        options = " ".join(tokens[:start])
        return options, resolved
    raise AssemblyError("Cannot resolve {} texture operand: {}".format(directive, arguments))


def merge_materials(assets: Sequence[Asset], output_root: Path, output_mtl: Path) -> Dict[str, Dict[str, str]]:
    material_maps: Dict[str, Dict[str, str]] = {}
    output_lines = ["# Deterministically assembled by tools/openhk3d_assemble.py\n"]
    seen_output_names = set()
    copied_texture_targets: Dict[str, Path] = {}
    for asset in assets:
        asset_slug = _slug(asset.name)
        mapping: Dict[str, str] = {}
        for library in _material_libraries(asset):
            with library.open("r", encoding="utf-8", errors="strict") as stream:
                for line_number, raw_line in enumerate(stream, 1):
                    stripped = raw_line.strip()
                    if not stripped or stripped.startswith("#"):
                        output_lines.append(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                        continue
                    keyword = stripped.split(maxsplit=1)[0]
                    lowered = keyword.lower()
                    remainder = stripped[len(keyword) :].strip()
                    if lowered == "newmtl":
                        if not remainder:
                            raise AssemblyError("{}:{} has an empty material name".format(library, line_number))
                        output_name = "{}__{}".format(asset_slug, _slug(remainder))
                        if output_name in seen_output_names:
                            raise AssemblyError("Output material collision: {}".format(output_name))
                        seen_output_names.add(output_name)
                        mapping[remainder] = output_name
                        output_lines.append("newmtl {}\n".format(output_name))
                    elif lowered in TEXTURE_DIRECTIVES:
                        options, texture = _resolve_texture(library.parent, remainder, keyword)
                        if lowered == "map_kd" and options:
                            raise AssemblyError(
                                "MAGICIAN expects map_Kd followed directly by a path; unsupported options in {}:{}"
                                .format(library, line_number)
                            )
                        relative_source = texture.relative_to(asset.root.resolve())
                        safe_relative_source = Path(*(_slug(part) for part in relative_source.parts))
                        destination_relative = Path("textures") / asset_slug / safe_relative_source
                        casefolded_target = destination_relative.as_posix().casefold()
                        previous_source = copied_texture_targets.get(casefolded_target)
                        if previous_source is not None and previous_source != texture:
                            raise AssemblyError(
                                "Case-insensitive output texture collision: {} and {}"
                                .format(previous_source, texture)
                            )
                        copied_texture_targets[casefolded_target] = texture
                        destination = output_root / destination_relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(texture), str(destination))
                        rendered_options = (options + " ") if options else ""
                        output_lines.append(
                            "{} {}{}\n".format(keyword, rendered_options, destination_relative.as_posix())
                        )
                    else:
                        output_lines.append(raw_line if raw_line.endswith("\n") else raw_line + "\n")
        missing = sorted(set(asset.stats.used_materials) - set(mapping))
        if missing:
            raise AssemblyError(
                "OBJ {} uses undefined materials: {}".format(asset.obj_path, ", ".join(missing))
            )
        material_maps[asset.name] = mapping
    with output_mtl.open("w", encoding="utf-8", newline="\n") as stream:
        stream.writelines(output_lines)
    return material_maps


def _remap_index(value: str, offset: int) -> str:
    if value == "":
        return value
    parsed = int(value)
    if parsed == 0:
        raise AssemblyError("OBJ indices are one-based; found zero")
    return str(parsed + offset) if parsed > 0 else str(parsed)


def _remap_element(value: str, offsets: Tuple[int, int, int]) -> str:
    fields = value.split("/")
    if len(fields) > 3:
        raise AssemblyError("Unsupported OBJ index tuple: {}".format(value))
    return "/".join(_remap_index(field, offsets[i]) for i, field in enumerate(fields))


def _format_float(value: float) -> str:
    if abs(value) < 5e-13:
        value = 0.0
    return "{:.9g}".format(value)


def merge_objects(
    assets: Sequence[Asset],
    transform: SharedTransform,
    material_maps: Dict[str, Dict[str, str]],
    output_obj: Path,
    output_mtl_name: str,
) -> ObjStats:
    offsets = [0, 0, 0]
    with output_obj.open("w", encoding="utf-8", newline="\n") as output:
        output.write("# Deterministically assembled by tools/openhk3d_assemble.py\n")
        output.write("mtllib {}\n".format(output_mtl_name))
        for asset in assets:
            output.write("\n# Begin asset {}\n".format(asset.name))
            prefix = _slug(asset.name)
            mapping = material_maps[asset.name]
            prior_offsets = tuple(offsets)
            with asset.obj_path.open("r", encoding="utf-8", errors="strict") as stream:
                for line_number, raw_line in enumerate(stream, 1):
                    stripped = raw_line.strip()
                    if not stripped:
                        output.write("\n")
                        continue
                    if stripped.startswith("#"):
                        output.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                        continue
                    parts = stripped.split()
                    keyword = parts[0]
                    lowered = keyword.lower()
                    if lowered == "mtllib":
                        continue
                    if lowered == "v":
                        if len(parts) < 4:
                            raise AssemblyError("{}:{} has an invalid vertex".format(asset.obj_path, line_number))
                        point = tuple(float(value) for value in parts[1:4])
                        mapped = point if asset.already_transformed else transform.point(point)
                        suffix = " " + " ".join(parts[4:]) if len(parts) > 4 else ""
                        output.write("v {}{}\n".format(" ".join(_format_float(v) for v in mapped), suffix))
                    elif lowered == "vn":
                        normal = tuple(float(value) for value in parts[1:4])
                        mapped_normal = normal if asset.already_transformed else transform.normal(normal)
                        output.write("vn {}\n".format(" ".join(_format_float(v) for v in mapped_normal)))
                    elif lowered == "usemtl":
                        original = stripped[len(keyword) :].strip()
                        try:
                            output.write("usemtl {}\n".format(mapping[original]))
                        except KeyError as exc:
                            raise AssemblyError(
                                "{}:{} uses undefined material {}".format(asset.obj_path, line_number, original)
                            ) from exc
                    elif lowered in {"o", "g"}:
                        names = parts[1:] or ["unnamed"]
                        output.write("{} {}\n".format(keyword, " ".join(prefix + "__" + _slug(v) for v in names)))
                    elif lowered in {"f", "l", "p"}:
                        remapped = [_remap_element(value, prior_offsets) for value in parts[1:]]
                        output.write("{} {}\n".format(keyword, " ".join(remapped)))
                    elif lowered in {"curv", "curv2", "surf", "trim", "hole", "scrv", "sp", "con"}:
                        raise AssemblyError(
                            "{}:{} uses unsupported free-form geometry {}"
                            .format(asset.obj_path, line_number, keyword)
                        )
                    else:
                        output.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
            offsets[0] += asset.stats.vertices
            offsets[1] += asset.stats.texture_vertices
            offsets[2] += asset.stats.normals
            output.write("# End asset {}\n".format(asset.name))
    return scan_obj(output_obj)


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _expanded_axis(
    old_min: float,
    old_max: float,
    old_count: int,
    wanted_min: float,
    wanted_max: float,
) -> Tuple[float, float, int, float]:
    if old_count <= 0 or old_max <= old_min:
        raise AssemblyError("Settings contain a non-positive grid extent")
    step = (old_max - old_min) / old_count
    new_min = min(old_min, wanted_min)
    new_max = max(old_max, wanted_max)
    new_count = max(1, int(math.ceil((new_max - new_min) / step)))
    # Preserve the lower bound and the original approximate step exactly; the
    # upper bound is allowed to expand slightly so cells do not get denser.
    new_max = new_min + new_count * step
    return new_min, new_max, new_count, step


def generate_settings(
    template: Dict[str, object],
    base_bounds: Bounds,
    output_bounds: Bounds,
    scene_padding: float,
) -> Dict[str, object]:
    result = copy.deepcopy(template)
    try:
        scene = result["scene"]
        camera = result["camera"]
        old_scene_min = [float(value) for value in scene["x_min"]]
        old_scene_max = [float(value) for value in scene["x_max"]]
        old_scene_counts = [int(scene[key]) for key in ("grid_l", "grid_w", "grid_h")]
        old_camera_min = [float(value) for value in camera["x_min"]]
        old_camera_max = [float(value) for value in camera["x_max"]]
        old_camera_counts = [int(camera[key]) for key in ("pose_l", "pose_w", "pose_h")]
        old_starts = camera["start_positions"]
    except (KeyError, TypeError, ValueError) as exc:
        raise AssemblyError("Settings template does not match MAGICIAN's scene/camera schema") from exc

    wanted_scene = output_bounds.padded(scene_padding)
    new_scene_min: List[float] = []
    new_scene_max: List[float] = []
    new_scene_counts: List[int] = []
    for axis in range(3):
        low, high, count, _ = _expanded_axis(
            old_scene_min[axis],
            old_scene_max[axis],
            old_scene_counts[axis],
            wanted_scene.minimum[axis],
            wanted_scene.maximum[axis],
        )
        new_scene_min.append(low)
        new_scene_max.append(high)
        new_scene_counts.append(count)

    # Carry the template's camera margin around the original base mesh over to
    # the enlarged mesh. Negative margins (intentional clipping) are retained.
    wanted_camera_min = []
    wanted_camera_max = []
    for axis in range(3):
        lower_margin = base_bounds.minimum[axis] - old_camera_min[axis]
        upper_margin = old_camera_max[axis] - base_bounds.maximum[axis]
        wanted_camera_min.append(output_bounds.minimum[axis] - lower_margin)
        wanted_camera_max.append(output_bounds.maximum[axis] + upper_margin)

    new_camera_min: List[float] = []
    new_camera_max: List[float] = []
    new_camera_counts: List[int] = []
    old_camera_steps: List[float] = []
    new_camera_steps: List[float] = []
    for axis in range(3):
        old_step = (old_camera_max[axis] - old_camera_min[axis]) / old_camera_counts[axis]
        low, high, count, _ = _expanded_axis(
            old_camera_min[axis],
            old_camera_max[axis],
            old_camera_counts[axis],
            wanted_camera_min[axis],
            wanted_camera_max[axis],
        )
        new_camera_min.append(low)
        new_camera_max.append(high)
        new_camera_counts.append(count)
        old_camera_steps.append(old_step)
        new_camera_steps.append((high - low) / count)

    remapped_starts = []
    for start in old_starts:
        if not isinstance(start, list) or len(start) != 5:
            raise AssemblyError("Each start position must contain five indices")
        remapped = list(start)
        for axis in range(3):
            old_world = old_camera_min[axis] + (float(start[axis]) + 0.5) * old_camera_steps[axis]
            index = int(math.floor((old_world - new_camera_min[axis]) / new_camera_steps[axis]))
            remapped[axis] = _clamp(index, 0, new_camera_counts[axis] - 1)
        remapped_starts.append(remapped)

    for key, value in zip(("grid_l", "grid_w", "grid_h"), new_scene_counts):
        scene[key] = value
    scene["x_min"] = [round(value, 9) for value in new_scene_min]
    scene["x_max"] = [round(value, 9) for value in new_scene_max]
    for key, value in zip(("pose_l", "pose_w", "pose_h"), new_camera_counts):
        camera[key] = value
    camera["x_min"] = [round(value, 9) for value in new_camera_min]
    camera["x_max"] = [round(value, 9) for value in new_camera_max]
    camera["start_positions"] = remapped_starts
    return result


class KDNode:
    __slots__ = ("axis", "left", "point", "right")

    def __init__(
        self,
        point: Tuple[float, float, float],
        axis: int,
        left: Optional["KDNode"],
        right: Optional["KDNode"],
    ) -> None:
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right


def _build_kd(points: List[Tuple[float, float, float]], depth: int = 0) -> Optional[KDNode]:
    if not points:
        return None
    axis = depth % 3
    points.sort(key=lambda point: point[axis])
    middle = len(points) // 2
    return KDNode(
        points[middle],
        axis,
        _build_kd(points[:middle], depth + 1),
        _build_kd(points[middle + 1 :], depth + 1),
    )


def _nearest_squared(node: Optional[KDNode], target: Sequence[float], best: float) -> float:
    if node is None:
        return best
    distance = sum((node.point[i] - target[i]) ** 2 for i in range(3))
    best = min(best, distance)
    delta = target[node.axis] - node.point[node.axis]
    first, second = (node.left, node.right) if delta < 0 else (node.right, node.left)
    best = _nearest_squared(first, target, best)
    if delta * delta < best:
        best = _nearest_squared(second, target, best)
    return best


def _reservoir_append(
    sample: List[Tuple[float, float, float]],
    point: Tuple[float, float, float],
    seen: int,
    limit: int,
) -> None:
    if len(sample) < limit:
        sample.append(point)
        return
    # Deterministic replacement, avoiding randomness in diagnostic artifacts.
    slot = int(hashlib.sha256("{}:{}:{}:{}".format(seen, *point).encode("utf-8")).hexdigest()[:16], 16) % seen
    if slot < limit:
        sample[slot] = point


def _boundary_points(
    asset: Asset,
    axis: int,
    boundary: float,
    band: float,
    limit: int,
) -> Tuple[List[Tuple[float, float, float]], int]:
    points: List[Tuple[float, float, float]] = []
    seen = 0
    with asset.obj_path.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            parts = line.split()
            if not parts or parts[0].lower() != "v" or len(parts) < 4:
                continue
            point = tuple(float(value) for value in parts[1:4])
            if abs(point[axis] - boundary) <= band:
                seen += 1
                _reservoir_append(points, point, seen, limit)
    return points, seen


def seam_diagnostic(
    left: Asset,
    right: Asset,
    transform: SharedTransform,
    band: float,
    sample_limit: int,
) -> Dict[str, object]:
    if left.already_transformed or right.already_transformed:
        return {
            "status": "not_applicable_incremental_base",
            "reason": "The assembled base no longer exposes member-specific boundary vertices.",
        }
    left_coord = left.members[0][1]
    right_coord = right.members[0][1]
    dx = right_coord.x - left_coord.x
    dy = right_coord.y - left_coord.y
    if abs(dx) + abs(dy) != 1:
        raise AssemblyError("Seam diagnostic requires directly adjacent tiles")
    axis = 0 if dx else 1
    if (dx or dy) > 0:
        first, second = left, right
    else:
        first, second = right, left
    first_edge = first.stats.bounds.maximum[axis]
    second_edge = second.stats.bounds.minimum[axis]
    boundary = (first_edge + second_edge) / 2.0
    first_points, first_seen = _boundary_points(first, axis, boundary, band, sample_limit)
    second_points, second_seen = _boundary_points(second, axis, boundary, band, sample_limit)
    distances: List[float] = []
    if first_points and second_points:
        tree = _build_kd(list(second_points))
        distances = [math.sqrt(_nearest_squared(tree, point, float("inf"))) * transform.scale for point in first_points]
    perpendicular = 1 - axis
    tangential_overlap = max(
        0.0,
        min(first.stats.bounds.maximum[perpendicular], second.stats.bounds.maximum[perpendicular])
        - max(first.stats.bounds.minimum[perpendicular], second.stats.bounds.minimum[perpendicular]),
    )
    result: Dict[str, object] = {
        "status": "diagnostic_only",
        "shared_source_axis": "x" if axis == 0 else "y",
        "source_axis_gap": second_edge - first_edge,
        "magician_axis_gap": (second_edge - first_edge) * transform.scale,
        "tangential_source_overlap": tangential_overlap,
        "band_source_units": band,
        "first_boundary_vertices": first_seen,
        "second_boundary_vertices": second_seen,
        "sample_limit_per_side": sample_limit,
    }
    if distances:
        ordered = sorted(distances)
        result["nearest_distance_magician_units"] = {
            "minimum": ordered[0],
            "median": statistics.median(ordered),
            "p95": ordered[min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)],
            "maximum": ordered[-1],
            "samples": len(ordered),
        }
    else:
        result["nearest_distance_magician_units"] = None
    return result


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise AssemblyError("Expected a JSON object in {}".format(path))
    return value


def _run_blender(
    blender: str,
    scene_root: Path,
    obj_name: str,
    occupancy_clearance: float,
) -> None:
    backend = Path(__file__).with_name("openhk3d_blender_validate.py")
    command = [
        blender,
        "--background",
        "--factory-startup",
        "--python",
        str(backend),
        "--",
        "--scene-dir",
        str(scene_root),
        "--obj",
        obj_name,
        "--occupancy-clearance",
        str(occupancy_clearance),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise AssemblyError("Headless Blender validation failed with exit code {}".format(completed.returncode))


def _run_magician_loader(python: str, scene_root: Path) -> None:
    smoke_test = Path(__file__).with_name("openhk3d_magician_smoke.py")
    command = [python, str(smoke_test), "--scene-dir", str(scene_root)]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise AssemblyError(
            "MAGICIAN loader smoke test failed with exit code {}".format(completed.returncode)
        )


def _write_occupied_pose(scene_root: Path) -> None:
    occupancy_path = scene_root / "occupied_pose.json"
    payload = _load_json(occupancy_path)
    try:
        import torch
    except ImportError as exc:
        raise AssemblyError(
            "PyTorch is required to convert occupied_pose.json to occupied_pose.pt; run in the MAGICIAN environment"
        ) from exc
    indices = payload.get("X_idx")
    occupied = payload.get("occupied")
    if not isinstance(indices, list) or not isinstance(occupied, list) or len(indices) != len(occupied):
        raise AssemblyError("Blender produced an invalid occupied_pose.json")
    torch.save(
        {
            "X_idx": torch.tensor(indices, dtype=torch.long),
            "occupied": torch.tensor(occupied, dtype=torch.bool),
        },
        scene_root / "occupied_pose.pt",
    )


def _verify_output(output_root: Path, obj_name: str, mtl_name: str, expected: ObjStats) -> Dict[str, object]:
    actual = scan_obj(output_root / obj_name)
    for field in ("vertices", "texture_vertices", "normals", "faces"):
        if getattr(actual, field) != getattr(expected, field):
            raise AssemblyError(
                "Output {} count changed: expected {}, found {}".format(
                    field, getattr(expected, field), getattr(actual, field)
                )
            )
    if actual.used_materials != expected.used_materials:
        raise AssemblyError("Output material assignments do not match the merged inputs")
    if actual.material_libraries != [mtl_name]:
        raise AssemblyError("Output OBJ does not reference exactly its generated MTL")
    _safe_relative(output_root, mtl_name, "output MTL")
    texture_count = 0
    with (output_root / mtl_name).open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            stripped = raw_line.strip()
            if not stripped:
                continue
            keyword = stripped.split(maxsplit=1)[0]
            if keyword.lower() in TEXTURE_DIRECTIVES:
                remainder = stripped[len(keyword) :].strip()
                _, texture = _resolve_texture(output_root, remainder, keyword)
                texture.relative_to(output_root.resolve())
                texture_count += 1
    return {
        "status": "passed",
        "counts": actual.as_dict(),
        "resolved_texture_directives": texture_count,
    }


def assemble(args: argparse.Namespace) -> Dict[str, object]:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise AssemblyError("Dataset root does not exist: {}".format(dataset_root))
    base_root = resolve_asset_path(dataset_root, args.base)
    add_root = resolve_asset_path(dataset_root, args.add)
    base, inherited_transform, inherited_manifest = load_base(base_root)
    added = _load_raw_tile(add_root)
    if {coord for _, coord in base.members} & {coord for _, coord in added.members}:
        raise AssemblyError("Added tile is already a member of the base scene")
    adjacent_members = [name for name, coord in base.members if coord.manhattan_distance(added.members[0][1]) == 1]
    if not adjacent_members:
        raise AssemblyError("Added tile is not edge-adjacent to any base-scene member")

    if inherited_transform is not None:
        transform = inherited_transform
    else:
        origin = tuple(args.origin) if args.origin else base.stats.bounds.center()
        transform = SharedTransform(scale=args.scale if args.scale is not None else 0.1, origin=origin)
    if inherited_transform is not None and args.origin:
        raise AssemblyError("--origin cannot override an incremental base scene's frozen transform")
    if (
        inherited_transform is not None
        and args.scale is not None
        and not math.isclose(args.scale, inherited_transform.scale)
    ):
        raise AssemblyError("--scale cannot override an incremental base scene's frozen transform")

    output_root = Path(args.output).expanduser().resolve()
    if output_root.exists():
        raise AssemblyError("Output already exists; choose a new directory: {}".format(output_root))
    output_root.parent.mkdir(parents=True, exist_ok=True)
    scene_name = _slug(args.scene_name or output_root.name)
    obj_name = scene_name + ".obj"
    mtl_name = scene_name + ".mtl"
    assets = [base, added]

    base_output_bounds = base.stats.bounds if base.already_transformed else transform.bounds(base.stats.bounds)
    added_output_bounds = transform.bounds(added.stats.bounds)
    output_bounds = base_output_bounds.union(added_output_bounds)

    settings_template_path = (
        Path(args.settings_template).expanduser().resolve()
        if args.settings_template
        else base.root / "settings.json"
    )
    if not settings_template_path.is_file():
        raise AssemblyError("A base MAGICIAN settings template is required: {}".format(settings_template_path))
    settings_template = _load_json(settings_template_path)
    settings = generate_settings(settings_template, base_output_bounds, output_bounds, args.scene_padding)

    with tempfile.TemporaryDirectory(prefix=".openhk3d-assemble-", dir=str(output_root.parent)) as temporary:
        staging = Path(temporary) / scene_name
        staging.mkdir()
        output_mtl = staging / mtl_name
        output_obj = staging / obj_name
        material_maps = merge_materials(assets, staging, output_mtl)
        merged_stats = merge_objects(assets, transform, material_maps, output_obj, mtl_name)
        expected = ObjStats(
            bounds=output_bounds,
            vertices=sum(asset.stats.vertices for asset in assets),
            texture_vertices=sum(asset.stats.texture_vertices for asset in assets),
            normals=sum(asset.stats.normals for asset in assets),
            faces=sum(asset.stats.faces for asset in assets),
            material_libraries=[mtl_name],
            used_materials=sorted(
                material_maps[asset.name][name]
                for asset in assets
                for name in asset.stats.used_materials
            ),
        )
        python_validation = _verify_output(staging, obj_name, mtl_name, expected)
        _write_json(staging / "settings.json", settings)
        seam = seam_diagnostic(base, added, transform, args.seam_band, args.seam_sample_limit)
        members = list(base.members) + list(added.members)
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "scene_name": scene_name,
            "members": [{"name": name, "grid": coord.as_dict()} for name, coord in members],
            "incremental_base": {
                "name": base.name,
                "manifest_sha256": _sha256(base.root / MANIFEST_NAME) if inherited_manifest is not None else None,
                "obj_sha256": _sha256(base.obj_path),
            },
            "added_tile": {
                "name": added.name,
                "adjacent_to": adjacent_members,
                "obj_sha256": _sha256(added.obj_path),
            },
            "shared_transform": transform.as_dict(),
            "output": {
                "obj": obj_name,
                "mtl": mtl_name,
                "settings": "settings.json",
                "settings_template_sha256": _sha256(settings_template_path),
                "occupied_pose": "occupied_pose.pt",
                "bounds": output_bounds.as_dict(),
                "counts": merged_stats.as_dict(),
            },
            "validation": {
                "python_structure_and_references": python_validation,
                "seam": seam,
                "blender": "pending",
                "magician_loader": "not_run",
            },
            "claims": {
                "assembly_structure": "validated_by_python",
                "seam_continuity": "diagnostic_only",
                "magician_compatibility": "unknown_until_loader_smoke_test",
            },
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        if not args.skip_blender:
            _run_blender(args.blender, staging, obj_name, args.occupancy_clearance)
            _write_occupied_pose(staging)
            blender_report = _load_json(staging / "validation_report.json")
            manifest["validation"]["blender"] = blender_report
            manifest["claims"]["assembly_structure"] = "python_and_blender_import_validated"
            if not args.skip_magician_loader:
                _run_magician_loader(args.magician_python, staging)
                manifest["validation"]["magician_loader"] = _load_json(
                    staging / "magician_loader_report.json"
                )
                manifest["claims"]["magician_compatibility"] = "loader_smoke_passed"
            else:
                manifest["validation"]["magician_loader"] = "skipped_by_request"
        else:
            manifest["output"]["occupied_pose"] = None
            manifest["validation"]["blender"] = "skipped_by_request"
            manifest["validation"]["magician_loader"] = "not_run_without_blender_gate"
        artifact_names = [obj_name, mtl_name, "settings.json"]
        if (staging / "occupied_pose.pt").is_file():
            artifact_names.append("occupied_pose.pt")
        manifest["output"]["sha256"] = {
            name: _sha256(staging / name) for name in artifact_names
        }
        _write_json(staging / MANIFEST_NAME, manifest)
        os.replace(str(staging), str(output_root))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="List edge-adjacent raw tiles")
    discover.add_argument("--dataset-root", required=True)
    discover.add_argument("--base", required=True, help="Raw tile or previously assembled scene")

    assemble_parser = subparsers.add_parser("assemble", help="Build one incremental N -> N+1 scene")
    assemble_parser.add_argument("--dataset-root", required=True)
    assemble_parser.add_argument("--base", required=True, help="Raw tile or previously assembled scene")
    assemble_parser.add_argument("--add", required=True, help="One raw edge-adjacent tile")
    assemble_parser.add_argument("--output", required=True, help="New output directory; must not exist")
    assemble_parser.add_argument("--scene-name")
    assemble_parser.add_argument("--settings-template")
    assemble_parser.add_argument(
        "--scale",
        type=float,
        help="Initial source-to-MAGICIAN scale (default 0.1); inherited for N -> N+1",
    )
    assemble_parser.add_argument("--origin", nargs=3, type=float, metavar=("X", "Y", "Z"))
    assemble_parser.add_argument("--scene-padding", type=float, default=0.05)
    assemble_parser.add_argument("--seam-band", type=float, default=0.5, help="Source-coordinate units")
    assemble_parser.add_argument("--seam-sample-limit", type=int, default=50000)
    assemble_parser.add_argument("--blender", default="blender", help="Blender 4.2 executable")
    assemble_parser.add_argument("--occupancy-clearance", type=float, default=0.2)
    assemble_parser.add_argument("--magician-python", default=sys.executable)
    assemble_parser.add_argument("--skip-magician-loader", action="store_true")
    assemble_parser.add_argument(
        "--skip-blender",
        action="store_true",
        help="Create an unvalidated intermediate without occupied_pose.pt",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "discover":
            result = adjacent_tiles(Path(args.dataset_root).expanduser().resolve(), args.base)
        else:
            if args.scale is not None and args.scale <= 0:
                raise AssemblyError("--scale must be positive")
            if args.scene_padding < 0 or args.seam_band < 0 or args.seam_sample_limit <= 0:
                raise AssemblyError("Padding/band must be non-negative and sample limit positive")
            result = assemble(args)
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    except (AssemblyError, OSError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
