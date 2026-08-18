#!/usr/bin/env python3
"""Convert one HUGE-Bench mesh scene into MAGICIAN's scene-data layout.

The converter deliberately treats the OBJ mesh as geometry truth.  A 3DGS PLY
is audited and recorded, but is never substituted for MAGICIAN's mesh input.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import trimesh
from plyfile import PlyData


CONVERTER_VERSION = "1.0.0"
OFFICIAL_DATA_REVISION = "f9bed5c1da172aecd5e3942848ee9174599ec59a"

# HUGE publishes local ENU/UTM-relative data in metres with +Z up. MAGICIAN's
# PyTorch3D scenes are +Y up. This proper rotation maps (E, N, U) to
# (X, Y, Z) = (E, U, -N); det(R)=+1, so handedness is retained.
T_HUGE_TO_MAGIC = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def finite_list(values: np.ndarray) -> list[float]:
    return [float(v) for v in values.tolist()]


def resolve_obj_index(raw: int, count: int) -> int:
    if raw == 0:
        raise ValueError("OBJ index 0 is invalid")
    resolved = raw - 1 if raw > 0 else count + raw
    if not 0 <= resolved < count:
        raise ValueError(f"OBJ index {raw} resolves outside 0..{count - 1}")
    return resolved


def parse_face_vertex(token: str, n_vertices: int) -> int:
    return resolve_obj_index(int(token.split("/", 1)[0]), n_vertices)


def parse_obj(path: Path) -> dict[str, Any]:
    vertices: list[list[float]] = []
    normals: list[list[float]] = []
    triangles: list[list[int]] = []
    mtllibs: list[str] = []
    materials: set[str] = set()
    counts = {
        "vertices": 0,
        "texture_coordinates": 0,
        "normals": 0,
        "faces": 0,
        "triangles_after_fan": 0,
        "objects": 0,
        "groups": 0,
        "non_triangular_faces": 0,
        "negative_index_faces": 0,
        "faces_missing_any_uv": 0,
        "faces_missing_any_normal": 0,
    }

    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            kind = fields[0]
            try:
                if kind == "v":
                    if len(fields) < 4:
                        raise ValueError("vertex has fewer than 3 coordinates")
                    vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
                    counts["vertices"] += 1
                elif kind == "vt":
                    counts["texture_coordinates"] += 1
                elif kind == "vn":
                    if len(fields) < 4:
                        raise ValueError("normal has fewer than 3 coordinates")
                    normals.append([float(fields[1]), float(fields[2]), float(fields[3])])
                    counts["normals"] += 1
                elif kind == "f":
                    if len(fields) < 4:
                        raise ValueError("face has fewer than 3 vertices")
                    if any(part.split("/", 1)[0].startswith("-") for part in fields[1:]):
                        counts["negative_index_faces"] += 1
                    face_tokens = fields[1:]
                    face = [parse_face_vertex(part, len(vertices)) for part in face_tokens]
                    has_uv = []
                    has_normal = []
                    for token in face_tokens:
                        components = token.split("/")
                        token_has_uv = len(components) > 1 and bool(components[1])
                        token_has_normal = len(components) > 2 and bool(components[2])
                        has_uv.append(token_has_uv)
                        has_normal.append(token_has_normal)
                        if token_has_uv:
                            resolve_obj_index(int(components[1]), counts["texture_coordinates"])
                        if token_has_normal:
                            resolve_obj_index(int(components[2]), len(normals))
                    if not all(has_uv):
                        counts["faces_missing_any_uv"] += 1
                    if not all(has_normal):
                        counts["faces_missing_any_normal"] += 1
                    counts["faces"] += 1
                    if len(face) != 3:
                        counts["non_triangular_faces"] += 1
                    for idx in range(1, len(face) - 1):
                        triangles.append([face[0], face[idx], face[idx + 1]])
                    counts["triangles_after_fan"] += len(face) - 2
                elif kind == "mtllib":
                    mtllibs.extend(shlex.split(stripped[len("mtllib") :].strip()))
                elif kind == "usemtl":
                    materials.add(stripped[len("usemtl") :].strip())
                elif kind == "o":
                    counts["objects"] += 1
                elif kind == "g":
                    counts["groups"] += 1
            except (ValueError, IndexError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc

    verts = np.asarray(vertices, dtype=np.float64)
    norms = np.asarray(normals, dtype=np.float64).reshape((-1, 3))
    tris = np.asarray(triangles, dtype=np.int64).reshape((-1, 3))
    if len(verts) == 0 or len(tris) == 0:
        raise ValueError(f"OBJ must contain non-empty vertices and faces: {path}")
    if not np.isfinite(verts).all() or not np.isfinite(norms).all():
        raise ValueError(f"OBJ contains NaN or infinity: {path}")

    tri_xyz = verts[tris]
    double_areas = np.linalg.norm(
        np.cross(tri_xyz[:, 1] - tri_xyz[:, 0], tri_xyz[:, 2] - tri_xyz[:, 0]), axis=1
    )
    counts["degenerate_triangles"] = int(np.count_nonzero(double_areas <= 1e-12))

    abnormal_normals = 0
    zero_normals = 0
    if len(norms):
        normal_lengths = np.linalg.norm(norms, axis=1)
        zero_normals = int(np.count_nonzero(normal_lengths < 1e-6))
        abnormal_normals = int(
            np.count_nonzero((normal_lengths < 0.99) | (normal_lengths > 1.01))
        )
    counts["zero_normals"] = zero_normals
    counts["abnormal_normals"] = abnormal_normals

    return {
        "vertices": verts,
        "normals": norms,
        "triangles": tris,
        "counts": counts,
        "mtllibs": sorted(set(mtllibs)),
        "materials_used": sorted(materials),
        "aabb_min": verts.min(axis=0),
        "aabb_max": verts.max(axis=0),
    }


def transform_points(points: np.ndarray) -> np.ndarray:
    return points @ T_HUGE_TO_MAGIC[:3, :3].T + T_HUGE_TO_MAGIC[:3, 3]


def parse_frame_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    root = ET.parse(path).getroot()
    srs = root.findtext("SRS")
    origin_text = root.findtext("SRSOrigin")
    if not srs or not origin_text:
        raise ValueError(f"metadata must define SRS and SRSOrigin: {path}")
    origin = [float(part.strip()) for part in origin_text.split(",")]
    if len(origin) != 3 or not np.isfinite(origin).all():
        raise ValueError(f"invalid SRSOrigin in {path}: {origin_text!r}")
    return {"path": str(path), "srs": srs.strip(), "origin": origin}


def audit_ply_xyz(path: Path, chunk_size: int = 1_000_000) -> dict[str, Any]:
    ply = PlyData.read(str(path), mmap=True)
    try:
        vertex_element = ply["vertex"]
    except KeyError as exc:
        raise ValueError(f"PLY has no vertex element: {path}") from exc
    properties = [prop.name for prop in vertex_element.properties]
    missing = sorted({"x", "y", "z"} - set(properties))
    if missing:
        raise ValueError(f"PLY is missing coordinate properties {missing}: {path}")
    data = vertex_element.data
    if len(data) == 0:
        raise ValueError(f"PLY has zero vertices: {path}")
    xyz_min = np.full(3, np.inf, dtype=np.float64)
    xyz_max = np.full(3, -np.inf, dtype=np.float64)
    nonfinite = 0
    for start in range(0, len(data), chunk_size):
        chunk = data[start : start + chunk_size]
        xyz = np.column_stack((chunk["x"], chunk["y"], chunk["z"])).astype(
            np.float64, copy=False
        )
        finite = np.isfinite(xyz).all(axis=1)
        nonfinite += int((~finite).sum())
        if finite.any():
            finite_xyz = xyz[finite]
            xyz_min = np.minimum(xyz_min, finite_xyz.min(axis=0))
            xyz_max = np.maximum(xyz_max, finite_xyz.max(axis=0))
    if not np.isfinite(xyz_min).all() or nonfinite:
        raise ValueError(f"PLY has {nonfinite} non-finite vertices: {path}")
    magic_corners = transform_points(
        np.asarray(list(product(*zip(xyz_min, xyz_max))), dtype=np.float64)
    )
    return {
        "format": (
            "ascii"
            if ply.text
            else f"binary_{'little' if ply.byte_order == '<' else 'big'}_endian"
        ),
        "vertex_count": len(data),
        "properties": properties,
        "nonfinite_vertices": nonfinite,
        "aabb_huge": [finite_list(xyz_min), finite_list(xyz_max)],
        "aabb_magic": [
            finite_list(magic_corners.min(axis=0)),
            finite_list(magic_corners.max(axis=0)),
        ],
    }


def transform_obj(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rotation = T_HUGE_TO_MAGIC[:3, :3]
    with source.open("r", encoding="utf-8", errors="strict") as src, destination.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line_no, line in enumerate(src, 1):
            ending = "\n" if line.endswith(("\n", "\r")) else ""
            stripped = line.strip()
            fields = stripped.split()
            if fields and fields[0] in {"v", "vn"}:
                if len(fields) < 4:
                    raise ValueError(f"{source}:{line_no}: malformed {fields[0]} record")
                xyz = np.array([float(fields[1]), float(fields[2]), float(fields[3])])
                if fields[0] == "v":
                    xyz = transform_points(xyz.reshape(1, 3))[0]
                else:
                    xyz = rotation @ xyz
                    length = np.linalg.norm(xyz)
                    if length > 0:
                        xyz /= length
                suffix = fields[4:]
                values = [fields[0], *(format(float(v), ".12g") for v in xyz), *suffix]
                dst.write(" ".join(values) + ending)
            else:
                dst.write(line.replace("\r\n", "\n").replace("\r", "\n"))


def safe_output_relative(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe external material path: {path_text!r}")
    return path


def parse_mtl_textures(path: Path) -> tuple[int, list[str], list[str]]:
    material_count = 0
    textures: list[str] = []
    diffuse_textures: list[str] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = shlex.split(stripped)
            if not fields:
                continue
            if fields[0] == "newmtl":
                material_count += 1
            if fields[0].lower().startswith("map_") or fields[0].lower() in {
                "bump",
                "disp",
                "decal",
                "refl",
            }:
                if len(fields) < 2:
                    raise ValueError(f"empty texture reference in {path}")
                textures.append(fields[-1])
                if fields[0].lower() == "map_kd":
                    diffuse_textures.append(fields[-1])
    return material_count, textures, diffuse_textures


def copy_material_assets(
    source_obj: Path, output_root: Path, mtllibs: Iterable[str]
) -> dict[str, Any]:
    copied: list[Path] = []
    texture_paths: list[str] = []
    diffuse_texture_paths: list[str] = []
    material_count = 0
    for mtl_text in mtllibs:
        relative_mtl = safe_output_relative(mtl_text)
        source_mtl = source_obj.parent / relative_mtl
        if not source_mtl.is_file():
            raise FileNotFoundError(f"OBJ references missing MTL: {source_mtl}")
        output_mtl = output_root / relative_mtl
        output_mtl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_mtl, output_mtl)
        copied.append(output_mtl)
        n_materials, refs, diffuse_refs = parse_mtl_textures(source_mtl)
        material_count += n_materials
        for texture_text in refs:
            relative_texture = safe_output_relative(texture_text)
            source_texture = source_mtl.parent / relative_texture
            if not source_texture.is_file():
                raise FileNotFoundError(f"MTL references missing texture: {source_texture}")
            output_texture = output_mtl.parent / relative_texture
            output_texture.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_texture, output_texture)
            copied.append(output_texture)
            texture_paths.append(str(output_texture.relative_to(output_root)))
            if texture_text in diffuse_refs:
                diffuse_texture_paths.append(str(output_texture.relative_to(output_root)))
    return {
        "copied": copied,
        "material_count": material_count,
        "texture_paths": sorted(set(texture_paths)),
        "diffuse_texture_paths": sorted(set(diffuse_texture_paths)),
    }


def read_landmarks(env_dir: Path) -> tuple[Path | None, np.ndarray, list[str]]:
    candidates = [
        env_dir / "location_gen" / "landmark_merged_s.txt",
        env_dir / "location_gen" / "landmark_merged.txt",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        points: list[list[float]] = []
        labels: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                fields = stripped.split("\t")
                if len(fields) < 4:
                    fields = stripped.split(maxsplit=3)
                if len(fields) < 4:
                    raise ValueError(f"{path}:{line_no}: expected x y z label")
                points.append([float(fields[0]), float(fields[1]), float(fields[2])])
                labels.append(fields[3])
        if points:
            return path, np.asarray(points, dtype=np.float64), labels
    return None, np.empty((0, 3), dtype=np.float64), []


def grid_count(span: float, target_step: float, minimum: int = 1) -> int:
    return max(minimum, int(math.ceil(span / target_step)))


def grid_centers(bounds_min: np.ndarray, bounds_max: np.ndarray, counts: np.ndarray) -> np.ndarray:
    axes = [
        bounds_min[i] + (np.arange(int(counts[i]), dtype=np.float64) + 0.5) * (
            bounds_max[i] - bounds_min[i]
        )
        / counts[i]
        for i in range(3)
    ]
    return np.asarray(list(product(*axes)), dtype=np.float64)


def classify_occupied(
    vertices: np.ndarray,
    triangles: np.ndarray,
    points: np.ndarray,
    clearance: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False, validate=False)
    _, distances, _ = trimesh.proximity.closest_point(mesh, points)
    occupied = distances <= clearance
    watertight = bool(mesh.is_watertight)
    if watertight:
        occupied |= mesh.contains(points)
    return occupied.astype(bool), distances, watertight


def pose_direction(elevation_deg: float, azimuth_deg: float) -> np.ndarray:
    elev = math.radians(elevation_deg)
    azim = math.radians(azimuth_deg)
    return np.array(
        [math.cos(elev) * math.sin(azim), math.sin(elev), math.cos(elev) * math.cos(azim)],
        dtype=np.float64,
    )


def best_orientation(
    position: np.ndarray, target: np.ndarray, n_elev: int, n_azim: int
) -> tuple[int, int]:
    desired = target - position
    desired /= np.linalg.norm(desired)
    best: tuple[float, int, int] | None = None
    for i_elev in range(n_elev):
        elevation = -90.0 + 180.0 * (1 + i_elev) / (n_elev + 1)
        for i_azim in range(n_azim):
            azimuth = 360.0 * i_azim / n_azim
            score = float(np.dot(pose_direction(elevation, azimuth), desired))
            candidate = (score, -i_elev, -i_azim)
            if best is None or candidate > best:
                best = candidate
    assert best is not None
    return -best[1], -best[2]


def choose_start_positions(
    counts: np.ndarray,
    centers: np.ndarray,
    occupied: np.ndarray,
    target: np.ndarray,
    n_elev: int,
    n_azim: int,
) -> list[list[int]]:
    l, w, h = (int(v) for v in counts)
    preferred = [
        (0, w - 1, 0),
        (l - 1, w - 1, 0),
        (l - 1, w - 1, h - 1),
        (0, w - 1, h - 1),
        (l // 2, w - 1, 0),
    ]
    all_indices = np.asarray(list(product(range(l), range(w), range(h))), dtype=np.int64)
    free_indices = all_indices[~occupied]
    if len(free_indices) < 5:
        raise RuntimeError(
            f"camera envelope has only {len(free_indices)} collision-free grid positions"
        )
    selected: list[np.ndarray] = []
    for wanted in preferred:
        wanted_array = np.asarray(wanted, dtype=np.int64)
        order = np.lexsort(
            (
                free_indices[:, 2],
                free_indices[:, 1],
                free_indices[:, 0],
                np.sum((free_indices - wanted_array) ** 2, axis=1),
            )
        )
        chosen = next(
            candidate
            for candidate in free_indices[order]
            if not any(np.array_equal(candidate, existing) for existing in selected)
        )
        selected.append(chosen)

    result: list[list[int]] = []
    for index in selected:
        flat = int(index[0] * w * h + index[1] * h + index[2])
        i_elev, i_azim = best_orientation(centers[flat], target, n_elev, n_azim)
        result.append([int(index[0]), int(index[1]), int(index[2]), i_elev, i_azim])
    return result


def make_preview(
    path: Path,
    vertices: np.ndarray,
    camera_centers: np.ndarray,
    starts: list[list[int]],
    camera_counts: np.ndarray,
    target: np.ndarray,
    n_elev: int,
    n_azim: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stride = max(1, len(vertices) // 100_000)
    sample = vertices[::stride]
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sample[:, 0], sample[:, 2], sample[:, 1], s=0.05, c=sample[:, 1], cmap="terrain")
    w, h = int(camera_counts[1]), int(camera_counts[2])
    for pose in starts:
        flat = pose[0] * w * h + pose[1] * h + pose[2]
        center = camera_centers[flat]
        elev = -90.0 + 180.0 * (1 + pose[3]) / (n_elev + 1)
        azim = 360.0 * pose[4] / n_azim
        direction = pose_direction(elev, azim)
        ax.scatter(center[0], center[2], center[1], s=35, c="red")
        ax.quiver(
            center[0], center[2], center[1], direction[0], direction[2], direction[1],
            length=50.0, normalize=True, color="red",
        )
    ax.scatter(target[0], target[2], target[1], s=55, c="blue", marker="x")
    ax.set_xlabel("MAGICIAN X / HUGE East (m)")
    ax.set_ylabel("MAGICIAN Z / -HUGE North (m)")
    ax.set_zlabel("MAGICIAN Y / HUGE Up (m)")
    ax.set_title("HUGE 1_office mesh and five discrete MAGICIAN start cameras")
    ax.view_init(elev=32, azim=-52)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace generated data: {path}")
    manifest_path = path / "conversion_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"refusing to overwrite directory without conversion_manifest.json: {path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("converter", {}).get("name") != Path(__file__).name:
        raise RuntimeError(f"refusing to overwrite directory owned by another producer: {path}")
    declared = {entry["path"] for entry in manifest.get("outputs", [])}
    declared.add("conversion_manifest.json")
    actual = {str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()}
    unknown = sorted(actual - declared)
    if unknown:
        raise RuntimeError(f"refusing to overwrite undeclared/manual files: {unknown}")


def install_output_directory(temporary: Path, output: Path) -> None:
    previous: Path | None = None
    if output.exists():
        previous = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.previous-", dir=output.parent)
        )
        previous.rmdir()
        os.replace(output, previous)
    try:
        os.replace(temporary, output)
    except BaseException:
        if previous is not None:
            os.replace(previous, output)
        raise
    if previous is not None:
        shutil.rmtree(previous)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--huge-data-root", required=True, type=Path)
    parser.add_argument("--env-id", default="1_office")
    parser.add_argument("--magician-data-root", required=True, type=Path)
    parser.add_argument("--output-scene-name", default="huge_1_office")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--random-seed", type=int, default=670)
    parser.add_argument("--collision-clearance-m", type=float, default=2.0)
    # HUGE task-0 for 1_office flies from z=20..40 m down to z=-20..0 m.
    # Keep that complete published activity band in the discrete envelope.
    parser.add_argument("--camera-height-min-m", type=float, default=-20.0)
    parser.add_argument("--camera-height-max-m", type=float, default=40.0)
    parser.add_argument("--camera-horizontal-margin-m", type=float, default=50.0)
    parser.add_argument("--huge-commit", default="unknown")
    parser.add_argument("--magician-commit", default="unknown")
    parser.add_argument("--data-revision", default=OFFICIAL_DATA_REVISION)
    parser.add_argument("--archive-path", type=Path)
    parser.add_argument("--no-preview", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for label, value in (
        ("env-id", args.env_id),
        ("output-scene-name", args.output_scene_name),
    ):
        if value in {"", ".", ".."} or Path(value).name != value:
            raise ValueError(f"--{label} must be one safe path component: {value!r}")
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    env_dir = args.huge_data_root.resolve() / args.env_id
    source_obj = env_dir / "terra_ply" / "simplified_mesh.obj"
    gaussian_ply = env_dir / "3dgs_ply" / "point_cloud_utm50.ply"
    mesh_metadata_path = env_dir / "terra_ply" / "metadata.xml"
    gaussian_metadata_path = env_dir / "3dgs_ply" / "metadata.xml"
    camera_annotation_path = env_dir / "BlocksExchangeUndistortAT_WithoutTiePoints.xml"
    if not source_obj.is_file():
        raise FileNotFoundError(source_obj)
    if not gaussian_ply.is_file():
        raise FileNotFoundError(gaussian_ply)

    mesh_metadata = parse_frame_metadata(mesh_metadata_path)
    gaussian_metadata = parse_frame_metadata(gaussian_metadata_path)
    if not mesh_metadata["srs"].upper().startswith("EPSG:"):
        raise ValueError(f"expected projected mesh SRS, got {mesh_metadata['srs']!r}")
    if not gaussian_metadata["srs"].upper().startswith("ENU:"):
        raise ValueError(f"expected ENU Gaussian SRS, got {gaussian_metadata['srs']!r}")
    origin_height_delta = abs(mesh_metadata["origin"][2] - gaussian_metadata["origin"][2])
    if origin_height_delta > 1e-3:
        raise ValueError(
            "mesh and Gaussian metadata origins disagree in height by "
            f"{origin_height_delta} m"
        )

    source = parse_obj(source_obj)
    ply_audit = audit_ply_xyz(gaussian_ply)
    transformed_vertices = transform_points(source["vertices"])
    rotation = T_HUGE_TO_MAGIC[:3, :3]
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=1e-12):
        raise RuntimeError(f"coordinate transform is not a proper rotation: det={determinant}")
    inverse = np.linalg.inv(T_HUGE_TO_MAGIC)
    sample_indices = np.linspace(
        0,
        len(source["vertices"]) - 1,
        min(100, len(source["vertices"])),
        dtype=int,
    )
    sample_magic_h = np.column_stack(
        (transformed_vertices[sample_indices], np.ones(len(sample_indices)))
    )
    recovered = (sample_magic_h @ inverse.T)[:, :3]
    inverse_error = float(np.max(np.abs(recovered - source["vertices"][sample_indices])))

    landmark_path, landmarks_huge, landmark_labels = read_landmarks(env_dir)
    landmarks_magic = transform_points(landmarks_huge) if len(landmarks_huge) else landmarks_huge
    mesh_min = transformed_vertices.min(axis=0)
    mesh_max = transformed_vertices.max(axis=0)
    ply_min = np.asarray(ply_audit["aabb_magic"][0], dtype=np.float64)
    ply_max = np.asarray(ply_audit["aabb_magic"][1], dtype=np.float64)
    mesh_ply_aabb_intersects = bool(
        np.all(np.maximum(mesh_min, ply_min) <= np.minimum(mesh_max, ply_max))
    )
    if not mesh_ply_aabb_intersects:
        raise RuntimeError(
            "mesh and 3DGS PLY AABBs do not intersect after the same frame transform"
        )
    landmark_mesh_count = 0
    landmark_ply_count = 0
    landmark_inverse_error = 0.0
    if len(landmarks_huge):
        landmark_mesh_count = int(
            np.count_nonzero(
                np.all(landmarks_huge >= source["aabb_min"], axis=1)
                & np.all(landmarks_huge <= source["aabb_max"], axis=1)
            )
        )
        ply_min_huge = np.asarray(ply_audit["aabb_huge"][0], dtype=np.float64)
        ply_max_huge = np.asarray(ply_audit["aabb_huge"][1], dtype=np.float64)
        landmark_ply_count = int(
            np.count_nonzero(
                np.all(landmarks_huge >= ply_min_huge, axis=1)
                & np.all(landmarks_huge <= ply_max_huge, axis=1)
            )
        )
        landmark_magic_h = np.column_stack((landmarks_magic, np.ones(len(landmarks_magic))))
        landmarks_recovered = (landmark_magic_h @ inverse.T)[:, :3]
        landmark_inverse_error = float(np.max(np.abs(landmarks_recovered - landmarks_huge)))

    if len(landmarks_magic):
        horizontal_min = landmarks_magic[:, [0, 2]].min(axis=0) - args.camera_horizontal_margin_m
        horizontal_max = landmarks_magic[:, [0, 2]].max(axis=0) + args.camera_horizontal_margin_m
        target = np.median(landmarks_magic, axis=0)
    else:
        horizontal_min = mesh_min[[0, 2]] - args.camera_horizontal_margin_m
        horizontal_max = mesh_max[[0, 2]] + args.camera_horizontal_margin_m
        target = (mesh_min + mesh_max) / 2.0
    target[1] = (
        float(np.median(landmarks_magic[:, 1]))
        if len(landmarks_magic)
        else float(mesh_min[1])
    )

    camera_min = np.array([horizontal_min[0], args.camera_height_min_m, horizontal_min[1]])
    camera_max = np.array([horizontal_max[0], args.camera_height_max_m, horizontal_max[1]])
    if np.any(camera_max <= camera_min):
        raise ValueError(f"invalid camera bounds: {camera_min} .. {camera_max}")
    camera_counts = np.array(
        [
            grid_count(camera_max[0] - camera_min[0], 75.0, 3),
            grid_count(camera_max[1] - camera_min[1], 10.0, 2),
            grid_count(camera_max[2] - camera_min[2], 75.0, 3),
        ],
        dtype=np.int64,
    )
    centers = grid_centers(camera_min, camera_max, camera_counts)
    occupied, surface_distances, watertight = classify_occupied(
        transformed_vertices,
        source["triangles"],
        centers,
        args.collision_clearance_m,
    )
    pose_n_elev = 7
    pose_n_azim = 12
    starts = choose_start_positions(
        camera_counts,
        centers,
        occupied,
        target,
        pose_n_elev,
        pose_n_azim,
    )

    scene_span = mesh_max - mesh_min
    scene_counts = np.array(
        [max(1, int(round(float(span) / 50.0))) for span in scene_span],
        dtype=np.int64,
    )
    settings = {
        "scene": {
            "grid_l": int(scene_counts[0]),
            "grid_w": int(scene_counts[1]),
            "grid_h": int(scene_counts[2]),
            "cell_capacity": 1000,
            "cell_resolution": 0.5,
            "x_min": finite_list(mesh_min),
            "x_max": finite_list(mesh_max),
            "visibility_ratio": 1.0,
        },
        "camera": {
            "pose_l": int(camera_counts[0]),
            "pose_w": int(camera_counts[1]),
            "pose_h": int(camera_counts[2]),
            "pose_n_theta": pose_n_elev,
            "pose_n_azim": pose_n_azim,
            "x_min": finite_list(camera_min),
            "x_max": finite_list(camera_max),
            "start_positions": starts,
            "contrast_factor": 1.0,
        },
    }

    indices = np.asarray(list(product(*(range(int(v)) for v in camera_counts))), dtype=np.int64)
    occupied_payload = {
        "X_idx": torch.from_numpy(indices),
        "occupied": torch.from_numpy(occupied),
    }

    output_dir = args.magician_data_root.resolve() / args.output_scene_name
    summary = {
        "output_directory": str(output_dir),
        "dry_run": args.dry_run,
        "source_obj": str(source_obj),
        "source_obj_counts": source["counts"],
        "source_aabb_huge": [finite_list(source["aabb_min"]), finite_list(source["aabb_max"])],
        "output_aabb_magic": [finite_list(mesh_min), finite_list(mesh_max)],
        "rotation_determinant": determinant,
        "inverse_sample_max_abs_error": inverse_error,
        "mesh_srs": mesh_metadata["srs"],
        "gaussian_srs": gaussian_metadata["srs"],
        "origin_height_delta_m": origin_height_delta,
        "camera_grid": camera_counts.tolist(),
        "camera_grid_cell_size_m": finite_list((camera_max - camera_min) / camera_counts),
        "scene_grid": scene_counts.tolist(),
        "scene_grid_cell_size_m": finite_list(scene_span / scene_counts),
        "occupied_positions": int(occupied.sum()),
        "free_positions": int((~occupied).sum()),
        "nearest_surface_distance_m": {
            "min": float(surface_distances.min()),
            "max": float(surface_distances.max()),
        },
        "mesh_watertight": watertight,
        "ply_vertex_count": ply_audit["vertex_count"],
        "ply_aabb_huge": ply_audit["aabb_huge"],
        "ply_aabb_magic": ply_audit["aabb_magic"],
        "mesh_ply_aabb_intersects": mesh_ply_aabb_intersects,
        "landmarks_in_mesh_aabb": landmark_mesh_count,
        "landmarks_in_ply_aabb": landmark_ply_count,
        "landmark_inverse_max_abs_error": landmark_inverse_error,
        "start_positions": starts,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    prepare_output_directory(output_dir, args.overwrite)
    temp_parent = output_dir.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{args.output_scene_name}.tmp-", dir=temp_parent)
    )
    try:
        output_obj = temporary / f"{args.output_scene_name}.obj"
        transform_obj(source_obj, output_obj)
        material_info = copy_material_assets(source_obj, temporary, source["mtllibs"])
        json_dump(temporary / "settings.json", settings)
        torch.save(occupied_payload, temporary / "occupied_pose.pt")
        if not args.no_preview:
            make_preview(
                temporary / "camera_mesh_preview.png",
                transformed_vertices,
                centers,
                starts,
                camera_counts,
                target,
                pose_n_elev,
                pose_n_azim,
            )

        input_paths = [
            source_obj,
            gaussian_ply,
            mesh_metadata_path,
            gaussian_metadata_path,
        ]
        if camera_annotation_path.is_file():
            input_paths.append(camera_annotation_path)
        if landmark_path is not None:
            input_paths.append(landmark_path)
        input_paths.extend(source_obj.parent / path for path in source["mtllibs"])
        archive_info = None
        if args.archive_path:
            archive = args.archive_path.resolve()
            if not archive.is_file():
                raise FileNotFoundError(archive)
            archive_info = {
                "path": str(archive),
                "size_bytes": archive.stat().st_size,
                "sha256": sha256_file(archive),
            }

        output_entries = []
        for output in sorted(item for item in temporary.rglob("*") if item.is_file()):
            output_entries.append(
                {
                    "path": str(output.relative_to(temporary)),
                    "size_bytes": output.stat().st_size,
                    "sha256": sha256_file(output),
                }
            )
        input_entries = []
        for input_path in sorted(set(input_paths)):
            input_entries.append(
                {
                    "path": str(input_path),
                    "size_bytes": input_path.stat().st_size,
                    "sha256": sha256_file(input_path),
                }
            )

        rgb_mesh_compatible = bool(material_info["diffuse_texture_paths"]) and (
            source["counts"]["texture_coordinates"] > 0
            and source["counts"]["faces_missing_any_uv"] == 0
        )
        converter_path = Path(__file__).resolve()
        manifest = {
            "converter": {
                "name": converter_path.name,
                "path": str(converter_path),
                "sha256": sha256_file(converter_path),
                "version": CONVERTER_VERSION,
            },
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "random_seed": args.random_seed,
            "repositories": {
                "MAGICIAN": args.magician_commit,
                "HUGE-Bench": args.huge_commit,
            },
            "data_revision": args.data_revision,
            "archive": archive_info,
            "inputs": input_entries,
            "outputs": output_entries,
            "coordinate_system": {
                "source": "HUGE local ENU/UTM-relative, right-handed, +Z up, metres",
                "destination": "MAGICIAN/PyTorch3D world, right-handed, +Y up, metres",
                "T_huge_to_magic": T_HUGE_TO_MAGIC.tolist(),
                "scale": 1.0,
                "rotation_determinant": determinant,
                "inverse_sample_max_abs_error": inverse_error,
                "frame_metadata": {
                    "mesh": mesh_metadata,
                    "gaussian": gaussian_metadata,
                    "origin_height_delta_m": origin_height_delta,
                },
                "rationale": (
                    "(E,N,U)->(X,Y,Z)=(E,U,-N), validated against mesh, "
                    "landmarks and PLY frame metadata"
                ),
            },
            "geometry": {
                "source": "terra_ply/simplified_mesh.obj",
                "source_counts": source["counts"],
                "source_aabb_huge": [
                    finite_list(source["aabb_min"]),
                    finite_list(source["aabb_max"]),
                ],
                "output_aabb_magic": [finite_list(mesh_min), finite_list(mesh_max)],
                "mesh_watertight": watertight,
                "gaussian_ply": ply_audit,
                "mesh_ply_aabb_intersects": mesh_ply_aabb_intersects,
            },
            "materials": {
                "obj_mtllibs": source["mtllibs"],
                "materials_used": source["materials_used"],
                "material_count": material_info["material_count"],
                "texture_paths": material_info["texture_paths"],
                "diffuse_texture_paths": material_info["diffuse_texture_paths"],
                "diffuse_texture_count": len(material_info["diffuse_texture_paths"]),
                "rgb_mesh_compatible": rgb_mesh_compatible,
                "status": (
                    "texture references preserved and validated"
                    if rgb_mesh_compatible
                    else (
                        "OBJ lacks complete UV-backed map_Kd textures: geometry loader "
                        "compatible, SoftPhong RGB rendering incompatible"
                    )
                ),
            },
            "camera": {
                "bounds_magic": [finite_list(camera_min), finite_list(camera_max)],
                "grid_shape": camera_counts.tolist(),
                "grid_cell_size_m": finite_list((camera_max - camera_min) / camera_counts),
                "start_positions": starts,
                "target_magic": finite_list(target),
                "huge_height_evidence_m": [args.camera_height_min_m, args.camera_height_max_m],
            },
            "occupancy": {
                "method": "nearest mesh-surface distance plus contains() when watertight",
                "clearance_m": args.collision_clearance_m,
                "shape": list(occupied_payload["occupied"].shape),
                "occupied": int(occupied.sum()),
                "free": int((~occupied).sum()),
                "nearest_surface_distance_m": {
                    "min": float(surface_distances.min()),
                    "max": float(surface_distances.max()),
                },
            },
            "landmarks": {
                "source_path": str(landmark_path) if landmark_path else None,
                "count": len(landmark_labels),
                "in_mesh_aabb": landmark_mesh_count,
                "in_gaussian_ply_aabb": landmark_ply_count,
                "inverse_max_abs_error": landmark_inverse_error,
                "first_three": [
                    {
                        "label": landmark_labels[i],
                        "huge": finite_list(landmarks_huge[i]),
                        "magic": finite_list(landmarks_magic[i]),
                    }
                    for i in range(min(3, len(landmark_labels)))
                ],
            },
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "trimesh": trimesh.__version__,
                "platform": platform.platform(),
            },
            "reproducibility": {
                "numeric_outputs_deterministic": True,
                "time_varying_manifest_fields": ["generated_at_utc", "command"],
            },
        }
        json_dump(temporary / "conversion_manifest.json", manifest)
        install_output_directory(temporary, output_dir)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)

    summary["manifest"] = str(output_dir / "conversion_manifest.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
