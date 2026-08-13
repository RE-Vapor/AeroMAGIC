#!/usr/bin/env python3
"""Recompute the Eiffel scene metric calibration from OBJ geometry."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_obj(path):
    vertices = []
    faces = []
    with path.open(errors="replace") as source:
        for line in source:
            if line.startswith("v "):
                fields = line.split()
                vertices.append(tuple(float(value) for value in fields[1:4]))
            elif line.startswith("f "):
                fields = line.split()[1:]
                if len(fields) != 3:
                    raise ValueError("Calibration expects a triangle-only OBJ.")
                faces.append(tuple(int(value.split("/", 1)[0]) - 1 for value in fields))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _tower_bounds(vertices, faces):
    rows = np.concatenate((faces[:, 0], faces[:, 1], faces[:, 2]))
    cols = np.concatenate((faces[:, 1], faces[:, 2], faces[:, 0]))
    graph = coo_matrix(
        (np.ones(rows.shape[0], dtype=np.uint8), (rows, cols)),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    mins = np.full((component_count, 3), np.inf)
    maxs = np.full((component_count, 3), -np.inf)
    for axis in range(3):
        np.minimum.at(mins[:, axis], labels, vertices[:, axis])
        np.maximum.at(maxs[:, axis], labels, vertices[:, axis])

    selected = (
        (maxs[:, 1] > 1.5)
        & (mins[:, 0] > -3.0)
        & (maxs[:, 0] < 3.0)
        & (mins[:, 2] > -3.0)
        & (maxs[:, 2] < 3.0)
    )
    selected_vertices = vertices[selected[labels]]
    return int(selected.sum()), selected_vertices.min(axis=0), selected_vertices.max(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mesh", type=Path, default=ROOT / "data/Macarons++/eiffel/eiffel_rgb.obj"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs/test/scene_metric_calibrations.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    calibration = json.loads(args.manifest.read_text())["calibrations"]["eiffel"]
    measurement = calibration["mesh_measurement"]
    vertices, faces = _load_obj(args.mesh)
    selected_count, minimum, maximum = _tower_bounds(vertices, faces)
    height_obj_units = float(maximum[1] - minimum[1])
    scene_units_per_meter = (
        height_obj_units
        * measurement["scene_scale_factor"]
        / calibration["physical_reference"]["meters"]
    )
    checks = {
        "mesh_sha256": _sha256(args.mesh) == measurement["sha256"],
        "triangle_count": len(faces) == 1419054,
        "selected_component_count": selected_count
        == measurement["selected_component_count"],
        "base_y": bool(
            np.isclose(minimum[1], measurement["base_y_obj_units"], atol=1e-12)
        ),
        "top_y": bool(
            np.isclose(maximum[1], measurement["top_y_obj_units"], atol=1e-12)
        ),
        "scene_units_per_meter": bool(
            np.isclose(
                scene_units_per_meter,
                calibration["scene_units_per_meter"],
                atol=1e-12,
            )
        ),
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "mesh": str(args.mesh),
        "vertices": len(vertices),
        "triangles": len(faces),
        "selected_components": selected_count,
        "selected_bounds_obj_units": {
            "minimum": minimum.tolist(),
            "maximum": maximum.tolist(),
        },
        "height_obj_units": height_obj_units,
        "scene_units_per_meter": scene_units_per_meter,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
