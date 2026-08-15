#!/usr/bin/env python3
"""Load one assembled scene through MAGICIAN's real dataset/mesh path."""

import argparse
import json
import sys
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", required=True)
    args = parser.parse_args()
    scene_root = Path(args.scene_dir).resolve()
    if not scene_root.is_dir():
        raise RuntimeError("Scene directory does not exist: {}".format(scene_root))

    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root))
    import torch

    from macarons.utility.CustomDataset import SceneDataset
    from macarons.utility.macarons_utils import load_scene

    dataset = SceneDataset(
        data_path=str(scene_root.parent),
        scene_names=[scene_root.name],
        use_occupied_pose=True,
    )
    item = dataset[0]
    obj_path = scene_root / item["obj_name"]
    mesh = load_scene(
        mesh_path=str(obj_path),
        scene_scale_factor=1.0,
        device=torch.device("cpu"),
        texture_atlas_size=1,
    )
    occupied_pose = item["occupied_pose"]
    if set(occupied_pose) != {"X_idx", "occupied"}:
        raise RuntimeError("occupied_pose.pt has unexpected keys")
    if len(occupied_pose["X_idx"]) != len(occupied_pose["occupied"]):
        raise RuntimeError("occupied_pose.pt arrays have different lengths")
    report = {
        "status": "passed",
        "device": "cpu",
        "scene_name": item["scene_name"],
        "obj_name": item["obj_name"],
        "mesh": {
            "vertices": int(mesh.verts_packed().shape[0]),
            "faces": int(mesh.faces_packed().shape[0]),
            "textures": type(mesh.textures).__name__,
            "texture_atlas_size": 1,
        },
        "occupied_pose": {
            "positions": len(occupied_pose["X_idx"]),
            "occupied": int(occupied_pose["occupied"].sum().item()),
        },
        "settings_sections": sorted(item["settings"].keys()),
        "limitations": [
            "The smoke test uses CPU and a 1x1 per-face texture atlas to bound memory.",
            "It validates the production dataset/mesh loader, not a planning trajectory.",
        ],
    }
    write_json(scene_root / "magician_loader_report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
