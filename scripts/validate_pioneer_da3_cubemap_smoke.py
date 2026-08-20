#!/usr/bin/env python3
"""Run one real DA3 six-face inference against an existing RGB bundle.

The renderer supplies recorded RGB but an unmistakable negative sentinel
z-buffer.  A successful report therefore exercises the production cubemap
capture/provider boundary while proving that planning depth did not come from
the renderer fragments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from unittest import mock

import torch
from pytorch3d.renderer import FoVPerspectiveCameras

from macarons.utility.da3_adapter import (
    DA3_DEFAULT_MODEL,
    DA3_DEFAULT_MODEL_REVISION,
    DA3_SOURCE_REVISION,
    DA3DepthProvider,
)
from macarons.utility.planning_observations import (
    CUBEMAP_FACE_NAMES,
    CUBEMAP_RIG_FRAME_WORLD,
    capture_cubemap_observation,
)


def _load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bundle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene", default="HKUST")
    parser.add_argument("--scene-units-per-meter", type=float, default=0.2)
    parser.add_argument("--camera-znear", type=float, default=1.0)
    parser.add_argument("--provider-znear", type=float, default=0.5)
    parser.add_argument("--zfar", type=float, default=750.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This smoke requires a CUDA device.")
    if args.scene_units_per_meter <= 0.0:
        raise ValueError("scene-units-per-meter must be positive.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output-dir must be empty: {args.output_dir}")
    if args.report.exists():
        raise FileExistsError(f"report already exists: {args.report}")

    recorded = {
        name: _load(args.input_bundle_dir / f"{name}.pt", device)
        for name in CUBEMAP_FACE_NAMES
    }
    first = recorded[CUBEMAP_FACE_NAMES[0]]
    face_size = int(first["rgb"].shape[1])
    for name, payload in recorded.items():
        if tuple(payload["rgb"].shape) != (1, face_size, face_size, 3):
            raise ValueError(f"recorded face {name} is not square RGB")

    reference_camera = FoVPerspectiveCameras(
        R=first["R"],
        T=first["T"],
        znear=args.camera_znear,
        zfar=args.zfar,
        fov=90.0,
        aspect_ratio=1.0,
        device=device,
    )
    camera = SimpleNamespace(
        fov_camera=reference_camera,
        device=device,
        zfar=args.zfar,
        contrast_factor=1.0,
        n_frames_captured=0,
        save_dir_path=str(args.output_dir),
        last_observation_bundle=None,
    )

    class RecordedRgbSentinelDepthRenderer:
        def __init__(self):
            self.calls = 0

        def __call__(self, mesh, cameras):
            name = CUBEMAP_FACE_NAMES[self.calls]
            self.calls += 1
            rgb = recorded[name]["rgb"]
            alpha = torch.ones_like(rgb[..., :1])
            sentinel = torch.full(
                (1, face_size, face_size, 1),
                -9999.0,
                dtype=rgb.dtype,
                device=rgb.device,
            )
            return torch.cat((rgb, alpha), dim=-1), SimpleNamespace(zbuf=sentinel)

    provider = DA3DepthProvider(
        config={
            "da3_cache_dir": str(args.cache_dir),
            "da3_model_id": DA3_DEFAULT_MODEL,
            "da3_model_revision": DA3_DEFAULT_MODEL_REVISION,
            "da3_source_revision": DA3_SOURCE_REVISION,
            "da3_window_size": 3,
            "da3_process_res": 504,
            "da3_process_res_method": "upper_bound_resize",
            "da3_output_height": face_size,
            "da3_output_width": face_size,
            "da3_confidence_percentile": None,
            "da3_cache_enabled": True,
            "scene_name": args.scene,
            "scene_units_per_meter": args.scene_units_per_meter,
            "znear": args.provider_znear,
            "zfar": args.zfar,
        },
        device=device,
    )
    renderer = RecordedRgbSentinelDepthRenderer()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with mock.patch(
        "macarons.utility.planning_observations._build_square_renderer",
        return_value=renderer,
    ):
        bundle = capture_cubemap_observation(
            camera=camera,
            mesh=object(),
            face_size=face_size,
            depth_provider=provider,
            save_png=False,
            dir_path=str(args.output_dir),
            rig_frame=CUBEMAP_RIG_FRAME_WORLD,
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    repo_root = Path(__file__).resolve().parents[1]
    git_commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    import depth_anything_3.api as da3_api

    face_rows = []
    total_planning_pixels = 0
    for face in bundle.faces:
        metadata = face.metadata["depth_cache_metadata"]
        finite = bool(torch.isfinite(face.depth_z).all().item())
        planning_pixels = int(face.valid_mask.sum().item())
        sentinel_pixels = int((face.depth_z == -9999.0).sum().item())
        if not finite or sentinel_pixels:
            raise RuntimeError(f"DA3 smoke depth validation failed for {face.name}")
        total_planning_pixels += planning_pixels
        face_rows.append(
            {
                "face_name": face.name,
                "cache_key": metadata["cache_key"],
                "cache_hit": metadata["cache_hit"],
                "stream_id": metadata["stream"]["id"],
                "pose_conditioned": metadata["camera"]["pose_conditioned"],
                "planning_pixels": planning_pixels,
                "depth_min": float(face.depth_z.min().item()),
                "depth_max": float(face.depth_z.max().item()),
                "sentinel_pixels": sentinel_pixels,
            }
        )

    report = {
        "schema_version": 1,
        "accepted": True,
        "git_commit": git_commit,
        "implementation_sha256": {
            str(path.relative_to(repo_root)): _sha256(path)
            for path in (
                Path(__file__).resolve(),
                repo_root / "macarons" / "utility" / "da3_adapter.py",
                repo_root / "macarons" / "utility" / "planning_observations.py",
            )
        },
        "da3_import_origin": str(Path(da3_api.__file__).resolve()),
        "input_bundle_dir": str(args.input_bundle_dir.resolve()),
        "input_face_sha256": {
            name: _sha256(args.input_bundle_dir / f"{name}.pt")
            for name in CUBEMAP_FACE_NAMES
        },
        "depth_source": bundle.metadata["depth_source"],
        "rgb_source": bundle.metadata["rgb_source"],
        "renderer_zbuf_read": bundle.metadata["renderer_zbuf_read"],
        "renderer_zbuf_sentinel": -9999.0,
        "depth_inference_count": bundle.metadata["depth_inference_count"],
        "face_size": face_size,
        "scene": args.scene,
        "scene_units_per_meter": args.scene_units_per_meter,
        "camera_znear": args.camera_znear,
        "provider_znear": args.provider_znear,
        "zfar": args.zfar,
        "model": {
            "id": DA3_DEFAULT_MODEL,
            "revision": DA3_DEFAULT_MODEL_REVISION,
            "source_revision": DA3_SOURCE_REVISION,
        },
        "elapsed_seconds": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        "faces": face_rows,
        "total_planning_pixels": total_planning_pixels,
        "empty_planning_faces": [
            row["face_name"] for row in face_rows if row["planning_pixels"] == 0
        ],
    }
    if (
        report["depth_source"] != "DA3"
        or report["renderer_zbuf_read"] is not False
        or report["depth_inference_count"] != 6
        or renderer.calls != 6
        or total_planning_pixels <= 0
    ):
        raise RuntimeError("DA3 smoke provenance contract failed.")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
