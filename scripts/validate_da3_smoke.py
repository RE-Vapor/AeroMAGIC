#!/usr/bin/env python3
"""Run a real-checkpoint, RGB-only DA3 adapter smoke test.

This is intentionally separate from the dependency-free unit suite.  It loads
the pinned Hugging Face checkpoint, uses 2--8 synthetic captured RGB/pose
frames (never zbuf or renderer masks), compares cache miss/hit/off outputs, and
records latency, peak CUDA memory, shapes, and the actual runtime environment.
"""

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from macarons.utility.da3_adapter import (
    DA3_DEFAULT_MODEL,
    DA3_DEFAULT_MODEL_REVISION,
    DA3_SOURCE_REVISION,
    DA3DepthProvider,
)
from macarons.utility.depth_sources import DepthObservation


class _FovCamera:
    fov = np.asarray([60.0], dtype=np.float32)
    aspect_ratio = np.asarray([1.0], dtype=np.float32)


class _Camera:
    def __init__(self, save_dir_path: Path, frame_count: int):
        self.save_dir_path = str(save_dir_path)
        self.n_frames_captured = frame_count
        self.fov_camera = _FovCamera()


def _rgb_frame(index: int, height: int, width: int):
    y = torch.linspace(0.0, 1.0, height).view(height, 1).expand(height, width)
    x = torch.linspace(0.0, 1.0, width).view(1, width).expand(height, width)
    rgb = torch.stack((x, y, torch.full_like(x, index / 8.0)), dim=-1).unsqueeze(0)
    return {
        "rgb": rgb,
        "R": torch.eye(3, dtype=torch.float32).unsqueeze(0),
        "T": torch.tensor(
            [[0.1 * index, 0.1 if index >= 2 else 0.0, 0.0]],
            dtype=torch.float32,
        ),
    }


def _elapsed(callable_):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = callable_()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def _frame_arrays(frame):
    return {
        name: getattr(frame, name).detach().cpu().numpy()
        for name in ("rgb", "depth_z", "valid_mask", "error_mask", "confidence")
    }


def _nvidia_driver():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.strip().splitlines()
    return lines[0].strip() if result.returncode == 0 and lines else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=3, choices=range(2, 9))
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=456)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--scene-units-per-meter", type=float, default=1.0)
    parser.add_argument("--model-id", default=DA3_DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DA3_DEFAULT_MODEL_REVISION)
    args = parser.parse_args()

    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        raise ValueError(f"Smoke work directory must be empty: {args.work_dir}")
    frames_dir = args.work_dir / "frames"
    cache_dir = args.work_dir / "cache"
    frames_dir.mkdir(parents=True)
    for index in range(args.frames):
        torch.save(_rgb_frame(index, args.height, args.width), frames_dir / f"{index}.pt")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.init()
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    camera = _Camera(frames_dir, args.frames)
    base_config = {
        "scene_name": "synthetic_metric_smoke",
        "scene_units_per_meter": args.scene_units_per_meter,
        "da3_model_id": args.model_id,
        "da3_model_revision": args.model_revision,
        "da3_window_size": args.frames,
        "da3_process_res": args.process_res,
        "da3_output_height": args.height,
        "da3_output_width": args.width,
        "da3_confidence_percentile": None,
        "da3_cache_dir": str(cache_dir),
        "znear": 0.01,
        "zfar": 1000.0,
    }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    cached = DA3DepthProvider(
        config={**base_config, "da3_cache_enabled": True}, device=device
    )
    miss, miss_seconds = _elapsed(
        lambda: cached.get_frame(DepthObservation(camera=camera, device=device))
    )
    hit, hit_seconds = _elapsed(
        lambda: cached.get_frame(DepthObservation(camera=camera, device=device))
    )
    uncached = DA3DepthProvider(
        config={**base_config, "da3_cache_enabled": False}, device=device
    )
    off, off_seconds = _elapsed(
        lambda: uncached.get_frame(DepthObservation(camera=camera, device=device))
    )

    miss_arrays = _frame_arrays(miss)
    hit_arrays = _frame_arrays(hit)
    off_arrays = _frame_arrays(off)
    equivalent = {
        "miss_vs_hit": all(
            np.array_equal(miss_arrays[name], hit_arrays[name]) for name in miss_arrays
        ),
        "miss_vs_off": all(
            np.allclose(miss_arrays[name], off_arrays[name], rtol=1e-5, atol=1e-6)
            for name in miss_arrays
        ),
    }
    expected_shapes = {
        "rgb": [1, args.height, args.width, 3],
        "depth_z": [1, args.height, args.width, 1],
        "valid_mask": [1, args.height, args.width, 1],
        "error_mask": [1, args.height, args.width, 1],
        "confidence": [1, args.height, args.width, 1],
    }
    actual_shapes = {name: list(value.shape) for name, value in miss_arrays.items()}
    checks = {
        "rgb_only_inputs": all(
            set(torch.load(frames_dir / f"{index}.pt", map_location="cpu").keys())
            == {"rgb", "R", "T"}
            for index in range(args.frames)
        ),
        "source_is_da3": miss.source == "DA3",
        "expected_shapes": actual_shapes == expected_shapes,
        "finite_depth": bool(np.isfinite(miss_arrays["depth_z"]).all()),
        "cache_miss_then_hit": (
            miss.cache_metadata["cache_hit"] is False
            and hit.cache_metadata["cache_hit"] is True
        ),
        "cache_off_no_hit": off.cache_metadata["cache_hit"] is False,
        "cache_equivalence": all(equivalent.values()),
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "equivalence": equivalent,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device_index),
            "gpu_total_memory_mib": torch.cuda.get_device_properties(device_index).total_memory
            / (1024**2),
            "nvidia_driver": _nvidia_driver(),
        },
        "model": {
            "id": args.model_id,
            "revision": args.model_revision,
            "source_revision": DA3_SOURCE_REVISION,
        },
        "frames": args.frames,
        "output_shapes": actual_shapes,
        "timing_seconds": {
            "model_load_and_cache_miss_total": miss_seconds,
            "cache_miss_per_input_frame": miss_seconds / args.frames,
            "cache_hit_total": hit_seconds,
            "cache_off_total": off_seconds,
            "cache_off_per_input_frame": off_seconds / args.frames,
        },
        "peak_cuda_memory_mib": {
            "allocated": torch.cuda.max_memory_allocated(device_index) / (1024**2),
            "reserved": torch.cuda.max_memory_reserved(device_index) / (1024**2),
        },
        "cache": {
            "key": miss.cache_metadata["cache_key"],
            "files": len(tuple(cache_dir.glob("*.npz"))),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
