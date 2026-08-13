#!/usr/bin/env python3
"""Verify that PyTorch3D's compiled CUDA rasterizer works in this runtime."""

import argparse
import json
from pathlib import Path
import platform
import sys

import torch
import pytorch3d
import pytorch3d._C
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.structures import Meshes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This validation requires a CUDA device.")
    torch.cuda.set_device(device)

    verts = torch.tensor(
        [[-0.8, -0.8, 1.0], [0.8, -0.8, 1.0], [0.0, 0.8, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
    mesh = Meshes(verts=[verts], faces=[faces])
    pix_to_face, zbuf, barycentric, distances = rasterize_meshes(
        mesh,
        image_size=32,
        blur_radius=0.0,
        faces_per_pixel=1,
        perspective_correct=False,
    )
    covered_pixels = int((pix_to_face >= 0).sum().item())
    finite_depth_pixels = int(torch.isfinite(zbuf[pix_to_face >= 0]).sum().item())
    checks = {
        "cuda_available": torch.cuda.is_available(),
        "compiled_extension_loaded": str(pytorch3d._C.__file__).endswith(".so"),
        "cuda_rasterizer_covered_pixels": covered_pixels > 0,
        "finite_rasterized_depth": finite_depth_pixels == covered_pixels,
        "cuda_outputs": all(
            value.is_cuda for value in (pix_to_face, zbuf, barycentric, distances)
        ),
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "pytorch3d": pytorch3d.__version__,
            "pytorch3d_extension": str(pytorch3d._C.__file__),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
        },
        "rasterizer": {
            "image_size": 32,
            "covered_pixels": covered_pixels,
            "finite_depth_pixels": finite_depth_pixels,
        },
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
