"""Official HUGE 3DGS RGB observations for MAGICIAN planning.

The adapter supplies RGB only. Depth remains owned by the configured planning
depth provider (DA3 or the benchmark's perfect mesh z-buffer), keeping the two
Stage-4 conditions scientifically distinct.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping


def _value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


@dataclass(frozen=True)
class Huge3DGSConfig:
    enabled: bool
    ply_path: Path | None = None
    manifest_path: Path | None = None
    render_device: str = "cuda:0"
    alpha_threshold: float = 0.1
    chunk_size: int = 1_000_000
    frustum_center_margin: float = 0.15
    kernel_size: float = 0.0
    render_height: int | None = None
    render_width: int | None = None
    diagnostics_path: Path | None = None

    @classmethod
    def from_config(cls, config: Any) -> "Huge3DGSConfig":
        enabled = _value(config, "huge_3dgs_rgb_enabled", False)
        if type(enabled) is not bool:
            raise ValueError("huge_3dgs_rgb_enabled must be a boolean.")
        if not enabled:
            return cls(enabled=False)

        ply = Path(str(_value(config, "huge_3dgs_ply_path", ""))).expanduser()
        manifest = Path(
            str(_value(config, "huge_3dgs_manifest_path", ""))
        ).expanduser()
        if not ply.is_file():
            raise FileNotFoundError(f"Official HUGE 3DGS PLY not found: {ply}")
        if not manifest.is_file():
            raise FileNotFoundError(f"HUGE conversion manifest not found: {manifest}")

        render_device = _value(config, "huge_3dgs_render_device", "cuda:0")
        if not isinstance(render_device, str) or not render_device.startswith("cuda"):
            raise ValueError("huge_3dgs_render_device must name a CUDA device.")
        alpha = _value(config, "huge_3dgs_alpha_threshold", 0.1)
        chunk_size = _value(config, "huge_3dgs_chunk_size", 1_000_000)
        margin = _value(config, "huge_3dgs_frustum_center_margin", 0.15)
        kernel_size = _value(config, "huge_3dgs_kernel_size", 0.0)
        render_height = _value(config, "huge_3dgs_render_height", None)
        render_width = _value(config, "huge_3dgs_render_width", None)
        if not isinstance(alpha, (int, float)) or not 0 <= alpha <= 1:
            raise ValueError("huge_3dgs_alpha_threshold must be in [0, 1].")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("huge_3dgs_chunk_size must be a positive integer.")
        if not isinstance(margin, (int, float)) or margin < 0:
            raise ValueError("huge_3dgs_frustum_center_margin must be non-negative.")
        if not isinstance(kernel_size, (int, float)) or kernel_size < 0:
            raise ValueError("huge_3dgs_kernel_size must be non-negative.")
        if (render_height is None) != (render_width is None):
            raise ValueError(
                "huge_3dgs_render_height and huge_3dgs_render_width must be set together."
            )
        for name, value in (
            ("huge_3dgs_render_height", render_height),
            ("huge_3dgs_render_width", render_width),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        diagnostics = _value(config, "huge_3dgs_diagnostics_path", None)
        return cls(
            enabled=True,
            ply_path=ply.resolve(),
            manifest_path=manifest.resolve(),
            render_device=render_device,
            alpha_threshold=float(alpha),
            chunk_size=chunk_size,
            frustum_center_margin=float(margin),
            kernel_size=float(kernel_size),
            render_height=render_height,
            render_width=render_width,
            diagnostics_path=Path(diagnostics).expanduser().resolve()
            if diagnostics
            else None,
        )


class Huge3DGSRGBProvider:
    """Render one official-Ply RGB observation at the current camera pose."""

    source = "HUGE_OFFICIAL_3DGS"

    def __init__(self, config: Huge3DGSConfig, *, output_device: Any):
        if not config.enabled or config.ply_path is None or config.manifest_path is None:
            raise ValueError("Huge3DGSRGBProvider requires an enabled config.")
        self.config = config
        self.output_device = output_device
        manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
        self.transform = manifest["coordinate_system"]["T_huge_to_magic"]
        scale = manifest["coordinate_system"].get("scale")
        destination = manifest["coordinate_system"].get("destination", "")
        if scale != 1.0 or "metres" not in destination:
            raise ValueError(
                "HUGE 3DGS planning requires a scale-1 metric conversion manifest."
            )
        self._header = None

    def _append_diagnostic(self, value: Mapping[str, Any]) -> None:
        path = self.config.diagnostics_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")

    def render(self, camera: Any) -> Mapping[str, Any]:
        import torch
        from pytorch3d.renderer import FoVPerspectiveCameras

        from tools.probe_huge_3dgs_alignment import (
            load_official_gaussians,
            parse_binary_ply_header,
            prefilter_gaussians_for_view,
            render_gaussians,
        )
        from .gaussian_utils import convert_camera_from_pytorch3d_to_gs

        started = time.perf_counter()
        render_device = torch.device(self.config.render_device)
        if not torch.cuda.is_available():
            raise RuntimeError("HUGE official 3DGS observations require CUDA.")
        if self._header is None:
            self._header = parse_binary_ply_header(self.config.ply_path)

        with torch.cuda.device(render_device):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(render_device)
            q = torch.tensor(
                self.transform, dtype=torch.float32, device=render_device
            )[:3, :3]
            source_camera = camera.fov_camera
            output_height = int(camera.image_height)
            output_width = int(camera.image_width)
            render_height = self.config.render_height or output_height
            render_width = self.config.render_width or output_width
            r_magic = source_camera.R.to(render_device)
            t_magic = source_camera.T.to(render_device)
            r_huge = torch.matmul(q.transpose(0, 1).unsqueeze(0), r_magic)
            p3d_huge = FoVPerspectiveCameras(
                R=r_huge,
                T=t_magic,
                znear=float(getattr(camera, "znear", 1.0)),
                zfar=float(camera.zfar),
                device=render_device,
            )
            p3d_huge.K = p3d_huge.get_projection_transform().get_matrix().transpose(
                -1, -2
            )
            gs_camera = convert_camera_from_pytorch3d_to_gs(
                p3d_huge,
                height=render_height,
                width=render_width,
                device=render_device,
            )[0]
            gs_camera.znear = float(getattr(camera, "znear", 1.0))
            gs_camera.zfar = float(camera.zfar)

            loaded_at = time.perf_counter()
            buffers = load_official_gaussians(
                self.config.ply_path,
                self._header,
                render_device,
                self.config.chunk_size,
            )
            load_seconds = time.perf_counter() - loaded_at
            buffers, prefilter = prefilter_gaussians_for_view(
                buffers,
                gs_camera,
                self.config.chunk_size,
                self.config.frustum_center_margin,
            )
            rendered_at = time.perf_counter()
            rendered = render_gaussians(
                buffers,
                gs_camera,
                render_height,
                render_width,
                self.config.kernel_size,
            )
            torch.cuda.synchronize(render_device)
            render_seconds = time.perf_counter() - rendered_at
            rgb_chw = rendered["rgb"].clamp(0, 1).unsqueeze(0)
            alpha_chw = rendered["alpha"].unsqueeze(0)
            if (render_height, render_width) != (output_height, output_width):
                rgb_chw = torch.nn.functional.interpolate(
                    rgb_chw,
                    size=(output_height, output_width),
                    mode="bilinear",
                    align_corners=False,
                )
                alpha_chw = torch.nn.functional.interpolate(
                    alpha_chw,
                    size=(output_height, output_width),
                    mode="bilinear",
                    align_corners=False,
                )
            rgb = rgb_chw.permute(0, 2, 3, 1)
            alpha = alpha_chw.permute(0, 2, 3, 1)
            valid = alpha >= self.config.alpha_threshold
            metadata = {
                "source": self.source,
                "ply_path": str(self.config.ply_path),
                "render_device": str(render_device),
                "native_render_size": [render_height, render_width],
                "planner_rgb_size": [output_height, output_width],
                "rgb_resize": "bilinear" if render_height != output_height else "none",
                "load_seconds": load_seconds,
                "render_seconds": render_seconds,
                "total_seconds": time.perf_counter() - started,
                "valid_pixels": int(valid.sum().item()),
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(render_device)
                ),
                "peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(render_device)
                ),
                "prefilter": prefilter,
            }
            result = {
                "rgb": rgb.to(self.output_device),
                "alpha": alpha.to(self.output_device),
                "valid_mask": valid.to(self.output_device),
                "source": self.source,
                "metadata": metadata,
            }
            self._append_diagnostic(metadata)
            del rendered, buffers, rgb_chw, alpha_chw, rgb, alpha, valid
            torch.cuda.empty_cache()
            return result


def create_scene_rgb_providers(
    config: Any, *, scene_names: list[str] | tuple[str, ...], device: Any
) -> Mapping[str, Huge3DGSRGBProvider | None]:
    parsed = Huge3DGSConfig.from_config(config)
    if not parsed.enabled:
        return {name: None for name in scene_names}
    allowed_scene = _value(config, "huge_3dgs_scene_name", "huge_1_office")
    unsupported = [name for name in scene_names if name != allowed_scene]
    if unsupported:
        raise ValueError(
            "HUGE 3DGS RGB adapter is not calibrated for: " + ", ".join(unsupported)
        )
    return {
        name: Huge3DGSRGBProvider(parsed, output_device=device) for name in scene_names
    }


def capture_planning_observation(
    camera: Any, mesh: Any, rgb_provider: Huge3DGSRGBProvider | None
):
    """Capture benchmark geometry, then replace only RGB with official 3DGS."""
    import torch

    if rgb_provider is None:
        return camera.capture_image(mesh)

    # HUGE's simplified collision/evaluation mesh deliberately has no UV/MTL.
    # Calling the shaded MeshRenderer would therefore fail in sample_textures.
    # Rasterize geometry only for the benchmark z-buffer/mask, while the
    # observation RGB comes exclusively from the official Gaussian PLY.
    with torch.no_grad():
        fragments = camera.renderer.rasterizer(mesh, cameras=camera.fov_camera)
        depth = fragments.zbuf
        # PyTorch3D's mesh rasterizer launches asynchronously.  Synchronize the
        # planner GPU before entering the RGB provider on another CUDA device;
        # otherwise repeated captures can surface a stale illegal-access error
        # at an unrelated empty_cache()/device-copy call in the provider.
        torch.cuda.synchronize(camera.device)
    rendered = rgb_provider.render(camera)
    frame_id = camera.n_frames_captured
    frame_path = Path(camera.save_dir_path) / f"{frame_id}.pt"
    frame = {
        "rgb": rendered["rgb"],
        "zbuf": depth,
        "mask": depth > -1,
        "R": camera.fov_camera.R,
        "T": camera.fov_camera.T,
        "zfar": camera.zfar,
        "observation_rgb_source": rendered["source"],
        "observation_rgb_metadata": rendered["metadata"],
    }
    temporary = frame_path.with_suffix(".pt.tmp")
    torch.save(frame, temporary)
    os.replace(temporary, frame_path)

    from torchvision.transforms.functional import to_pil_image

    png_path = frame_path.parent.parent / "imgs" / f"{frame_id}.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(rendered["rgb"][0].permute(2, 0, 1).cpu()).save(png_path)
    camera.n_frames_captured += 1
    return rendered["rgb"], depth
