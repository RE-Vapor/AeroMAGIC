"""Depth-source selection and the shared planning-depth contract.

This module deliberately has no eager PyTorch import.  Source selection can
therefore fail fast while parsing configuration, before heavyweight model
dependencies are imported or constructed.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
import os
from typing import Any, Callable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class DepthObservation:
    """Inputs that identify the frame requested from a depth provider."""

    camera: Any
    device: Any = None
    frame_ids: Optional[Sequence[int]] = None
    frame_history: Optional[Sequence[Mapping[str, Any]]] = None
    cache_namespace: Optional[str] = None


@dataclass(frozen=True)
class DepthFrame:
    """Canonical depth result consumed by the planning testers.

    Tensor shapes follow the existing MAGICIAN convention: RGB is
    ``(1, H, W, 3)`` and depth/masks are ``(1, H, W, 1)``. ``depth_z`` is
    PyTorch3D view-space Z, not Euclidean ray distance.
    """

    rgb: Any
    depth_z: Any
    valid_mask: Any
    error_mask: Any
    R: Any
    T: Any
    confidence: Optional[Any] = None
    source: str = ""
    frame_id: Optional[int] = None
    cache_metadata: Mapping[str, Any] = field(default_factory=dict)


class DepthProvider(ABC):
    """Narrow interface implemented by every planning depth source."""

    source: str

    @abstractmethod
    def get_frame(self, observation: DepthObservation) -> DepthFrame:
        """Return depth for one captured RGB observation."""


DepthProviderFactory = Callable[..., DepthProvider]
_DEPTH_PROVIDER_REGISTRY = {}


def _normalize_kind(kind: str) -> str:
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("Depth backend names must be non-empty strings.")
    return kind.strip().upper()


def register_depth_provider(
    kind: str,
    provider_factory: DepthProviderFactory,
    *,
    registry=None,
    allow_override: bool = False,
) -> None:
    """Register a provider factory under a case-insensitive backend name.

    Factories receive keyword arguments ``config`` and ``device``.  Passing a
    registry is useful for isolated tests and plugin composition.
    """

    target = _DEPTH_PROVIDER_REGISTRY if registry is None else registry
    normalized = _normalize_kind(kind)
    if not callable(provider_factory):
        raise TypeError("provider_factory must be callable.")
    if normalized in target and not allow_override:
        raise ValueError(f"Depth backend '{normalized}' is already registered.")
    target[normalized] = provider_factory


def registered_depth_providers(*, registry=None):
    """Return registered backend names in deterministic order."""

    target = _DEPTH_PROVIDER_REGISTRY if registry is None else registry
    return tuple(sorted(target))


def _config_value(config: Any, name: str, missing: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, missing)
    return getattr(config, name, missing)


def _supported_non_gt_backends(registry) -> str:
    names = sorted(name for name in registry if name != "GT")
    return ", ".join(names) if names else "(none registered)"


def create_depth_provider(config: Any, *, device: Any = None, registry=None) -> DepthProvider:
    """Resolve and construct the configured planning depth source.

    ``use_perfect_depth_map=True`` always selects GT and intentionally ignores
    ``kind_depth_map``.  When it is false, ``kind_depth_map`` is mandatory and
    must name a registered non-GT backend; there is no GT fallback.
    """

    target = _DEPTH_PROVIDER_REGISTRY if registry is None else registry
    missing = object()
    use_gt = _config_value(config, "use_perfect_depth_map", missing)
    if use_gt is missing:
        raise ValueError("Missing required config field 'use_perfect_depth_map'.")
    if type(use_gt) is not bool:
        raise ValueError("Config field 'use_perfect_depth_map' must be a boolean.")

    if use_gt:
        factory = target.get("GT")
        if factory is None:
            raise RuntimeError("GT depth provider is not registered.")
        return factory(config=config, device=device)

    raw_kind = _config_value(config, "kind_depth_map", missing)
    if raw_kind is missing or raw_kind is None:
        raise ValueError(
            "Missing required config field 'kind_depth_map' when "
            "use_perfect_depth_map is false."
        )
    if not isinstance(raw_kind, str) or not raw_kind.strip():
        raise ValueError(
            "Config field 'kind_depth_map' must be a non-empty string when "
            "use_perfect_depth_map is false."
        )

    kind = raw_kind.strip().upper()
    factory = target.get(kind)
    if kind == "GT" or factory is None:
        supported = _supported_non_gt_backends(target)
        raise ValueError(
            f"Unsupported kind_depth_map={raw_kind!r}; supported: {supported}. "
            "Set use_perfect_depth_map=true to select GT."
        )
    return factory(config=config, device=device)


def _load_torch_frame(path: str, device: Any):
    import torch

    return torch.load(path, map_location=device)


def _clamp_torch_depth(depth: Any, minimum: float, maximum: float):
    import torch

    return torch.clamp(depth, min=minimum, max=maximum)


def _torch_bool(mask: Any):
    return mask.bool()


class GTDepthProvider(DepthProvider):
    """Adapter preserving the legacy perfect-depth tester behavior exactly."""

    source = "GT"

    def __init__(
        self,
        *,
        config: Any = None,
        device: Any = None,
        frame_loader: Callable[[str, Any], Mapping[str, Any]] = _load_torch_frame,
        clamp_depth: Callable[[Any, float, float], Any] = _clamp_torch_depth,
        mask_to_bool: Callable[[Any], Any] = _torch_bool,
    ):
        self.device = device
        self._frame_loader = frame_loader
        self._clamp_depth = clamp_depth
        self._mask_to_bool = mask_to_bool
        self.depth_minimum = _config_value(config, "gt_depth_minimum", 0.5)
        self.depth_maximum = _config_value(config, "gt_depth_maximum", 750.0)
        for name, value in (
            ("gt_depth_minimum", self.depth_minimum),
            ("gt_depth_maximum", self.depth_maximum),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite positive number.")
            if value <= 0:
                raise ValueError(f"{name} must be a finite positive number.")
        if self.depth_maximum <= self.depth_minimum:
            raise ValueError("gt_depth_maximum must exceed gt_depth_minimum.")

    def get_frame(self, observation: DepthObservation) -> DepthFrame:
        camera = observation.camera
        frame_id = camera.n_frames_captured - 1
        frame_path = os.path.join(camera.save_dir_path, f"{frame_id}.pt")
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Current frame does not exist: {frame_path}")

        device = observation.device if observation.device is not None else self.device
        frame = self._frame_loader(frame_path, device)
        required = ("rgb", "zbuf", "mask", "R", "T")
        missing = [name for name in required if name not in frame]
        if missing:
            raise KeyError(f"GT frame {frame_path} is missing keys: {', '.join(missing)}")

        valid_mask = self._mask_to_bool(frame["mask"])
        depth_z = self._clamp_depth(
            frame["zbuf"], self.depth_minimum, self.depth_maximum
        )
        return DepthFrame(
            rgb=frame["rgb"],
            depth_z=depth_z,
            valid_mask=valid_mask,
            error_mask=valid_mask,
            R=frame["R"],
            T=frame["T"],
            confidence=None,
            source=self.source,
            frame_id=frame_id,
            cache_metadata={"source": self.source, "frame_id": frame_id},
        )


register_depth_provider("GT", GTDepthProvider)


def _create_da3_provider(*, config: Any, device: Any) -> DepthProvider:
    # Keep the optional DA3 implementation and its heavyweight dependencies
    # out of the GT-only import path.
    from .da3_adapter import DA3DepthProvider

    return DA3DepthProvider(config=config, device=device)


register_depth_provider("DA3", _create_da3_provider)
