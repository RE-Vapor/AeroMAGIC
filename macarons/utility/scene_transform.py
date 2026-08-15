"""Validated, opt-in transforms for scene mesh vertices.

Dataset settings and meshes normally share the same coordinate frame.  Some
custom assets retain exporter/world coordinates and need an explicit transform
before MAGICIAN's existing ``scene_scale_factor`` is applied.  The helpers in
this module keep that adaptation scene-scoped and default to the identity.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


IDENTITY_SCENE_MESH_TRANSFORM = {
    "axis_order": (0, 1, 2),
    "axis_signs": (1.0, 1.0, 1.0),
    "translation": (0.0, 0.0, 0.0),
    "scale": 1.0,
}


def _config_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _finite_vector(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three finite numbers.")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain exactly three finite numbers.")
    return result


def resolve_scene_mesh_transform(config: Any, scene_name: str) -> dict[str, Any]:
    """Return a validated transform for ``scene_name`` or the identity.

    ``axis_order`` and ``axis_signs`` are applied first, followed by
    ``translation`` and ``scale``. MAGICIAN's existing
    ``scene_scale_factor`` remains a separate final step.
    """

    transforms = _config_value(config, "scene_mesh_transforms", {})
    if transforms is None:
        transforms = {}
    if not isinstance(transforms, Mapping):
        raise ValueError("scene_mesh_transforms must be an object keyed by scene name.")
    raw = transforms.get(scene_name)
    if raw is None:
        return dict(IDENTITY_SCENE_MESH_TRANSFORM)
    if not isinstance(raw, Mapping):
        raise ValueError(f"scene_mesh_transforms.{scene_name} must be an object.")

    axis_order = raw.get("axis_order", (0, 1, 2))
    if (
        not isinstance(axis_order, (list, tuple))
        or len(axis_order) != 3
        or any(isinstance(axis, bool) or not isinstance(axis, int) for axis in axis_order)
        or sorted(axis_order) != [0, 1, 2]
    ):
        raise ValueError(
            f"scene_mesh_transforms.{scene_name}.axis_order must be a permutation of [0, 1, 2]."
        )

    translation = _finite_vector(
        raw.get("translation", (0.0, 0.0, 0.0)),
        f"scene_mesh_transforms.{scene_name}.translation",
    )
    axis_signs = _finite_vector(
        raw.get("axis_signs", (1.0, 1.0, 1.0)),
        f"scene_mesh_transforms.{scene_name}.axis_signs",
    )
    if any(abs(sign) != 1.0 for sign in axis_signs):
        raise ValueError(
            f"scene_mesh_transforms.{scene_name}.axis_signs must contain only -1 or 1."
        )
    try:
        scale = float(raw.get("scale", 1.0))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"scene_mesh_transforms.{scene_name}.scale must be a positive finite number."
        ) from error
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"scene_mesh_transforms.{scene_name}.scale must be a positive finite number."
        )

    unknown = sorted(
        set(raw) - {"axis_order", "axis_signs", "translation", "scale"}
    )
    if unknown:
        raise ValueError(
            f"Unsupported scene mesh transform fields for {scene_name}: {', '.join(unknown)}"
        )
    return {
        "axis_order": tuple(axis_order),
        "axis_signs": axis_signs,
        "translation": translation,
        "scale": scale,
    }


def transform_scene_vertices(
    vertices: Any,
    transform: Mapping[str, Any],
    *,
    scene_scale_factor: float = 1.0,
) -> Any:
    """Apply a validated transform to NumPy- or Torch-like ``vertices``."""

    axis_order = list(transform["axis_order"])
    reordered = vertices[..., axis_order]
    axis_signs = transform["axis_signs"]
    translation = transform["translation"]
    if hasattr(reordered, "new_tensor"):
        axis_signs = reordered.new_tensor(axis_signs)
        translation = reordered.new_tensor(translation)
    else:
        # NumPy arrays and trimesh TrackedArray accept a same-length tuple.
        axis_signs = list(axis_signs)
        translation = list(translation)
    return (reordered * axis_signs + translation) * (
        float(transform["scale"]) * float(scene_scale_factor)
    )
