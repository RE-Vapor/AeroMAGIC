"""Position-only planner state for full-sphere PIONEER observations.

The legacy camera lattice remains five-dimensional because rendering still
needs a concrete camera pose.  This module keeps that representation behind a
small adapter: beam state and observation history contain XYZ indices only,
while renderer calls receive one explicitly configured canonical orientation.

An XYZ position becomes observed only through :meth:`capture_and_commit`, after
the capture callable has returned one complete six-face cubemap bundle.  Merely
generating a candidate, traversing a path, or updating the legacy ``Camera``
object cannot mutate the position-only observation history.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Callable, FrozenSet, Mapping, Sequence, Tuple, TypeVar

import torch


POSITION_ONLY_STATE_VERSION = "pioneer-position-only-v1"

FULL_SPHERE_FACE_NAMES: Tuple[str, ...] = (
    "front",
    "back",
    "left",
    "right",
    "up",
    "down",
)

POSITION_ACTIONS: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)

_Bundle = TypeVar("_Bundle")


def _positive_shape(values: Sequence[int], *, size: int, label: str) -> Tuple[int, ...]:
    normalized = tuple(values)
    if len(normalized) != size:
        raise ValueError(f"{label} must contain exactly {size} values.")
    if any(type(value) is not int or value < 1 for value in normalized):
        raise ValueError(f"{label} values must be positive integers.")
    return normalized


@dataclass(frozen=True)
class PositionOnlySpec:
    """Validated bridge between a 3D planner lattice and a 5D camera lattice."""

    position_shape: Tuple[int, int, int]
    orientation_shape: Tuple[int, int]
    canonical_orientation_index: Tuple[int, int] = (2, 0)

    def __post_init__(self) -> None:
        position_shape = _positive_shape(
            self.position_shape, size=3, label="position_shape"
        )
        orientation_shape = _positive_shape(
            self.orientation_shape, size=2, label="orientation_shape"
        )
        canonical = tuple(self.canonical_orientation_index)
        if len(canonical) != 2 or any(type(value) is not int for value in canonical):
            raise ValueError(
                "canonical_orientation_index must contain exactly two integers."
            )
        if any(
            value < 0 or value >= bound
            for value, bound in zip(canonical, orientation_shape)
        ):
            raise ValueError(
                "canonical_orientation_index must lie inside orientation_shape."
            )
        object.__setattr__(self, "position_shape", position_shape)
        object.__setattr__(self, "orientation_shape", orientation_shape)
        object.__setattr__(self, "canonical_orientation_index", canonical)

    @property
    def position_state_count(self) -> int:
        return int(prod(self.position_shape))

    @property
    def orientation_state_multiplier(self) -> int:
        return int(prod(self.orientation_shape))

    @property
    def legacy_pose_state_count(self) -> int:
        return self.position_state_count * self.orientation_state_multiplier


def _index_tensor(value: Any, *, size: int, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 0 or tensor.shape[-1] != size:
        raise ValueError(f"{label} must have final dimension {size}.")
    if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
        raise TypeError(f"{label} must contain integer indices.")
    return tensor.to(dtype=torch.long)


def _validate_xyz_bounds(xyz_index: torch.Tensor, spec: PositionOnlySpec) -> None:
    bounds = torch.tensor(
        spec.position_shape, dtype=xyz_index.dtype, device=xyz_index.device
    )
    if bool(((xyz_index < 0) | (xyz_index >= bounds)).any()):
        raise ValueError("XYZ index lies outside position_shape.")


def pose_index_to_xyz(pose_index: Any) -> torch.Tensor:
    """Project a legacy ``[..., 5]`` camera index to planner ``[..., 3]`` state."""

    pose = _index_tensor(pose_index, size=5, label="pose_index")
    return pose[..., :3].clone()


def xyz_to_canonical_pose_index(
    xyz_index: Any, spec: PositionOnlySpec
) -> torch.Tensor:
    """Attach the configured canonical elevation/azimuth to XYZ camera indices."""

    xyz = _index_tensor(xyz_index, size=3, label="xyz_index")
    _validate_xyz_bounds(xyz, spec)
    orientation = torch.tensor(
        spec.canonical_orientation_index, dtype=xyz.dtype, device=xyz.device
    )
    orientation = orientation.expand(*xyz.shape[:-1], 2)
    return torch.cat((xyz, orientation), dim=-1)


def position_key(xyz_index: Any, spec: PositionOnlySpec) -> Tuple[int, int, int]:
    """Return the immutable key used by position-only observation history."""

    xyz = _index_tensor(xyz_index, size=3, label="xyz_index")
    if xyz.ndim != 1:
        raise ValueError("position_key accepts exactly one XYZ index.")
    _validate_xyz_bounds(xyz, spec)
    return tuple(int(value) for value in xyz.detach().cpu().tolist())


def position_neighbors(xyz_index: Any, spec: PositionOnlySpec) -> torch.Tensor:
    """Return unique, in-bounds, six-connected XYZ neighbors.

    Out-of-bounds actions are rejected rather than clamped, so a boundary move
    never becomes a duplicate of the current state.
    """

    xyz = _index_tensor(xyz_index, size=3, label="xyz_index")
    if xyz.ndim != 1:
        raise ValueError("position_neighbors accepts exactly one XYZ index.")
    _validate_xyz_bounds(xyz, spec)
    actions = torch.tensor(POSITION_ACTIONS, dtype=xyz.dtype, device=xyz.device)
    candidates = xyz.reshape(1, 3) + actions
    bounds = torch.tensor(
        spec.position_shape, dtype=xyz.dtype, device=xyz.device
    ).reshape(1, 3)
    in_bounds = ((candidates >= 0) & (candidates < bounds)).all(dim=1)
    return candidates[in_bounds]


def _bundle_face_names(bundle: Any) -> Tuple[str, ...]:
    try:
        return tuple(str(face.name) for face in bundle.faces)
    except (AttributeError, TypeError) as error:
        raise ValueError("Capture result must expose an ordered faces collection.") from error


def validate_full_sphere_bundle(bundle: Any) -> None:
    """Reject partial or mislabeled capture results before seen-state mutation."""

    names = _bundle_face_names(bundle)
    if names != FULL_SPHERE_FACE_NAMES:
        raise ValueError(
            "Full-sphere capture must contain exactly the canonical six faces "
            f"{FULL_SPHERE_FACE_NAMES}; received {names}."
        )
    try:
        render_count = int(bundle.render_count)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Full-sphere capture must expose render_count=6.") from error
    if render_count != len(FULL_SPHERE_FACE_NAMES):
        raise ValueError("Full-sphere capture must report render_count=6.")
    metadata = getattr(bundle, "metadata", None)
    if not isinstance(metadata, Mapping) or metadata.get("observation_mode") != "cubemap6":
        raise ValueError(
            "Full-sphere capture metadata must declare observation_mode='cubemap6'."
        )


class PositionOnlyPlannerState:
    """XYZ observation history with transactionally committed cubemap captures."""

    def __init__(self, spec: PositionOnlySpec) -> None:
        if not isinstance(spec, PositionOnlySpec):
            raise TypeError("spec must be a PositionOnlySpec.")
        self.spec = spec
        self._observed_positions: set[Tuple[int, int, int]] = set()

    @property
    def observed_positions(self) -> FrozenSet[Tuple[int, int, int]]:
        return frozenset(self._observed_positions)

    def is_observed(self, xyz_index: Any) -> bool:
        return position_key(xyz_index, self.spec) in self._observed_positions

    def unseen_neighbors(self, xyz_index: Any) -> torch.Tensor:
        neighbors = position_neighbors(xyz_index, self.spec)
        if neighbors.shape[0] == 0:
            return neighbors
        unseen = [not self.is_observed(neighbor) for neighbor in neighbors]
        mask = torch.tensor(unseen, dtype=torch.bool, device=neighbors.device)
        return neighbors[mask]

    def valid_neighbors(
        self,
        xyz_index: Any,
        *,
        is_available: Callable[[torch.Tensor], bool] = None,
    ) -> torch.Tensor:
        """Prefer unseen available XYZ states, then backtrack through available ones."""

        neighbors = position_neighbors(xyz_index, self.spec)
        if neighbors.shape[0] == 0:
            return neighbors
        if is_available is not None:
            if not callable(is_available):
                raise TypeError("is_available must be callable.")
            available = [bool(is_available(neighbor)) for neighbor in neighbors]
            mask = torch.tensor(
                available, dtype=torch.bool, device=neighbors.device
            )
            neighbors = neighbors[mask]
            if neighbors.shape[0] == 0:
                return neighbors
        unseen_mask = torch.tensor(
            [not self.is_observed(neighbor) for neighbor in neighbors],
            dtype=torch.bool,
            device=neighbors.device,
        )
        unseen = neighbors[unseen_mask]
        return unseen if unseen.shape[0] > 0 else neighbors

    def commit_full_sphere_capture(self, xyz_index: Any, bundle: Any) -> None:
        """Commit one XYZ only after validating its completed six-face bundle."""

        key = position_key(xyz_index, self.spec)
        validate_full_sphere_bundle(bundle)
        self._observed_positions.add(key)

    def capture_and_commit(
        self,
        xyz_index: Any,
        capture_fn: Callable[..., _Bundle],
        *args: Any,
        **kwargs: Any,
    ) -> _Bundle:
        """Run capture first and commit only after its full-sphere contract passes."""

        if not callable(capture_fn):
            raise TypeError("capture_fn must be callable.")
        bundle = capture_fn(*args, **kwargs)
        self.commit_full_sphere_capture(xyz_index, bundle)
        return bundle


def position_only_structure_audit(spec: PositionOnlySpec) -> Mapping[str, Any]:
    """Describe structural state/action changes without claiming runtime speedup."""

    if not isinstance(spec, PositionOnlySpec):
        raise TypeError("spec must be a PositionOnlySpec.")
    legacy_orientation_actions = 4
    position_actions = len(POSITION_ACTIONS)
    return {
        "schema_version": 1,
        "state_version": POSITION_ONLY_STATE_VERSION,
        "legacy_pose_state_dimension": 5,
        "position_only_state_dimension": 3,
        "position_shape": list(spec.position_shape),
        "orientation_shape": list(spec.orientation_shape),
        "canonical_orientation_index": list(spec.canonical_orientation_index),
        "legacy_pose_state_count": spec.legacy_pose_state_count,
        "position_only_state_count": spec.position_state_count,
        "orientation_state_multiplier": spec.orientation_state_multiplier,
        "legacy_raw_action_branch_count": position_actions
        + legacy_orientation_actions,
        "pan10_effective_action_branch_count": position_actions,
        "position_only_action_branch_count": position_actions,
        "removed_orientation_action_branch_count": legacy_orientation_actions,
    }


__all__ = [
    "FULL_SPHERE_FACE_NAMES",
    "POSITION_ACTIONS",
    "POSITION_ONLY_STATE_VERSION",
    "PositionOnlyPlannerState",
    "PositionOnlySpec",
    "pose_index_to_xyz",
    "position_key",
    "position_neighbors",
    "position_only_structure_audit",
    "validate_full_sphere_bundle",
    "xyz_to_canonical_pose_index",
]
