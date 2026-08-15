"""Opt-in diagnostics for experiments that cross a scene-space tile seam.

The helpers in this module never influence candidate filtering or ranking.  They
only serialize state that the planner already computed and derive coverage and
trajectory summaries after the corresponding observations exist.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


NEIGHBOR_SHIFTS = (
    (1, 0, 0, 0, 0),
    (-1, 0, 0, 0, 0),
    (0, 1, 0, 0, 0),
    (0, -1, 0, 0, 0),
    (0, 0, 1, 0, 0),
    (0, 0, -1, 0, 0),
    (0, 0, 0, 1, 0),
    (0, 0, 0, -1, 0),
    (0, 0, 0, 0, 1),
    (0, 0, 0, 0, -1),
)


def validate_tile_partition(partition: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate an ordered N-tile partition along any runtime world axis."""

    if not isinstance(partition, Mapping):
        raise ValueError("tile partition must be an object")
    axis = partition.get("axis")
    if isinstance(axis, bool) or not isinstance(axis, int) or axis not in (0, 1, 2):
        raise ValueError("tile partition axis must be 0, 1, or 2")
    boundaries = partition.get("boundaries")
    tile_ids = partition.get("tile_ids")
    if not isinstance(boundaries, Sequence) or isinstance(boundaries, (str, bytes)):
        raise ValueError("tile partition boundaries must be an array")
    if not isinstance(tile_ids, Sequence) or isinstance(tile_ids, (str, bytes)):
        raise ValueError("tile partition tile_ids must be an array")
    normalized_boundaries = []
    for value in boundaries:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("tile partition boundaries must be finite numbers")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("tile partition boundaries must be finite numbers")
        normalized_boundaries.append(value)
    if any(
        right <= left
        for left, right in zip(normalized_boundaries, normalized_boundaries[1:])
    ):
        raise ValueError("tile partition boundaries must be strictly increasing")
    normalized_ids = [str(value) for value in tile_ids]
    if len(normalized_ids) != len(normalized_boundaries) + 1:
        raise ValueError("tile_ids must contain exactly len(boundaries) + 1 values")
    if any(not value for value in normalized_ids) or len(set(normalized_ids)) != len(
        normalized_ids
    ):
        raise ValueError("tile_ids must be non-empty and unique")
    return {
        "axis": axis,
        "boundaries": normalized_boundaries,
        "tile_ids": normalized_ids,
        "interval_semantics": "(-inf,b0], (b0,b1], ..., (bN,+inf)",
    }


def _list(value: Any) -> list:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value).tolist()


def audit_neighbor_generation(
    pose_index: Sequence[int], pose_shape: Sequence[int]
) -> Mapping[str, Any]:
    """Describe the ten intended move/rotation actions before clamping.

    ``Camera.get_neighboring_poses`` clamps spatial/elevation coordinates and
    wraps azimuth.  Explicitly retaining the pre-clamp attempts lets an
    experiment distinguish a boundary rejection from a missing candidate.
    """

    current = np.asarray(_list(pose_index), dtype=np.int64)
    shape = np.asarray(_list(pose_shape), dtype=np.int64)
    if current.shape != (5,) or shape.shape != (5,):
        raise ValueError("pose_index and pose_shape must each contain five values")
    attempts = []
    for shift in NEIGHBOR_SHIFTS:
        raw = current + np.asarray(shift, dtype=np.int64)
        boundary = bool(np.any(raw[:4] < 0) or np.any(raw[:4] >= shape[:4]))
        wrapped = raw.copy()
        wrapped[4] %= shape[4]
        attempts.append(
            {
                "shift": list(shift),
                "raw_pose_index": raw.tolist(),
                "pose_index": wrapped.tolist(),
                "rejection_reason": "boundary" if boundary else None,
            }
        )
    return {
        "attempted_count": len(attempts),
        "boundary_rejected_count": sum(
            item["rejection_reason"] == "boundary" for item in attempts
        ),
        "attempts": attempts,
    }


def compute_tiled_scene_coverage(
    gt_scene: Any,
    recovered_scene: Any,
    *,
    tile_partition: Mapping[str, Any],
    surface_epsilon: float,
    normalization: float,
    reconstruction_points: Any,
) -> Mapping[str, Any]:
    """Compute exact N-tile coverage using the native cell/reference contract.

    ``raw`` is the covered-reference fraction within a tile. ``normalized`` is
    divided by the same visibility calibration as global coverage, so the two
    tile numerators/denominators recombine exactly into the global reference.
    """

    if normalization <= 0:
        raise ValueError("normalization must be positive")
    import torch

    partition = validate_tile_partition(tile_partition)
    tile_ids = partition["tile_ids"]
    counts = {
        tile_id: {"covered_points": 0, "reference_points": 0}
        for tile_id in tile_ids
    }
    boundaries = torch.tensor(
        partition["boundaries"], dtype=torch.float64
    )
    epsilon = float(surface_epsilon)
    for key, gt_cell in gt_scene.cells.items():
        gt_points = gt_cell.cell_pts
        if len(gt_points) == 0:
            continue
        recovered_points = recovered_scene.cells[key].cell_pts
        if len(recovered_points) > 0:
            distance = torch.cdist(
                gt_points.double(), recovered_points.double(), p=2.0
            ).min(dim=-1)[0]
            covered = distance < epsilon
        else:
            covered = torch.zeros(len(gt_points), dtype=torch.bool, device=gt_points.device)
        device_boundaries = boundaries.to(gt_points.device)
        assignments = torch.bucketize(
            gt_points[:, partition["axis"]].double(),
            device_boundaries,
            right=False,
        )
        for tile_index, tile_id in enumerate(tile_ids):
            mask = assignments == tile_index
            counts[tile_id]["reference_points"] += int(mask.sum().item())
            counts[tile_id]["covered_points"] += int((covered & mask).sum().item())

    reconstruction = np.asarray(_list(reconstruction_points), dtype=np.float64)
    if reconstruction.size:
        reconstruction = reconstruction.reshape(-1, 3)
        assignments = np.searchsorted(
            np.asarray(partition["boundaries"], dtype=np.float64),
            reconstruction[:, partition["axis"]],
            side="left",
        )
        reconstructed_counts = {
            tile_id: int(np.sum(assignments == tile_index))
            for tile_index, tile_id in enumerate(tile_ids)
        }
    else:
        reconstructed_counts = {tile_id: 0 for tile_id in tile_ids}

    total_covered = 0
    total_reference = 0
    for tile_id in tile_ids:
        numerator = counts[tile_id]["covered_points"]
        denominator = counts[tile_id]["reference_points"]
        raw = numerator / denominator if denominator else 0.0
        counts[tile_id].update(
            {
                "raw": float(raw),
                "normalized": float(raw / normalization),
                "reconstruction_points": reconstructed_counts[tile_id],
            }
        )
        total_covered += numerator
        total_reference += denominator
    combined_raw = total_covered / total_reference if total_reference else 0.0
    return {
        "tile_partition": partition,
        "normalization": float(normalization),
        "raw_definition": "covered_reference_points / tile_reference_points",
        "normalized_definition": "raw / global_visibility_ratio",
        "tiles": counts,
        "combined": {
            "covered_points": total_covered,
            "reference_points": total_reference,
            "raw": float(combined_raw),
            "normalized": float(combined_raw / normalization),
            "reconstruction_points": sum(reconstructed_counts.values()),
        },
    }


def compute_partitioned_scene_coverage(
    gt_scene: Any,
    recovered_scene: Any,
    *,
    seam_x: float,
    surface_epsilon: float,
    normalization: float,
    reconstruction_points: Any,
) -> Mapping[str, Any]:
    """Backward-compatible two-tile wrapper around N-tile coverage."""

    result = dict(
        compute_tiled_scene_coverage(
            gt_scene,
            recovered_scene,
            tile_partition={
                "axis": 0,
                "boundaries": [float(seam_x)],
                "tile_ids": ["tile_1", "tile_2"],
            },
            surface_epsilon=surface_epsilon,
            normalization=normalization,
            reconstruction_points=reconstruction_points,
        )
    )
    tiles = result.pop("tiles")
    result.update(
        {
            "seam_x": float(seam_x),
            "tile_1": tiles["tile_1"],
            "tile_2": tiles["tile_2"],
        }
    )
    return result


def _longest_true_run(values: Iterable[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def summarize_cross_tile_trajectory(
    positions: Any,
    pose_indices: Any,
    *,
    seam_x: float,
    tile_coverage: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    positions_array = np.asarray(_list(positions), dtype=np.float64).reshape(-1, 3)
    indices_array = np.asarray(_list(pose_indices), dtype=np.int64)
    if indices_array.size:
        indices_array = indices_array.reshape(-1, 5)
    else:
        indices_array = np.zeros((0, 5), dtype=np.int64)
    if len(indices_array) not in (0, len(positions_array)):
        raise ValueError("pose_indices must be empty or match positions")

    tile_2_mask = positions_array[:, 0] > seam_x if len(positions_array) else np.zeros(0, bool)
    crossing = np.flatnonzero(tile_2_mask)
    returns = int(np.sum(tile_2_mask[:-1] & ~tile_2_mask[1:])) if len(tile_2_mask) > 1 else 0
    differences = np.diff(positions_array, axis=0) if len(positions_array) > 1 else np.zeros((0, 3))
    lengths = np.linalg.norm(differences, axis=1)

    positive_x = negative_x = other_translation = in_place_rotation = 0
    x_visits = Counter()
    if len(indices_array):
        x_visits.update(int(value) for value in indices_array[:, 0])
        for delta in np.diff(indices_array, axis=0):
            if delta[0] > 0:
                positive_x += 1
            elif delta[0] < 0:
                negative_x += 1
            elif np.any(delta[1:3] != 0):
                other_translation += 1
            elif np.any(delta[3:] != 0):
                in_place_rotation += 1

    tile_2_reconstruction = [
        int(frame["tile_2"]["reconstruction_points"]) > 0
        for frame in tile_coverage
    ]
    tile_2_raw = [float(frame["tile_2"]["raw"]) for frame in tile_coverage]
    tile_2_normalized = [
        float(frame["tile_2"]["normalized"]) for frame in tile_coverage
    ]
    nonzero_coverage = [index for index, value in enumerate(tile_2_raw) if value > 0]
    return {
        "first_crossing_frame": int(crossing[0]) if len(crossing) else None,
        "tile_2_observations": int(tile_2_mask.sum()),
        "tile_2_observation_fraction": (
            float(tile_2_mask.mean()) if len(tile_2_mask) else 0.0
        ),
        "tile_2_longest_consecutive_stay": _longest_true_run(tile_2_mask),
        "returns_to_tile_1": returns,
        "x_min": float(positions_array[:, 0].min()) if len(positions_array) else None,
        "x_max": float(positions_array[:, 0].max()) if len(positions_array) else None,
        "final_tile": (
            "tile_2" if len(tile_2_mask) and tile_2_mask[-1] else "tile_1"
        ),
        "actions": {
            "positive_x": positive_x,
            "negative_x": negative_x,
            "other_translation": other_translation,
            "in_place_rotation": in_place_rotation,
        },
        "path_length_scene_units": float(lengths.sum()),
        "effective_translation_length_scene_units": float(lengths[lengths > 0].sum()),
        "x_grid_level_visits": {str(key): value for key, value in sorted(x_visits.items())},
        "tile_2_coverage_first_nonzero_frame": (
            nonzero_coverage[0] if nonzero_coverage else None
        ),
        "tile_2_raw_coverage_delta": (
            tile_2_raw[-1] - tile_2_raw[0] if tile_2_raw else 0.0
        ),
        "tile_2_normalized_coverage_delta": (
            tile_2_normalized[-1] - tile_2_normalized[0]
            if tile_2_normalized
            else 0.0
        ),
        "tile_2_longest_nonempty_reconstruction_run": _longest_true_run(
            tile_2_reconstruction
        ),
    }
