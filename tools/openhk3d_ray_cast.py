#!/usr/bin/env python3
"""Numerically robust helpers for Blender BVH ray traversal."""

from typing import Any


def count_ray_intersections(
    tree: Any,
    point: Any,
    direction: Any,
    epsilon: float = 1e-6,
    max_iterations: int = 10000,
    max_sticky_hits: int = 32,
) -> int:
    """Count distinct surface crossings while escaping sticky BVH hits.

    Blender can return the same triangle again when a new ray starts only a
    floating-point epsilon beyond its previous intersection.  A triangle can
    intersect a straight ray at most once, so those hits are numerical repeats,
    not additional parity crossings.  Grow the nudge geometrically until the
    ray leaves that surface and count the crossing once.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_iterations <= 0 or max_sticky_hits <= 0:
        raise ValueError("ray traversal limits must be positive")

    count = 0
    origin = point.copy()
    previous_face = None
    last_crossing = None
    sticky_hits = 0
    for _ in range(max_iterations):
        location, _normal, face_index, _distance = tree.ray_cast(origin, direction)
        if location is None:
            return count

        same_face = previous_face is not None and face_index == previous_face
        same_crossing = (
            last_crossing is not None
            and (location - last_crossing).length <= epsilon * 4
        )
        if same_face or same_crossing:
            sticky_hits += 1
            if sticky_hits > max_sticky_hits:
                raise RuntimeError(
                    "BVH ray remained on one surface after {} escape attempts".format(
                        max_sticky_hits
                    )
                )
            nudge = epsilon * (2 ** min(sticky_hits, 20))
            origin = location + direction * nudge
            continue

        count += 1
        previous_face = face_index
        last_crossing = location.copy()
        sticky_hits = 0
        origin = location + direction * epsilon

    raise RuntimeError("BVH ray exceeded the intersection safety limit")
