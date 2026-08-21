"""Geometry checks for validated UE5 six-face observation bundles."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np

from .ue5_observation_contract import CanonicalFace, CanonicalObservationBundle


def camera_ray_directions(face: CanonicalFace) -> np.ndarray:
    """Return unit camera-frame rays with OpenCV x-right/y-down/z-forward."""

    height, width = face.image_size
    rows, columns = np.indices((height, width), dtype=np.float64)
    intrinsic = np.asarray(face.K_pixel, dtype=np.float64)
    rays = np.stack(
        (
            (columns - intrinsic[0, 2]) / intrinsic[0, 0],
            (rows - intrinsic[1, 2]) / intrinsic[1, 1],
            np.ones((height, width), dtype=np.float64),
        ),
        axis=-1,
    )
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def backproject_face(face: CanonicalFace) -> Tuple[np.ndarray, np.ndarray]:
    """Backproject valid canonical ray ranges to right-handed world points."""

    rays = camera_ray_directions(face)
    ranges = np.asarray(face.depth_range_m, dtype=np.float64)
    mask = np.asarray(face.valid_mask, dtype=bool)
    camera_points = rays[mask] * ranges[mask, None]
    transform = np.asarray(face.T_world_from_cam, dtype=np.float64)
    world_points = camera_points @ transform[:3, :3].T + transform[:3, 3]
    colors = np.asarray(face.rgb_uint8, dtype=np.uint8)[mask]
    return world_points, colors


def bundle_pointcloud(
    bundle: CanonicalObservationBundle,
) -> Tuple[np.ndarray, np.ndarray]:
    point_sets = []
    color_sets = []
    for face in bundle.faces:
        points, colors = backproject_face(face)
        point_sets.append(points)
        color_sets.append(colors)
    return np.concatenate(point_sets), np.concatenate(color_sets)


def pointcloud_summary(bundle: CanonicalObservationBundle) -> Dict[str, Any]:
    points, _ = bundle_pointcloud(bundle)
    center = np.asarray(bundle.position_world_m, dtype=np.float64)
    radii = np.linalg.norm(points - center, axis=1)
    return {
        "point_count": int(points.shape[0]),
        "bounds_min_m": points.min(axis=0).tolist(),
        "bounds_max_m": points.max(axis=0).tolist(),
        "radius_percentiles_m": {
            str(percentile): float(np.percentile(radii, percentile))
            for percentile in (1, 50, 95, 99, 100)
        },
    }


def _boundary_coordinates(height: int, width: int) -> np.ndarray:
    coordinates = []
    coordinates.extend((0, column) for column in range(width))
    coordinates.extend((height - 1, column) for column in range(width))
    coordinates.extend((row, 0) for row in range(1, height - 1))
    coordinates.extend((row, width - 1) for row in range(1, height - 1))
    return np.asarray(coordinates, dtype=np.int64)


def _boundary_payload(face: CanonicalFace) -> Dict[str, np.ndarray]:
    height, width = face.image_size
    coordinates = _boundary_coordinates(height, width)
    rows, columns = coordinates[:, 0], coordinates[:, 1]
    camera_rays = camera_ray_directions(face)[rows, columns]
    transform = np.asarray(face.T_world_from_cam, dtype=np.float64)
    world_rays = camera_rays @ transform[:3, :3].T
    ranges = np.asarray(face.depth_range_m, dtype=np.float64)[rows, columns]
    mask = np.asarray(face.valid_mask, dtype=bool)[rows, columns]
    points = (
        world_rays * np.nan_to_num(ranges, nan=0.0)[:, None]
        + transform[:3, 3]
    )
    return {"rays": world_rays, "ranges": ranges, "mask": mask, "points": points}


def seam_summary(bundle: CanonicalObservationBundle) -> Dict[str, Any]:
    """Compare duplicated boundary rays across every adjacent face pair."""

    payloads = {face.face_name: _boundary_payload(face) for face in bundle.faces}
    seams = []
    names = [face.face_name for face in bundle.faces]
    for first_index, first_name in enumerate(names):
        first = payloads[first_name]
        for second_name in names[first_index + 1 :]:
            second = payloads[second_name]
            first_forward = first["rays"].mean(axis=0)
            second_forward = second["rays"].mean(axis=0)
            if abs(float(first_forward @ second_forward)) > 1e-6:
                continue
            similarities = first["rays"] @ second["rays"].T
            nearest = np.argmax(similarities, axis=1)
            best = similarities[np.arange(similarities.shape[0]), nearest]
            # Adjacent 90-degree UE faces do not duplicate the mathematical
            # boundary ray: their outer pixel centres are one raster sample
            # apart. Select the N closest boundary samples for this seam.
            seam_sample_count = max(bundle.faces[0].image_size)
            matched_first = np.argsort(best)[-seam_sample_count:]
            matched_second = nearest[matched_first]
            first_valid = first["mask"][matched_first]
            second_valid = second["mask"][matched_second]
            both_valid = first_valid & second_valid
            agreement = first_valid == second_valid
            distances = np.linalg.norm(
                first["points"][matched_first[both_valid]]
                - second["points"][matched_second[both_valid]],
                axis=1,
            )
            mean_ranges = 0.5 * (
                first["ranges"][matched_first[both_valid]]
                + second["ranges"][matched_second[both_valid]]
            )
            relative_distances = distances / np.maximum(mean_ranges, 1e-12)
            seams.append(
                {
                    "faces": [first_name, second_name],
                    "matched_boundary_ray_count": int(matched_first.size),
                    "angular_gap_degrees_median": float(
                        np.degrees(np.arccos(np.clip(np.median(best[matched_first]), -1.0, 1.0)))
                    ),
                    "angular_gap_degrees_max": float(
                        np.degrees(np.arccos(np.clip(np.min(best[matched_first]), -1.0, 1.0)))
                    ),
                    "validity_agreement_fraction": float(np.mean(agreement)),
                    "both_valid_count": int(both_valid.sum()),
                    "world_point_distance_median_m": (
                        float(np.median(distances)) if distances.size else None
                    ),
                    "world_point_distance_p95_m": (
                        float(np.percentile(distances, 95)) if distances.size else None
                    ),
                    "world_point_distance_max_m": (
                        float(distances.max()) if distances.size else None
                    ),
                    "relative_world_point_distance_p95": (
                        float(np.percentile(relative_distances, 95))
                        if relative_distances.size
                        else None
                    ),
                }
            )
    comparable = [item for item in seams if item["both_valid_count"] > 0]
    return {
        "seams": seams,
        "seam_count": len(seams),
        "comparable_seam_count": len(comparable),
        "min_validity_agreement_fraction": min(
            item["validity_agreement_fraction"] for item in seams
        ),
        "max_world_point_distance_p95_m": max(
            item["world_point_distance_p95_m"] for item in comparable
        ),
        "max_world_point_distance_median_m": max(
            item["world_point_distance_median_m"] for item in comparable
        ),
        "max_relative_world_point_distance_p95": max(
            item["relative_world_point_distance_p95"] for item in comparable
        ),
    }


def forward_directions(bundle: CanonicalObservationBundle) -> np.ndarray:
    return np.stack(
        [np.asarray(face.T_world_from_cam, dtype=np.float64)[:3, 2] for face in bundle.faces]
    )


def direction_gram(directions: Iterable[Iterable[float]]) -> np.ndarray:
    values = np.asarray(tuple(directions), dtype=np.float64)
    return values @ values.T


__all__ = [
    "backproject_face",
    "bundle_pointcloud",
    "camera_ray_directions",
    "direction_gram",
    "forward_directions",
    "pointcloud_summary",
    "seam_summary",
]
