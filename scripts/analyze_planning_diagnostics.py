#!/usr/bin/env python3
"""Compute post-run depth/geometry diagnostics against renderer GT.

This command is deliberately separate from both online planners.  It reads a
completed online metrics record plus retained capture files, and its outputs
are never consumed by DA3 or by either next-best-view policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _load_capture(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _flatten_map(value: Any) -> np.ndarray:
    array = _numpy(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    while array.ndim > 2 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected a single HxW map, got {array.shape}.")
    return array


def _prediction(
    frame_metric: Mapping[str, Any], capture: Mapping[str, Any], capture_dir: Path
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    source = frame_metric["source"].upper()
    if source == "GT":
        depth = np.clip(_flatten_map(capture["zbuf"]), 0.5, 750.0)
        mask = _flatten_map(capture["mask"]).astype(bool)
        return depth, mask, None
    if source != "DA3":
        raise ValueError(f"Unsupported diagnostic source: {source}")
    cache_key = frame_metric.get("cache_key")
    if not cache_key:
        raise ValueError("DA3 diagnostic requires an online cache_key.")
    cache_path = capture_dir / ".da3_cache" / f"{cache_key}.npz"
    with np.load(cache_path, allow_pickle=False) as cached:
        depth = _flatten_map(cached["depth_z"]).astype(np.float64)
        valid = _flatten_map(cached["valid_mask"]).astype(bool)
        accepted = _flatten_map(cached["error_mask"]).astype(bool)
        confidence = _flatten_map(cached["confidence"]).astype(np.float64)
    return depth, valid & accepted, confidence


def _error_metrics(prediction: np.ndarray, target: np.ndarray) -> Mapping[str, float]:
    error = prediction - target
    absolute = np.abs(error)
    ratio = np.maximum(prediction / target, target / prediction)
    return {
        "mae": float(np.mean(absolute)),
        "median_ae": float(np.median(absolute)),
        "p90_ae": float(np.percentile(absolute, 90)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "abs_rel": float(np.mean(absolute / target)),
        "delta_1_25": float(np.mean(ratio < 1.25)),
    }


def _fit_metrics(prediction: np.ndarray, target: np.ndarray) -> Mapping[str, Any]:
    denominator = float(np.dot(prediction, prediction))
    scale = float(np.dot(prediction, target) / denominator) if denominator else 1.0
    design = np.column_stack((prediction, np.ones_like(prediction)))
    affine, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return {
        "scale_only_factor": scale,
        "scale_only": _error_metrics(prediction * scale, target),
        "affine_scale": float(affine[0]),
        "affine_shift": float(affine[1]),
        "affine": _error_metrics(design @ affine, target),
    }


def _default_intrinsics(height: int, width: int) -> np.ndarray:
    scale = min(height, width) / 2.0
    focal = scale / np.tan(np.deg2rad(60.0) / 2.0)
    return np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _ray_geometry_error(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    intrinsics: Optional[Any],
    scene_units_per_meter: float,
) -> np.ndarray:
    height, width = prediction.shape
    K = (
        np.asarray(intrinsics, dtype=np.float64)
        if intrinsics is not None
        else _default_intrinsics(height, width)
    )
    yy, xx = np.indices((height, width), dtype=np.float64)
    ray_norm = np.sqrt(
        ((xx - K[0, 2]) / K[0, 0]) ** 2
        + ((yy - K[1, 2]) / K[1, 1]) ** 2
        + 1.0
    )
    return np.abs(prediction[mask] - target[mask]) * ray_norm[mask] / scene_units_per_meter


def _confidence_metrics(confidence: Optional[np.ndarray], error: np.ndarray) -> Optional[Mapping[str, Any]]:
    if confidence is None:
        return None
    confidence = confidence.reshape(-1)
    error = error.reshape(-1)
    finite = np.isfinite(confidence) & np.isfinite(error)
    confidence = confidence[finite]
    error = error[finite]
    if not confidence.size:
        return None
    correlation = None
    if np.std(confidence) > 0 and np.std(error) > 0:
        correlation = float(np.corrcoef(confidence, error)[0, 1])
    edges = np.percentile(confidence, [0, 25, 50, 75, 100])
    bins = []
    for index in range(4):
        selected = (confidence >= edges[index]) & (
            confidence <= edges[index + 1] if index == 3 else confidence < edges[index + 1]
        )
        bins.append(
            {
                "confidence_min": float(edges[index]),
                "confidence_max": float(edges[index + 1]),
                "pixel_count": int(selected.sum()),
                "mae_meters": float(np.mean(error[selected])) if selected.any() else None,
            }
        )
    return {"pearson_confidence_vs_abs_error": correlation, "quartiles": bins}


def analyze_online_metrics(online: Mapping[str, Any]) -> Mapping[str, Any]:
    if not online.get("online_only") or online.get("renderer_gt_read") is not False:
        raise ValueError("Input must be a leakage-safe online metrics record.")
    capture_dir = Path(online["capture_dir"])
    units_per_meter = float(online["scene_units_per_meter"])
    frame_results = []
    all_prediction_m = []
    all_target_m = []
    all_geometry_m = []
    all_confidence = []
    all_confidence_error_m = []

    for frame_metric in online["frames"]:
        frame_id = int(frame_metric["frame_id"])
        capture = _load_capture(capture_dir / f"{frame_id}.pt")
        gt_depth = _flatten_map(capture["zbuf"]).astype(np.float64)
        gt_mask = _flatten_map(capture["mask"]).astype(bool)
        prediction, prediction_mask, confidence = _prediction(
            frame_metric, capture, capture_dir
        )
        mask = (
            gt_mask
            & prediction_mask
            & np.isfinite(gt_depth)
            & np.isfinite(prediction)
            & (gt_depth > 0)
            & (prediction > 0)
        )
        if not mask.any():
            frame_results.append({"frame_id": frame_id, "paired_pixels": 0})
            continue
        pred_m = prediction[mask] / units_per_meter
        target_m = gt_depth[mask] / units_per_meter
        geometry_m = _ray_geometry_error(
            prediction,
            gt_depth,
            mask,
            frame_metric.get("intrinsics"),
            units_per_meter,
        )
        confidence_values = confidence[mask] if confidence is not None else None
        result = {
            "frame_id": frame_id,
            "paired_pixels": int(mask.sum()),
            "paired_gt_pixel_fraction": float(mask.sum() / max(gt_mask.sum(), 1)),
            "depth_meters": _error_metrics(pred_m, target_m),
            "test_fit_diagnostic_only": _fit_metrics(pred_m, target_m),
            "paired_ray_geometry_meters": {
                "mae": float(np.mean(geometry_m)),
                "median": float(np.median(geometry_m)),
                "p90": float(np.percentile(geometry_m, 90)),
                "rmse": float(np.sqrt(np.mean(geometry_m**2))),
            },
            "confidence": _confidence_metrics(
                confidence_values, np.abs(pred_m - target_m)
            ),
        }
        frame_results.append(result)
        all_prediction_m.append(pred_m)
        all_target_m.append(target_m)
        all_geometry_m.append(geometry_m)
        if confidence_values is not None:
            all_confidence.append(confidence_values)
            all_confidence_error_m.append(np.abs(pred_m - target_m))

    aggregate = None
    if all_prediction_m:
        prediction_m = np.concatenate(all_prediction_m)
        target_m = np.concatenate(all_target_m)
        geometry_m = np.concatenate(all_geometry_m)
        aggregate = {
            "paired_pixels": int(len(prediction_m)),
            "depth_meters": _error_metrics(prediction_m, target_m),
            "test_fit_diagnostic_only": _fit_metrics(prediction_m, target_m),
            "paired_ray_geometry_meters": {
                "mae": float(np.mean(geometry_m)),
                "median": float(np.median(geometry_m)),
                "p90": float(np.percentile(geometry_m, 90)),
                "rmse": float(np.sqrt(np.mean(geometry_m**2))),
            },
            "confidence": (
                _confidence_metrics(
                    np.concatenate(all_confidence), np.concatenate(all_confidence_error_m)
                )
                if all_confidence
                else None
            ),
        }
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "computed_after_planner_exit": True,
        "feedback_to_online_planner": False,
        "renderer_zbuf_read": True,
        "source_online_metrics": online.get("run", {}).get("run_id"),
        "planner": online["planner"],
        "scene": online["scene"],
        "start_index": online["start_index"],
        "scene_units_per_meter": units_per_meter,
        "frames": frame_results,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    online = json.loads(Path(args.online_json).read_text(encoding="utf-8"))
    result = analyze_online_metrics(online)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
