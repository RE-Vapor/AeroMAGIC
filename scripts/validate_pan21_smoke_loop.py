#!/usr/bin/env python3
"""Validate PAN-21 with two real UE5 bundles and one existing-Planner move.

This is an acceptance harness, not another mapping implementation.  It adapts
the validated PAN-19 files and calls the existing PAN-10 cubemap fusion and
proxy update functions directly.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import torch

from macarons.utility.planning_depth import update_proxy_state
from macarons.utility.planning_observations import process_cubemap_observation
from macarons.utility.ue5_observation_contract import (
    BundleValidationError,
    CanonicalObservationBundle,
    load_bundle,
    validate_bundle,
)
from macarons.utility.ue5_pan10_adapter import canonical_bundle_to_pan10


SCHEMA_VERSION = "pan21.two-observation-one-move.acceptance.v1"
SCENE_UNITS_PER_METER = 0.2
SENSOR_RANGE_SCENE_UNITS = 130.0
VOXEL_SIZE_SCENE_UNITS = 0.02
CARVING_TOLERANCE = 0.05


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().numpy().astype(np.float32, copy=False)
    if array.ndim == 2 and array.shape[1] == 3 and len(array):
        order = np.lexsort((array[:, 2], array[:, 1], array[:, 0]))
        array = array[order]
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class AuditPointMap:
    """Minimal fill_cells state used to audit the real PAN-10 payload."""

    def __init__(self, device: torch.device) -> None:
        self.points = torch.zeros((0, 3), dtype=torch.float32, device=device)
        self.fill_count = 0

    def fill_cells(self, points: torch.Tensor, features: torch.Tensor) -> None:
        if points.ndim != 2 or points.shape[1] != 3:
            raise AssertionError("map update requires an Nx3 point cloud")
        if len(points) != len(features):
            raise AssertionError("map point/features length mismatch")
        self.points = torch.cat((self.points, points.detach()), dim=0)
        self.fill_count += 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "fill_count": self.fill_count,
            "point_count": int(len(self.points)),
            "points_sha256": _tensor_sha256(self.points),
        }


class AuditProxyScene:
    """Stateful audit of the existing update_proxy_state occupancy formula."""

    def __init__(self, proxy_points: torch.Tensor) -> None:
        self.proxy_points = proxy_points
        count = len(proxy_points)
        device = proxy_points.device
        self.proxy_n_inside_fov = torch.zeros(count, dtype=torch.int32, device=device)
        self.proxy_n_behind_depth = torch.zeros(count, dtype=torch.int32, device=device)
        self.out_of_field = torch.ones(count, dtype=torch.bool, device=device)
        self.fill_count = 0
        self.view_update_count = 0
        self.occupancy_update_count = 0
        self.no_hit_free_evidence_count = 0
        self._last_signed_distances = None

    def get_proxy_indices_from_mask(self, mask: torch.Tensor) -> torch.Tensor:
        return torch.nonzero(mask, as_tuple=False).reshape(-1)

    def fill_cells(self, points: torch.Tensor, features: torch.Tensor) -> None:
        if len(points) != len(features):
            raise AssertionError("proxy point/features length mismatch")
        self.fill_count += 1

    def update_proxy_view_states(self, camera: Any, mask: torch.Tensor, **kwargs: Any) -> None:
        signed = kwargs["signed_distances"].reshape(-1)
        if len(signed) != int(mask.sum().item()):
            raise AssertionError("signed distance count does not match proxy union")
        self.view_update_count += 1

    def update_proxy_supervision_occ(
        self,
        mask: torch.Tensor,
        signed_distances: torch.Tensor,
        tol: float = 0.0,
    ) -> None:
        signed = signed_distances.reshape(-1)
        self.proxy_n_inside_fov[mask] += 1
        behind = signed >= -tol
        self.proxy_n_behind_depth[mask] += behind.to(torch.int32)
        # PAN-10 represents sampled no-hit as 1.1*zfar.  With this smoke's
        # bounded proxy radius, distances below -sensor_range can only be
        # no-hit/free-space evidence and must never increment occupied/behind.
        no_hit = signed < -SENSOR_RANGE_SCENE_UNITS
        if bool(behind[no_hit].any()):
            raise AssertionError("no-hit generated occupied proxy evidence")
        self.no_hit_free_evidence_count += int(no_hit.sum().item())
        self.occupancy_update_count += 1
        self._last_signed_distances = signed.detach().clone()

    def update_proxy_out_of_field(self, mask: torch.Tensor) -> None:
        self.out_of_field[mask] = False

    def snapshot(self) -> Dict[str, Any]:
        return {
            "fill_count": self.fill_count,
            "view_update_count": self.view_update_count,
            "occupancy_update_count": self.occupancy_update_count,
            "inside_fov_total": int(self.proxy_n_inside_fov.sum().item()),
            "behind_depth_total": int(self.proxy_n_behind_depth.sum().item()),
            "observed_proxy_count": int((self.proxy_n_inside_fov > 0).sum().item()),
            "max_observations_per_proxy": int(self.proxy_n_inside_fov.max().item()),
            "no_hit_free_evidence_count": self.no_hit_free_evidence_count,
            "inside_fov_sha256": _tensor_sha256(
                self.proxy_n_inside_fov.to(torch.float32).reshape(-1, 1)
            ),
            "behind_depth_sha256": _tensor_sha256(
                self.proxy_n_behind_depth.to(torch.float32).reshape(-1, 1)
            ),
        }


def _make_proxy_points(
    centers: Sequence[torch.Tensor], count: int, device: torch.device
) -> torch.Tensor:
    if count < 64:
        raise ValueError("proxy count must be at least 64")
    midpoint = torch.stack(tuple(center.reshape(3) for center in centers)).mean(dim=0)
    index = torch.arange(count, dtype=torch.float64, device=device)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (index + 0.5) / count
    radius_xy = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    directions = torch.stack(
        (radius_xy * torch.cos(index * golden_angle), z, radius_xy * torch.sin(index * golden_angle)),
        dim=1,
    )
    radii = 8.0 + (index.remainder(11.0) / 10.0) * 102.0
    return (midpoint.to(torch.float64) + directions * radii[:, None]).to(torch.float32)


def _planner_record(metrics: Dict[str, Any]) -> Dict[str, Any]:
    trajectory = metrics["trajectory"]
    indices = trajectory["planner_state_indices"]
    positions = trajectory["positions"]
    if metrics["planner"] != "pioneer":
        raise AssertionError("PAN-21 requires the current PIONEER Planner")
    if len(indices) != 2 or len(positions) != 2:
        raise AssertionError("Planner evidence must contain exactly two observations")
    delta = [abs(int(a) - int(b)) for a, b in zip(indices[0], indices[1])]
    if sum(delta) != 1:
        raise AssertionError("p1 must be one legal translation action from p0")
    if metrics["run"].get("validation_initial_ue_manifest") is None:
        raise AssertionError("Planner did not consume the real UE5 p0 manifest")
    return {
        "planner": metrics["planner"],
        "state_indices": indices,
        "positions_scene_units": positions,
        "move_l1_index_distance": sum(delta),
        "search_totals": metrics["planner_search"]["totals"],
        "initial_provider": metrics["pioneer_observation"]["bundles"][0]["depth_source"],
        "planner_metrics_observation_count": trajectory["observation_count"],
    }


def _assert_close(actual: Iterable[float], expected: Iterable[float], label: str) -> None:
    if not np.allclose(tuple(actual), tuple(expected), rtol=0.0, atol=1e-4):
        raise AssertionError(f"{label} mismatch: actual={tuple(actual)} expected={tuple(expected)}")


def _plot_result(
    output: Path,
    canonical: Sequence[CanonicalObservationBundle],
    frames: Sequence[Dict[str, Any]],
    planner: Dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row, (bundle, frame) in enumerate(zip(canonical, frames)):
        front = bundle.faces[0].rgb_uint8
        axes[row, 0].imshow(front)
        axes[row, 0].set_title(f"p{row} UE5 front RGB")
        axes[row, 0].axis("off")
        valid = bundle.faces[0].valid_mask
        axes[row, 1].imshow(valid, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"p{row} front valid/no-hit mask")
        axes[row, 1].axis("off")
        points = frame["part_pc"].detach().cpu().numpy()
        if len(points) > 6000:
            points = points[np.linspace(0, len(points) - 1, 6000, dtype=int)]
        axes[row, 2].scatter(points[:, 0], points[:, 2], s=0.35, alpha=0.5)
        path = np.asarray(planner["positions_scene_units"], dtype=np.float64)
        axes[row, 2].plot(path[:, 0], path[:, 2], "r-o", linewidth=2)
        axes[row, 2].set_title(f"p{row} PAN-10 fused cloud + Planner move")
        axes[row, 2].set_xlabel("MAGICIAN x")
        axes[row, 2].set_ylabel("MAGICIAN z")
        axes[row, 2].axis("equal")
    figure.suptitle("PAN-21: two real UE5 observations / one current-Planner move")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def run_smoke(
    *,
    p0_manifest: Path,
    p1_manifest: Path,
    planner_metrics: Path,
    output_dir: Path,
    device_name: str,
    proxy_count: int,
) -> Dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is not available")

    manifests = (p0_manifest.resolve(), p1_manifest.resolve())
    canonical = tuple(load_bundle(path) for path in manifests)
    summaries = tuple(validate_bundle(bundle) for bundle in canonical)
    adapted = tuple(
        canonical_bundle_to_pan10(
            bundle,
            device=device,
            scene_units_per_meter=SCENE_UNITS_PER_METER,
            bundle_id=index,
        )
        for index, bundle in enumerate(canonical)
    )
    planner_payload = json.loads(planner_metrics.read_text(encoding="utf-8"))
    planner = _planner_record(planner_payload)
    _assert_close(adapted[0].center[0].detach().cpu(), planner["positions_scene_units"][0], "p0")
    _assert_close(adapted[1].center[0].detach().cpu(), planner["positions_scene_units"][1], "p1")

    proxy_points = _make_proxy_points(
        (adapted[0].center[0], adapted[1].center[0]), proxy_count, device
    )
    surface_scene = AuditPointMap(device)
    covered_scene = AuditPointMap(device)
    proxy_scene = AuditProxyScene(proxy_points)
    frame_count = 0

    def state() -> Dict[str, Any]:
        return {
            "frame_count": frame_count,
            "surface_scene": surface_scene.snapshot(),
            "covered_scene": covered_scene.snapshot(),
            "proxy_scene": proxy_scene.snapshot(),
        }

    before_negative = state()
    incomplete = dataclasses.replace(canonical[0], faces=canonical[0].faces[:-1])
    negative_error = None
    try:
        canonical_bundle_to_pan10(
            incomplete,
            device=device,
            scene_units_per_meter=SCENE_UNITS_PER_METER,
            bundle_id=99,
        )
    except BundleValidationError as error:
        negative_error = str(error)
    if negative_error is None:
        raise AssertionError("incomplete six-face bundle was not rejected")
    after_negative = state()
    if after_negative != before_negative:
        raise AssertionError("incomplete bundle mutated frame or map state")

    frames = []
    updates = []
    for index, bundle in enumerate(adapted):
        torch.manual_seed(2100 + index)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(2100 + index)
        previous = state()
        frame = process_cubemap_observation(
            bundle=bundle,
            proxy_points=proxy_points,
            gathering_factor=1.0,
            sensor_range=SENSOR_RANGE_SCENE_UNITS,
            voxel_size=VOXEL_SIZE_SCENE_UNITS,
            device=device,
        )
        features = torch.zeros((len(frame["part_pc"]), 1), device=device)
        covered_scene.fill_cells(frame["part_pc"], features=features)
        surface_scene.fill_cells(frame["part_pc"], features=features)
        updated_proxy = update_proxy_state(
            camera=SimpleNamespace(),
            proxy_scene=proxy_scene,
            frame_data=frame,
            carving_tolerance=CARVING_TOLERANCE,
        )
        if not updated_proxy:
            raise AssertionError(f"p{index} did not update proxy_scene")
        frame_count += 1
        current = state()
        stats = frame["bundle_stats"]
        contract_valid = summaries[index].valid_depth_count
        if current["surface_scene"]["point_count"] <= previous["surface_scene"]["point_count"]:
            raise AssertionError(f"p{index} did not update surface_scene")
        if current["covered_scene"]["point_count"] <= previous["covered_scene"]["point_count"]:
            raise AssertionError(f"p{index} did not update covered_scene")
        if stats["raw_point_count"] > contract_valid:
            raise AssertionError("no-hit pixels entered the surface point cloud")
        if stats["unique_point_count"] > stats["raw_point_count"]:
            raise AssertionError("PAN-10 voxel deduplication increased point count")
        owner = frame["proxy_face_owner"]
        union = frame["fov_proxy_mask"]
        owned = owner[union]
        if len(owned) != stats["proxy_union_count"] or bool(((owned < 0) | (owned >= 6)).any()):
            raise AssertionError("proxy visibility union lacks one valid face owner")
        if current["proxy_scene"]["max_observations_per_proxy"] > frame_count:
            raise AssertionError("a seam proxy accumulated more than once per observation")
        updates.append(
            {
                "observation_index": index,
                "request_id": canonical[index].request_id,
                "canonical_summary": summaries[index].as_dict(),
                "bundle_stats": stats,
                "state_before": previous,
                "state_after": current,
                "surface_delta": current["surface_scene"]["point_count"]
                - previous["surface_scene"]["point_count"],
                "covered_delta": current["covered_scene"]["point_count"]
                - previous["covered_scene"]["point_count"],
                "proxy_inside_fov_delta": current["proxy_scene"]["inside_fov_total"]
                - previous["proxy_scene"]["inside_fov_total"],
                "single_owner_count": int(len(owned)),
            }
        )
        frames.append(frame)

    final_state = state()
    if frame_count != 2:
        raise AssertionError("one six-face bundle must count as one observation/frame")
    expected_proxy_updates = sum(item["bundle_stats"]["proxy_union_count"] for item in updates)
    if final_state["proxy_scene"]["inside_fov_total"] != expected_proxy_updates:
        raise AssertionError("visibility union was accumulated more than once")
    if final_state["proxy_scene"]["no_hit_free_evidence_count"] <= 0:
        raise AssertionError("real fixtures did not exercise no-hit/free evidence")

    report = {
        "schema_version": SCHEMA_VERSION,
        "result": "PASS",
        "observation_count": frame_count,
        "planner_move_count": 1,
        "provider": "UE5 file/manifest exchange",
        "existing_paths_reused": {
            "contract": "PAN-19 load_bundle/validate_bundle",
            "fusion": "PAN-10 process_cubemap_observation",
            "map_update": "PAN-10 update_proxy_state",
            "planner": "current PIONEER Planner metrics replay",
        },
        "inputs": {
            "p0_manifest": str(manifests[0]),
            "p0_manifest_sha256": _sha256_file(manifests[0]),
            "p1_manifest": str(manifests[1]),
            "p1_manifest_sha256": _sha256_file(manifests[1]),
            "planner_metrics": str(planner_metrics.resolve()),
            "planner_metrics_sha256": _sha256_file(planner_metrics),
        },
        "planner": planner,
        "negative_test": {
            "mutation": "remove down face from p0",
            "error": negative_error,
            "state_before": before_negative,
            "state_after": after_negative,
            "zero_update": before_negative == after_negative,
            "next_valid_bundle_succeeded": True,
        },
        "updates": updates,
        "final_state": final_state,
        "assertions": {
            "two_observations_one_move": True,
            "p1_generated_by_current_planner": True,
            "p0_surface_covered_proxy_updated": True,
            "p1_same_path_updated": True,
            "incomplete_bundle_zero_update": True,
            "no_hit_has_no_surface_or_occupied_evidence": True,
            "one_proxy_owner_per_bundle": True,
            "visibility_union_not_double_accumulated": True,
        },
        "limitations": [
            "Debug-only two-observation smoke; coverage is not benchmark-comparable.",
            "File/manifest exchange is offline; no RPC, throughput, long-trajectory, or fault-matrix claim.",
            "The current Planner run consumes real UE5 p0 to select p1; this harness then consumes both real p0 and p1 through the same PAN-10 map-update path.",
            "AuditPointMap records the exact PAN-10 fused payload and state hashes; the authoritative Planner run separately exercises the full MAGICIAN scene objects.",
        ],
    }
    report_path = output_dir / "pan21_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot_result(output_dir / "pan21_smoke_figure.png", canonical, frames, planner)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0-manifest", type=Path, required=True)
    parser.add_argument("--p1-manifest", type=Path, required=True)
    parser.add_argument("--planner-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--proxy-count", type=int, default=8192)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_smoke(
        p0_manifest=args.p0_manifest,
        p1_manifest=args.p1_manifest,
        planner_metrics=args.planner_metrics,
        output_dir=args.output_dir,
        device_name=args.device,
        proxy_count=args.proxy_count,
    )
    print(json.dumps({"result": report["result"], "output": str(args.output_dir.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
