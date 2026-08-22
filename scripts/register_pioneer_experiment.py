#!/usr/bin/env python3
"""Register one live PIONEER run without inventing missing evidence.

The v1.0 registry predates PAN-10/PAN-11.  This helper adds or replaces exactly one
record, hashes the artifact bytes that are still present, and maintains a
separate generated Markdown section for live PIONEER runs.  A requested PASS
is accepted only when the wrapper exit status, online metrics, planner, and
six-face observation contract all agree.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional, Sequence


LIVE_START = "<!-- PIONEER_LIVE_SECTION_START -->"
LIVE_END = "<!-- PIONEER_LIVE_SECTION_END -->"
SUPPORTED_ISSUES = ("PAN-10", "PAN-11", "PAN-30")
PASS_STATUSES = {"PASS", "SUCCESS", "COMPLETED"}
FAIL_STATUSES = {"FAIL", "FAILED", "RUNTIME_FAIL", "SCIENTIFIC_FAIL"}
BUNDLE_TRANSACTION_VERSION = "pioneer-bundle-commit-v1"


def _read_json(path: Path) -> Optional[Mapping[str, Any]]:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _read_key_values(path: Path) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    if not path.is_file():
        return result
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return result
    for line in lines:
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _read_evidence_file(path: Path) -> Mapping[str, Any]:
    if path.suffix.lower() == ".json":
        return _read_json(path) or {}
    return _read_key_values(path)


def _first_file(root: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    return None


def _nested(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _pick(sources: Iterable[Mapping[str, Any]], *paths: Sequence[str]) -> Any:
    for source in sources:
        for path in paths:
            value = _nested(source, path)
            if value is not None and value != "":
                return value
    return None


def _as_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "cubemap6", "six-face"}:
            return True
        if normalized in {"false", "no", "0", "single"}:
            return False
    return None


def _last_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, list) and value and isinstance(value[-1], Mapping):
        return value[-1]
    return {}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_transaction_verified(
    capture_dir: Path,
    bundle: Mapping[str, Any],
    expected_face_names: Sequence[str],
) -> bool:
    bundle_id = _as_int(bundle.get("bundle_id"))
    if (
        bundle_id is None
        or bundle.get("artifact_transaction_version")
        != BUNDLE_TRANSACTION_VERSION
        or bundle.get("artifact_committed") is not True
    ):
        return False
    marker_path = (
        capture_dir.parent / ".pioneer_bundle_commits" / f"{bundle_id:06d}.json"
    )
    marker = _read_json(marker_path)
    if not isinstance(marker, Mapping):
        return False
    frame_hashes = marker.get("frame_sha256")
    image_hashes = marker.get("image_sha256")
    expected_frames = {"bundle.pt", *(f"{name}.pt" for name in expected_face_names)}
    expected_images = {f"{name}.png" for name in expected_face_names}
    if (
        marker.get("transaction_version") != BUNDLE_TRANSACTION_VERSION
        or _as_int(marker.get("bundle_id")) != bundle_id
        or list(marker.get("face_names") or []) != list(expected_face_names)
        or marker.get("png_committed") is not True
        or not isinstance(frame_hashes, Mapping)
        or set(frame_hashes) != expected_frames
        or not isinstance(image_hashes, Mapping)
        or set(image_hashes) != expected_images
    ):
        return False
    for filename, expected_sha in frame_hashes.items():
        path = capture_dir / f"{bundle_id:06d}" / filename
        if (
            not isinstance(expected_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            or not path.is_file()
            or _file_sha256(path) != expected_sha
        ):
            return False
    for filename, expected_sha in image_hashes.items():
        path = capture_dir.parent / "imgs" / f"{bundle_id:06d}" / filename
        if (
            not isinstance(expected_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            or not path.is_file()
            or _file_sha256(path) != expected_sha
        ):
            return False
    return True


def _path_is_within(path: Any, root: Any) -> bool:
    if not isinstance(path, str) or not path or not isinstance(root, str) or not root:
        return False
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except (OSError, ValueError):
        return False
    return True


def _resolve_config(
    *,
    run_dir: Path,
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Optional[Path]]:
    """Load the run config when its bytes are still locatable.

    Relative manifest references are searched beside the run, in the current
    checkout, and below ``configs/test``.  The selected debug profile is then
    merged exactly as an effective semantic overlay, while the original config
    file remains the artifact whose bytes are hashed.
    """

    repo_root = Path(__file__).resolve().parents[1]
    references: list[Path] = [run_dir / "config.json"]
    config_ref = manifest.get("config")
    if isinstance(config_ref, str) and config_ref.strip():
        raw = Path(config_ref.strip())
        if raw.is_absolute():
            references.append(raw)
        else:
            references.extend(
                [run_dir / raw, repo_root / raw, repo_root / "configs" / "test" / raw]
            )

    selected: Optional[Path] = None
    config: dict[str, Any] = {}
    seen: set[Path] = set()
    for candidate in references:
        try:
            candidate = candidate.resolve()
        except OSError:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        loaded = _read_json(candidate)
        if loaded is not None:
            config.update(loaded)
            selected = candidate
            break

    profile = _pick(
        [metrics, config],
        ("run", "debug_profile"),
        ("debug_profile",),
    )
    if isinstance(profile, str) and re.fullmatch(r"[A-Za-z0-9._-]+", profile):
        profile_ref = manifest.get("debug_profile_snapshot")
        profile_snapshot = (
            run_dir / profile_ref
            if isinstance(profile_ref, str)
            and re.fullmatch(r"[A-Za-z0-9._-]+", profile_ref)
            else run_dir / "debug_profile.json"
        )
        profile_document = _read_json(profile_snapshot)
        if profile_document is None:
            profile_document = _read_json(
                repo_root / "configs" / "debug" / f"{profile}.json"
            )
        if profile_document is not None:
            overrides = profile_document.get("overrides")
            if isinstance(overrides, Mapping):
                config.update(overrides)
            for key in ("name", "debug_only", "coverage_comparable"):
                if key in profile_document:
                    target = "debug_profile" if key == "name" else key
                    config[target] = profile_document[key]
    return config, selected


def _artifact_paths(
    run_dir: Path,
    online_metrics: Path,
    config_path: Optional[Path],
    artifact_roots: Sequence[Path],
    excluded: set[Path],
) -> list[Path]:
    candidates: list[Path] = []
    if run_dir.is_dir():
        candidates.extend(
            path for path in sorted(run_dir.rglob("*")) if path.is_file() and not path.is_symlink()
        )
    for path in (online_metrics, config_path):
        if path is not None and path.is_file() and not path.is_symlink():
            candidates.append(path)
    for root in artifact_roots:
        try:
            resolved_root = root.resolve()
        except OSError:
            continue
        # A capture locator comes from a metrics file. Refuse filesystem roots
        # or suspiciously broad paths before recursively hashing it.
        if resolved_root == Path(resolved_root.anchor) or len(resolved_root.parts) < 4:
            continue
        if resolved_root.is_file() and not resolved_root.is_symlink():
            candidates.append(resolved_root)
        elif resolved_root.is_dir():
            candidates.extend(
                path
                for path in sorted(resolved_root.rglob("*"))
                if path.is_file() and not path.is_symlink()
            )

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in excluded or resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _artifact_set(
    *,
    issue: str,
    experiment_id: str,
    run_dir: Path,
    online_metrics: Path,
    config_path: Optional[Path],
    metrics: Mapping[str, Any],
    explicit_artifact_roots: Sequence[Path],
    registry_paths: set[Path],
) -> tuple[str, Mapping[str, Any]]:
    artifact_id = "ART-{}-LIVE".format(
        re.sub(r"[^A-Za-z0-9._-]+", "-", experiment_id).strip("-") or "PIONEER"
    )
    artifacts: dict[str, Any] = {}
    used_keys: set[str] = set()
    artifact_roots = list(explicit_artifact_roots)
    capture_dir = metrics.get("capture_dir")
    if isinstance(capture_dir, str) and capture_dir.strip():
        capture_path = Path(capture_dir.strip())
        artifact_roots.append(capture_path)
        commit_root = capture_path.parent / ".pioneer_bundle_commits"
        if commit_root.is_dir():
            artifact_roots.append(commit_root)
    for path in _artifact_paths(
        run_dir,
        online_metrics,
        config_path,
        artifact_roots,
        excluded=registry_paths,
    ):
        try:
            relative = path.relative_to(run_dir)
            locator = str(relative)
            key = str(relative).replace("/", "__")
        except ValueError:
            locator = str(path)
            key = path.name
        base_key = key
        suffix = 2
        while key in used_keys:
            key = f"{base_key}_{suffix}"
            suffix += 1
        used_keys.add(key)
        try:
            artifacts[key] = {
                "path": locator,
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        except OSError:
            continue
    return artifact_id, {
        "issue": issue,
        "experiment_id": experiment_id,
        "hash_semantics": "SHA256 of the live artifact file bytes",
        "artifacts": artifacts,
    }


def _status_from_evidence(
    requested_status: str,
    *,
    exit_code: Optional[int],
    metrics_present: bool,
    planner: Optional[str],
    cubemap6: Optional[bool],
    cubemap6_metrics_verified: bool,
) -> tuple[str, bool, str]:
    requested = (requested_status or "UNKNOWN").strip().upper()
    if requested in PASS_STATUSES:
        verified = (
            exit_code == 0
            and metrics_present
            and (planner or "").strip().lower() == "pioneer"
            and cubemap6 is True
            and cubemap6_metrics_verified
        )
        if verified:
            return "PASS", True, "wrapper exit 0 plus PIONEER cubemap6 online metrics"
        return (
            "UNKNOWN",
            False,
            "requested PASS was withheld because completion or measured six-face bundle evidence is missing",
        )
    if requested in FAIL_STATUSES:
        if exit_code is not None and exit_code != 0:
            return requested, True, "non-zero wrapper exit status"
        return "UNKNOWN", False, "requested failure lacks a non-zero wrapper exit status"
    if requested in {"RUNNING", "PENDING", "BLOCKED", "UNKNOWN"}:
        return requested, False, "non-success lifecycle state"
    return "UNKNOWN", False, "unrecognized requested status"


def _ordered_counts(existing: Any, counter: Counter[str]) -> Mapping[str, int]:
    ordered: dict[str, int] = {}
    if isinstance(existing, Mapping):
        for key in existing:
            if key in counter:
                ordered[str(key)] = int(counter[key])
    for key in sorted(counter):
        if key not in ordered:
            ordered[key] = int(counter[key])
    return ordered


def _short_commit(value: Optional[str]) -> Optional[str]:
    if not value or value.lower() in {"unknown", "null", "none"}:
        return None
    return value[:7]


def _full_commit(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    return value if re.fullmatch(r"[0-9a-fA-F]{40}", value) else None


def build_record(
    *,
    issue: str,
    experiment_id: str,
    run_dir: Path,
    online_metrics_path: Path,
    scientific_run_commit: str,
    final_branch_commit: str,
    requested_status: str,
    artifact_set_id: str,
    artifact_set: Mapping[str, Any],
) -> Mapping[str, Any]:
    manifest_path = _first_file(run_dir, ("manifest.json", "manifest.txt"))
    status_path = _first_file(run_dir, ("status.json", "status.txt"))
    manifest = _read_evidence_file(manifest_path) if manifest_path else {}
    status = _read_evidence_file(status_path) if status_path else {}
    metrics = _read_json(online_metrics_path) or {}
    config, config_path = _resolve_config(
        run_dir=run_dir, manifest=manifest, metrics=metrics
    )
    sources = [metrics, manifest, config]

    manifest_config_sha = manifest.get("config_sha256")
    config_snapshot_verified = bool(
        config_path is not None
        and isinstance(manifest_config_sha, str)
        and re.fullmatch(r"[0-9a-f]{64}", manifest_config_sha)
        and _file_sha256(config_path) == manifest_config_sha
    )
    profile_ref = manifest.get("debug_profile_snapshot")
    profile_snapshot = (
        run_dir / profile_ref
        if isinstance(profile_ref, str)
        and re.fullmatch(r"[A-Za-z0-9._-]+", profile_ref)
        else run_dir / "debug_profile.json"
    )
    manifest_profile_sha = manifest.get("debug_profile_sha256")
    profile_snapshot_verified = bool(
        profile_snapshot.is_file()
        and isinstance(manifest_profile_sha, str)
        and re.fullmatch(r"[0-9a-f]{64}", manifest_profile_sha)
        and _file_sha256(profile_snapshot) == manifest_profile_sha
    )
    position_policy_ref = manifest.get("position_policy_snapshot")
    position_policy_snapshot = (
        run_dir / position_policy_ref
        if isinstance(position_policy_ref, str)
        and re.fullmatch(r"[A-Za-z0-9._/-]+", position_policy_ref)
        and not Path(position_policy_ref).is_absolute()
        and ".." not in Path(position_policy_ref).parts
        else run_dir / "flight_policy.json"
    )
    manifest_position_policy_sha = manifest.get("position_policy_sha256")
    position_policy_snapshot_verified = bool(
        position_policy_snapshot.is_file()
        and isinstance(manifest_position_policy_sha, str)
        and re.fullmatch(r"[0-9a-f]{64}", manifest_position_policy_sha)
        and _file_sha256(position_policy_snapshot) == manifest_position_policy_sha
    )
    macarons_params_ref = manifest.get("macarons_params_snapshot")
    macarons_params_snapshot = (
        run_dir / macarons_params_ref
        if isinstance(macarons_params_ref, str)
        and re.fullmatch(r"[A-Za-z0-9._-]+", macarons_params_ref)
        else run_dir / "macarons_params.json"
    )
    manifest_macarons_params_sha = manifest.get("macarons_params_sha256")
    macarons_params_snapshot_verified = bool(
        macarons_params_snapshot.is_file()
        and isinstance(manifest_macarons_params_sha, str)
        and re.fullmatch(r"[0-9a-f]{64}", manifest_macarons_params_sha)
        and _file_sha256(macarons_params_snapshot) == manifest_macarons_params_sha
    )
    manifest_run_commit = _full_commit(
        str(manifest.get("git_commit") or "")
    )
    scientific_full = _full_commit(scientific_run_commit)
    run_commit_verified = bool(
        scientific_full is not None and manifest_run_commit == scientific_full
    )
    runtime_snapshot_integrity_verified = bool(
        manifest.get("runtime_snapshot_integrity_contract")
        == "pre-and-post-v1"
        and status.get("snapshot_integrity_preflight") == "PASS"
        and status.get("snapshot_integrity_postflight") == "PASS"
    )

    planner_value = _pick(sources, ("planner",), ("run", "planner"))
    planner = str(planner_value) if planner_value is not None else None
    scene_value = _pick(sources, ("scene",), ("run", "scene"))
    scene = str(scene_value) if scene_value is not None else None
    observation_mode_value = _pick(
        sources,
        ("run", "planning_observation_mode"),
        ("planning_observation_mode",),
        ("observation_mode",),
    )
    observation_mode = (
        str(observation_mode_value) if observation_mode_value is not None else None
    )
    face_count = _as_int(
        _pick(
            sources,
            ("pioneer_observation", "bundles", "face_count"),
            ("run", "pioneer_face_count"),
            ("pioneer_face_count",),
        )
    )
    pioneer_metrics = metrics.get("pioneer_observation")
    if isinstance(pioneer_metrics, Mapping):
        face_count = _as_int(pioneer_metrics.get("real_face_render_count")) or face_count
        bundle_count_for_faces = _as_int(pioneer_metrics.get("bundle_count"))
        if face_count is not None and bundle_count_for_faces:
            face_count = face_count // bundle_count_for_faces
    mode_cubemap = (
        observation_mode is not None and observation_mode.strip().lower() == "cubemap6"
    )
    cubemap6: Optional[bool] = True if mode_cubemap else (True if face_count == 6 else None)

    seed_random = _as_int(
        _pick(sources, ("run", "seed"), ("random_seed",), ("seed_random",))
    )
    seed_torch = _as_int(
        _pick(sources, ("run", "torch_seed"), ("torch_seed",), ("seed_torch",))
    )
    budget = _as_int(
        _pick(
            sources,
            ("run", "budget_observations"),
            ("experiment_budget_observations",),
            ("observations",),
        )
    )
    beam_width = _as_int(_pick(sources, ("beam_width",), ("run", "beam_width")))
    beam_steps = _as_int(_pick(sources, ("beam_steps",), ("run", "beam_steps")))
    proxy_points = _as_int(
        _pick(
            sources,
            ("validation_n_proxy_points",),
            ("n_proxy_points",),
            ("proxy_points",),
            ("run", "proxy_points"),
        )
    )

    last_coverage = _last_mapping(metrics.get("coverage"))
    final_normalized_coverage = _as_float(last_coverage.get("normalized"))
    final_raw_coverage = _as_float(last_coverage.get("raw"))
    trajectory = metrics.get("trajectory") if isinstance(metrics.get("trajectory"), Mapping) else {}
    observation_count = _as_int(trajectory.get("observation_count"))
    latency = metrics.get("latency") if isinstance(metrics.get("latency"), Mapping) else {}
    cuda = metrics.get("cuda") if isinstance(metrics.get("cuda"), Mapping) else {}
    pioneer = pioneer_metrics if isinstance(pioneer_metrics, Mapping) else {}
    bundle_count = _as_int(pioneer.get("bundle_count"))
    real_face_render_count = _as_int(pioneer.get("real_face_render_count"))
    bundles = pioneer.get("bundles")
    expected_face_names = ("front", "back", "left", "right", "up", "down")
    bundle_ids = (
        [
            _as_int(bundle.get("bundle_id"))
            if isinstance(bundle, Mapping)
            else None
            for bundle in bundles
        ]
        if isinstance(bundles, list)
        else []
    )
    bundle_rows_verified = (
        isinstance(bundles, list)
        and bundle_count is not None
        and len(bundles) == bundle_count
        and bundle_ids == list(range(bundle_count))
        and all(
            isinstance(bundle, Mapping)
            and _as_int(bundle.get("face_count")) == 6
            and tuple(bundle.get("face_names") or ()) == expected_face_names
            for bundle in bundles
        )
    )
    cubemap6_metrics_verified = bool(
        bundle_count is not None
        and bundle_count > 0
        and real_face_render_count == 6 * bundle_count
        and bundle_rows_verified
        and observation_count == bundle_count
        and (budget is None or bundle_count == budget)
    )
    exit_code = _as_int(_pick([status], ("exit_code",), ("returncode",)))
    effective_status, completion_verified, status_reason = _status_from_evidence(
        requested_status,
        exit_code=exit_code,
        metrics_present=bool(metrics),
        planner=planner,
        cubemap6=cubemap6,
        cubemap6_metrics_verified=cubemap6_metrics_verified,
    )

    face_size = _as_int(
        _pick(sources, ("run", "pioneer_face_size"), ("pioneer_face_size",))
    )
    face_fov = _as_float(
        _pick(
            sources,
            ("run", "pioneer_face_fov_degrees"),
            ("pioneer_face_fov_degrees",),
        )
    )
    debug_profile = _pick(sources, ("run", "debug_profile"), ("debug_profile",))
    coverage_comparable = _as_bool(
        _pick(sources, ("run", "coverage_comparable"), ("coverage_comparable",))
    )
    collision = _as_bool(
        _pick(sources, ("run", "compute_collision"), ("compute_collision",))
    )
    depth_source_value = _pick(
        sources,
        ("run", "depth_source"),
        ("depth_source",),
        ("kind_depth_map",),
    )
    depth_source = (
        str(depth_source_value).strip().upper()
        if depth_source_value is not None
        else None
    )
    start_index = _as_int(_pick(sources, ("start_index",), ("start",)))

    planner_state_mode_value = _pick(
        sources,
        ("run", "pioneer_planner_state_mode"),
        ("planner_search", "state_mode"),
        ("pioneer_planner_state_mode",),
    )
    planner_state_mode = (
        str(planner_state_mode_value)
        if planner_state_mode_value is not None
        else None
    )
    planner_state_dimension = _as_int(
        _pick(
            sources,
            ("run", "planner_state_dimension"),
            ("planner_search", "state_dimension"),
            ("planner_state_dimension",),
        )
    )
    cubemap_rig_frame_value = _pick(
        sources,
        ("run", "pioneer_cubemap_rig_frame"),
        ("planner_search", "cubemap_rig_frame"),
        ("pioneer_cubemap_rig_frame",),
    )
    cubemap_rig_frame = (
        str(cubemap_rig_frame_value)
        if cubemap_rig_frame_value is not None
        else None
    )
    cubemap_extrinsics_version_value = _pick(
        sources,
        ("run", "pioneer_cubemap_extrinsics_version"),
        ("planner_search", "cubemap_extrinsics_version"),
        ("pioneer_cubemap_extrinsics_version",),
    )
    cubemap_extrinsics_version = (
        str(cubemap_extrinsics_version_value)
        if cubemap_extrinsics_version_value is not None
        else None
    )
    canonical_orientation_indices = _pick(
        sources,
        ("run", "pioneer_canonical_orientation_indices"),
        ("planner_search", "canonical_orientation_indices"),
        ("pioneer_canonical_orientation_indices",),
    )
    filter_occupied_position_candidates = _as_bool(
        _pick(
            sources,
            ("run", "pioneer_filter_occupied_position_candidates"),
            ("pioneer_filter_occupied_position_candidates",),
        )
    )
    require_complete_occupied_pose = _as_bool(
        _pick(
            sources,
            ("run", "validation_require_complete_occupied_pose"),
            ("validation_require_complete_occupied_pose",),
        )
    )
    position_policy_value = _pick(
        sources,
        ("run", "validation_position_policy"),
        ("validation_position_policy",),
    )
    position_policy = (
        dict(position_policy_value)
        if isinstance(position_policy_value, Mapping)
        else {}
    )
    planner_search = (
        metrics.get("planner_search")
        if isinstance(metrics.get("planner_search"), Mapping)
        else {}
    )
    planner_search_totals_value = planner_search.get("totals")
    planner_search_totals = (
        dict(planner_search_totals_value)
        if isinstance(planner_search_totals_value, Mapping)
        else {}
    )
    expected_state_dimension = {
        "legacy_pose5d": 5,
        "position_only": 3,
    }.get(planner_state_mode)
    expected_rig_frame = {
        "legacy_pose5d": "body",
        "position_only": "world",
    }.get(planner_state_mode)
    expected_extrinsics_version = {
        "legacy_pose5d": "pytorch3d-body-aligned-v1",
        "position_only": "pytorch3d-world-axes-v1",
    }.get(planner_state_mode)
    orientation_proposals = _as_int(
        planner_search_totals.get("orientation_action_proposal_count")
    )
    rendered_candidates = _as_int(
        planner_search_totals.get("rendered_candidate_count")
    )
    imagined_candidate_bundles = _as_int(
        pioneer.get("imagined_candidate_bundle_render_count")
    )
    required_search_count_fields = (
        "parent_beam_count",
        "raw_action_proposal_count",
        "translation_action_proposal_count",
        "orientation_action_proposal_count",
        "generated_candidate_count",
        "valid_state_candidate_count",
        "observed_rejected_candidate_count",
        "collision_rejected_candidate_count",
        "rendered_candidate_count",
        "retained_beam_count",
    )
    occupied_rejections = _as_int(
        planner_search_totals.get("occupied_rejected_candidate_count")
    )
    bound_rejections = _as_int(
        planner_search_totals.get("validation_bound_rejected_candidate_count")
    )
    if filter_occupied_position_candidates is True:
        required_search_count_fields += ("occupied_rejected_candidate_count",)
    if issue == "PAN-30":
        required_search_count_fields += ("validation_bound_rejected_candidate_count",)
    search_counts_verified = all(
        type(planner_search_totals.get(field)) is int
        and planner_search_totals[field] >= 0
        for field in required_search_count_fields
    )
    search_seconds = planner_search_totals.get("search_seconds")
    search_seconds_verified = (
        not isinstance(search_seconds, bool)
        and isinstance(search_seconds, (int, float))
        and search_seconds >= 0
    )
    effective_occupied_rejections = (
        occupied_rejections if occupied_rejections is not None else 0
    )
    effective_bound_rejections = bound_rejections if bound_rejections is not None else 0
    search_closure_verified = bool(
        search_counts_verified
        and planner_search_totals["generated_candidate_count"]
        == planner_search_totals["valid_state_candidate_count"]
        + planner_search_totals["observed_rejected_candidate_count"]
        + effective_occupied_rejections
        + effective_bound_rejections
        and planner_search_totals["valid_state_candidate_count"]
        == planner_search_totals["collision_rejected_candidate_count"]
        + planner_search_totals["rendered_candidate_count"]
    )
    position_planner_contract_verified = bool(
        issue in {"PAN-11", "PAN-30"}
        and expected_state_dimension is not None
        and planner_state_dimension == expected_state_dimension
        and cubemap_rig_frame == expected_rig_frame
        and cubemap_extrinsics_version == expected_extrinsics_version
        and (
            canonical_orientation_indices == [2, 0]
            if planner_state_mode == "position_only"
            else True
        )
        and planner_search.get("state_mode") == planner_state_mode
        and _as_int(planner_search.get("state_dimension"))
        == planner_state_dimension
        and planner_search.get("cubemap_rig_frame") == cubemap_rig_frame
        and planner_search.get("cubemap_extrinsics_version")
        == cubemap_extrinsics_version
        and search_counts_verified
        and search_closure_verified
        and search_seconds_verified
        and (
            filter_occupied_position_candidates is not True
            or (
                require_complete_occupied_pose is True
                and occupied_rejections is not None
            )
        )
        and rendered_candidates is not None
        and imagined_candidate_bundles is not None
        and rendered_candidates == imagined_candidate_bundles
        and (
            orientation_proposals == 0
            if planner_state_mode == "position_only"
            else orientation_proposals is not None and orientation_proposals > 0
        )
    )
    pan30_low_altitude_contract_verified = bool(
        issue == "PAN-30"
        and position_planner_contract_verified
        and debug_profile == "pioneer-low-altitude-50"
        and budget == 50
        and position_policy_snapshot_verified
        and position_policy.get("source_policy_sha256")
        == manifest_position_policy_sha
        and _as_float(position_policy.get("hard_ceiling_agl_m")) == 120.0
        and _as_float(position_policy.get("start_agl_m")) is not None
        and _as_float(position_policy.get("start_agl_m")) <= 10.0
        and _as_int(position_policy.get("verified_free_position_count")) == 63
    )
    issue_contract_verified = (
        pan30_low_altitude_contract_verified
        if issue == "PAN-30"
        else position_planner_contract_verified
    )
    if (
        issue in {"PAN-11", "PAN-30"}
        and effective_status == "PASS"
        and not issue_contract_verified
    ):
        effective_status = "UNKNOWN"
        completion_verified = False
        status_reason = f"{issue} planner-state/search telemetry contract is incomplete."

    metrics_run = metrics.get("run") if isinstance(metrics.get("run"), Mapping) else {}
    da3_model_id = metrics_run.get("da3_model_id")
    da3_model_revision = metrics_run.get("da3_model_revision")
    da3_model_config_sha = metrics_run.get("da3_model_config_sha256")
    da3_model_weights_sha = metrics_run.get("da3_model_weights_sha256")
    da3_source_revision = metrics_run.get("da3_source_revision")
    da3_source_tree_sha = metrics_run.get("da3_source_tree_sha256")
    da3_window_size = _as_int(metrics_run.get("da3_window_size"))
    da3_process_res = _as_int(metrics_run.get("da3_process_res"))
    da3_process_res_method = metrics_run.get("da3_process_res_method")
    da3_confidence_percentile = metrics_run.get("da3_confidence_percentile")
    da3_confidence_declared = "da3_confidence_percentile" in metrics_run
    da3_cache_enabled = metrics_run.get("da3_cache_enabled")
    da3_cache_dir = metrics_run.get("da3_cache_dir")
    da3_scale_map = metrics_run.get("da3_scene_units_per_meter")
    da3_scale = (
        da3_scale_map.get(scene)
        if isinstance(da3_scale_map, Mapping) and scene is not None
        else None
    )
    da3_output_height = _as_int(metrics_run.get("da3_output_height"))
    da3_output_width = _as_int(metrics_run.get("da3_output_width"))
    calibration_snapshot = run_dir / "scene_metric_calibrations.json"
    manifest_calibration_sha = manifest.get("da3_calibration_sha256")
    calibration_snapshot_verified = bool(
        calibration_snapshot.is_file()
        and isinstance(manifest_calibration_sha, str)
        and re.fullmatch(r"[0-9a-f]{64}", manifest_calibration_sha)
        and _file_sha256(calibration_snapshot) == manifest_calibration_sha
    )
    calibration_document = _read_json(calibration_snapshot) or {}
    calibration_rows = calibration_document.get("calibrations")
    scene_calibration = (
        calibration_rows.get(scene)
        if isinstance(calibration_rows, Mapping) and scene is not None
        else None
    )
    calibration_semantics_verified = bool(
        isinstance(scene_calibration, Mapping)
        and _as_float(scene_calibration.get("scene_units_per_meter")) == da3_scale
    )
    import_tree_sha = manifest.get("da3_import_package_tree_sha256")
    import_source_root = manifest.get("da3_import_source_root")
    import_origin = manifest.get("da3_import_origin")
    import_source_verified = bool(
        isinstance(import_tree_sha, str)
        and re.fullmatch(r"[0-9a-f]{64}", import_tree_sha)
        and _path_is_within(import_origin, import_source_root)
        and isinstance(da3_source_revision, str)
        and da3_source_revision in str(import_source_root)
    )
    try:
        asset_provenance = json.loads(
            str(manifest.get("scene_asset_provenance_json") or "")
        )
    except json.JSONDecodeError:
        asset_provenance = None
    asset_hashes_verified = bool(
        _as_bool(manifest.get("scene_asset_hashes_verified")) is True
        and isinstance(asset_provenance, Mapping)
        and set(asset_provenance)
        >= {
            "adaptation_manifest", "mesh", "settings", "occupied_pose",
            "planner_weight", "da3_model_config", "da3_model_weights",
        }
        and all(
            isinstance(item, Mapping)
            and isinstance(item.get("path"), str)
            and bool(item.get("path"))
            and isinstance(item.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
            and Path(item["path"]).is_file()
            and _file_sha256(Path(item["path"])) == item["sha256"]
            for item in asset_provenance.values()
        )
    )
    if asset_hashes_verified and isinstance(scene_calibration, Mapping):
        repo_root = Path(__file__).resolve().parents[1]
        for calibration_key in ("adaptation_manifest", "mesh", "settings"):
            calibration_item = scene_calibration.get(calibration_key)
            asset_item = asset_provenance.get(calibration_key)
            if not isinstance(calibration_item, Mapping) or not isinstance(
                asset_item, Mapping
            ):
                asset_hashes_verified = False
                break
            calibration_path = Path(str(calibration_item.get("path") or ""))
            if not calibration_path.is_absolute():
                calibration_path = repo_root / calibration_path
            if (
                calibration_path.resolve() != Path(asset_item["path"]).resolve()
                or calibration_item.get("sha256") != asset_item.get("sha256")
            ):
                asset_hashes_verified = False
                break
    else:
        asset_hashes_verified = False
    manifest_texture_tree_sha = manifest.get("scene_texture_tree_sha256")
    texture_tree_verified = False
    if asset_hashes_verified:
        texture_items = [
            item
            for name, item in asset_provenance.items()
            if name.startswith("material_") or name.startswith("texture_")
        ]
        material_count = sum(name.startswith("material_") for name in asset_provenance)
        texture_count = sum(name.startswith("texture_") for name in asset_provenance)
        try:
            mesh_root = Path(asset_provenance["mesh"]["path"]).resolve().parent
            tree = hashlib.sha256()
            for item in sorted(texture_items, key=lambda value: value["path"]):
                path = Path(item["path"]).resolve()
                tree.update(path.relative_to(mesh_root).as_posix().encode("utf-8"))
                tree.update(b"\0")
                tree.update(item["sha256"].encode("ascii"))
                tree.update(b"\n")
            texture_tree_verified = bool(
                material_count > 0
                and texture_count > 0
                and isinstance(manifest_texture_tree_sha, str)
                and re.fullmatch(r"[0-9a-f]{64}", manifest_texture_tree_sha)
                and tree.hexdigest() == manifest_texture_tree_sha
                and config.get("scene_texture_tree_sha256")
                == manifest_texture_tree_sha
                and metrics_run.get("scene_texture_tree_sha256")
                == manifest_texture_tree_sha
            )
        except (KeyError, OSError, ValueError, TypeError):
            texture_tree_verified = False
    model_cache_hashes_verified = bool(
        asset_hashes_verified
        and isinstance(da3_model_config_sha, str)
        and isinstance(da3_model_weights_sha, str)
        and asset_provenance["da3_model_config"].get("sha256")
        == da3_model_config_sha
        and asset_provenance["da3_model_weights"].get("sha256")
        == da3_model_weights_sha
    )
    da3_bundle_cache_hits = 0
    da3_face_rows_verified = depth_source != "DA3"
    capture_path_value = metrics.get("capture_dir")
    capture_path = (
        Path(capture_path_value).expanduser().resolve()
        if isinstance(capture_path_value, str) and capture_path_value.strip()
        else None
    )
    if depth_source == "DA3" and isinstance(bundles, list):
        da3_face_rows_verified = True
        for bundle in bundles:
            depth_faces = bundle.get("depth_faces") if isinstance(bundle, Mapping) else None
            if (
                not isinstance(depth_faces, list)
                or len(depth_faces) != 6
                or str(bundle.get("depth_source", "")).upper() != "DA3"
                or bundle.get("rgb_source") != "gt_mesh"
                or bundle.get("renderer_zbuf_read") is not False
                or _as_int(bundle.get("depth_inference_count")) != 6
                or capture_path is None
                or not _bundle_transaction_verified(
                    capture_path, bundle, expected_face_names
                )
            ):
                da3_face_rows_verified = False
                break
            bundle_cache_hits = 0
            for expected_face_name, face in zip(expected_face_names, depth_faces):
                model = face.get("model") if isinstance(face, Mapping) else None
                preprocess = (
                    face.get("preprocess") if isinstance(face, Mapping) else None
                )
                scale_metadata = (
                    face.get("scale") if isinstance(face, Mapping) else None
                )
                cache_hit = face.get("cache_hit") if isinstance(face, Mapping) else None
                stream_id = face.get("stream_id") if isinstance(face, Mapping) else None
                provider_seconds = (
                    face.get("provider_seconds") if isinstance(face, Mapping) else None
                )
                count_fields = (
                    face.get("provider_valid_pixels"),
                    face.get("provider_error_pixels"),
                    face.get("planning_pixels"),
                ) if isinstance(face, Mapping) else ()
                if (
                    not isinstance(face, Mapping)
                    or face.get("face_name") != expected_face_name
                    or str(face.get("depth_source", "")).upper() != "DA3"
                    or not isinstance(face.get("cache_key"), str)
                    or not face["cache_key"]
                    or type(cache_hit) is not bool
                    or stream_id
                    != f"pioneer/cubemap6/{cubemap_rig_frame}/{expected_face_name}"
                    or type(face.get("pose_conditioned")) is not bool
                    or any(type(value) is not int or value < 0 for value in count_fields)
                    or isinstance(provider_seconds, bool)
                    or not isinstance(provider_seconds, (int, float))
                    or provider_seconds < 0
                    or not isinstance(face.get("adapter_version"), str)
                    or not face["adapter_version"]
                    or face.get("source_revision") != da3_source_revision
                    or face.get("source_tree_sha256") != da3_source_tree_sha
                    or not isinstance(model, Mapping)
                    or model.get("id") != da3_model_id
                    or model.get("revision") != da3_model_revision
                    or model.get("config_sha256") != da3_model_config_sha
                    or model.get("weights_sha256") != da3_model_weights_sha
                    or not isinstance(preprocess, Mapping)
                    or _as_int(preprocess.get("window_size")) != da3_window_size
                    or _as_int(preprocess.get("process_res")) != da3_process_res
                    or preprocess.get("process_res_method")
                    != da3_process_res_method
                    or preprocess.get("output_size")
                    != [da3_output_height, da3_output_width]
                    or preprocess.get("confidence_percentile")
                    != da3_confidence_percentile
                    or not isinstance(scale_metadata, Mapping)
                    or scale_metadata.get("scene") != scene
                    or _as_float(scale_metadata.get("scene_units_per_meter"))
                    != da3_scale
                    or _as_float(scale_metadata.get("znear")) is None
                    or _as_float(scale_metadata.get("zfar")) is None
                    or _as_float(scale_metadata.get("zfar"))
                    <= _as_float(scale_metadata.get("znear"))
                ):
                    da3_face_rows_verified = False
                    break
                bundle_cache_hits += int(cache_hit)
            if not da3_face_rows_verified:
                break
            if _as_int(bundle.get("depth_cache_hit_count")) != bundle_cache_hits:
                da3_face_rows_verified = False
                break
            da3_bundle_cache_hits += bundle_cache_hits
    da3_cubemap_provenance_verified = bool(
        depth_source != "DA3"
        or (
            metrics.get("renderer_gt_read") is False
            and metrics_run.get("depth_source") == "DA3"
            and metrics_run.get("use_perfect_depth_map") is False
            and str(metrics_run.get("kind_depth_map", "")).upper() == "DA3"
            and metrics_run.get("renderer_zbuf_role")
            == "rgb_geometry_render_depth_discarded"
            and metrics_run.get("gt_feedback_to_da3") is False
            and isinstance(da3_model_id, str)
            and bool(da3_model_id)
            and isinstance(da3_model_revision, str)
            and bool(da3_model_revision)
            and isinstance(da3_model_config_sha, str)
            and re.fullmatch(r"[0-9a-f]{64}", da3_model_config_sha)
            and isinstance(da3_model_weights_sha, str)
            and re.fullmatch(r"[0-9a-f]{64}", da3_model_weights_sha)
            and config.get("da3_model_config_sha256") == da3_model_config_sha
            and config.get("da3_model_weights_sha256") == da3_model_weights_sha
            and manifest.get("da3_model_config_sha256") == da3_model_config_sha
            and manifest.get("da3_model_weights_sha256") == da3_model_weights_sha
            and isinstance(da3_source_revision, str)
            and bool(da3_source_revision)
            and isinstance(da3_source_tree_sha, str)
            and re.fullmatch(r"[0-9a-f]{64}", da3_source_tree_sha)
            and import_tree_sha == da3_source_tree_sha
            and da3_window_size is not None
            and da3_window_size > 0
            and da3_process_res is not None
            and da3_process_res > 0
            and isinstance(da3_process_res_method, str)
            and bool(da3_process_res_method)
            and da3_confidence_declared
            and (
                da3_confidence_percentile is None
                or (
                    not isinstance(da3_confidence_percentile, bool)
                    and isinstance(da3_confidence_percentile, (int, float))
                    and 0 <= da3_confidence_percentile <= 100
                )
            )
            and type(da3_cache_enabled) is bool
            and (
                not da3_cache_enabled
                or (isinstance(da3_cache_dir, str) and bool(da3_cache_dir))
            )
            and not isinstance(da3_scale, bool)
            and isinstance(da3_scale, (int, float))
            and da3_scale > 0
            and da3_output_height == face_size
            and da3_output_width == face_size
            and pioneer.get("depth_source") == "DA3"
            and bundle_count is not None
            and _as_int(pioneer.get("depth_inference_count"))
            == 6 * bundle_count
            and _as_int(pioneer.get("artifact_committed_bundle_count"))
            == bundle_count
            and _as_int(pioneer.get("depth_cache_hit_count"))
            == da3_bundle_cache_hits
            and da3_face_rows_verified
            and config_snapshot_verified
            and profile_snapshot_verified
            and macarons_params_snapshot_verified
            and config.get("macarons_params_sha256")
            == manifest_macarons_params_sha
            and metrics_run.get("macarons_params_sha256")
            == manifest_macarons_params_sha
            and run_commit_verified
            and runtime_snapshot_integrity_verified
            and calibration_snapshot_verified
            and calibration_semantics_verified
            and import_source_verified
            and asset_hashes_verified
            and texture_tree_verified
            and model_cache_hashes_verified
        )
    )
    if (
        depth_source == "DA3"
        and effective_status == "PASS"
        and not da3_cubemap_provenance_verified
    ):
        effective_status = "UNKNOWN"
        completion_verified = False
        status_reason = "DA3 cubemap RGB-only face provenance is incomplete."

    normalized_config = {
        "scene": scene,
        "planner": planner,
        "observation_mode": observation_mode,
        "cubemap6": cubemap6,
        "face_count": face_count,
        "face_size": face_size,
        "face_fov_degrees": face_fov,
        "start": start_index,
        "seed_random": seed_random,
        "seed_torch": seed_torch,
        "budget_observations": budget,
        "beam_width": beam_width,
        "beam_steps": beam_steps,
        "proxy_points": proxy_points,
        "depth_source": depth_source,
        "collision": collision,
        "debug_profile": debug_profile,
        "coverage_comparable": coverage_comparable,
    }
    if depth_source == "DA3":
        normalized_config.update(
            {
                "da3_model_id": da3_model_id,
                "da3_model_revision": da3_model_revision,
                "da3_model_config_sha256": da3_model_config_sha,
                "da3_model_weights_sha256": da3_model_weights_sha,
                "da3_source_revision": da3_source_revision,
                "da3_source_tree_sha256": da3_source_tree_sha,
                "da3_window_size": da3_window_size,
                "da3_process_res": da3_process_res,
                "da3_process_res_method": da3_process_res_method,
                "da3_scene_units_per_meter": da3_scale,
                "da3_output_height": da3_output_height,
                "da3_output_width": da3_output_width,
                "da3_confidence_percentile": da3_confidence_percentile,
                "da3_cache_enabled": da3_cache_enabled,
                "macarons_params_sha256": manifest_macarons_params_sha,
                "scene_texture_tree_sha256": manifest_texture_tree_sha,
            }
        )
    if issue in {"PAN-11", "PAN-30"}:
        normalized_config.update(
            {
                "pioneer_planner_state_mode": planner_state_mode,
                "planner_state_dimension": planner_state_dimension,
                "pioneer_cubemap_rig_frame": cubemap_rig_frame,
                "pioneer_cubemap_extrinsics_version": cubemap_extrinsics_version,
                "pioneer_canonical_orientation_indices": canonical_orientation_indices,
                "pioneer_filter_occupied_position_candidates": (
                    filter_occupied_position_candidates
                ),
                "validation_require_complete_occupied_pose": (
                    require_complete_occupied_pose
                ),
                **(
                    {
                        "validation_position_policy": position_policy,
                        "position_policy_sha256": manifest_position_policy_sha,
                    }
                    if issue == "PAN-30"
                    else {}
                ),
            }
        )
    normalized_hash = _canonical_sha256(normalized_config)

    artifact_count = len(artifact_set.get("artifacts", {}))
    final_full = _full_commit(final_branch_commit)
    checks = {
        "run_commit_full": scientific_full is not None,
        "final_branch_commit_full": final_full is not None,
        "normalized_config_fingerprint": True,
        "artifact_hash_set": artifact_count > 0,
        "completion_status_evidence": completion_verified,
        "run_commit_matches_manifest": (
            run_commit_verified if depth_source == "DA3" or issue == "PAN-30" else True
        ),
        "runtime_snapshot_integrity": (
            runtime_snapshot_integrity_verified
            if depth_source == "DA3" or issue == "PAN-30"
            else True
        ),
        "config_snapshot_hash": (
            config_snapshot_verified if depth_source == "DA3" or issue == "PAN-30" else True
        ),
        "debug_profile_snapshot_hash": (
            profile_snapshot_verified if depth_source == "DA3" or issue == "PAN-30" else True
        ),
        "position_policy_snapshot_hash": (
            position_policy_snapshot_verified if issue == "PAN-30" else True
        ),
        "macarons_params_snapshot_hash": (
            macarons_params_snapshot_verified if depth_source == "DA3" else True
        ),
        "da3_import_source_binding": (
            import_source_verified if depth_source == "DA3" else True
        ),
        "scene_asset_hashes": (
            asset_hashes_verified if depth_source == "DA3" else True
        ),
        "scene_texture_tree_hash": (
            texture_tree_verified if depth_source == "DA3" else True
        ),
        "da3_model_cache_hashes": (
            model_cache_hashes_verified if depth_source == "DA3" else True
        ),
        "scene_calibration_semantics": (
            calibration_semantics_verified if depth_source == "DA3" else True
        ),
    }
    score = sum(bool(value) for value in checks.values()) / len(checks)
    grade = "A" if score >= 0.8 else ("B" if score >= 0.6 else "C")

    render_counts = {
        "real_bundle_count": _as_int(pioneer.get("bundle_count")),
        "real_face_render_count": _as_int(pioneer.get("real_face_render_count")),
        "imagined_bundle_render_count": _as_int(
            pioneer.get("imagined_bundle_render_count")
        ),
        "imagined_face_render_count": _as_int(pioneer.get("imagined_face_render_count")),
        "imagined_history_bundle_render_count": _as_int(
            pioneer.get("imagined_history_bundle_render_count")
        ),
        "imagined_history_face_render_count": _as_int(
            pioneer.get("imagined_history_face_render_count")
        ),
        "imagined_candidate_bundle_render_count": _as_int(
            pioneer.get("imagined_candidate_bundle_render_count")
        ),
        "imagined_candidate_face_render_count": _as_int(
            pioneer.get("imagined_candidate_face_render_count")
        ),
    }
    timing = {
        "trajectory_seconds": _as_float(latency.get("trajectory_seconds")),
        "provider_seconds": _as_float(latency.get("provider_seconds")),
        "geometry_seconds": _as_float(latency.get("geometry_seconds")),
    }
    vram = {
        "peak_allocated_mib": _as_float(cuda.get("peak_allocated_mib")),
        "peak_reserved_mib": _as_float(cuda.get("peak_reserved_mib")),
    }

    relation = "unknown"
    if scientific_full and final_full:
        relation = "same_as_run_commit" if scientific_full == final_full else "different_from_run_commit"
    conclusion = (
        "Verified PIONEER six-face cubemap run completed with exit code 0."
        if effective_status == "PASS"
        else f"No PASS claim: {status_reason}."
    )
    notes = "Debug-only coverage is not comparable." if coverage_comparable is False else None
    if issue in {"PAN-11", "PAN-30"}:
        notes = f"Issue provenance: {issue}." + (f" {notes}" if notes else "")
    now = datetime.now(timezone.utc)
    return {
        "experiment_id": experiment_id,
        "date": now.date().isoformat(),
        "pr": None,
        "commit": _short_commit(scientific_run_commit),
        "multica_issue": issue,
        "category": "real_mesh_smoke",
        "scene": scene,
        "planner": planner,
        "observation_source": "six-face cubemap" if cubemap6 is True else None,
        "depth_source": depth_source,
        "start": start_index,
        "seed_random": seed_random,
        "seed_torch": seed_torch,
        "observations": observation_count if observation_count is not None else budget,
        "beam_width": beam_width,
        "beam_steps": beam_steps,
        "mapping_multiplier": None,
        "sensor_range": _as_float(_pick(sources, ("sensor_range",), ("run", "sensor_range"))),
        "proxy_points": proxy_points,
        "gt_surface_points": None,
        "collision": collision,
        "changed_params": {
            "planning_observation_mode": observation_mode,
            "pioneer_face_count": face_count,
            "pioneer_face_size": face_size,
            "pioneer_face_fov_degrees": face_fov,
            **(
                {
                    "pioneer_planner_state_mode": planner_state_mode,
                    "planner_state_dimension": planner_state_dimension,
                    "pioneer_cubemap_rig_frame": cubemap_rig_frame,
                    "pioneer_cubemap_extrinsics_version": cubemap_extrinsics_version,
                    "pioneer_filter_occupied_position_candidates": (
                        filter_occupied_position_candidates
                    ),
                    "validation_require_complete_occupied_pose": (
                        require_complete_occupied_pose
                    ),
                }
                if issue in {"PAN-11", "PAN-30"}
                else {}
            ),
        },
        "baseline": None,
        "final_normalized_coverage": final_normalized_coverage,
        "final_raw_coverage": final_raw_coverage,
        "path_length": _as_float(trajectory.get("path_length_scene_units")),
        "final_points": _as_int(trajectory.get("final_point_count")),
        "status": effective_status,
        "conclusion": conclusion,
        "artifacts": artifact_set_id,
        "evidence": {
            "run_dir": str(run_dir),
            "online_metrics": str(online_metrics_path) if online_metrics_path.is_file() else None,
            "wrapper_exit_code": exit_code,
            "requested_status": requested_status,
            "status_reason": status_reason,
        },
        "comparability_group": (
            None if coverage_comparable is not True else f"{issue}-PIONEER"
        ),
        "notes": notes,
        "pioneer": {
            "observation_mode": observation_mode,
            "cubemap6": cubemap6,
            "face_count": face_count,
            "face_size": face_size,
            "face_fov_degrees": face_fov,
            "seed_random": seed_random,
            "seed_torch": seed_torch,
            "budget_observations": budget,
            "beam_width": beam_width,
            "beam_steps": beam_steps,
            "proxy_points": proxy_points,
            "depth_source": depth_source,
            "depth_inference_count": _as_int(
                pioneer.get("depth_inference_count")
            ),
            "depth_cache_hit_count": _as_int(
                pioneer.get("depth_cache_hit_count")
            ),
            "artifact_committed_bundle_count": _as_int(
                pioneer.get("artifact_committed_bundle_count")
            ),
            "da3_cubemap_provenance_verified": (
                da3_cubemap_provenance_verified
            ),
            **(
                {
                    "da3_window_size": da3_window_size,
                    "da3_process_res": da3_process_res,
                    "da3_process_res_method": da3_process_res_method,
                    "da3_output_size": [da3_output_height, da3_output_width],
                    "da3_confidence_percentile": da3_confidence_percentile,
                    "da3_cache_enabled": da3_cache_enabled,
                    "da3_cache_dir": da3_cache_dir,
                    "da3_import_package_tree_sha256": import_tree_sha,
                    "da3_calibration_sha256": manifest_calibration_sha,
                    "da3_model_config_sha256": da3_model_config_sha,
                    "da3_model_weights_sha256": da3_model_weights_sha,
                    "macarons_params_sha256": manifest_macarons_params_sha,
                    "scene_texture_tree_sha256": manifest_texture_tree_sha,
                }
                if depth_source == "DA3"
                else {}
            ),
            **(
                {
                    "pioneer_planner_state_mode": planner_state_mode,
                    "planner_state_dimension": planner_state_dimension,
                    "pioneer_cubemap_rig_frame": cubemap_rig_frame,
                    "pioneer_cubemap_extrinsics_version": cubemap_extrinsics_version,
                    "pioneer_canonical_orientation_indices": canonical_orientation_indices,
                    "pioneer_filter_occupied_position_candidates": (
                        filter_occupied_position_candidates
                    ),
                    "validation_require_complete_occupied_pose": (
                        require_complete_occupied_pose
                    ),
                    "planner_search": {
                        "state_mode": planner_search.get("state_mode"),
                        "state_dimension": _as_int(
                            planner_search.get("state_dimension")
                        ),
                        "cubemap_rig_frame": planner_search.get(
                            "cubemap_rig_frame"
                        ),
                        "cubemap_extrinsics_version": planner_search.get(
                            "cubemap_extrinsics_version"
                        ),
                        "totals": planner_search_totals,
                    },
                    **(
                        {
                            "pan11_contract_verified": position_planner_contract_verified
                        }
                        if issue == "PAN-11"
                        else {
                            "pan30_contract_verified": pan30_low_altitude_contract_verified,
                            "validation_position_policy": position_policy,
                            "position_policy_sha256": manifest_position_policy_sha,
                        }
                    ),
                }
                if issue in {"PAN-11", "PAN-30"}
                else {}
            ),
            "coverage": {
                "final_normalized": final_normalized_coverage,
                "final_raw": final_raw_coverage,
                "comparable": coverage_comparable,
            },
            "timing": timing,
            "vram": vram,
            "render_counts": render_counts,
            "cubemap6_metrics_verified": cubemap6_metrics_verified,
        },
        "provenance": {
            **({"issue": issue} if issue in {"PAN-11", "PAN-30"} else {}),
            "scientific_run_commit": scientific_run_commit or None,
            "scientific_run_commit_full": scientific_full,
            "scientific_run_commit_resolution": "user_supplied_not_resolved",
            "pr_final_head_sha": final_full,
            "final_branch_commit": final_branch_commit or None,
            "config_delta": {},
            "config_diff_status": "semantic fields extracted from live run evidence",
            "traceability_level": "complete" if grade == "A" else "partial",
            "artifact_set_ids": [artifact_set_id] if artifact_count else [],
            "normalized_config": normalized_config,
            "normalized_config_sha256": normalized_hash,
            "normalized_config_hash_semantics": (
                "SHA256 of canonical registry-normalized experiment-defining fields; "
                "NOT a byte hash of config.json"
            ),
            "artifact_provenance_status": (
                "verified_sha256_from_live_files" if artifact_count else "no_readable_artifacts"
            ),
            "pr_head_at_run": scientific_full,
            "pr_head_at_run_status": "user_supplied_not_resolved",
            "pr_final_head_relation": relation,
            "provenance_checks": checks,
            "provenance_completeness_score": score,
            "provenance_grade": grade,
        },
        "repair_links": [],
        "repair_ids": [],
        "intervention_ids": [],
    }


def _markdown_value(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _live_section(registry: Mapping[str, Any]) -> str:
    records = [
        record
        for record in registry.get("experiments", [])
        if isinstance(record, Mapping)
        and (
            record.get("multica_issue") in SUPPORTED_ISSUES
            or str(record.get("planner") or "").lower() == "pioneer"
        )
    ]
    lines = [
        LIVE_START,
        "## Live PIONEER experiments",
        "",
        "This section is generated from live artifacts. A PASS appears only after wrapper exit 0 and PIONEER cubemap6 online metrics are both present.",
        "",
        "| Experiment ID | Scene | Status | Cubemap6 | Observations | Coverage | Time (s) | Peak VRAM (MiB) | Real / imagined face renders | Artifact set |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        pioneer = record.get("pioneer") if isinstance(record.get("pioneer"), Mapping) else {}
        coverage = pioneer.get("coverage") if isinstance(pioneer.get("coverage"), Mapping) else {}
        timing = pioneer.get("timing") if isinstance(pioneer.get("timing"), Mapping) else {}
        vram = pioneer.get("vram") if isinstance(pioneer.get("vram"), Mapping) else {}
        renders = pioneer.get("render_counts") if isinstance(pioneer.get("render_counts"), Mapping) else {}
        render_summary = "{} / {}".format(
            _markdown_value(renders.get("real_face_render_count")),
            _markdown_value(renders.get("imagined_face_render_count")),
        )
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _markdown_value(record.get("experiment_id")),
                _markdown_value(record.get("scene")),
                _markdown_value(record.get("status")),
                _markdown_value(pioneer.get("cubemap6")),
                _markdown_value(record.get("observations")),
                _markdown_value(coverage.get("final_normalized")),
                _markdown_value(timing.get("trajectory_seconds")),
                _markdown_value(vram.get("peak_reserved_mib")),
                render_summary,
                _markdown_value(record.get("artifacts")),
            )
        )
    if not records:
        lines.append("| unknown | unknown | UNKNOWN | unknown | unknown | unknown | unknown | unknown | unknown / unknown | unknown |")
    lines.extend(["", f"Generated at: `{registry.get('generated_at', 'unknown')}`", LIVE_END])
    return "\n".join(lines)


def _update_markdown(markdown: str, registry: Mapping[str, Any]) -> str:
    live_count = sum(
        isinstance(record, Mapping)
        and (
            record.get("multica_issue") in SUPPORTED_ISSUES
            or str(record.get("planner") or "").lower() == "pioneer"
        )
        for record in registry.get("experiments", [])
    )
    historical_count = int(registry["record_count"]) - live_count
    record_line = (
        "Experiment records：**{}**（{} historical v1.0 IDs + {} live PIONEER IDs）".format(
            registry["record_count"], historical_count, live_count
        )
    )
    markdown = re.sub(
        r"(?m)^(?P<prefix>- )?Experiment records：\*\*\d+\*\*(?:（[^\n]*）)?$",
        lambda match: (match.group("prefix") or "") + record_line,
        markdown,
    )
    grade_counts = registry.get("provenance_grade_counts", {})
    grade_line = "Provenance grades：**A={} / B={} / C={}**".format(
        grade_counts.get("A", 0), grade_counts.get("B", 0), grade_counts.get("C", 0)
    )
    markdown = re.sub(
        r"(?m)^(?P<prefix>- )?Provenance grades：\*\*[^\n]*\*\*$",
        lambda match: (match.group("prefix") or "") + grade_line,
        markdown,
    )
    real_mesh_count = registry.get("category_counts", {}).get("real_mesh_smoke", 0)
    markdown = re.sub(
        r"(?m)^\| real_mesh_smoke \| \d+ \|$",
        f"| real_mesh_smoke | {real_mesh_count} |",
        markdown,
    )

    section = _live_section(registry)
    pattern = re.compile(
        re.escape(LIVE_START) + r".*?" + re.escape(LIVE_END), re.DOTALL
    )
    if pattern.search(markdown):
        markdown = pattern.sub(section, markdown)
    else:
        markdown = markdown.rstrip() + "\n\n" + section + "\n"
    return markdown


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def register_experiment(
    *,
    registry_json: Path,
    registry_md: Path,
    run_dir: Path,
    online_metrics: Path,
    experiment_id: str,
    scientific_run_commit: str,
    final_branch_commit: str,
    status: str,
    issue: str = "PAN-10",
    artifact_roots: Sequence[Path] = (),
) -> Mapping[str, Any]:
    if issue not in SUPPORTED_ISSUES:
        allowed = ", ".join(SUPPORTED_ISSUES)
        raise ValueError(f"issue must be one of: {allowed}")
    registry = _read_json(registry_json)
    if registry is None:
        raise ValueError(f"Registry JSON is missing or invalid: {registry_json}")
    experiments_value = registry.get("experiments")
    if not isinstance(experiments_value, list):
        raise ValueError("Registry JSON experiments must be a list.")
    mutable = dict(registry)
    experiments = [dict(record) for record in experiments_value if isinstance(record, Mapping)]

    manifest_path = _first_file(run_dir, ("manifest.json", "manifest.txt"))
    manifest = _read_evidence_file(manifest_path) if manifest_path else {}
    metrics_document = _read_json(online_metrics) or {}
    _, config_path = _resolve_config(
        run_dir=run_dir, manifest=manifest, metrics=metrics_document
    )
    excluded: set[Path] = set()
    for path in (registry_json, registry_md):
        try:
            excluded.add(path.resolve())
        except OSError:
            pass
    artifact_set_id, artifact_set = _artifact_set(
        issue=issue,
        experiment_id=experiment_id,
        run_dir=run_dir,
        online_metrics=online_metrics,
        config_path=config_path,
        metrics=metrics_document,
        explicit_artifact_roots=artifact_roots,
        registry_paths=excluded,
    )
    record = build_record(
        issue=issue,
        experiment_id=experiment_id,
        run_dir=run_dir,
        online_metrics_path=online_metrics,
        scientific_run_commit=scientific_run_commit,
        final_branch_commit=final_branch_commit,
        requested_status=status,
        artifact_set_id=artifact_set_id,
        artifact_set=artifact_set,
    )

    matches = [index for index, item in enumerate(experiments) if item.get("experiment_id") == experiment_id]
    if matches:
        experiments[matches[0]] = dict(record)
        for index in reversed(matches[1:]):
            del experiments[index]
    else:
        experiments.append(dict(record))
    mutable["experiments"] = experiments
    mutable["record_count"] = len(experiments)
    mutable["generated_at"] = datetime.now(timezone.utc).isoformat()
    category_counter = Counter(str(item.get("category") or "unknown") for item in experiments)
    status_counter = Counter(str(item.get("status") or "UNKNOWN") for item in experiments)
    grade_counter = Counter(
        str(_nested(item, ("provenance", "provenance_grade")) or "C")
        for item in experiments
    )
    mutable["category_counts"] = _ordered_counts(registry.get("category_counts"), category_counter)
    mutable["status_counts"] = _ordered_counts(registry.get("status_counts"), status_counter)
    mutable["provenance_grade_counts"] = {
        grade: int(grade_counter.get(grade, 0)) for grade in ("A", "B", "C")
    }
    artifact_sets = dict(registry.get("artifact_sets") or {})
    artifact_sets[artifact_set_id] = artifact_set
    mutable["artifact_sets"] = artifact_sets

    current_markdown = registry_md.read_text(encoding="utf-8") if registry_md.is_file() else "# PIONEER Experiment Registry\n"
    updated_markdown = _update_markdown(current_markdown, mutable)
    _atomic_write(
        registry_json,
        json.dumps(mutable, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
    )
    _atomic_write(registry_md, updated_markdown)
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-json", required=True, type=Path)
    parser.add_argument("--registry-md", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--online-metrics", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--scientific-run-commit", required=True)
    parser.add_argument("--final-branch-commit", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument(
        "--issue",
        choices=SUPPORTED_ISSUES,
        default="PAN-10",
        help="Linear issue provenance (default: PAN-10 for legacy commands).",
    )
    parser.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        type=Path,
        help="Additional file or directory tree whose artifact bytes must be hashed.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    record = register_experiment(
        registry_json=args.registry_json.resolve(),
        registry_md=args.registry_md.resolve(),
        run_dir=args.run_dir.resolve(),
        online_metrics=args.online_metrics.resolve(),
        experiment_id=args.experiment_id,
        scientific_run_commit=args.scientific_run_commit,
        final_branch_commit=args.final_branch_commit,
        status=args.status,
        issue=args.issue,
        artifact_roots=[path.resolve() for path in args.artifact_root],
    )
    print(
        json.dumps(
            {
                "experiment_id": record["experiment_id"],
                "status": record["status"],
                "artifact_set": record["artifacts"],
                "provenance_grade": record["provenance"]["provenance_grade"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
