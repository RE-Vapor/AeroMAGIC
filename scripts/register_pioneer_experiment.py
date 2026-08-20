#!/usr/bin/env python3
"""Register one live PIONEER run without inventing missing evidence.

The v1.0 registry predates PAN-10.  This helper adds or replaces exactly one
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
PASS_STATUSES = {"PASS", "SUCCESS", "COMPLETED"}
FAIL_STATUSES = {"FAIL", "FAILED", "RUNTIME_FAIL", "SCIENTIFIC_FAIL"}


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
        profile_document = _read_json(repo_root / "configs" / "debug" / f"{profile}.json")
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
    experiment_id: str,
    run_dir: Path,
    online_metrics: Path,
    config_path: Optional[Path],
    metrics: Mapping[str, Any],
    registry_paths: set[Path],
) -> tuple[str, Mapping[str, Any]]:
    artifact_id = "ART-{}-LIVE".format(
        re.sub(r"[^A-Za-z0-9._-]+", "-", experiment_id).strip("-") or "PIONEER"
    )
    artifacts: dict[str, Any] = {}
    used_keys: set[str] = set()
    artifact_roots: list[Path] = []
    capture_dir = metrics.get("capture_dir")
    if isinstance(capture_dir, str) and capture_dir.strip():
        artifact_roots.append(Path(capture_dir.strip()))
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
        "issue": "PAN-10",
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
    config, _ = _resolve_config(run_dir=run_dir, manifest=manifest, metrics=metrics)
    sources = [metrics, manifest, config]

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
    bundle_rows_verified = (
        isinstance(bundles, list)
        and bundle_count is not None
        and len(bundles) == bundle_count
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
    depth_source_value = _pick(sources, ("kind_depth_map",), ("depth_source",))
    depth_source = str(depth_source_value) if depth_source_value is not None else None
    start_index = _as_int(_pick(sources, ("start_index",), ("start",)))

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
    normalized_hash = _canonical_sha256(normalized_config)

    artifact_count = len(artifact_set.get("artifacts", {}))
    scientific_full = _full_commit(scientific_run_commit)
    final_full = _full_commit(final_branch_commit)
    checks = {
        "run_commit_full": scientific_full is not None,
        "final_branch_commit_full": final_full is not None,
        "normalized_config_fingerprint": True,
        "artifact_hash_set": artifact_count > 0,
        "completion_status_evidence": completion_verified,
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
        "Verified PIONEER six-face cubemap quick smoke completed with exit code 0."
        if effective_status == "PASS"
        else f"No PASS claim: {status_reason}."
    )
    now = datetime.now(timezone.utc)
    return {
        "experiment_id": experiment_id,
        "date": now.date().isoformat(),
        "pr": None,
        "commit": _short_commit(scientific_run_commit),
        "multica_issue": "PAN-10",
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
        "comparability_group": None if coverage_comparable is not True else "PAN-10-PIONEER",
        "notes": "Debug-only coverage is not comparable." if coverage_comparable is False else None,
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
            record.get("multica_issue") == "PAN-10"
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
            record.get("multica_issue") == "PAN-10"
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
) -> Mapping[str, Any]:
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
        experiment_id=experiment_id,
        run_dir=run_dir,
        online_metrics=online_metrics,
        config_path=config_path,
        metrics=metrics_document,
        registry_paths=excluded,
    )
    record = build_record(
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
