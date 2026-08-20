"""Load bounded debug-profile overlays for planning entry points.

The selected profile is applied after the base test config is loaded.  This
keeps scene, depth-provider, calibration, and model choices in the base config
while making compute-related debug limits explicit and repeatable.
"""

import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional


DEBUG_PROFILE_NAMES = ("quick", "magician", "large-scene")

_PROFILE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
_OUTPUT_SUFFIX_PATTERN = re.compile(r"^[a-z0-9_]+$")
_OVERRIDE_MINIMUMS = {
    "beam_steps": 1,
    "beam_width": 1,
    "experiment_budget_observations": 1,
    "random_seed": 0,
    "torch_seed": 0,
    "validation_max_start_positions": 1,
    "validation_n_interpolation_steps": 1,
    "validation_n_poses_in_trajectory": 0,
    "validation_n_proxy_points": 1,
}
_PROFILE_KEYS = {
    "coverage_comparable",
    "debug_only",
    "description",
    "name",
    "output_suffix",
    "overrides",
}


def _config_value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _set_config_value(config: Any, name: str, value: Any) -> None:
    if isinstance(config, dict):
        config[name] = value
        return
    setattr(config, name, value)


def _suffixed_directory(config: Any, key: str, default: str, suffix: str) -> None:
    base = _config_value(config, key, default)
    if not isinstance(base, str) or not base:
        raise ValueError(f"{key} must be a non-empty string before applying a debug profile.")
    _set_config_value(config, key, f"{base}_{suffix}")


def _suffixed_filename(config: Any, key: str, default: str, suffix: str) -> None:
    base = _config_value(config, key, default)
    if not isinstance(base, str) or not base:
        raise ValueError(f"{key} must be a non-empty string before applying a debug profile.")
    path = Path(base)
    filename = f"{path.stem}_{suffix}{path.suffix}"
    _set_config_value(config, key, str(path.with_name(filename)))


def load_debug_profile(profile_name: str, profiles_dir: str) -> Mapping[str, Any]:
    """Load and validate one named profile without importing the GPU stack."""

    if profile_name not in DEBUG_PROFILE_NAMES or not _PROFILE_NAME_PATTERN.fullmatch(
        profile_name
    ):
        allowed = ", ".join(DEBUG_PROFILE_NAMES)
        raise ValueError(f"Unknown debug profile {profile_name!r}; choose one of: {allowed}.")

    path = Path(profiles_dir) / f"{profile_name}.json"
    with path.open("r", encoding="utf-8") as stream:
        profile = json.load(stream)
    if not isinstance(profile, dict):
        raise ValueError(f"Debug profile {path} must contain a JSON object.")

    unknown_profile_keys = sorted(set(profile) - _PROFILE_KEYS)
    missing_profile_keys = sorted(_PROFILE_KEYS - set(profile))
    if unknown_profile_keys or missing_profile_keys:
        details = []
        if unknown_profile_keys:
            details.append("unknown=" + ",".join(unknown_profile_keys))
        if missing_profile_keys:
            details.append("missing=" + ",".join(missing_profile_keys))
        raise ValueError(f"Invalid debug profile {path}: {'; '.join(details)}.")

    if profile["name"] != profile_name:
        raise ValueError(f"Debug profile {path} must declare name={profile_name!r}.")
    if not isinstance(profile["description"], str) or not profile["description"]:
        raise ValueError(f"Debug profile {path} requires a non-empty description.")
    if profile["debug_only"] is not True:
        raise ValueError(f"Debug profile {path} must declare debug_only=true.")
    if profile["coverage_comparable"] is not False:
        raise ValueError(f"Debug profile {path} must declare coverage_comparable=false.")

    output_suffix = profile["output_suffix"]
    if not isinstance(output_suffix, str) or not _OUTPUT_SUFFIX_PATTERN.fullmatch(
        output_suffix
    ):
        raise ValueError(
            f"Debug profile {path} output_suffix must use lowercase letters, digits, or underscores."
        )

    overrides = profile["overrides"]
    if not isinstance(overrides, dict):
        raise ValueError(f"Debug profile {path} overrides must be a JSON object.")
    unknown_overrides = sorted(set(overrides) - set(_OVERRIDE_MINIMUMS))
    missing_overrides = sorted(set(_OVERRIDE_MINIMUMS) - set(overrides))
    if unknown_overrides or missing_overrides:
        details = []
        if unknown_overrides:
            details.append("unknown=" + ",".join(unknown_overrides))
        if missing_overrides:
            details.append("missing=" + ",".join(missing_overrides))
        raise ValueError(f"Invalid overrides in {path}: {'; '.join(details)}.")

    for key, minimum in _OVERRIDE_MINIMUMS.items():
        value = overrides[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Debug profile {path} {key} must be an integer >= {minimum}.")

    expected_observations = (
        overrides["validation_n_poses_in_trajectory"]
        * overrides["validation_n_interpolation_steps"]
        + 1
    )
    if overrides["experiment_budget_observations"] != expected_observations:
        raise ValueError(
            f"Debug profile {path} experiment_budget_observations must equal "
            "validation_n_poses_in_trajectory * validation_n_interpolation_steps + 1."
        )
    return profile


def apply_debug_profile(
    config: Any,
    *,
    cli_profile_name: Optional[str],
    profiles_dir: str,
) -> Optional[Mapping[str, Any]]:
    """Apply a CLI- or config-selected profile and isolate its output paths.

    The CLI selector has the same meaning as a top-level ``debug_profile`` key
    in the base test config.  Supplying both is allowed only when they agree.
    """

    config_profile_name = _config_value(config, "debug_profile", None)
    if config_profile_name is not None and not isinstance(config_profile_name, str):
        raise ValueError("debug_profile in the base config must be a string.")
    if (
        cli_profile_name is not None
        and config_profile_name is not None
        and cli_profile_name != config_profile_name
    ):
        raise ValueError(
            "CLI --debug-profile and base config debug_profile must match when both are set."
        )

    profile_name = cli_profile_name or config_profile_name
    if profile_name is None:
        return None

    profile = load_debug_profile(profile_name, profiles_dir)
    for key, value in profile["overrides"].items():
        _set_config_value(config, key, value)

    suffix = profile["output_suffix"]
    _set_config_value(config, "debug_profile", profile_name)
    _set_config_value(config, "debug_profile_description", profile["description"])
    _set_config_value(config, "debug_only", True)
    _set_config_value(config, "coverage_comparable", False)
    _suffixed_directory(config, "lmdb_dir_name", "magician_lmdb", suffix)
    _suffixed_directory(config, "scone_lmdb_dir_name", "macarons_lmdb", suffix)
    _suffixed_directory(
        config,
        "validation_memory_dir_name",
        "test_memory",
        suffix,
    )
    _suffixed_filename(config, "results_json_name", "results.json", suffix)

    run_id = _config_value(config, "experiment_run_id", None)
    if run_id is not None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("experiment_run_id must be a non-empty string when present.")
        _set_config_value(config, "experiment_run_id", f"{run_id}_{suffix}")
    metrics_dir = _config_value(config, "experiment_metrics_dir", None)
    if metrics_dir is not None:
        if not isinstance(metrics_dir, str) or not metrics_dir:
            raise ValueError("experiment_metrics_dir must be a non-empty string when present.")
        _set_config_value(config, "experiment_metrics_dir", f"{metrics_dir}_{suffix}")

    return profile
