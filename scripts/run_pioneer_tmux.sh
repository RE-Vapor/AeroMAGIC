#!/usr/bin/env bash
set -euo pipefail

DEFAULT_PYTHON=/home/ubuntu/anaconda3/envs/magician_mve/bin/python
DEFAULT_CONFIG=test_pioneer_eiffel_quick_config.json
DEFAULT_DA3_HF_HOME=/home/ubuntu/Projects/Pioneer/experiments/runtime/MAGICIAN_MVE/.venv-myl12-hf
DEFAULT_DA3_APPEND_PATHS=/home/ubuntu/Projects/Pioneer/experiments/runtime/MAGICIAN_MVE/.venv-myl12-source/Depth-Anything-3-3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src:/home/ubuntu/Projects/Pioneer/experiments/runtime/MAGICIAN_MVE/.venv-myl12-deps

manifest_value() {
  local manifest_path="$1"
  local key="$2"
  sed -n "s/^${key}=//p" "$manifest_path" | tail -n 1
}

verify_run_snapshot() {
  local run_dir="$1"
  local python_bin="$2"
  local repo_root="$3"
  local manifest_path="$run_dir/manifest.txt"
  if [[ ! -f "$manifest_path" ]]; then
    printf 'run manifest is missing: %s\n' "$manifest_path" >&2
    return 2
  fi
  local expected_commit
  expected_commit="$(manifest_value "$manifest_path" git_commit)"
  if [[ "$(git -C "$repo_root" rev-parse HEAD)" != "$expected_commit" ]]; then
    printf 'run checkout HEAD changed after manifest creation\n' >&2
    return 2
  fi
  if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
    printf 'run checkout became dirty after manifest creation\n' >&2
    return 2
  fi
  local config_snapshot profile_snapshot
  config_snapshot="$run_dir/$(manifest_value "$manifest_path" config_snapshot)"
  profile_snapshot="$run_dir/$(manifest_value "$manifest_path" debug_profile_snapshot)"
  if [[ "$(sha256sum "$config_snapshot" | awk '{print $1}')" != \
        "$(manifest_value "$manifest_path" config_sha256)" ]]; then
    printf 'config snapshot changed after manifest creation\n' >&2
    return 2
  fi
  if [[ "$(sha256sum "$profile_snapshot" | awk '{print $1}')" != \
        "$(manifest_value "$manifest_path" debug_profile_sha256)" ]]; then
    printf 'debug profile snapshot changed after manifest creation\n' >&2
    return 2
  fi
  local macarons_params_snapshot
  macarons_params_snapshot="$run_dir/$(manifest_value "$manifest_path" macarons_params_snapshot)"
  if [[ "$(sha256sum "$macarons_params_snapshot" | awk '{print $1}')" != \
        "$(manifest_value "$manifest_path" macarons_params_sha256)" ]]; then
    printf 'base Macarons parameter snapshot changed after manifest creation\n' >&2
    return 2
  fi

  if [[ "$(manifest_value "$manifest_path" depth_source)" == "DA3" ]]; then
    local calibration_snapshot expected_tree probe probe_contract
    calibration_snapshot="$run_dir/scene_metric_calibrations.json"
    if [[ "$(sha256sum "$calibration_snapshot" | awk '{print $1}')" != \
          "$(manifest_value "$manifest_path" da3_calibration_sha256)" ]]; then
      printf 'scene calibration snapshot changed after manifest creation\n' >&2
      return 2
    fi
    if ! "$python_bin" -c '
import hashlib, json, sys
from pathlib import Path
payload = json.loads(sys.argv[1])
expected_texture_tree = sys.argv[2]
def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
for name, item in payload.items():
    if digest(item["path"]) != item["sha256"]:
        raise SystemExit(f"runtime asset changed after manifest creation: {name}")
mesh_root = Path(payload["mesh"]["path"]).resolve().parent
texture_tree = hashlib.sha256()
texture_items = [
    item for name, item in payload.items()
    if name.startswith("material_") or name.startswith("texture_")
]
for item in sorted(texture_items, key=lambda value: value["path"]):
    path = Path(item["path"]).resolve()
    texture_tree.update(path.relative_to(mesh_root).as_posix().encode("utf-8"))
    texture_tree.update(b"\0")
    texture_tree.update(item["sha256"].encode("ascii"))
    texture_tree.update(b"\n")
if not texture_items or texture_tree.hexdigest() != expected_texture_tree:
    raise SystemExit("scene material/texture tree changed after manifest creation")
' "$(manifest_value "$manifest_path" scene_asset_provenance_json)" \
  "$(manifest_value "$manifest_path" scene_texture_tree_sha256)"; then
      return 2
    fi
    expected_tree="$(manifest_value "$manifest_path" da3_source_tree_sha256)"
    probe="$(
      HF_HOME="$(manifest_value "$manifest_path" hf_home)" HF_HUB_OFFLINE=1 \
        MAGICIAN_DA3_APPEND_PATHS="$(manifest_value "$manifest_path" da3_append_paths)" \
        "$python_bin" "$repo_root/scripts/run_with_da3_overlay.py" \
        --probe-da3-import
    )"
    probe_contract="$("$python_bin" -c '
import json, sys
p = json.loads(sys.argv[1])
if p.get("source_bound") is not True:
    raise SystemExit("DA3 import is no longer source-bound")
print(p["python_tree_sha256"] + "|" + p["origin"])
' "$probe")"
    if [[ "${probe_contract%%|*}" != "$expected_tree" ]]; then
      printf 'DA3 source tree changed after manifest creation\n' >&2
      return 2
    fi
  fi
}

finish_status() {
  local return_code="$?"
  printf 'finished_at_utc=%s\nexit_code=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$return_code" \
    >> "$pioneer_status_file"
}

run_inside_tmux() {
  local gpu="$1"
  local run_dir="$2"
  local python_bin="$3"
  local config_name="$4"
  local debug_profile="$5"
  local repo_root="$6"

  cd "$repo_root"
  local config_argument="$config_name"
  if [[ "${PIONEER_ENFORCE_RUN_SNAPSHOT:-0}" == "1" ]]; then
    verify_run_snapshot "$run_dir" "$python_bin" "$repo_root"
    printf 'snapshot_integrity_preflight=PASS\n' >> "$run_dir/status.txt"
    config_argument="$run_dir/$(manifest_value "$run_dir/manifest.txt" config_snapshot)"
  fi
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$run_dir/status.txt"
  set +e
  set -o pipefail
  local entrypoint=(
    "$repo_root/test_magician_planning.py"
    -c "$config_argument"
    --debug-profile "$debug_profile"
  )
  if [[ "${PIONEER_ENFORCE_RUN_SNAPSHOT:-0}" == "1" ]]; then
    entrypoint+=(
      --debug-profiles-dir "$run_dir"
      --macarons-params-path "$run_dir/macarons_params.json"
    )
  fi
  if [[ "${PIONEER_USE_DA3:-0}" == "1" ]]; then
    entrypoint=(
      "$repo_root/scripts/run_with_da3_overlay.py"
      "${entrypoint[@]}"
    )
  fi
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$python_bin" \
    "${entrypoint[@]}" \
    2>&1 | tee "$run_dir/run.log"
  local return_code="${PIPESTATUS[0]}"
  if [[ "${PIONEER_ENFORCE_RUN_SNAPSHOT:-0}" == "1" ]]; then
    if verify_run_snapshot "$run_dir" "$python_bin" "$repo_root"; then
      printf 'snapshot_integrity_postflight=PASS\n' >> "$run_dir/status.txt"
    else
      printf 'snapshot_integrity_postflight=FAIL\n' >> "$run_dir/status.txt"
      return_code=2
    fi
  fi
  set -e
  return "$return_code"
}

if [[ "${1:-}" == "--inside-tmux" ]]; then
  shift
  if [[ $# -eq 5 ]]; then
    set -- "$1" "$2" "$3" "$4" quick "$5"
  elif [[ $# -ne 6 ]]; then
    printf 'inside-tmux requires GPU RUN_DIR PYTHON CONFIG_NAME DEBUG_PROFILE REPO_ROOT\n' >&2
    exit 2
  fi
  pioneer_status_file="$2/status.txt"
  trap finish_status EXIT
  run_inside_tmux "$@"
  exit $?
fi

if [[ $# -lt 3 || $# -gt 5 ]]; then
  printf 'usage: %s SESSION GPU RUN_DIR [CONFIG_NAME] [DEBUG_PROFILE]\n' "$0" >&2
  exit 2
fi

session="$1"
gpu="$2"
run_dir="$3"
config_name="${4:-$DEFAULT_CONFIG}"
debug_profile="${5:-}"
python_bin="${PIONEER_PYTHON:-$DEFAULT_PYTHON}"

if [[ ! "$session" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'invalid tmux session name: %s\n' "$session" >&2
  exit 2
fi
if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
  printf 'GPU must be a non-negative integer\n' >&2
  exit 2
fi
if [[ "$config_name" == */* || "$config_name" != *.json ]]; then
  printf 'CONFIG_NAME must be a JSON filename under configs/test\n' >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  printf 'Python runtime is not executable: %s\n' "$python_bin" >&2
  exit 2
fi
if tmux has-session -t "$session" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$session" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
  printf 'refusing to run from a dirty worktree\n' >&2
  exit 2
fi
config_path="$repo_root/configs/test/$config_name"
if [[ ! -f "$config_path" ]]; then
  printf 'config does not exist: %s\n' "$config_path" >&2
  exit 2
fi
if ! scene="$("$python_bin" -c '
import json, sys
scenes = json.load(open(sys.argv[1], encoding="utf-8")).get("test_scenes")
if (
    not isinstance(scenes, list)
    or len(scenes) != 1
    or not isinstance(scenes[0], str)
    or not scenes[0].strip()
):
    raise SystemExit("config test_scenes must contain exactly one non-empty scene")
print(scenes[0])
' "$config_path")"; then
  exit 2
fi
config_profile="$(
  "$python_bin" -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("debug_profile", ""))' \
    "$config_path"
)"
if [[ -z "$debug_profile" ]]; then
  debug_profile="${config_profile:-quick}"
elif [[ -n "$config_profile" && "$debug_profile" != "$config_profile" ]]; then
  printf 'DEBUG_PROFILE %s does not match config debug_profile %s\n' \
    "$debug_profile" "$config_profile" >&2
  exit 2
fi
if [[ ! "$debug_profile" =~ ^[a-z][a-z0-9-]*$ ]]; then
  printf 'DEBUG_PROFILE must use lowercase letters, digits, and hyphens\n' >&2
  exit 2
fi
profile_path="$repo_root/configs/debug/$debug_profile.json"
if [[ ! -f "$profile_path" ]]; then
  printf 'debug profile does not exist: %s\n' "$profile_path" >&2
  exit 2
fi

params_name="$("$python_bin" -c '
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8")).get("params_name", "")
if not isinstance(value, str) or not value.endswith(".json") or "/" in value or "\\\\" in value:
    raise SystemExit("params_name must be a JSON filename under configs/macarons")
print(value)
' "$config_path")"
macarons_params_path="$repo_root/configs/macarons/$params_name"
if [[ ! -f "$macarons_params_path" ]]; then
  printf 'base Macarons parameter file is missing: %s\n' "$macarons_params_path" >&2
  exit 2
fi
macarons_params_sha256="$(sha256sum "$macarons_params_path" | awk '{print $1}')"
declared_macarons_params_sha256="$("$python_bin" -c '
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("macarons_params_sha256", ""))
' "$config_path")"
if [[ -n "$declared_macarons_params_sha256" && \
      "$macarons_params_sha256" != "$declared_macarons_params_sha256" ]]; then
  printf 'base Macarons parameter hash does not match config\n' >&2
  exit 2
fi

depth_contract="$(
  "$python_bin" -c '
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
scene = sys.argv[2]
use_gt = c.get("use_perfect_depth_map", True)
if type(use_gt) is not bool:
    raise SystemExit("use_perfect_depth_map must be a boolean")
kind = str(c.get("kind_depth_map", "GT" if use_gt else "")).strip().upper()
if not use_gt and "da3_confidence_percentile" not in c:
    raise SystemExit("DA3 config must declare da3_confidence_percentile explicitly")
calibrations = c.get("da3_scene_units_per_meter", {})
scale = calibrations.get(scene, "") if isinstance(calibrations, dict) else ""
values = (
    str(use_gt).lower(), kind,
    str(c.get("da3_model_id", "")), str(c.get("da3_model_revision", "")),
    str(c.get("da3_model_config_sha256", "")),
    str(c.get("da3_model_weights_sha256", "")),
    str(c.get("da3_source_revision", "")), str(c.get("da3_source_tree_sha256", "")),
    str(c.get("da3_window_size", "")),
    str(c.get("da3_process_res", "")), str(c.get("da3_process_res_method", "")),
    str(c.get("da3_output_height", "")), str(c.get("da3_output_width", "")),
    json.dumps(c.get("da3_confidence_percentile"), separators=(",", ":")),
    str(c.get("da3_cache_enabled", "")).lower(),
    str(c.get("da3_cache_dir", "")),
    str(scale),
)
print("|".join(values))
' "$config_path" "$scene"
)"
IFS='|' read -r use_perfect_depth_map kind_depth_map da3_model_id \
  da3_model_revision da3_model_config_sha256 da3_model_weights_sha256 \
  da3_source_revision da3_source_tree_sha256 \
  da3_window_size da3_process_res \
  da3_process_res_method da3_output_height da3_output_width \
  da3_confidence_percentile da3_cache_enabled da3_cache_dir \
  da3_scene_units_per_meter <<< "$depth_contract"
depth_source=GT
use_da3=0
da3_hf_home=""
da3_append_paths=""
da3_cache_dir_resolved=""
da3_import_origin=""
da3_import_package_tree_sha256=""
da3_import_source_root=""
da3_asset_provenance_json=""
da3_calibration_sha256=""
scene_texture_tree_sha256=""
if [[ "$use_perfect_depth_map" == "false" ]]; then
  if [[ "$kind_depth_map" != "DA3" ]]; then
    printf 'PIONEER non-GT depth currently supports only DA3\n' >&2
    exit 2
  fi
  for required_value in "$da3_model_id" "$da3_model_revision" \
    "$da3_model_config_sha256" "$da3_model_weights_sha256" \
    "$da3_source_revision" "$da3_source_tree_sha256" \
    "$da3_window_size" "$da3_process_res" \
    "$da3_process_res_method" "$da3_output_height" "$da3_output_width" \
    "$da3_cache_enabled" "$da3_scene_units_per_meter"; do
    if [[ -z "$required_value" ]]; then
      printf 'DA3 config contract is incomplete\n' >&2
      exit 2
    fi
  done
  if [[ "$da3_cache_enabled" != "true" && "$da3_cache_enabled" != "false" ]]; then
    printf 'da3_cache_enabled must be a boolean\n' >&2
    exit 2
  fi
  if [[ "$da3_cache_enabled" == "true" && -z "$da3_cache_dir" ]]; then
    printf 'DA3 cache directory is required when cache is enabled\n' >&2
    exit 2
  fi
  depth_source=DA3
  use_da3=1
  da3_hf_home="${PIONEER_DA3_HF_HOME:-$DEFAULT_DA3_HF_HOME}"
  da3_append_paths="${PIONEER_DA3_APPEND_PATHS:-$DEFAULT_DA3_APPEND_PATHS}"
  if [[ ! -d "$da3_hf_home" ]]; then
    printf 'DA3 HF_HOME does not exist: %s\n' "$da3_hf_home" >&2
    exit 2
  fi
  IFS=':' read -r -a da3_overlay_entries <<< "$da3_append_paths"
  if [[ ${#da3_overlay_entries[@]} -lt 2 ]]; then
    printf 'DA3 append overlay must contain dependency and source directories\n' >&2
    exit 2
  fi
  for overlay_path in "${da3_overlay_entries[@]}"; do
    if [[ ! -d "$overlay_path" ]]; then
      printf 'DA3 append overlay path does not exist: %s\n' "$overlay_path" >&2
      exit 2
    fi
  done
  if [[ "${da3_overlay_entries[0]}" != *"$da3_source_revision"* ]]; then
    printf 'DA3 source overlay is not bound to configured source revision: %s\n' \
      "${da3_overlay_entries[0]}" >&2
    exit 2
  fi
  if [[ ! -f "$repo_root/scripts/run_with_da3_overlay.py" ]]; then
    printf 'DA3 overlay bootstrap is missing\n' >&2
    exit 2
  fi
  da3_probe="$(
    HF_HOME="$da3_hf_home" HF_HUB_OFFLINE=1 \
      MAGICIAN_DA3_APPEND_PATHS="$da3_append_paths" \
      "$python_bin" "$repo_root/scripts/run_with_da3_overlay.py" \
      --probe-da3-import
  )"
  if ! da3_probe_contract="$("$python_bin" -c '
import json, sys
p = json.loads(sys.argv[1])
if p.get("source_bound") is not True:
    raise SystemExit("DA3 import did not resolve from the pinned source overlay")
values = (p.get("origin"), p.get("python_tree_sha256"), p.get("source_root"))
if not all(isinstance(v, str) and v for v in values):
    raise SystemExit("DA3 import probe is incomplete")
print("|".join(values))
' "$da3_probe")"; then
    exit 2
  fi
  IFS='|' read -r da3_import_origin da3_import_package_tree_sha256 \
    da3_import_source_root <<< "$da3_probe_contract"
  if [[ ! "$da3_import_package_tree_sha256" =~ ^[0-9a-f]{64}$ ]]; then
    printf 'DA3 package tree hash is invalid\n' >&2
    exit 2
  fi
  if [[ "$da3_import_package_tree_sha256" != "$da3_source_tree_sha256" ]]; then
    printf 'DA3 package tree hash does not match configured source tree hash\n' >&2
    exit 2
  fi
  da3_cache_dir_resolved="$("$python_bin" -c \
    'from pathlib import Path; import sys; print((Path(sys.argv[1]) / sys.argv[2]).resolve())' \
    "$repo_root" "$da3_cache_dir")"

  calibration_path="$repo_root/configs/test/scene_metric_calibrations.json"
  if [[ ! -f "$calibration_path" ]]; then
    printf 'scene metric calibration file is missing: %s\n' "$calibration_path" >&2
    exit 2
  fi
  da3_calibration_sha256="$(sha256sum "$calibration_path" | awk '{print $1}')"
  if ! da3_asset_contract="$("$python_bin" -c '
import hashlib, json, shlex, sys
from pathlib import Path
repo, config_path, calibration_path, hf_home = map(Path, sys.argv[1:5])
scene = sys.argv[5]
c = json.load(open(config_path, encoding="utf-8"))
d = json.load(open(calibration_path, encoding="utf-8"))
cal = d.get("calibrations", {}).get(scene)
if not isinstance(cal, dict):
    raise SystemExit(f"missing scene calibration: {scene}")
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
assets = {}
for key in ("adaptation_manifest", "mesh", "settings"):
    item = cal.get(key)
    if not isinstance(item, dict):
        raise SystemExit(f"calibration is missing {key}")
    path = (repo / item["path"]).resolve()
    actual = digest(path)
    if actual != item.get("sha256"):
        raise SystemExit(f"scene asset hash mismatch: {path}")
    assets[key] = {"path": str(path), "sha256": actual}
occupied = (repo / "data" / "Macarons++" / scene / "occupied_pose.pt").resolve()
weight = (repo / "weights" / "macarons" / c.get("model_name", "trained_macarons.pth")).resolve()
assets["occupied_pose"] = {"path": str(occupied), "sha256": digest(occupied)}
assets["planner_weight"] = {"path": str(weight), "sha256": digest(weight)}
model_root = (
    hf_home / "hub" /
    ("models--" + c["da3_model_id"].replace("/", "--")) /
    "snapshots" / c["da3_model_revision"]
)
for key, filename, expected_key in (
    ("da3_model_config", "config.json", "da3_model_config_sha256"),
    ("da3_model_weights", "model.safetensors", "da3_model_weights_sha256"),
):
    path = model_root / filename
    actual = digest(path)
    if actual != c.get(expected_key):
        raise SystemExit(f"DA3 model cache hash mismatch: {path}")
    assets[key] = {"path": str(path.resolve()), "sha256": actual}

mesh = Path(assets["mesh"]["path"])
scene_root = mesh.parent
material_paths = []
for raw in mesh.read_text(encoding="utf-8", errors="replace").splitlines():
    stripped = raw.strip()
    if stripped.lower().startswith("mtllib "):
        material_paths.append((scene_root / stripped.split(None, 1)[1]).resolve())
if not material_paths:
    raise SystemExit(f"scene OBJ has no material library: {mesh}")
texture_paths = set()
for index, material in enumerate(sorted(set(material_paths))):
    assets[f"material_{index:03d}"] = {
        "path": str(material), "sha256": digest(material)
    }
    for raw in material.read_text(encoding="utf-8", errors="replace").splitlines():
        tokens = shlex.split(raw, comments=True)
        if tokens and tokens[0].lower().startswith("map_") and len(tokens) >= 2:
            texture_paths.add((material.parent / tokens[-1]).resolve())
if not texture_paths:
    raise SystemExit(f"scene material closure has no textures: {mesh}")
for index, texture in enumerate(sorted(texture_paths)):
    assets[f"texture_{index:04d}"] = {
        "path": str(texture), "sha256": digest(texture)
    }
tree = hashlib.sha256()
for path in sorted(set(material_paths) | texture_paths):
    relative = path.relative_to(scene_root).as_posix()
    tree.update(relative.encode("utf-8"))
    tree.update(b"\0")
    tree.update(digest(path).encode("ascii"))
    tree.update(b"\n")
tree_sha256 = tree.hexdigest()
if tree_sha256 != c.get("scene_texture_tree_sha256"):
    raise SystemExit("scene material/texture tree hash does not match config")
print(tree_sha256 + "|" + json.dumps(assets, sort_keys=True, separators=(",", ":")))
' "$repo_root" "$config_path" "$calibration_path" "$da3_hf_home" "$scene")"; then
    exit 2
  fi
  IFS='|' read -r scene_texture_tree_sha256 da3_asset_provenance_json \
    <<< "$da3_asset_contract"
fi

declared_run_dir="$(
  "$python_bin" -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("experiment_run_dir", ""))' \
    "$config_path"
)"
canonical_run_dir="$(
  "$python_bin" -c \
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' \
    "$run_dir"
)"
if [[ "$use_da3" == "1" ]]; then
  if ! "$python_bin" -c '
from pathlib import Path
import sys
Path(sys.argv[1]).resolve().relative_to(Path(sys.argv[2]).resolve())
' "$da3_cache_dir_resolved" "$canonical_run_dir"; then
    printf 'DA3 cache directory must be isolated below RUN_DIR\n' >&2
    exit 2
  fi
fi
if [[ -n "$declared_run_dir" ]]; then
  canonical_declared_run_dir="$(
    "$python_bin" -c \
      'from pathlib import Path; import sys; print((Path(sys.argv[1]) / sys.argv[2]).resolve())' \
      "$repo_root" "$declared_run_dir"
  )"
  if [[ "$canonical_run_dir" != "$canonical_declared_run_dir" ]]; then
    printf 'RUN_DIR %s does not match config experiment_run_dir %s\n' \
      "$canonical_run_dir" "$canonical_declared_run_dir" >&2
    exit 2
  fi
fi

mutable_targets="$("$python_bin" -c '
import json, sys
from pathlib import Path
repo, config_path, profile_path, scene = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
c = json.load(open(config_path, encoding="utf-8"))
p = json.load(open(profile_path, encoding="utf-8"))
suffix = p["output_suffix"]
targets = []
def suffixed_name(value):
    path = Path(value)
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")
if c.get("results_json_name"):
    targets.append(repo / "results" / "scene_exploration" / suffixed_name(c["results_json_name"]))
if c.get("lmdb_dir_name"):
    targets.append(repo / "results" / "scene_exploration" / (c["lmdb_dir_name"] + "_" + suffix))
if c.get("validation_memory_dir_name"):
    targets.append(repo / "data" / "Macarons++" / scene / (c["validation_memory_dir_name"] + "_" + suffix))
if c.get("experiment_metrics_dir"):
    targets.append((repo / (c["experiment_metrics_dir"] + "_" + suffix)).resolve())
if c.get("da3_cache_dir"):
    targets.append((repo / c["da3_cache_dir"]).resolve())
for target in targets:
    print(target.resolve())
' "$repo_root" "$config_path" "$profile_path" "$scene")"
while IFS= read -r mutable_target; do
  if [[ -n "$mutable_target" && -e "$mutable_target" ]]; then
    printf 'refusing to reuse mutable experiment output: %s\n' "$mutable_target" >&2
    exit 2
  fi
done <<< "$mutable_targets"

if [[ -e "$run_dir" ]]; then
  if [[ ! -d "$run_dir" ]]; then
    printf 'run path exists and is not a directory: %s\n' "$run_dir" >&2
    exit 2
  fi
  if [[ -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    printf 'refusing to reuse non-empty run directory: %s\n' "$run_dir" >&2
    exit 2
  fi
fi

mkdir -p "$run_dir"
run_dir="$(cd "$run_dir" && pwd)"
script_path="$repo_root/scripts/run_pioneer_tmux.sh"
commit_sha="$(git -C "$repo_root" rev-parse HEAD)"
config_sha="$(sha256sum "$config_path" | awk '{print $1}')"
profile_sha="$(sha256sum "$profile_path" | awk '{print $1}')"
cp -p -- "$config_path" "$run_dir/config.json"
profile_snapshot_name="$debug_profile.json"
cp -p -- "$profile_path" "$run_dir/$profile_snapshot_name"
cp -p -- "$macarons_params_path" "$run_dir/macarons_params.json"
if [[ "$(sha256sum "$run_dir/config.json" | awk '{print $1}')" != "$config_sha" ]]; then
  printf 'config snapshot hash mismatch\n' >&2
  exit 2
fi
if [[ "$(sha256sum "$run_dir/$profile_snapshot_name" | awk '{print $1}')" != "$profile_sha" ]]; then
  printf 'debug profile snapshot hash mismatch\n' >&2
  exit 2
fi
if [[ "$(sha256sum "$run_dir/macarons_params.json" | awk '{print $1}')" != "$macarons_params_sha256" ]]; then
  printf 'base Macarons parameter snapshot hash mismatch\n' >&2
  exit 2
fi
if [[ "$use_da3" == "1" ]]; then
  cp -p -- "$calibration_path" "$run_dir/scene_metric_calibrations.json"
  if [[ "$(sha256sum "$run_dir/scene_metric_calibrations.json" | awk '{print $1}')" != "$da3_calibration_sha256" ]]; then
    printf 'scene calibration snapshot hash mismatch\n' >&2
    exit 2
  fi
fi
expected_observations="$(
  "$python_bin" -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["overrides"]["experiment_budget_observations"])' \
    "$profile_path"
)"
expected_real_face_renders="$((expected_observations * 6))"
planner_contract="$(
  "$python_bin" -c \
    'import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
m = c.get("pioneer_planner_state_mode", "legacy_pose5d")
r = c.get("pioneer_cubemap_rig_frame", "world" if m == "position_only" else "body")
v = c.get("pioneer_cubemap_extrinsics_version", "")
o = json.dumps(c.get("pioneer_canonical_orientation_indices"), separators=(",", ":"))
f = c.get("pioneer_filter_occupied_position_candidates", False)
q = c.get("validation_require_complete_occupied_pose", False)
if type(f) is not bool or type(q) is not bool:
    raise SystemExit("occupied-position filter flags must be booleans")
print("|".join((m, r, v, o, str(f).lower(), str(q).lower())))' \
    "$config_path"
)"
IFS='|' read -r planner_state_mode cubemap_rig_frame \
  cubemap_extrinsics_version canonical_orientation_indices \
  filter_occupied_position_candidates require_complete_occupied_pose \
  <<< "$planner_contract"
if [[ "$planner_state_mode" == "position_only" ]]; then
  planner_state_dimension=3
  raw_action_branches_per_parent=6
  orientation_action_branches_per_parent=0
else
  planner_state_dimension=5
  raw_action_branches_per_parent=10
  orientation_action_branches_per_parent=4
fi

{
  printf 'schema_version=1\n'
  printf 'runtime_snapshot_integrity_contract=pre-and-post-v1\n'
  printf 'planner=pioneer\n'
  printf 'observation_mode=cubemap6\n'
  printf 'scene=%s\n' "$scene"
  printf 'tmux_session=%s\n' "$session"
  printf 'gpu=%s\n' "$gpu"
  printf 'git_commit=%s\n' "$commit_sha"
  printf 'config=%s\n' "$config_name"
  printf 'config_sha256=%s\n' "$config_sha"
  printf 'config_snapshot=config.json\n'
  printf 'debug_profile=%s\n' "$debug_profile"
  printf 'debug_profile_sha256=%s\n' "$profile_sha"
  printf 'debug_profile_snapshot=%s\n' "$profile_snapshot_name"
  printf 'macarons_params_name=%s\n' "$params_name"
  printf 'macarons_params_snapshot=macarons_params.json\n'
  printf 'macarons_params_sha256=%s\n' "$macarons_params_sha256"
  printf 'depth_source=%s\n' "$depth_source"
  printf 'use_perfect_depth_map=%s\n' "$use_perfect_depth_map"
  printf 'kind_depth_map=%s\n' "$kind_depth_map"
  printf 'da3_model_id=%s\n' "$da3_model_id"
  printf 'da3_model_revision=%s\n' "$da3_model_revision"
  printf 'da3_model_config_sha256=%s\n' "$da3_model_config_sha256"
  printf 'da3_model_weights_sha256=%s\n' "$da3_model_weights_sha256"
  printf 'da3_source_revision=%s\n' "$da3_source_revision"
  printf 'da3_source_tree_sha256=%s\n' "$da3_source_tree_sha256"
  printf 'da3_window_size=%s\n' "$da3_window_size"
  printf 'da3_process_res=%s\n' "$da3_process_res"
  printf 'da3_process_res_method=%s\n' "$da3_process_res_method"
  printf 'da3_output_size=%sx%s\n' "$da3_output_height" "$da3_output_width"
  printf 'da3_confidence_percentile=%s\n' "$da3_confidence_percentile"
  printf 'da3_cache_enabled=%s\n' "$da3_cache_enabled"
  printf 'da3_cache_dir=%s\n' "$da3_cache_dir_resolved"
  printf 'da3_scene_units_per_meter=%s\n' "$da3_scene_units_per_meter"
  printf 'hf_hub_offline=%s\n' "$use_da3"
  printf 'hf_home=%s\n' "$da3_hf_home"
  printf 'da3_append_paths=%s\n' "$da3_append_paths"
  printf 'da3_import_source_root=%s\n' "$da3_import_source_root"
  printf 'da3_import_origin=%s\n' "$da3_import_origin"
  printf 'da3_import_package_tree_sha256=%s\n' "$da3_import_package_tree_sha256"
  printf 'da3_calibration_sha256=%s\n' "$da3_calibration_sha256"
  printf 'scene_asset_hashes_verified=%s\n' "$use_da3"
  printf 'scene_asset_provenance_json=%s\n' "$da3_asset_provenance_json"
  printf 'scene_texture_tree_sha256=%s\n' "$scene_texture_tree_sha256"
  printf 'expected_observations=%s\n' "$expected_observations"
  printf 'expected_real_face_renders=%s\n' "$expected_real_face_renders"
  printf 'planner_state_mode=%s\n' "$planner_state_mode"
  printf 'planner_state_dimension=%s\n' "$planner_state_dimension"
  printf 'cubemap_rig_frame=%s\n' "$cubemap_rig_frame"
  printf 'cubemap_extrinsics_version=%s\n' "$cubemap_extrinsics_version"
  printf 'canonical_orientation_indices=%s\n' "$canonical_orientation_indices"
  printf 'raw_action_branches_per_parent=%s\n' "$raw_action_branches_per_parent"
  printf 'orientation_action_branches_per_parent=%s\n' "$orientation_action_branches_per_parent"
  printf 'filter_occupied_position_candidates=%s\n' "$filter_occupied_position_candidates"
  printf 'require_complete_occupied_pose=%s\n' "$require_complete_occupied_pose"
  printf 'experiment_run_dir=%s\n' "$run_dir"
  printf 'python=%s\n' "$python_bin"
  printf 'command='
  if [[ "$use_da3" == "1" ]]; then
    printf '%q ' env "HF_HOME=$da3_hf_home" HF_HUB_OFFLINE=1 \
      "MAGICIAN_DA3_APPEND_PATHS=$da3_append_paths" "$python_bin" \
      "$repo_root/scripts/run_with_da3_overlay.py" \
      "$repo_root/test_magician_planning.py" -c "$run_dir/config.json" \
      --debug-profile "$debug_profile" --debug-profiles-dir "$run_dir" \
      --macarons-params-path "$run_dir/macarons_params.json"
  else
    printf '%q ' "$python_bin" "$repo_root/test_magician_planning.py" \
      -c "$run_dir/config.json" --debug-profile "$debug_profile" \
      --debug-profiles-dir "$run_dir" \
      --macarons-params-path "$run_dir/macarons_params.json"
  fi
  printf '\n'
} > "$run_dir/manifest.txt"

if [[ "$use_da3" == "1" ]]; then
  printf -v tmux_command '%q ' env PIONEER_ENFORCE_RUN_SNAPSHOT=1 \
    PIONEER_USE_DA3=1 \
    "HF_HOME=$da3_hf_home" HF_HUB_OFFLINE=1 \
    "MAGICIAN_DA3_APPEND_PATHS=$da3_append_paths" \
    "$script_path" --inside-tmux "$gpu" "$run_dir" "$python_bin" \
    "$config_name" "$debug_profile" "$repo_root"
else
  printf -v tmux_command '%q ' env PIONEER_ENFORCE_RUN_SNAPSHOT=1 \
    "$script_path" --inside-tmux "$gpu" "$run_dir" "$python_bin" \
    "$config_name" "$debug_profile" "$repo_root"
fi
tmux new-session -d -s "$session" -c "$repo_root" "$tmux_command"
printf 'started tmux session %s; artifacts: %s\n' "$session" "$run_dir"
