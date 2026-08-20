#!/usr/bin/env bash
set -euo pipefail

DEFAULT_PYTHON=/home/ubuntu/anaconda3/envs/magician_mve/bin/python
DEFAULT_CONFIG=test_pioneer_eiffel_quick_config.json

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
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$run_dir/status.txt"
  set +e
  set -o pipefail
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$python_bin" \
    test_magician_planning.py \
    -c "$config_name" \
    --debug-profile "$debug_profile" \
    2>&1 | tee "$run_dir/run.log"
  local return_code="${PIPESTATUS[0]}"
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
  printf 'planner=pioneer\n'
  printf 'observation_mode=cubemap6\n'
  printf 'scene=%s\n' "$scene"
  printf 'tmux_session=%s\n' "$session"
  printf 'gpu=%s\n' "$gpu"
  printf 'git_commit=%s\n' "$commit_sha"
  printf 'config=%s\n' "$config_name"
  printf 'config_sha256=%s\n' "$config_sha"
  printf 'debug_profile=%s\n' "$debug_profile"
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
  printf '%q ' "$python_bin" test_magician_planning.py -c "$config_name" --debug-profile "$debug_profile"
  printf '\n'
} > "$run_dir/manifest.txt"

printf -v tmux_command '%q ' \
  "$script_path" --inside-tmux "$gpu" "$run_dir" "$python_bin" \
  "$config_name" "$debug_profile" "$repo_root"
tmux new-session -d -s "$session" -c "$repo_root" "$tmux_command"
printf 'started tmux session %s; artifacts: %s\n' "$session" "$run_dir"
