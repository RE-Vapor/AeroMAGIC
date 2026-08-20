#!/usr/bin/env bash
set -euo pipefail

DEFAULT_PYTHON=/home/ubuntu/anaconda3/envs/magician_mve/bin/python
DEFAULT_CONFIG=test_pioneer_eiffel_quick_config.json

run_inside_tmux() {
  local gpu="$1"
  local run_dir="$2"
  local python_bin="$3"
  local config_name="$4"
  local repo_root="$5"

  finish_status() {
    local return_code="$?"
    printf 'finished_at_utc=%s\nexit_code=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$return_code" >> "$run_dir/status.txt"
  }
  trap finish_status EXIT

  cd "$repo_root"
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$run_dir/status.txt"
  set +e
  set -o pipefail
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$python_bin" \
    test_magician_planning.py \
    -c "$config_name" \
    --debug-profile quick \
    2>&1 | tee "$run_dir/run.log"
  local return_code="${PIPESTATUS[0]}"
  set -e
  return "$return_code"
}

if [[ "${1:-}" == "--inside-tmux" ]]; then
  shift
  run_inside_tmux "$@"
  exit $?
fi

if [[ $# -lt 3 || $# -gt 4 ]]; then
  printf 'usage: %s SESSION GPU RUN_DIR [CONFIG_NAME]\n' "$0" >&2
  exit 2
fi

session="$1"
gpu="$2"
run_dir="$3"
config_name="${4:-$DEFAULT_CONFIG}"
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
if [[ ! -f "$repo_root/configs/test/$config_name" ]]; then
  printf 'config does not exist: %s\n' "$repo_root/configs/test/$config_name" >&2
  exit 2
fi

mkdir -p "$run_dir"
run_dir="$(cd "$run_dir" && pwd)"
script_path="$repo_root/scripts/run_pioneer_tmux.sh"
commit_sha="$(git -C "$repo_root" rev-parse HEAD)"
config_sha="$(sha256sum "$repo_root/configs/test/$config_name" | awk '{print $1}')"

{
  printf 'schema_version=1\n'
  printf 'planner=pioneer\n'
  printf 'observation_mode=cubemap6\n'
  printf 'scene=eiffel\n'
  printf 'tmux_session=%s\n' "$session"
  printf 'gpu=%s\n' "$gpu"
  printf 'git_commit=%s\n' "$commit_sha"
  printf 'config=%s\n' "$config_name"
  printf 'config_sha256=%s\n' "$config_sha"
  printf 'python=%s\n' "$python_bin"
  printf 'command='
  printf '%q ' "$python_bin" test_magician_planning.py -c "$config_name" --debug-profile quick
  printf '\n'
} > "$run_dir/manifest.txt"

printf -v tmux_command '%q ' \
  "$script_path" --inside-tmux "$gpu" "$run_dir" "$python_bin" \
  "$config_name" "$repo_root"
tmux new-session -d -s "$session" -c "$repo_root" "$tmux_command"
printf 'started tmux session %s; artifacts: %s\n' "$session" "$run_dir"
