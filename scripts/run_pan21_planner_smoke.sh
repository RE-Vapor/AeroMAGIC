#!/usr/bin/env bash
set -euo pipefail

DEFAULT_PYTHON=/home/ubuntu/anaconda3/envs/magician_mve/bin/python
DEFAULT_BASE_CONFIG=test_pioneer_hkust_pan11_position_only_quick_config.json
PROFILE=pan21-two-observation

if [[ "${1:-}" == "--inside-tmux" ]]; then
  if [[ $# -ne 5 ]]; then
    printf 'inside-tmux requires GPU RUN_DIR PYTHON REPO_ROOT\n' >&2
    exit 2
  fi
  gpu="$2"
  run_dir="$3"
  python_bin="$4"
  repo_root="$5"
  cd "$repo_root"
  set +e
  set -o pipefail
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$python_bin" \
    "$repo_root/test_magician_planning.py" \
    -c "$run_dir/config.json" \
    --debug-profile "$PROFILE" \
    --debug-profiles-dir "$run_dir" \
    --macarons-params-path "$run_dir/macarons_params.json" \
    2>&1 | tee "$run_dir/run.log"
  return_code="${PIPESTATUS[0]}"
  set -e
  printf 'finished_at_utc=%s\nexit_code=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$return_code" \
    >> "$run_dir/status.txt"
  exit "$return_code"
fi

if [[ $# -lt 3 || $# -gt 4 ]]; then
  printf 'usage: %s SESSION GPU RUN_DIR [BASE_CONFIG]\n' "$0" >&2
  exit 2
fi

session="$1"
gpu="$2"
run_dir="$3"
base_config_name="${4:-$DEFAULT_BASE_CONFIG}"
python_bin="${PIONEER_PYTHON:-$DEFAULT_PYTHON}"
initial_ue_manifest="${PIONEER_INITIAL_UE_MANIFEST:-}"
repo_root="$(git rev-parse --show-toplevel)"

if [[ ! "$session" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'invalid tmux session name: %s\n' "$session" >&2
  exit 2
fi
if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
  printf 'GPU must be a non-negative integer\n' >&2
  exit 2
fi
if [[ "$base_config_name" == */* || "$base_config_name" != *.json ]]; then
  printf 'BASE_CONFIG must be one JSON filename under configs/test\n' >&2
  exit 2
fi
if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
  printf 'refusing to run from a dirty worktree\n' >&2
  exit 2
fi
if tmux has-session -t "$session" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$session" >&2
  exit 2
fi
if [[ -e "$run_dir" ]]; then
  printf 'refusing to reuse run path: %s\n' "$run_dir" >&2
  exit 2
fi

base_config="$repo_root/configs/test/$base_config_name"
profile_path="$repo_root/configs/debug/$PROFILE.json"
macarons_params="$repo_root/configs/macarons/macarons_default_training_config.json"
spatial_policy=/home/ubuntu/Projects/Pioneer/experiments/pan-13-hkust/ue_derived_inspection/hkust_spatial_policy.json
for required in "$base_config" "$profile_path" "$macarons_params" "$spatial_policy"; do
  if [[ ! -f "$required" ]]; then
    printf 'missing required input: %s\n' "$required" >&2
    exit 2
  fi
done
if [[ -n "$initial_ue_manifest" && ! -f "$initial_ue_manifest" ]]; then
  printf 'initial UE manifest does not exist: %s\n' "$initial_ue_manifest" >&2
  exit 2
fi

mkdir -p "$run_dir"
run_dir="$(cd "$run_dir" && pwd)"
shared_scene="$repo_root/data/Macarons++/HKUST"
overlay_scene="$run_dir/data/Macarons++/HKUST"
mkdir -p "$overlay_scene" "$run_dir/results"
while IFS= read -r -d '' source; do
  ln -s "$source" "$overlay_scene/$(basename "$source")"
done < <(find "$shared_scene" -mindepth 1 -maxdepth 1 -print0)
if [[ -e "$repo_root/results" || -L "$repo_root/results" ]]; then
  printf 'refusing to replace existing worktree results path: %s\n' \
    "$repo_root/results" >&2
  exit 2
fi
ln -s "$run_dir/results" "$repo_root/results"
cp -p -- "$profile_path" "$run_dir/$PROFILE.json"
cp -p -- "$macarons_params" "$run_dir/macarons_params.json"
cp -p -- "$base_config" "$run_dir/base_config.json"
cp -p -- "$spatial_policy" "$run_dir/hkust_spatial_policy.json"

"$python_bin" - "$run_dir/base_config.json" "$run_dir/config.json" "$run_dir" "$initial_ue_manifest" <<'PY'
import json
import sys
from pathlib import Path

source, output, run_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
initial_ue_manifest = sys.argv[4]
config = json.loads(source.read_text(encoding="utf-8"))
config.pop("debug_profile", None)
config["dataset_path"] = str(run_dir / "data" / "Macarons++")
config["experiment_run_id"] = "PAN-21-two-observation-one-move"
config["experiment_run_dir"] = str(run_dir)
config["experiment_metrics_dir"] = str(run_dir / "metrics")
config["debug_only"] = True
config["coverage_comparable"] = False
if initial_ue_manifest:
    config["validation_initial_ue_manifest"] = str(Path(initial_ue_manifest).resolve())
output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

commit_sha="$(git -C "$repo_root" rev-parse HEAD)"
base_sha="$(sha256sum "$run_dir/base_config.json" | awk '{print $1}')"
config_sha="$(sha256sum "$run_dir/config.json" | awk '{print $1}')"
profile_sha="$(sha256sum "$run_dir/$PROFILE.json" | awk '{print $1}')"
params_sha="$(sha256sum "$run_dir/macarons_params.json" | awk '{print $1}')"
settings_sha="$(sha256sum "$repo_root/data/Macarons++/HKUST/settings.json" | awk '{print $1}')"
occupied_sha="$(sha256sum "$repo_root/data/Macarons++/HKUST/occupied_pose.pt" | awk '{print $1}')"
weight_sha="$(sha256sum "$repo_root/weights/macarons/trained_macarons.pth" | awk '{print $1}')"
spatial_policy_sha="$(sha256sum "$run_dir/hkust_spatial_policy.json" | awk '{print $1}')"
expected_spatial_policy_sha="$(
  "$python_bin" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["validation_position_policy"]["source_policy_sha256"])' \
    "$run_dir/$PROFILE.json"
)"
if [[ "$spatial_policy_sha" != "$expected_spatial_policy_sha" ]]; then
  printf 'spatial policy hash does not match debug profile\n' >&2
  exit 2
fi
initial_ue_manifest_sha=none
if [[ -n "$initial_ue_manifest" ]]; then
  initial_ue_manifest_sha="$(sha256sum "$initial_ue_manifest" | awk '{print $1}')"
fi

{
  printf 'schema_version=pan21.planner-run.v1\n'
  printf 'git_commit=%s\n' "$commit_sha"
  printf 'base_config=%s\n' "$base_config_name"
  printf 'base_config_sha256=%s\n' "$base_sha"
  printf 'derived_config_sha256=%s\n' "$config_sha"
  printf 'debug_profile=%s\n' "$PROFILE"
  printf 'debug_profile_sha256=%s\n' "$profile_sha"
  printf 'macarons_params_sha256=%s\n' "$params_sha"
  printf 'settings_sha256=%s\n' "$settings_sha"
  printf 'occupied_pose_sha256=%s\n' "$occupied_sha"
  printf 'planner_weight_sha256=%s\n' "$weight_sha"
  printf 'planner_to_ue_cm=%s\n' '[planner_x*500,planner_z*500,planner_y*500]'
  printf 'validation_position_index_bounds=%s\n' '[0,3,0]..[11,3,9]'
  if [[ -n "$initial_ue_manifest" ]]; then
    printf 'initial_observation_provider=UE5\n'
  else
    printf 'initial_observation_provider=original\n'
  fi
  printf 'initial_ue_manifest=%s\n' "${initial_ue_manifest:-none}"
  printf 'initial_ue_manifest_sha256=%s\n' "$initial_ue_manifest_sha"
  printf 'expected_observations=2\n'
  printf 'expected_real_face_renders=12\n'
  printf 'validation_start_position_override=5,3,1,2,0\n'
  printf 'isolated_dataset_overlay=%s\n' "$run_dir/data/Macarons++"
  printf 'isolated_results_root=%s\n' "$run_dir/results"
  printf 'debug_only=true\ncoverage_comparable=false\n'
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'command=%q ' "$python_bin" "$repo_root/test_magician_planning.py" \
    -c "$run_dir/config.json" --debug-profile "$PROFILE" \
    --debug-profiles-dir "$run_dir" \
    --macarons-params-path "$run_dir/macarons_params.json"
  printf '\n'
} > "$run_dir/manifest.txt"

printf -v tmux_command '%q ' "$repo_root/scripts/run_pan21_planner_smoke.sh" \
  --inside-tmux "$gpu" "$run_dir" "$python_bin" "$repo_root"
tmux new-session -d -s "$session" -c "$repo_root" "$tmux_command"
printf 'started tmux session %s; artifacts: %s\n' "$session" "$run_dir"
