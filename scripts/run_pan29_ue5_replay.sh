#!/usr/bin/env bash
set -euo pipefail

DEFAULT_PYTHON=/home/ubuntu/anaconda3/envs/magician_mve/bin/python
UE_COMMAND=/usr/local/bin/unreal-editor-cmd-5.4
SOURCE_RUN=/home/ubuntu/Projects/Pioneer/experiments/runs/integration-all-experiments/PAN-11-pioneer-hkust-position-only-20obs/attempt-001
SOURCE_METRICS="$SOURCE_RUN/metrics_debug_pioneer20/pioneer_HKUST_0.online.json"
SOURCE_CAPTURE=/home/ubuntu/Projects/Pioneer/experiments/inputs/shared/data/Macarons++/HKUST/pan11_pioneer_hkust_position_only_20obs_a1_memory_debug_pioneer20/training/0
SOURCE_LMDB=/home/ubuntu/Projects/Pioneer/experiments/runs/integration-all-experiments/results/scene_exploration/pan11_pioneer_hkust_position_only_20obs_a1_lmdb_debug_pioneer20
SOURCE_PREVIEW="$SOURCE_RUN/hkust_position_only_20obs_preview.png"
SOURCE_PREVIEW_SIDECAR="${SOURCE_PREVIEW%.png}.json"
SOURCE_REGISTRY_ID=PAN-11-PIONEER-HKUST-POSITION-ONLY-20OBS-20260821
SOURCE_CONFIG=/home/ubuntu/Projects/Pioneer/repository/configs/test/test_pioneer_hkust_pan11_position_only_20obs_a1_config.json
SPATIAL_POLICY=/home/ubuntu/Projects/Pioneer/experiments/pan-13-hkust/ue_derived_inspection/hkust_spatial_policy.json
UE_INSPECTION=/home/ubuntu/Projects/Pioneer/experiments/pan-13-hkust/ue_derived_inspection/ue_derived_inspection.json
UE_PROJECT=/home/ubuntu/Projects/Pioneer/experiments/pan-13-hkust/ue_project/HKUSTPan13/HKUSTPan13.uproject
UE_DEFAULT_ENGINE=/home/ubuntu/Projects/Pioneer/experiments/pan-13-hkust/ue_project/HKUSTPan13/Config/DefaultEngine.ini
UE_LEVEL=/home/ubuntu/Projects/Pioneer/experiments/pan-13-hkust/ue_project/HKUSTPan13/Content/PAN13_Derived/HKUST_ZUp_QA.umap
OBSERVATION_IDS=(0 5 10 14 19)
OBSERVATION_IDS_CSV=0,5,10,14,19
REPLAY_TASK=PAN-29
REPLAY_REQUEST_PREFIX=pan29-hkust-gt20obs
PREVIEW_FILENAME=hkust_gt20obs_pan29_ue5_replay_preview.png

load_replay_spec() {
  local spec="$1" python_bin="$2"
  local values
  mapfile -t values < <("$python_bin" - "$spec" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p.get("schema_version")=="pioneer.ue5-postrun-replay-source.v1"
ids=p.get("observation_ids")
assert isinstance(ids,list) and len(ids)==5 and ids==sorted(set(ids)) and ids[0]>=0
for key in (
 "task","source_run","source_metrics","source_capture","source_lmdb",
 "source_preview","source_registry_id","source_config","replay_request_prefix",
 "preview_filename",
):
 value=p.get(key); assert isinstance(value,str) and value
 print(value)
print(",".join(str(x) for x in ids))
PY
  )
  [[ "${#values[@]}" -eq 11 ]] || return 2
  REPLAY_TASK="${values[0]}"
  SOURCE_RUN="${values[1]}"
  SOURCE_METRICS="${values[2]}"
  SOURCE_CAPTURE="${values[3]}"
  SOURCE_LMDB="${values[4]}"
  SOURCE_PREVIEW="${values[5]}"
  SOURCE_REGISTRY_ID="${values[6]}"
  SOURCE_CONFIG="${values[7]}"
  REPLAY_REQUEST_PREFIX="${values[8]}"
  PREVIEW_FILENAME="${values[9]}"
  OBSERVATION_IDS_CSV="${values[10]}"
  IFS=',' read -r -a OBSERVATION_IDS <<< "$OBSERVATION_IDS_CSV"
  SOURCE_PREVIEW_SIDECAR="${SOURCE_PREVIEW%.png}.json"
}

manifest_value() {
  local key="$1" manifest="$2"
  sed -n "s/^${key}=//p" "$manifest"
}

verify_snapshot() {
  local run_dir="$1" repo_root="$2" python_bin="$3"
  local manifest="$run_dir/run_manifest.txt"
  [[ "$(git -C "$repo_root" rev-parse HEAD)" == "$(manifest_value git_commit "$manifest")" ]] || return 2
  [[ -z "$(git -C "$repo_root" status --porcelain)" ]] || return 2
  [[ "$(sha256sum "$run_dir/capture_config.json" | awk '{print $1}')" == "$(manifest_value capture_config_sha256 "$manifest")" ]] || return 2
  [[ "$(sha256sum "$UE_PROJECT" | awk '{print $1}')" == "$(manifest_value ue_project_sha256 "$manifest")" ]] || return 2
  [[ "$(sha256sum "$UE_DEFAULT_ENGINE" | awk '{print $1}')" == "$(manifest_value ue_default_engine_sha256 "$manifest")" ]] || return 2
  [[ "$(sha256sum "$UE_LEVEL" | awk '{print $1}')" == "$(manifest_value ue_level_sha256 "$manifest")" ]] || return 2
  [[ "$(sha256sum "$UE_INSPECTION" | awk '{print $1}')" == "$(manifest_value ue_inspection_sha256 "$manifest")" ]] || return 2
  [[ "$(sha256sum "$SPATIAL_POLICY" | awk '{print $1}')" == "$(manifest_value spatial_policy_sha256 "$manifest")" ]] || return 2
  [[ "$(sha256sum "$repo_root/unreal/PAN23/Scripts/capture_six_face_rgbd.py" | awk '{print $1}')" == "$(manifest_value capture_script_sha256 "$manifest")" ]] || return 2
  [[ "$(sha256sum "$repo_root/scripts/process_pan29_replay_capture.py" | awk '{print $1}')" == "$(manifest_value process_script_sha256 "$manifest")" ]] || return 2
  [[ "$(sha256sum "$repo_root/scripts/make_pan29_preview.py" | awk '{print $1}')" == "$(manifest_value preview_script_sha256 "$manifest")" ]] || return 2
  [[ "$(sha256sum "$run_dir/replay_plan.json" | awk '{print $1}')" == "$(manifest_value replay_plan_sha256 "$manifest")" ]] || return 2
  if [[ -f "$run_dir/replay_source_spec.json" ]]; then
    [[ "$(sha256sum "$run_dir/replay_source_spec.json" | awk '{print $1}')" == "$(manifest_value replay_source_spec_sha256 "$manifest")" ]] || return 2
  fi
  "$python_bin" - "$run_dir/replay_plan.json" "$OBSERVATION_IDS_CSV" <<'PY'
import hashlib,json,sys
from pathlib import Path
plan=json.load(open(sys.argv[1]))
expected=[int(x) for x in sys.argv[2].split(",")]
assert plan["selected_bundle_ids"] == expected
for key in ("config","metrics"):
    record=plan["source"][key]; path=Path(record["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record["sha256"]
lmdb=Path(plan["source"]["lmdb"]["path"])/"data.mdb"
assert hashlib.sha256(lmdb.read_bytes()).hexdigest()==plan["source"]["lmdb"]["data_sha256"]
preview=Path(plan["source"]["original_preview"]["path"])
assert hashlib.sha256(preview.read_bytes()).hexdigest()==plan["source"]["original_preview"]["sha256"]
PY
}

run_inside_tmux() {
  local gpu="$1" run_dir="$2" python_bin="$3" repo_root="$4"
  local status="$run_dir/status.txt"
  if [[ -f "$run_dir/replay_source_spec.json" ]]; then
    load_replay_spec "$run_dir/replay_source_spec.json" "$python_bin"
  fi
  if verify_snapshot "$run_dir" "$repo_root" "$python_bin"; then
    printf 'snapshot_integrity_preflight=PASS\n' >> "$status"
  else
    printf 'snapshot_integrity_preflight=FAIL\nfinished_at_utc=%s\nexit_code=2\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$status"
    return 2
  fi

  run_science() {
    local captures="$run_dir/captures" processed="$run_dir/processed"
    mkdir -p "$captures" "$processed" "$run_dir/attempts"
    for observation_id in "${OBSERVATION_IDS[@]}"; do
      local formatted temporary final_capture final_processed
      printf -v formatted '%06d' "$observation_id"
      final_capture="$captures/$formatted"
      final_processed="$processed/$formatted"
      [[ ! -e "$final_capture" && ! -e "$final_processed" ]] || {
        printf 'refusing to reuse observation output %s\n' "$formatted" >&2
        return 3
      }
      temporary="$(mktemp -d "$captures/.${formatted}.tmp-XXXXXX")"
      "$python_bin" "$repo_root/scripts/make_pan29_capture_request.py" \
        --plan "$run_dir/replay_plan.json" \
        --observation-id "$observation_id" \
        --output "$temporary/request.json" \
        > "$temporary/request.log" 2>&1

      set +e
      PAN15_RAW_BUNDLE_DIR="$temporary/raw_bundle" \
      PAN15_CAPTURE_CONFIG="$run_dir/capture_config.json" \
      PAN15_REQUEST_PATH="$temporary/request.json" \
      PAN15_PROJECT_COMMIT="$(git -C "$repo_root" rev-parse HEAD)" \
      PAN15_GPU_INDEX="$gpu" \
      OPENCV_IO_ENABLE_OPENEXR=1 \
      "$UE_COMMAND" "$UE_PROJECT" \
        -run=pythonscript \
        -script="$repo_root/unreal/PAN23/Scripts/capture_six_face_rgbd.py" \
        -unattended -nop4 -nosplash -vulkan -graphicsadapter="$gpu" \
        -RenderOffscreen -AllowCommandletRendering -ResX=256 -ResY=256 \
        -stdout -FullStdOutLogOutput \
        > "$temporary/ue_capture.log" 2>&1
      local ue_exit=$?
      set -e
      printf '%s\n' "$ue_exit" > "$temporary/ue_exit_code"
      if (( ue_exit != 0 )) || [[ ! -f "$temporary/raw_bundle/manifest.json" ]]; then
        mv "$temporary" "$run_dir/attempts/${formatted}-ue-failed-$(date -u +%Y%m%dT%H%M%SZ)"
        if (( ue_exit != 0 )); then
          return "$ue_exit"
        fi
        return 4
      fi

      set +e
      OPENCV_IO_ENABLE_OPENEXR=1 "$python_bin" \
        "$repo_root/scripts/adapt_pan15_ue_capture.py" \
        --raw-manifest "$temporary/raw_bundle/manifest.json" \
        --output "$temporary/canonical_bundle" \
        > "$temporary/adapter.log" 2>&1
      local adapter_exit=$?
      set -e
      printf '%s\n' "$adapter_exit" > "$temporary/adapter_exit_code"
      if (( adapter_exit != 0 )); then
        mv "$temporary" "$run_dir/attempts/${formatted}-adapter-failed-$(date -u +%Y%m%dT%H%M%SZ)"
        return "$adapter_exit"
      fi
      mv "$temporary" "$final_capture"

      set +e
      "$python_bin" -m scripts.process_pan29_replay_capture \
        --plan "$run_dir/replay_plan.json" \
        --observation-id "$observation_id" \
        --request "$final_capture/request.json" \
        --raw-manifest "$final_capture/raw_bundle/manifest.json" \
        --canonical-manifest "$final_capture/canonical_bundle/manifest.json" \
        --output "$final_processed" \
        --erp-height 512 \
        > "$final_capture/process.log" 2>&1
      local process_exit=$?
      set -e
      printf '%s\n' "$process_exit" > "$final_capture/process_exit_code"
      if (( process_exit != 0 )) || [[ ! -f "$final_processed/receipt.json" ]]; then
        local failed_process="$run_dir/attempts/${formatted}-process-failed-$(date -u +%Y%m%dT%H%M%SZ)"
        mkdir -p "$failed_process"
        mv "$final_capture" "$failed_process/capture"
        if [[ -e "$final_processed" ]]; then
          mv "$final_processed" "$failed_process/processed"
        fi
        if (( process_exit != 0 )); then
          return "$process_exit"
        fi
        return 5
      fi
      printf 'observation_id=%s result=PASS capture=%s processed=%s\n' \
        "$observation_id" "$final_capture" "$final_processed"
    done

    "$python_bin" "$repo_root/scripts/finalize_pan29_replay.py" \
      --plan "$run_dir/replay_plan.json" \
      --processed-root "$run_dir/processed" \
      --run-manifest "$run_dir/run_manifest.txt" \
      --output "$run_dir/replay_result.pending.json"
  }

  set +e
  ( set -euo pipefail; run_science ) 2>&1 | tee "$run_dir/run.log"
  local science_exit="${PIPESTATUS[0]}"
  set -e
  local postflight=PASS final_exit="$science_exit"
  if ! verify_snapshot "$run_dir" "$repo_root" "$python_bin"; then
    postflight=FAIL
    final_exit=2
  fi
  printf 'snapshot_integrity_postflight=%s\n' "$postflight" >> "$status"
  if (( final_exit == 0 )); then
    if [[ ! -f "$run_dir/replay_result.pending.json" ]]; then
      final_exit=6
    else
      mv "$run_dir/replay_result.pending.json" "$run_dir/replay_result.json"
      set +e
      "$python_bin" "$repo_root/scripts/make_pan29_preview.py" \
        --metrics "$SOURCE_METRICS" \
        --replay-result "$run_dir/replay_result.json" \
        --output "$run_dir/preview/$PREVIEW_FILENAME" \
        > "$run_dir/preview.log" 2>&1
      local preview_exit=$?
      set -e
      if (( preview_exit == 0 )); then
        set +e
        "$python_bin" - "$run_dir/preview/$PREVIEW_FILENAME" "$OBSERVATION_IDS_CSV" <<'PY'
import hashlib,json,sys
from pathlib import Path
png=Path(sys.argv[1]).resolve(); sidecar=png.with_suffix(".json"); commit=png.with_suffix(".commit.json")
expected=[int(x) for x in sys.argv[2].split(",")]
record=json.load(open(commit))
assert hashlib.sha256(png.read_bytes()).hexdigest()==record["preview_sha256"]
assert hashlib.sha256(sidecar.read_bytes()).hexdigest()==record["sidecar_sha256"]
payload=json.load(open(sidecar))
assert payload["summary"]["displayed_observation_ids"]==expected
assert payload["summary"]["planner_input_unchanged"] is True
PY
        preview_exit=$?
        set -e
      fi
      if (( preview_exit != 0 )); then
        final_exit="$preview_exit"
        local failed_preview="$run_dir/attempts/preview-failed-$(date -u +%Y%m%dT%H%M%SZ)"
        mkdir -p "$failed_preview"
        mv "$run_dir/replay_result.json" "$failed_preview/replay_result.json"
        [[ ! -e "$run_dir/preview" ]] || mv "$run_dir/preview" "$failed_preview/preview"
        [[ ! -e "$run_dir/preview.log" ]] || mv "$run_dir/preview.log" "$failed_preview/preview.log"
      fi
    fi
  elif [[ -f "$run_dir/replay_result.pending.json" ]]; then
    mv "$run_dir/replay_result.pending.json" \
      "$run_dir/attempts/replay_result-postflight-failed-$(date -u +%Y%m%dT%H%M%SZ).json"
  fi
  printf 'finished_at_utc=%s\nexit_code=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$final_exit" >> "$status"
  if (( final_exit == 0 )); then
    (
      cd "$run_dir"
      find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
    )
  fi
  return "$final_exit"
}

if [[ "${1:-}" == "--inside-tmux" ]]; then
  [[ $# -eq 5 ]] || { printf 'inside-tmux requires GPU RUN_DIR PYTHON REPO_ROOT\n' >&2; exit 2; }
  run_inside_tmux "$2" "$3" "$4" "$5"
  exit $?
fi

if [[ $# -lt 3 || $# -gt 4 ]]; then
  printf 'usage: %s SESSION GPU RUN_DIR [REPLAY_SOURCE_SPEC]\n' "$0" >&2
  exit 2
fi
session="$1"
gpu="$2"
run_dir="$3"
source_spec="${4:-}"
python_bin="${PIONEER_PYTHON:-$DEFAULT_PYTHON}"
repo_root="$(git rev-parse --show-toplevel)"

if [[ -n "$source_spec" ]]; then
  source_spec="$(cd "$(dirname "$source_spec")" && pwd)/$(basename "$source_spec")"
  [[ -f "$source_spec" ]] || { printf 'missing replay source spec: %s\n' "$source_spec" >&2; exit 2; }
  load_replay_spec "$source_spec" "$python_bin"
fi

[[ "$session" =~ ^[A-Za-z0-9._-]+$ ]] || { printf 'invalid tmux session name\n' >&2; exit 2; }
[[ "$gpu" =~ ^[0-9]+$ ]] || { printf 'GPU must be a non-negative integer\n' >&2; exit 2; }
[[ -z "$(git -C "$repo_root" status --porcelain)" ]] || { printf 'refusing to run from a dirty worktree\n' >&2; exit 2; }
tmux has-session -t "$session" 2>/dev/null && { printf 'tmux session already exists: %s\n' "$session" >&2; exit 2; }
[[ ! -e "$run_dir" ]] || { printf 'refusing to reuse run path: %s\n' "$run_dir" >&2; exit 2; }

source_config="$SOURCE_CONFIG"
source_manifest="$SOURCE_RUN/manifest.txt"
capture_config_source="$repo_root/unreal/PAN29/Config/pan29_hkust_replay.json"
for required in "$python_bin" "$UE_COMMAND" "$source_config" "$source_manifest" "$SOURCE_METRICS" "$SOURCE_CAPTURE/frames" "$SOURCE_LMDB/data.mdb" "$SOURCE_PREVIEW" "$SOURCE_PREVIEW_SIDECAR" "$SPATIAL_POLICY" "$UE_INSPECTION" "$UE_PROJECT" "$UE_DEFAULT_ENGINE" "$UE_LEVEL" "$capture_config_source"; do
  [[ -e "$required" ]] || { printf 'missing replay input: %s\n' "$required" >&2; exit 2; }
done

gpu_used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
[[ "$gpu_used_mib" =~ ^[0-9]+$ ]] || { printf 'could not read GPU memory use\n' >&2; exit 2; }
(( gpu_used_mib < 2048 )) || { printf 'GPU %s is already using %s MiB; refusing replay launch\n' "$gpu" "$gpu_used_mib" >&2; exit 2; }

mkdir -p "$run_dir"
run_dir="$(cd "$run_dir" && pwd)"
if [[ -n "$source_spec" ]]; then
  cp -p -- "$source_spec" "$run_dir/replay_source_spec.json"
  load_replay_spec "$run_dir/replay_source_spec.json" "$python_bin"
fi
"$python_bin" - "$capture_config_source" "$run_dir/capture_config.json" "$gpu" <<'PY'
import json,sys
source,output,gpu=sys.argv[1],sys.argv[2],int(sys.argv[3])
payload=json.load(open(source))
payload["gpu_index"]=gpu
open(output,"w").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY

"$python_bin" "$repo_root/scripts/make_pan29_replay_manifest.py" \
  --metrics "$SOURCE_METRICS" \
  --source-config "$source_config" \
  --source-run-manifest "$source_manifest" \
  --capture-root "$SOURCE_CAPTURE" \
  --spatial-policy "$SPATIAL_POLICY" \
  --source-lmdb "$SOURCE_LMDB" \
  --original-preview "$SOURCE_PREVIEW" \
  --bundle-ids "$OBSERVATION_IDS_CSV" \
  --source-registry-id "$SOURCE_REGISTRY_ID" \
  --task "$REPLAY_TASK" \
  --replay-request-prefix "$REPLAY_REQUEST_PREFIX" \
  --output "$run_dir/replay_plan.json"

commit_sha="$(git -C "$repo_root" rev-parse HEAD)"
{
  printf 'schema_version=pan29.ue5-postrun-run.v1\n'
  printf 'task=%s\n' "$REPLAY_TASK"
  printf 'git_commit=%s\n' "$commit_sha"
  printf 'branch=%s\n' "$(git -C "$repo_root" branch --show-current)"
  printf 'gpu_index=%s\n' "$gpu"
  printf 'source_scientific_commit=%s\n' "$(manifest_value git_commit "$source_manifest")"
  if [[ -f "$run_dir/replay_source_spec.json" ]]; then
    printf 'replay_source_spec_sha256=%s\n' "$(sha256sum "$run_dir/replay_source_spec.json" | awk '{print $1}')"
  fi
  printf 'source_config_sha256=%s\n' "$(sha256sum "$source_config" | awk '{print $1}')"
  printf 'source_metrics_sha256=%s\n' "$(sha256sum "$SOURCE_METRICS" | awk '{print $1}')"
  printf 'source_lmdb_data_sha256=%s\n' "$(sha256sum "$SOURCE_LMDB/data.mdb" | awk '{print $1}')"
  printf 'source_original_preview_sha256=%s\n' "$(sha256sum "$SOURCE_PREVIEW" | awk '{print $1}')"
  printf 'capture_config_source_sha256=%s\n' "$(sha256sum "$capture_config_source" | awk '{print $1}')"
  printf 'capture_config_sha256=%s\n' "$(sha256sum "$run_dir/capture_config.json" | awk '{print $1}')"
  printf 'capture_script_sha256=%s\n' "$(sha256sum "$repo_root/unreal/PAN23/Scripts/capture_six_face_rgbd.py" | awk '{print $1}')"
  printf 'process_script_sha256=%s\n' "$(sha256sum "$repo_root/scripts/process_pan29_replay_capture.py" | awk '{print $1}')"
  printf 'preview_script_sha256=%s\n' "$(sha256sum "$repo_root/scripts/make_pan29_preview.py" | awk '{print $1}')"
  printf 'replay_plan_sha256=%s\n' "$(sha256sum "$run_dir/replay_plan.json" | awk '{print $1}')"
  printf 'ue_project=%s\nue_project_sha256=%s\n' "$UE_PROJECT" "$(sha256sum "$UE_PROJECT" | awk '{print $1}')"
  printf 'ue_default_engine_sha256=%s\n' "$(sha256sum "$UE_DEFAULT_ENGINE" | awk '{print $1}')"
  printf 'ue_level=%s\nue_level_sha256=%s\n' "$UE_LEVEL" "$(sha256sum "$UE_LEVEL" | awk '{print $1}')"
  printf 'ue_inspection_sha256=%s\n' "$(sha256sum "$UE_INSPECTION" | awk '{print $1}')"
  printf 'spatial_policy_sha256=%s\n' "$(sha256sum "$SPATIAL_POLICY" | awk '{print $1}')"
  printf 'replay_observation_ids=[%s]\n' "$OBSERVATION_IDS_CSV"
  printf 'planner_input_unchanged=true\nue5_role=post_run_visualization_only\n'
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'tmux_session=%s\n' "$session"
} > "$run_dir/run_manifest.txt"
printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$run_dir/status.txt"

printf -v tmux_command '%q ' "$repo_root/scripts/run_pan29_ue5_replay.sh" \
  --inside-tmux "$gpu" "$run_dir" "$python_bin" "$repo_root"
tmux new-session -d -s "$session" -c "$repo_root" "$tmux_command"
printf 'started %s tmux session %s; artifacts: %s\n' "$REPLAY_TASK" "$session" "$run_dir"
