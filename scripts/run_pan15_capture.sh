#!/usr/bin/env bash
set -uo pipefail

scenario="${1:?usage: run_pan15_capture.sh analytic|hkust OUTPUT_DIR}"
output_dir="${2:?usage: run_pan15_capture.sh analytic|hkust OUTPUT_DIR}"
shift 2
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
capture_script="$repository_root/unreal/PAN23/Scripts/capture_six_face_rgbd.py"
capture_config="$repository_root/unreal/PAN23/Config/pan23_capture.json"
adapter_python=/home/ubuntu/anaconda3/envs/magician_mve/bin/python
gpu_index=1

case "$scenario" in
  analytic)
    project="$repository_root/unreal/PAN23/PioneerPAN23.uproject"
    level=/Game/PAN23/Analytic/PAN23_Analytic
    position_actor_label=
    ;;
  hkust)
    project=/home/ubuntu/Projects/Pioneer/experiments/pan-13-hkust/ue_project/HKUSTPan13/HKUSTPan13.uproject
    level=/Game/PAN13_Derived/HKUST_ZUp_QA
    position_actor_label=PAN13_Legal_Start_Camera
    ;;
  *)
    printf 'scenario must be analytic or hkust\n' >&2
    exit 2
    ;;
esac

position_ue_cm=()
if [[ $# -gt 0 ]]; then
  if [[ "${1:-}" != "--position-ue-cm" || $# -ne 4 ]]; then
    printf 'optional arguments must be --position-ue-cm X Y Z\n' >&2
    exit 2
  fi
  if [[ "$scenario" != "hkust" ]]; then
    printf 'explicit UE positions are only supported for hkust\n' >&2
    exit 2
  fi
  position_actor_label=
  position_ue_cm=("$2" "$3" "$4")
fi

if [[ -e "$output_dir" ]]; then
  printf 'refusing to overwrite %s\n' "$output_dir" >&2
  exit 3
fi
output_parent="$(dirname "$output_dir")"
output_name="$(basename "$output_dir")"
mkdir -p "$output_parent"
temporary="$(mktemp -d "$output_parent/.${output_name}.tmp-XXXXXX")"
request_id="pan15-${scenario}-$(date -u +%Y%m%dT%H%M%SZ)"
frame_id="${scenario}-fixed-pose-000000"
request="$temporary/request.json"

request_args=(
  --scenario "$scenario"
  --level "$level"
  --request-id "$request_id"
  --frame-id "$frame_id"
  --output "$request"
)
if [[ -n "$position_actor_label" ]]; then
  request_args+=(--position-actor-label "$position_actor_label")
fi
if (( ${#position_ue_cm[@]} == 3 )); then
  request_args+=(--position-ue-cm "${position_ue_cm[@]}")
fi
python3 "$repository_root/scripts/make_pan15_request.py" "${request_args[@]}"

project_hash_before="$(sha256sum "$project" | cut -d' ' -f1)"
config_path="$(dirname "$project")/Config/DefaultEngine.ini"
config_hash_before="$(sha256sum "$config_path" | cut -d' ' -f1)"
export PAN15_RAW_BUNDLE_DIR="$temporary/raw_bundle"
export PAN15_CAPTURE_CONFIG="$capture_config"
export PAN15_REQUEST_PATH="$request"
export PAN15_PROJECT_COMMIT="$(git -C "$repository_root" rev-parse HEAD)"
export PAN15_GPU_INDEX="$gpu_index"
export OPENCV_IO_ENABLE_OPENEXR=1

/usr/local/bin/unreal-editor-cmd-5.4 "$project" \
  -run=pythonscript \
  -script="$capture_script" \
  -unattended \
  -nop4 \
  -nosplash \
  -vulkan \
  -graphicsadapter="$gpu_index" \
  -RenderOffscreen \
  -AllowCommandletRendering \
  -ResX=256 \
  -ResY=256 \
  -stdout \
  -FullStdOutLogOutput \
  >"$temporary/ue_capture.log" 2>&1
ue_exit=$?
printf '%s\n' "$ue_exit" >"$temporary/ue_exit_code"
if ((ue_exit != 0)); then
  attempts="$output_parent/attempts"
  mkdir -p "$attempts"
  mv "$temporary" "$attempts/${output_name}-ue-failed-$(date -u +%Y%m%dT%H%M%SZ)"
  exit "$ue_exit"
fi

project_hash_after="$(sha256sum "$project" | cut -d' ' -f1)"
config_hash_after="$(sha256sum "$config_path" | cut -d' ' -f1)"
if [[ "$project_hash_before" != "$project_hash_after" || "$config_hash_before" != "$config_hash_after" ]]; then
  printf 'UE capture modified project descriptors\n' >&2
  exit 4
fi

"$adapter_python" "$repository_root/scripts/adapt_pan15_ue_capture.py" \
  --raw-manifest "$temporary/raw_bundle/manifest.json" \
  --output "$temporary/canonical_bundle" \
  >"$temporary/adapter.log" 2>&1
adapter_exit=$?
printf '%s\n' "$adapter_exit" >"$temporary/adapter_exit_code"
if ((adapter_exit != 0)); then
  attempts="$output_parent/attempts"
  mkdir -p "$attempts"
  mv "$temporary" "$attempts/${output_name}-adapter-failed-$(date -u +%Y%m%dT%H%M%SZ)"
  exit "$adapter_exit"
fi

{
  printf 'scenario=%s\n' "$scenario"
  printf 'project=%s\n' "$project"
  printf 'level=%s\n' "$level"
  printf 'gpu_index=%s\n' "$gpu_index"
  printf 'project_sha256=%s\n' "$project_hash_after"
  printf 'config_sha256=%s\n' "$config_hash_after"
  printf 'repository_commit=%s\n' "$PAN15_PROJECT_COMMIT"
  if (( ${#position_ue_cm[@]} == 3 )); then
    printf 'requested_position_ue_cm=%s,%s,%s\n' "${position_ue_cm[@]}"
  fi
} >"$temporary/provenance.txt"
mv "$temporary" "$output_dir"
printf 'PAN15_CAPTURE result=PASS scenario=%s output=%s\n' "$scenario" "$output_dir"
