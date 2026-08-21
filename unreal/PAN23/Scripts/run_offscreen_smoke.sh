#!/usr/bin/env bash
set -uo pipefail

output_dir="${1:?usage: run_offscreen_smoke.sh OUTPUT_DIR}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project="$project_root/PioneerPAN23.uproject"
script="$project_root/Scripts/capture_six_face_rgb.py"
PAN23_GPU_INDEX="${PAN23_GPU_INDEX:-1}"
export PAN23_GPU_INDEX
mkdir -p "$output_dir"
export PAN23_OUTPUT_DIR="$output_dir"
export PAN23_CAPTURE_CONFIG="$project_root/Config/pan23_capture.json"
export PAN23_PROJECT_COMMIT="$(git -C "$project_root" rev-parse HEAD 2>/dev/null || printf unknown)"

/usr/local/bin/unreal-editor-cmd-5.4 "$project" \
  -run=pythonscript \
  -script="$script" \
  -unattended \
  -nop4 \
  -nosplash \
  -vulkan \
  -graphicsadapter="$PAN23_GPU_INDEX" \
  -RenderOffscreen \
  -AllowCommandletRendering \
  -ResX=256 \
  -ResY=256 \
  -stdout \
  -FullStdOutLogOutput \
  >"$output_dir/offscreen_smoke.log" 2>&1
exit_code=$?
printf '%s\n' "$exit_code" >"$output_dir/exit_code"
exit "$exit_code"
