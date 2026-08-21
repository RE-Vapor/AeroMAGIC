#!/usr/bin/env bash
set -uo pipefail

output_dir="${1:?usage: run_build_analytic_level.sh OUTPUT_DIR}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
project="$project_root/PioneerPAN23.uproject"
script="$project_root/Scripts/build_analytic_level.py"
mkdir -p "$output_dir"
export PAN23_OUTPUT_DIR="$output_dir"
export PAN23_CAPTURE_CONFIG="$project_root/Config/pan23_capture.json"
export PAN23_SCENE_CONFIG="$project_root/Config/pan23_analytic_scene.json"

/usr/local/bin/unreal-editor-cmd-5.4 "$project" \
  -run=pythonscript \
  -script="$script" \
  -unattended \
  -nop4 \
  -nosplash \
  -NullRHI \
  -stdout \
  -FullStdOutLogOutput \
  >"$output_dir/build_analytic_level.log" 2>&1
exit_code=$?
printf '%s\n' "$exit_code" >"$output_dir/exit_code"
exit "$exit_code"
