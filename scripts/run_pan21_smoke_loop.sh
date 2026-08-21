#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  printf 'usage: %s P0_MANIFEST P1_MANIFEST PLANNER_METRICS OUTPUT_DIR [DEVICE]\n' "$0" >&2
  exit 2
fi

p0_manifest="$1"
p1_manifest="$2"
planner_metrics="$3"
output_dir="$4"
device="${5:-cuda:0}"
python_bin="${PIONEER_PYTHON:-/home/ubuntu/anaconda3/envs/magician_mve/bin/python}"
repo_root="$(git rev-parse --show-toplevel)"

if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
  printf 'refusing to run from a dirty worktree\n' >&2
  exit 2
fi
for required in "$p0_manifest" "$p1_manifest" "$planner_metrics"; do
  if [[ ! -f "$required" ]]; then
    printf 'missing required input: %s\n' "$required" >&2
    exit 2
  fi
done
if [[ -e "$output_dir" ]]; then
  printf 'refusing to reuse output path: %s\n' "$output_dir" >&2
  exit 2
fi

commit_sha="$(git -C "$repo_root" rev-parse HEAD)"
command=(
  "$python_bin" "$repo_root/scripts/validate_pan21_smoke_loop.py"
  --p0-manifest "$p0_manifest"
  --p1-manifest "$p1_manifest"
  --planner-metrics "$planner_metrics"
  --output-dir "$output_dir"
  --device "$device"
)

set +e
"${command[@]}" 2>&1 | tee "${output_dir}.runner.log"
exit_code="${PIPESTATUS[0]}"
set -e
if [[ "$exit_code" -ne 0 ]]; then
  printf 'PAN21_SMOKE result=FAIL exit_code=%s log=%s\n' "$exit_code" "${output_dir}.runner.log" >&2
  exit "$exit_code"
fi
mv -- "${output_dir}.runner.log" "$output_dir/run.log"
{
  printf 'schema_version=pan21.acceptance-run.v1\n'
  printf 'result=PASS\n'
  printf 'git_commit=%s\n' "$commit_sha"
  printf 'device=%s\n' "$device"
  printf 'finished_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'command='
  printf '%q ' "${command[@]}"
  printf '\n'
} > "$output_dir/status.txt"
(
  cd "$output_dir"
  find . -type f ! -name SHA256SUMS -print0 |
    sort -z |
    xargs -0 sha256sum > SHA256SUMS
)
printf 'PAN21_SMOKE result=PASS output=%s checksums=%s\n' "$output_dir" "$(sha256sum "$output_dir/SHA256SUMS" | awk '{print $1}')"
