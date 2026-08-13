#!/usr/bin/env bash

set -o pipefail

mode=${1:-}
project_root=/home/ubuntu/Projects/MAGICIAN_MYL18
case "$mode" in
  da3)
    config=test_huge_1_office_phase4_da3_3dgs.json
    run_id=myl18_huge_1_office_da3_3dgs
    ;;
  perfect)
    config=test_huge_1_office_phase4_perfect_3dgs.json
    run_id=myl18_huge_1_office_perfect_3dgs
    ;;
  *)
    echo "usage: $0 {da3|perfect}" >&2
    exit 2
    ;;
esac

result_root="$project_root/results/$run_id"
log_path="$result_root/run.log"
status_path="$result_root/status.txt"
lmdb_path="$project_root/results/scene_exploration/${run_id}_lmdb"
mkdir -p "$result_root"
cd "$project_root" || exit 1

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/ubuntu/Projects/MAGICIAN_MVE/.venv-myl12-source/Depth-Anything-3-3d835ec1a5802d64a8b8b15f817a1ab54809bfe4:/home/ubuntu/Projects/MAGICIAN_MVE/.venv-myl12-realmesh:/home/ubuntu/Projects/MAGICIAN_MVE/.venv-myl12-deps
export HF_HOME=/home/ubuntu/Projects/MAGICIAN_MVE/.venv-myl12-hf
export CUDA_HOME=/usr/local/cuda-12.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

{
  date --iso-8601=seconds
  git rev-parse HEAD
  git status --short --branch
  nvidia-smi
  python3 -c 'import sys, torch, pytorch3d, diff_gaussian_rasterization; print(sys.version); print(torch.__version__, torch.version.cuda, pytorch3d.__version__)'
} > "$result_root/environment.log" 2>&1

echo "running started_at=$(date --iso-8601=seconds)" > "$status_path"
python3 -u test_magician_planning.py -c "$config" 2>&1 | tee "$log_path"
run_status=${PIPESTATUS[0]}

if [ "$run_status" -eq 0 ]; then
  python3 scripts/plot_huge_office_start3.py \
    --lmdb "$lmdb_path" \
    --output "$result_root/start3_trajectory.png" \
    --label "$mode" 2>&1 | tee -a "$log_path"
  plot_status=${PIPESTATUS[0]}
  if [ "$plot_status" -eq 0 ]; then
    echo "completed exit_code=0 finished_at=$(date --iso-8601=seconds)" > "$status_path"
  else
    echo "failed exit_code=$plot_status stage=start3_plot finished_at=$(date --iso-8601=seconds)" > "$status_path"
    exit "$plot_status"
  fi
else
  echo "failed exit_code=$run_status stage=experiment finished_at=$(date --iso-8601=seconds)" > "$status_path"
fi
exit "$run_status"
