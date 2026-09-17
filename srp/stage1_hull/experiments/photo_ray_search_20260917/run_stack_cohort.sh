#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
PYTHON=${PYTHON:-/home/cho/.pyenv/versions/webots_visual_hull/bin/python3}
RUNNER="$REPO/srp/stage1_hull/experiments/photo_ray_search_20260917/run_ray_search.py"

mapfile -t SCENES < <(
  find "$REPO/data/eval/srp_hull_mv2_v12_am1" -mindepth 2 -maxdepth 2 -name hull.npz -printf '%h\n' \
    | sed 's#.*/##' | rg '^stack[345]_scene' | sort
)

export CAPTURES_ROOT="$REPO/data/captures_fast"
export SAM_ROOT="$REPO/data/eval/mobilesamv2_fast"
"$PYTHON" "$RUNNER" "${SCENES[@]}"
