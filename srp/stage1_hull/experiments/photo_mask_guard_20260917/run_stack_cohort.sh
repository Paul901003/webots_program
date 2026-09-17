#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "Usage: $0" >&2
  echo "Runs the fixed stack3/4/5 cohort for center and footprint hulls." >&2
  exit 2
fi

REPO=$(cd "$(dirname "$0")/../../../.." && pwd)
PYTHON=${PYTHON:-/home/cho/.pyenv/versions/webots_visual_hull/bin/python3}
RUNNER="$REPO/srp/stage1_hull/experiments/photo_mask_guard_20260917/run_fgpatch.py"
INPUT_CENTER=srp_hull_mv2_v12_am1
INPUT_FOOTPRINT=srp_hull_mv2_v12_am1_fp

mapfile -t SCENES < <(
  find "$REPO/data/eval/$INPUT_CENTER" -mindepth 2 -maxdepth 2 -name hull.npz -printf '%h\n' \
    | sed 's#.*/##' | rg '^stack[345]_scene' | sort
)

if [ "${#SCENES[@]}" -eq 0 ]; then
  echo "No legacy stack scenes found under $INPUT_CENTER" >&2
  exit 1
fi

export CAPTURES_ROOT="$REPO/data/captures_fast"
export SAM_ROOT="$REPO/data/eval/mobilesamv2_fast"

"$PYTHON" "$RUNNER" --in-root "$INPUT_CENTER" "${SCENES[@]}"
"$PYTHON" "$RUNNER" --in-root "$INPUT_FOOTPRINT" "${SCENES[@]}"
