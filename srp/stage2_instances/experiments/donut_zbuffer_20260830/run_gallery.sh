#!/bin/bash
# 批次產 367 場 hull gallery:report.html + reproj_instances.png + index.html
set -e
cd /home/cho/webots_program
ROOT=srp_hull_semcluster_surf_am1photo
PY=/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
export SAM_ROOT=$PWD/data/eval/mobilesamv2_fast
export CAPTURES_ROOT=$PWD/data/captures_fast

echo "=== [1/3] gen_hull_report 全部 ==="
$PY srp/stage2_instances/gen_hull_report.py --root $ROOT 2>&1 | grep -viE "Warning|QuickGELU" | tail -3

echo "=== [2/3] reproj_instances 每場(跳過已存在) ==="
n=0
for d in data/eval/$ROOT/*_scene*/instances.npz; do
  sc=$(basename $(dirname "$d"))
  if [ -f "data/eval/$ROOT/$sc/reproj_instances.png" ]; then continue; fi
  $PY srp/stage2_instances/reproj_instances.py "$sc" --inst-root $ROOT >/dev/null 2>&1 || echo "  [fail] $sc"
  n=$((n+1))
  if [ $((n % 30)) -eq 0 ]; then echo "  reproj $n ..."; fi
done
echo "  reproj 新產 $n 場"

echo "=== [3/3] gen_hull_index ==="
$PY srp/stage2_instances/gen_hull_index.py --root $ROOT 2>&1 | grep -viE "Warning|QuickGELU" | tail -2
echo "ALL DONE"
