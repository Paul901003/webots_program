#!/bin/bash
# 合併後 root 的 hull gallery:report.html + reproj_instances.png + index.html(跳過已存,可續跑)
cd /home/cho/webots_program
ROOT=srp_hull_semcluster_surf_am1photo_merged
PY=/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
export SAM_ROOT=$PWD/data/eval/mobilesamv2_fast
export CAPTURES_ROOT=$PWD/data/captures_fast

echo "=== [1/3] gen_hull_report(一次全部) ==="
$PY srp/stage2_instances/gen_hull_report.py --root $ROOT 2>&1 | grep -viE "Warning|QuickGELU" | tail -2
echo "  report: $(ls data/eval/$ROOT/*_scene*/report.html 2>/dev/null | wc -l)"

echo "=== [2/3] reproj_instances(跳過已存) ==="
for d in data/eval/$ROOT/*_scene*/instances.npz; do
  sc=$(basename $(dirname "$d"))
  [ -f "data/eval/$ROOT/$sc/reproj_instances.png" ] && continue
  $PY srp/stage2_instances/reproj_instances.py "$sc" --inst-root $ROOT >/dev/null 2>&1 || echo "  [rej fail] $sc"
done
echo "  reproj: $(ls data/eval/$ROOT/*_scene*/reproj_instances.png 2>/dev/null | wc -l)"

echo "=== [3/3] index ==="
$PY srp/stage2_instances/gen_hull_index.py --root $ROOT 2>&1 | grep -viE "Warning|QuickGELU" | tail -2
echo "ALL DONE"
