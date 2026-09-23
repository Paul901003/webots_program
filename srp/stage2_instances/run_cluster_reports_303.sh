#!/bin/bash
# run_cluster_reports_303.sh — 產 303 場「兩種遮罩語意分群 HTML 報告」(都去手臂夾爪)。
#
# 母體 303 場 = n3/n4/n5 各61 + occ3/occ4/occ5 各20 + stack3/stack4/stack5 各20;排除 n1 與 b 組(nb/occb/stkb)。
# A(原始mv2): DONUT=0 DROP_ARM=1 → root srp_hull_cluster_mv2_noarm   → gen_cluster_report → cluster_report.html
# B(donut)  : DONUT=1 DROP_ARM=1 → root srp_hull_cluster_donut_noarm → gen_donut_report   → donut_cluster_report.html
# 兩份差別只在 DONUT(挖不挖大遮罩);手臂+夾爪遮罩(≥ARM_DROP_THR=0.5 落 FK 剪影 srp_arm_masks)在分群前丟掉,
# 背景/桌面由 masks.kept_object_masks 排除。報告只讀 instances.json 的 mask_clusters(連通前語意群),
# 與 voxel 投票模式(中心/fp)無關。
#
# 輸入一致性:SAM_ROOT=mobilesamv2_fast、HULL_ROOT=srp_hull_mv2_v12_am1、CAPTURES_ROOT=captures_fast。
# 續跑保護:reassign_soliddrop 與 gen_donut_report 遇既有輸出會跳過(FORCE=1 可強制重做)。
# 用法: ./run_cluster_reports_303.sh   (log 見 $LOG)
set -u
REPO=/home/cho/webots_program
HERE=$REPO/srp/stage2_instances
PY=/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
LOG=${LOG:-$HERE/run_cluster_reports_303.log}

export SAM_ROOT=$REPO/data/eval/mobilesamv2_fast
export HULL_ROOT=$REPO/data/eval/srp_hull_mv2_v12_am1
export CAPTURES_ROOT=$REPO/data/captures_fast

ROOT_A=srp_hull_cluster_mv2_noarm
ROOT_B=srp_hull_cluster_donut_noarm

# 場景清單:9 組,排除 n1 與所有 b 組
SCENES=$(ls -d "$HULL_ROOT"/*_scene*/ 2>/dev/null | xargs -n1 basename \
         | grep -E '^(n3|n4|n5|occ3|occ4|occ5|stack3|stack4|stack5)_scene' | sort)
N=$(echo "$SCENES" | wc -l)

{
  echo "=== 開始 $(date '+%F %T') ==="
  echo "場景數=$N  rootA=$ROOT_A  rootB=$ROOT_B"
  echo "SAM_ROOT=$SAM_ROOT"
  echo "HULL_ROOT=$HULL_ROOT"
  echo "CAPTURES_ROOT=$CAPTURES_ROOT"
} >> "$LOG"

i=0
for sc in $SCENES; do
  i=$((i+1))
  echo "[$i/$N] $sc  $(date '+%T')" >> "$LOG"
  DONUT=0 DROP_ARM=1 OUT_ROOT=$ROOT_A $PY "$HERE/voxel_sem_cluster_reassign_soliddrop.py" "$sc" >> "$LOG" 2>&1
  DONUT=1 DROP_ARM=1 OUT_ROOT=$ROOT_B $PY "$HERE/voxel_sem_cluster_reassign_soliddrop.py" "$sc" >> "$LOG" 2>&1
  $PY "$HERE/gen_cluster_report.py" "$sc" --root $ROOT_A >> "$LOG" 2>&1
  $PY "$HERE/gen_donut_report.py"   "$sc" --root $ROOT_B >> "$LOG" 2>&1
done

{
  echo "=== 完成 $(date '+%F %T') ==="
  echo "A 報告數=$(ls $REPO/data/eval/$ROOT_A/*/cluster_report.html 2>/dev/null | wc -l)"
  echo "B 報告數=$(ls $REPO/data/eval/$ROOT_B/*/donut_cluster_report.html 2>/dev/null | wc -l)"
} >> "$LOG"
