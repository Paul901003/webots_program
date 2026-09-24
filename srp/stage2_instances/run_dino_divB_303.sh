#!/bin/bash
# run_dino_divB_303.sh — 用 DINOv2 特徵重現目前最佳方法,產 srp_hull_divB_t50_reNNcSd_am1_dino 並評估。
#
# 目的:與 baseline srp_hull_divB_t50_reNNcSd_am1(CLIP)做【單一變因】比較 —— 只換 ④ 遮罩語意特徵。
# baseline build_meta(2026-09-14):vote=center, donut=True, nest_thr=0.8, feat=recomputed_clip_donut,
#   debias=True, sem_thr=0.4, min_vox=50, drop_arm=True, arm_drop_thr=0.5, hull=srp_hull_mv2_v12_am1, n_views=12
#   → div: merge_div_guard.py, method=B_span_guard, theta=0.5, span_thr=0.6
# 本腳本除 FEAT=dino / DEBIAS=0 外,其餘參數逐項相同(DEBIAS=0 因 DINO 不摳圖填灰,無 F_BG 偏置可扣)。
#
# 母體 303 場 = n3/n4/n5 各61 + occ3/occ4/occ5 各20 + stack3/stack4/stack5 各20;排除 n1 與 b 組。
# 步驟:① semcluster(DINO) ② div(B,θ=0.5) ③ reproj_iou(2D 重投影 vs GT modal 遮罩,排 GEX)
# 續跑保護:reassign_soliddrop 遇既有 instances.npz 會跳過(FORCE=1 強制重做)。
# 用法: ./run_dino_divB_303.sh   (log 見 $LOG)
set -u
REPO=/home/cho/webots_program
HERE=$REPO/srp/stage2_instances
PY=/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
LOG=${LOG:-$HERE/run_dino_divB_303.log}

export SAM_ROOT=$REPO/data/eval/mobilesamv2_fast
export HULL_ROOT=$REPO/data/eval/srp_hull_mv2_v12_am1
export CAPTURES_ROOT=$REPO/data/captures_fast

SEM=srp_hull_semcluster_reNNcSd_am1_dino
DIV=srp_hull_divB_t50_reNNcSd_am1_dino

SCENES=$(ls -d "$HULL_ROOT"/*_scene*/ 2>/dev/null | xargs -n1 basename \
         | grep -E '^(n3|n4|n5|occ3|occ4|occ5|stack3|stack4|stack5)_scene' | sort)
N=$(echo "$SCENES" | wc -l)

{ echo "=== 開始 $(date '+%F %T') ==="; echo "場景數=$N  sem=$SEM  div=$DIV"; } >> "$LOG"

echo "--- 步驟1 semcluster(FEAT=dino DEBIAS=0 VOTE=center DONUT=1) $(date '+%T') ---" >> "$LOG"
FEAT=dino DEBIAS=0 VOTE=center DONUT=1 DROP_ARM=1 OUT_ROOT=$SEM \
  $PY "$HERE/voxel_sem_cluster_reassign_soliddrop.py" $SCENES >> "$LOG" 2>&1
echo "  semcluster 場數=$(ls $REPO/data/eval/$SEM/*/instances.npz 2>/dev/null | wc -l)" >> "$LOG"

echo "--- 步驟2 div(B_span_guard θ=0.5 span_thr=0.6) $(date '+%T') ---" >> "$LOG"
BASE_ROOT=$SEM THETAS=0.5 OUT_SUFFIX=_reNNcSd_am1_dino \
  $PY "$HERE/merge_div_guard.py" $SCENES >> "$LOG" 2>&1
echo "  divB 場數=$(ls $REPO/data/eval/$DIV/*/instances.npz 2>/dev/null | wc -l)" >> "$LOG"

echo "--- 步驟3 reproj_iou(zbuffer,排 GEX) $(date '+%T') ---" >> "$LOG"
$PY "$HERE/reproj_iou.py" --inst-root $DIV --hull-root srp_hull_mv2_v12_am1 >> "$LOG" 2>&1

echo "=== 完成 $(date '+%F %T') ===" >> "$LOG"
