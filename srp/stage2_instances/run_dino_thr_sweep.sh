#!/bin/bash
# run_dino_thr_sweep.sh — DINOv2 的 sem_thr 掃描:0.20/0.25/0.30/0.35/0.40,各跑到 divB 並評估。
#
# 動機:baseline 的 sem_thr=0.4 是為 CLIP 調的;實測(20場 stack 遮罩兩兩 cosine 距離)
#   同樣 thr=0.4 之下 CLIP 併 20.2% 的配對、DINO 併 38.6%(1.9 倍)→ 對 DINO 明顯偏寬,比較不公平。
#   要等化合併比例,DINO 門檻約需降到 0.22~0.25。本掃描實測下游 IoU,排除這個變因。
#
# 步驟① sweep_sem_thr.py 一次產 5 個 semcluster root(重用與門檻無關的計算;已驗證 thr=0.4 逐 voxel 重現
#        既有 srp_hull_semcluster_reNNcSd_am1_dino,4 場 0 差異)
#      ② 每個 root 跑 div(B_span_guard θ=0.5 span_thr=0.6)
#      ③ 每個 divB root 跑 reproj_iou(zbuffer,排 GEX,vs GT modal 遮罩)
# 母體 303 場;其餘參數與 baseline 逐項相同(vote=center, donut=1, nest_thr=0.8, min_vox=50,
#   drop_arm=1, arm_drop_thr=0.5, n_views=12, hull=srp_hull_mv2_v12_am1)。
# 用法: ./run_dino_thr_sweep.sh   (log 見 $LOG)
set -u
REPO=/home/cho/webots_program
HERE=$REPO/srp/stage2_instances
PY=/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
LOG=${LOG:-$HERE/run_dino_thr_sweep.log}
PREFIX=srp_hull_semcluster_reNNcSd_am1_dino_sw
THRS=${THRS:-0.20,0.25,0.30,0.35,0.40}          # 可由 env 指定要掃的門檻
TAGS=$(echo "$THRS" | tr ',' '\n' | awk '{printf "%02d ", $1*100}')   # 0.45 -> 45

export SAM_ROOT=$REPO/data/eval/mobilesamv2_fast
export HULL_ROOT=$REPO/data/eval/srp_hull_mv2_v12_am1
export CAPTURES_ROOT=$REPO/data/captures_fast

{ echo "=== 開始 $(date '+%F %T') ==="; } >> "$LOG"

echo "--- 步驟1 sweep semcluster(5 門檻)$(date '+%T') ---" >> "$LOG"
FEAT=dino DEBIAS=0 THRS=$THRS OUT_PREFIX=$PREFIX \
  $PY "$HERE/sweep_sem_thr.py" --verify-root srp_hull_semcluster_reNNcSd_am1_dino --verify-thr 0.40 \
  >> "$LOG" 2>&1

for T in $TAGS; do
  SEM=${PREFIX}_t${T}
  echo "--- 步驟2 div t=0.${T} $(date '+%T')  src=$SEM ---" >> "$LOG"
  BASE_ROOT=$SEM THETAS=0.5 OUT_SUFFIX=_reNNcSd_am1_dino_t${T} \
    $PY "$HERE/merge_div_guard.py" >> "$LOG" 2>&1
  DIV=srp_hull_divB_t50_reNNcSd_am1_dino_t${T}
  echo "  divB($DIV) 場數=$(ls $REPO/data/eval/$DIV/*/instances.npz 2>/dev/null | wc -l)" >> "$LOG"
  echo "--- 步驟3 reproj_iou t=0.${T} $(date '+%T') ---" >> "$LOG"
  $PY "$HERE/reproj_iou.py" --inst-root $DIV --hull-root srp_hull_mv2_v12_am1 >> "$LOG" 2>&1
done

echo "=== 完成 $(date '+%F %T') ===" >> "$LOG"
