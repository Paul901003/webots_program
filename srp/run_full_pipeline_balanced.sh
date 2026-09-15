#!/bin/bash
# run_full_pipeline_balanced.sh — 平衡資料集(nb/occb/stkb)整條 GT+srp 管線,逐子組串跑。
# 每步用正確直譯器 + skip-guard;可續跑(重下即跳過已完成)。逐組把 recall/precision 寫 summary。
# 用法: setsid nohup srp/run_full_pipeline_balanced.sh <子組...> > log 2>&1 < /dev/null &
set -u
cd ~/webots_program
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc
PYR=/home/cho/.pyenv/versions/3.10.10/bin/python3          # pyrender(amodal)
PYG=/home/cho/.pyenv/versions/grounded_sam/bin/python3      # MobileSAM
PYW=/home/cho/.pyenv/versions/webots_visual_hull/bin/python3 # srp 全鏈
CAPF=$PWD/data/captures_fast
MV2=$PWD/data/eval/mobilesamv2_fast
HULLR=srp_hull_mv2_v12_am1; HULL=$PWD/data/eval/$HULLR
SEMR=srp_hull_semcluster_clip
SUMMARY=$PWD/cap_pipeline_summary.txt
SUBS="${*:-nb4 nb5 nb6 occb3 occb4 occb5 occb6 stkb3 stkb4 stkb5 stkb6}"

alabel(){ local sc="$1" h="${sc%%_scene*}" c="${h%%[0-9]*}" n="${h#$c}"; echo "$PWD/data/labels/$c/$n/$sc"; }

for g in $SUBS; do
  echo "########## $g START $(date +%H:%M:%S) ##########"
  scenes=$(for d in data/captures_fast/multi_$g/${g}_scene*/; do basename "$d"; done)
  [ -z "$scenes" ] && { echo "[$g] 無場景,跳過"; continue; }

  echo "--- [$g] 1/10 modal GT ---"
  CAPTURES_ROOT=$CAPF LABELS=$PWD/data/labels MODE=actual JOBS=6 \
    bash tools/run_generate_labels_parallel.sh "$g" 2>&1 | grep -E "待生成|全部結束"

  echo "--- [$g] 2/10 amodal GT(補缺)---"
  miss=""; for s in $scenes; do [ -f "$(alabel "$s")/amodal/annotations.json" ] || miss="$miss $s"; done
  if [ -n "$miss" ]; then CAPTURES_ROOT=$CAPF $PYR tools/generate_amodal_masks.py $miss 2>&1 | tail -1; else echo "全有"; fi

  echo "--- [$g] 3/10 relations ---"
  CAPTURES_ROOT=$CAPF $PYW srp/stage3_graph/gt_relations.py $scenes 2>&1 | grep "場景 |"

  echo "--- [$g] 4/10 MobileSAM 遮罩 ---"
  CAPTURES_ROOT=$CAPF SAM_OUT_ROOT=$MV2 $PYG mobilesamv2/mobilesamv2_seg.py "$g" 2>&1 | tail -1

  echo "--- [$g] 5/10 Stage1 hull ---"
  SAM_ROOT=$MV2 CAPTURES_ROOT=$CAPF $PYW srp/stage1_hull/run_scene.py $scenes \
    --num-views 12 --allow-miss 1 --root $HULLR 2>&1 | tail -1

  echo "--- [$g] 6/10 clip feats ---"
  SAM_ROOT=$MV2 CAPTURES_ROOT=$CAPF HULL_ROOT=$HULL \
    $PYW srp/stage2_instances/precompute_clip_mean.py "$g" --n-views 12 2>&1 | tail -1

  echo "--- [$g] 7/10 gt_reproj ---"
  $PYW srp/stage2_instances/build_gt.py "$g" --hull-root $HULLR 2>&1 | tail -1

  echo "--- [$g] 8/10 scene_graph ---"
  HULL_ROOT=$HULL CAPTURES_ROOT=$CAPF $PYW srp/stage3_graph/scene_graph_gt.py "$g" 2>&1 | tail -1

  echo "--- [$g] 9/10 Stage2 semcluster ---"
  HULL_ROOT=$HULL SAM_ROOT=$MV2 CAPTURES_ROOT=$CAPF OUT_ROOT=$SEMR \
    $PYW srp/stage2_instances/voxel_sem_cluster.py "$g" --n-views 12 --sem-thr 0.40 2>&1 | tail -1

  echo "--- [$g] 10/10 eval ---"
  $PYW srp/stage2_instances/build_hull_gt.py --root $SEMR "$g" 2>&1 | tail -1
  for th in 0.6 0.7; do
    line=$($PYW srp/stage2_instances/eval_hull_gt.py --root $SEMR --hit-iou $th "$g" 2>&1 | grep -E "全體")
    echo "$g @$th: $line" | tee -a "$SUMMARY"
  done
  echo "########## $g DONE $(date +%H:%M:%S) ##########"
done
echo "===== ALL SUBGROUPS DONE $(date +%H:%M:%S) ====="
