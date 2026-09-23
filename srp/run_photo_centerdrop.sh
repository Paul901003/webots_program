#!/bin/bash
# run_photo_centerdrop.sh — am1_photo / am1_fp_photo 各跑「中心投票+drop+div」→ leak 評估。
# 一次性 runner(2026-09-16);可續跑(各步已存則 skip)。log 到 photo_centerdrop.log。
set -u
cd /home/cho/webots_program
PYW=/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
MV2=$PWD/data/eval/mobilesamv2_fast
CAPF=$PWD/data/captures_fast
ARM=$PWD/data/eval/srp_arm_masks
GRPS="n3 n4 n5 occ3 occ4 occ5 stack3 stack4 stack5"   # 303 多物,排 n1
S2=srp/stage2_instances

echo "########## START $(date +%H:%M:%S) ##########"

echo "--- 0b. am1_fp_photo 補 surface(am1_photo 已有)---"
$PYW $S2/add_surface_mask.py $GRPS --root srp_hull_mv2_v12_am1_fp_photo 2>&1 | tail -1

echo "--- 1. gtlabel(2 顆 photo hull)---"
$PYW $S2/build_gtlabel.py --hull srp_hull_mv2_v12_am1_photo    --out srp_hull_gtlabel_am1photo   2>&1 | tail -2
$PYW $S2/build_gtlabel.py --hull srp_hull_mv2_v12_am1_fp_photo --out srp_hull_gtlabel_am1fpphoto 2>&1 | tail -2

echo "--- 2. semcluster 中心+drop(2 顆)---"
HULL_ROOT=$PWD/data/eval/srp_hull_mv2_v12_am1_photo SAM_ROOT=$MV2 CAPTURES_ROOT=$CAPF ARM_MASK_ROOT=$ARM \
  VOTE=center OUT_ROOT=srp_hull_semcluster_reNNcSd_am1photo \
  $PYW $S2/voxel_sem_cluster_reassign_soliddrop.py $GRPS --n-views 12 --sem-thr 0.40 2>&1 | tail -2
HULL_ROOT=$PWD/data/eval/srp_hull_mv2_v12_am1_fp_photo SAM_ROOT=$MV2 CAPTURES_ROOT=$CAPF ARM_MASK_ROOT=$ARM \
  VOTE=center OUT_ROOT=srp_hull_semcluster_reNNcSd_am1fpphoto \
  $PYW $S2/voxel_sem_cluster_reassign_soliddrop.py $GRPS --n-views 12 --sem-thr 0.40 2>&1 | tail -2

echo "--- 3. div B θ0.5(2 顆)---"
BASE_ROOT=srp_hull_semcluster_reNNcSd_am1photo HULL_ROOT_NAME=srp_hull_mv2_v12_am1_photo \
  OUT_SUFFIX=_reNNcSd_am1photo THETAS=0.5 $PYW $S2/merge_div_guard.py 2>&1 | tail -1
BASE_ROOT=srp_hull_semcluster_reNNcSd_am1fpphoto HULL_ROOT_NAME=srp_hull_mv2_v12_am1_fp_photo \
  OUT_SUFFIX=_reNNcSd_am1fpphoto THETAS=0.5 $PYW $S2/merge_div_guard.py 2>&1 | tail -1

echo "--- 4. eval leak(2 顆 photo)---"
echo "===== am1_photo 中心drop ====="
$PYW $S2/stack_leak_nosep.py --inst-root srp_hull_divB_t50_reNNcSd_am1photo   --gt-root srp_hull_gtlabel_am1photo   2>&1
echo "===== am1_fp_photo 中心drop ====="
$PYW $S2/stack_leak_nosep.py --inst-root srp_hull_divB_t50_reNNcSd_am1fpphoto --gt-root srp_hull_gtlabel_am1fpphoto 2>&1

echo "########## DONE $(date +%H:%M:%S) ##########"
