#!/bin/bash
set -u; cd /home/cho/webots_program
PY=/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc
MV2=$PWD/data/eval/mobilesamv2_fast; CAP=$PWD/data/captures_fast
# 展開場景名(run_scene不接群名)
SCENES=$(for g in stack3 stack4 stack5; do ls -d data/eval/mobilesamv2_fast/${g}_scene* 2>/dev/null | xargs -n1 basename; done)
Q(){ grep -viE "QuickGELU|UserWarning|warnings.warn|Warning"; }
echo "######## voxel大小實驗 10/7/5/3/1mm × stack60 START $(date '+%m-%d %H:%M') ########"
echo "場景數: $(echo "$SCENES"|wc -l)"
for MM in 10 7 5 3 1; do
  V=$(python3 -c "print($MM/1000)")
  HR=srp_hull_vox${MM}mm
  echo "==== ${MM}mm ($(date '+%H:%M')) ===="
  CK=4000000; [ $MM -le 3 ] && CK=2000000; [ $MM -eq 1 ] && CK=1500000
  # 1.建hull(逐場,run_scene吃展開場景名)
  for sc in $SCENES; do
    SAM_ROOT=$MV2 CAPTURES_ROOT=$CAP CARVE_CHUNKED=1 CARVE_CHUNK=$CK \
      $PY srp/stage1_hull/run_scene.py $sc --num-views 12 --allow-miss 1 --voxel $V --root $HR >/dev/null 2>&1
  done
  echo "   hull: $(ls data/eval/$HR/*_scene*/hull.npz 2>/dev/null|wc -l) 場 $(date '+%H:%M')"
  # 2.補surface(接群名可空,但保險傳場景)
  $PY srp/stage2_instances/add_surface_mask.py stack3 stack4 stack5 --root $HR >/dev/null 2>&1
  # 3.語意分群
  FORCE=1 HULL_ROOT=$PWD/data/eval/$HR SAM_ROOT=$MV2 CAPTURES_ROOT=$CAP OUT_ROOT=srp_hull_semdonut_vox${MM}mm \
    $PY srp/stage2_instances/voxel_sem_cluster_surf_donut.py stack3 stack4 stack5 --n-views 12 --sem-thr 0.4 >/dev/null 2>&1
  echo "   分群: $(ls data/eval/srp_hull_semdonut_vox${MM}mm/*_scene*/instances.json 2>/dev/null|wc -l) 場 $(date '+%H:%M')"
done
echo "######## hull+分群 DONE $(date '+%H:%M') ########"
