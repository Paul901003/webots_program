#!/usr/bin/env bash
# run_all.sh — B 方法整批跑:sam_only → associate → carve_instances
#
# 用法:
#   ./instance_hull/run_all.sh                # 預設跑 n1 n3 n4 n5 全部
#   ./instance_hull/run_all.sh 3              # 只跑 n3 整組
#   ./instance_hull/run_all.sh n3_scene0001   # 單一場景
#   FORCE=1 ./instance_hull/run_all.sh 3      # sam_only 重做(忽略已存在)
#
# 三步在兩個環境:sam_only/associate 用 grounded_sam,carve 用 webots_visual_hull。
set -e
cd "$(dirname "$0")/.."

GS_PY=/home/cho/.pyenv/versions/grounded_sam/bin/python3
VH_PY=/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc

TARGETS=("${@:-1 3 4 5}")

echo "######## [1/3] SAM 切所有遮罩(最慢)########"
$GS_PY sam_only/sam_only.py ${TARGETS[@]}

echo "######## [2/3] 幾何關聯 associate ########"
$GS_PY instance_hull/associate.py ${TARGETS[@]}

echo "######## [3/3] per-object 雕殼 carve ########"
$VH_PY instance_hull/carve_instances.py ${TARGETS[@]}

echo "######## 完成。輸出: data/eval/instance_hull/<scene>/ ########"
