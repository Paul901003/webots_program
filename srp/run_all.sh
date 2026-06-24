#!/bin/bash
# 整批建 hull + instance 關聯 + D1/D2 評估(所有 n1/n3/n4/n5 場景)。
# 用法: bash srp/run_all.sh   (建議背景執行,輸出見 data/eval/srp_hull/run_all.log)
set -e
cd "$(dirname "$0")/.."
PY=/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
mkdir -p data/eval/srp_hull

for g in n1 n3 n4 n5; do
  scenes=$(ls data/eval/sam_only/ | grep "^${g}_")
  n=$(echo $scenes | wc -w)
  echo "========== GROUP $g  ($n scenes) =========="
  echo "--- Stage 1 carve ---"
  $PY srp/stage1_hull/run_scene.py $scenes >/dev/null
  echo "--- Stage 2 associate ---"
  $PY srp/stage2_instances/associate.py $scenes >/dev/null
  echo "--- D1/D2 eval ---"
  $PY srp/stage2_instances/eval.py $scenes --csv data/eval/srp_hull/d1d2_${g}.csv 2>&1 | grep "=="
done
echo "========== 全部完成 =========="
