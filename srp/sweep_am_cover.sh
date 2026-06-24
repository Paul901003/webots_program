#!/bin/bash
# Sweep --allow-miss(Stage1) × --cover(Stage2) × --agree-frac(Stage2)。檔名後綴標記,絕不互蓋。
# 全部寫到 data/eval/srp_sweep/(baseline srp_hull/ 不動):
#   hull       : srp_sweep/<scene>/hull_am{am}.npz                      (只依 am)
#   instances  : srp_sweep/<scene>/instances_am{am}_cv{cov}_ag{NN}.*    (am+cover+agree_frac)
#   CSV        : srp_sweep/d1d2_am{am}_cv{cov}_ag{NN}.csv               (每列一場景;NN=agree_frac×100)
# 用法: bash srp/sweep_am_cover.sh ["場景 regex"]   (預設全部 n1/n3/n4/n5)
#   背景: bash srp/sweep_am_cover.sh > data/eval/srp_sweep/sweep.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
PY=/home/cho/.pyenv/versions/webots_visual_hull/bin/python3

AM_LIST="0 1 2"            # allow_miss(Stage1,絕對張數)
COV_LIST="small large"     # cover(Stage2)
AG_LIST="0.5 0.8 0.9"             # agree-frac(Stage2,以參與視角為基準);加值會倍增組合數
ROOT=srp_sweep
PAT="${1:-^n[1345]_}"
SCENES=$(ls data/eval/sam_only/ | grep -E "$PAT")
mkdir -p "data/eval/$ROOT"
ncomb=$(( $(echo $AM_LIST|wc -w) * $(echo $COV_LIST|wc -w) * $(echo $AG_LIST|wc -w) ))
echo "場景數: $(echo $SCENES | wc -w)  | AM=[$AM_LIST] COV=[$COV_LIST] AG=[$AG_LIST] = $ncomb 組 → data/eval/$ROOT/"

for am in $AM_LIST; do
  echo "########## Stage1 carve allow_miss=$am → hull_am${am}.npz ##########"
  $PY srp/stage1_hull/run_scene.py $SCENES --allow-miss "$am" --root "$ROOT" --tag "am${am}" >/dev/null
  for cov in $COV_LIST; do
    for ag in $AG_LIST; do
      NN=$(awk "BEGIN{printf \"%02d\", $ag*100}")
      TAG="am${am}_cv${cov}_ag${NN}"
      echo "===== Stage2 associate am=$am cover=$cov agree=$ag → instances_${TAG}.* ====="
      $PY srp/stage2_instances/associate.py $SCENES --root "$ROOT" \
          --hull-tag "am${am}" --tag "$TAG" --cover "$cov" --agree-frac "$ag" >/dev/null
      printf "  %s : " "$TAG"
      $PY srp/stage2_instances/eval.py $SCENES --root "$ROOT" --tag "$TAG" \
          --csv "data/eval/$ROOT/d1d2_${TAG}.csv" 2>&1 | grep "=="
    done
  done
done
echo "########## sweep 完成 → data/eval/$ROOT/ ##########"
