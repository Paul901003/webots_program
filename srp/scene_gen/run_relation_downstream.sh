#!/bin/bash
# 關係場景(stack/occ)下游管線:SAM 遮罩 → GT modal 標註 → GT amodal → GT 關係 → srp hull+instance。
# 場景名驅動(下游自動分開存到 stack{N}/occ{N});各階段用對的 python 環境。
# 用法: bash srp/scene_gen/run_relation_downstream.sh [stage]
#   stage = sam | labels | amodal | rel | hull | all(預設)
#   SAM 最耗時(120場×12視角):建議 `... sam > sam.log 2>&1 &` 單獨背景跑。
cd "$(dirname "$0")/../.."
PY_SAM=/home/cho/.pyenv/versions/grounded_sam/bin/python3
PY_LBL=/home/cho/.pyenv/versions/3.10.10/bin/python3
PY_VH=/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
GROUPS="stack3 stack4 stack5 occ3 occ4 occ5"
STAGE="${1:-all}"

SCENES=$(ls -d data/captures/multi_stack*/stack*_scene* data/captures/multi_occ*/occ*_scene* 2>/dev/null \
         | xargs -n1 basename | sort | tr '\n' ' ')
echo "場景數: $(echo $SCENES | wc -w)  | stage=$STAGE"
[ -z "$SCENES" ] && { echo "錯誤:收集到 0 場景,中止。"; exit 1; }

run() { echo; echo "===== $1 ====="; }

if [ "$STAGE" = "sam" ] || [ "$STAGE" = "all" ]; then
  run "1. SAM 遮罩 (grounded_sam)"
  $PY_SAM sam_only/sam_only.py $SCENES
fi
if [ "$STAGE" = "labels" ] || [ "$STAGE" = "all" ]; then
  run "2. GT modal 標註 (3.10.10)"
  for s in $SCENES; do
    g=${s%%_*}
    $PY_LBL tools/generate_labels.py \
      --manifest "data/captures/multi_$g/$s/scene_manifest.json" \
      --output data/labels --mode both || echo "[fail] labels $s"
  done
fi
if [ "$STAGE" = "amodal" ] || [ "$STAGE" = "all" ]; then
  run "3. GT amodal 遮罩 (3.10.10)"
  $PY_LBL tools/generate_amodal_masks.py $SCENES
fi
if [ "$STAGE" = "rel" ] || [ "$STAGE" = "all" ]; then
  run "4. GT 關係 on/blocks_access (webots_visual_hull)"
  $PY_VH srp/stage3_graph/gt_relations.py $SCENES
fi
if [ "$STAGE" = "hull" ] || [ "$STAGE" = "all" ]; then
  run "5. srp hull + instance (webots_visual_hull)"
  $PY_VH srp/stage1_hull/run_scene.py $SCENES
  $PY_VH srp/stage2_instances/associate.py $SCENES
fi
echo; echo "===== 完成 (stage=$STAGE) ====="
