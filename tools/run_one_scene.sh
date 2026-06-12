#!/bin/bash
# run_one_scene.sh <scene> [box text nms]
# 單一場景一鍵跑 pipeline A(Grounded-SAM):
#   C-2a 產遮罩(grounded_sam) → C-2b 評估(evaluate_masks) → C-3 建殼(build_torchhull)
# (標籤 C-1 須事先生成:data/labels/<scene>/actual/annotations.json)
#
# 用法:
#   ./tools/run_one_scene.sh n3_scene0001
#   ./tools/run_one_scene.sh n5_scene0010 0.3 0.3 0.7    # 自訂門檻
set -e

SCENE="${1:-}"
BOX="${2:-0.25}"; TEXT="${3:-0.25}"; NMS="${4:-0.8}"
if [ -z "$SCENE" ]; then
    echo "用法: $0 <scene> [box text nms]   例如: $0 n3_scene0001"
    exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

G="${SCENE%%_*}"                                   # n3_scene0001 → n3
CAP="data/captures/multi_$G/$SCENE"
[ -d "$CAP" ] || { echo "找不到場景: $CAP"; exit 1; }

SAM_PY=/home/cho/.pyenv/versions/webots_visual_hull/bin/python3   # torch/SAM/cv2
VH=Grounded-Segment-Anything/webots_visual_hull
W="$(python3 -c "import sys;b,t,n=map(float,sys.argv[1:4]);print(f'grounded_sam_{b:g}_{t:g}_{n:g}')" "$BOX" "$TEXT" "$NMS")"
MASK="$REPO/data/eval/$W/multi_$G/$SCENE"
ANN="data/labels/$SCENE/actual/annotations.json"
[ -f "$ANN" ] || { echo "找不到標籤: $ANN（請先生成 C-1 標籤）"; exit 1; }

echo "===== [$SCENE] 門檻 box=$BOX text=$TEXT nms=$NMS → $W ====="

echo "--- C-2a Grounded-SAM 產遮罩 ---"
"$SAM_PY" grounded_sam/run_grounded_sam.py "$SCENE" \
    --box-threshold "$BOX" --text-threshold "$TEXT" --nms-threshold "$NMS"

echo "--- C-2b 評估(讀遮罩 vs GT) ---"
"$SAM_PY" tools/evaluate_masks.py --labels "$ANN" --pred-dir "$MASK"

echo "--- C-3 Visual Hull ---"
PARTIAL_ARG=()
[ "${MASKS_PARTIAL:-0}" = "1" ] && PARTIAL_ARG=(--masks-partial)   # 預設關;MASKS_PARTIAL=1 開啟
env CUDA_HOME=/usr/local/cuda-12.6 CUDACXX=/usr/local/cuda-12.6/bin/nvcc \
    PATH=/usr/local/cuda-12.6/bin:$PATH \
    "$SAM_PY" "$VH/build_torchhull.py" \
    --scene-dir "$CAP" --mask-dir "$MASK" --device cuda "${PARTIAL_ARG[@]}"

echo ""
echo "===== 完成 ====="
echo "評估+mask+hull: $MASK/"
echo "驗證 3D : VH_SCENE=$SCENE${W:+ VH_MASKDIR=$W} webots worlds/ycb_visual_hull_view.wbt"
echo "驗證 2D : $SAM_PY $VH/project_visual_hull.py --scene-dir $CAP --box-threshold $BOX --text-threshold $TEXT --nms-threshold $NMS"
