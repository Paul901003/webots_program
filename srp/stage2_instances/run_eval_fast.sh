#!/bin/bash
# run_eval_fast.sh — captures_fast: instance vs GT 評估(report §2 D1)。
# ① 產 GT amodal 遮罩(缺的才產,py3.10 pyrender) ② eval.py 3D IoU 匈牙利配對 → found/recall/prec/mIoU。
# 讀 srp_hull_fast 的 instances;GT 快取用 gt_hull_cache_fast(不撞舊)。自帶 oomd 隔離 scope。
#
# 用法:  ./srp/stage2_instances/run_eval_fast.sh              # 全 10 組
#        ./srp/stage2_instances/run_eval_fast.sh occ3 stack3  # 指定組
# env:   IOU(預設 0.25)

set -u
if [ -z "${_EVAL_SCOPED:-}" ] && command -v systemd-run >/dev/null 2>&1; then
    exec systemd-run --user --scope -p ManagedOOMPreference=omit \
        --setenv=_EVAL_SCOPED=1 --setenv=IOU="${IOU:-}" -- "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$(dirname "$SCRIPT_DIR")")"
export CAPTURES_ROOT="$REPO/data/captures_fast"
export GT_CACHE="$REPO/data/eval/gt_hull_cache_fast"
AMODAL="$REPO/tools/generate_amodal_masks.py"
IOU="${IOU:-0.25}"

groups=("$@")
[ "${#groups[@]}" -eq 0 ] && groups=(n1 n3 n4 n5 occ3 occ4 occ5 stack3 stack4 stack5)

# 收集有 instances 的場景
scenes=()
for g in "${groups[@]}"; do
    for d in "$REPO/data/eval/srp_hull_fast/${g}_scene"*; do
        [ -f "$d/instances.npz" ] && scenes+=("$(basename "$d")")
    done
done
echo "評估 ${#scenes[@]} 場景  IoU門檻=$IOU"
[ "${#scenes[@]}" -eq 0 ] && { echo "無 instances 場景(先跑 Stage 2)"; exit 1; }

# ① 缺 amodal 的場景才產
need_amodal=()
for s in "${scenes[@]}"; do
    [ -f "$REPO/data/labels/$s/amodal/annotations.json" ] || need_amodal+=("$s")
done
echo ""
echo "===== 步驟 1/2: GT amodal 遮罩(缺 ${#need_amodal[@]} 場景要產) ====="
if [ "${#need_amodal[@]}" -gt 0 ]; then
    "$AMODAL" "${need_amodal[@]}" 2>&1 \
        | grep -viE 'warning|futurewarning|weights_only|torch.load|state_dict'
fi

echo ""
echo "===== 步驟 2/2: eval instance vs GT → srp_hull_fast/d1d2.csv ====="
"$SCRIPT_DIR/eval.py" "${scenes[@]}" --iou "$IOU" --root srp_hull_fast

echo ""
echo "完成 → $REPO/data/eval/srp_hull_fast/d1d2.csv"
