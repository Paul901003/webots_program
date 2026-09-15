#!/bin/bash
# run_downstream_mv2.sh — MobileSAMv2 遮罩的下游:hull → instance → vs GT 評估。
# SAM_ROOT=mobilesamv2_fast,輸出獨立 root srp_hull_mobilesamv2(不動 sam_only 的 srp_hull_fast)。
# 共用(與分割方法無關):arm 剪影 srp_arm_masks、GT amodal 快取 gt_hull_cache_fast。自帶 oomd 隔離 scope。
#
# 用法:  ./mobilesamv2/run_downstream_mv2.sh              # 全 10 組
#        ./mobilesamv2/run_downstream_mv2.sh occ3 stack3  # 指定組
# env:   ALLOW_MISS(預設 2)

set -u
if [ -z "${_MV2DS_SCOPED:-}" ] && command -v systemd-run >/dev/null 2>&1; then
    exec systemd-run --user --scope -p ManagedOOMPreference=omit \
        --setenv=_MV2DS_SCOPED=1 --setenv=ALLOW_MISS="${ALLOW_MISS:-}" --setenv=MV2_ENCODER="${MV2_ENCODER:-}" \
        -- "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
MV2_ENCODER="${MV2_ENCODER:-efficientvit_l2}"
case "$MV2_ENCODER" in
    efficientvit_l2) _mask="mobilesamv2_fast";        ROOT="srp_hull_mobilesamv2";;
    *)               _mask="mobilesamv2_${MV2_ENCODER}_fast"; ROOT="srp_hull_mv2_${MV2_ENCODER}";;
esac
export CAPTURES_ROOT="$REPO/data/captures_fast"
export SAM_ROOT="$REPO/data/eval/$_mask"
export ARM_MASK_ROOT="$REPO/data/eval/srp_arm_masks"
export GT_CACHE="$REPO/data/eval/gt_hull_cache_fast"
ALLOW_MISS="${ALLOW_MISS:-2}"
S1="$REPO/srp/stage1_hull/run_scene.py"
S2="$REPO/srp/stage2_instances/associate.py"
S3="$REPO/srp/stage2_instances/eval.py"

groups=("$@")
[ "${#groups[@]}" -eq 0 ] && groups=(n1 n3 n4 n5 occ3 occ4 occ5 stack3 stack4 stack5)

# 收集有 MobileSAMv2 遮罩的場景
scenes=()
for g in "${groups[@]}"; do
    for d in "$SAM_ROOT/${g}_scene"*; do
        [ -d "$d" ] && scenes+=("$(basename "$d")")
    done
done
echo "MobileSAMv2 下游: ${#scenes[@]} 場景  allow_miss=$ALLOW_MISS → data/eval/$ROOT/"
[ "${#scenes[@]}" -eq 0 ] && { echo "無 MobileSAMv2 遮罩場景(先跑 run_mobilesamv2_all.sh)"; exit 1; }

echo ""; echo "===== ①/3 voxel carving(排手臂,soft am$ALLOW_MISS)→ $ROOT ====="
"$S1" "${scenes[@]}" --voxel 0.005 --allow-miss "$ALLOW_MISS" --root "$ROOT"

echo ""; echo "===== ②/3 跨視角實例關聯 → $ROOT ====="
"$S2" "${scenes[@]}" --agree-frac 0.5 --root "$ROOT"

echo ""; echo "===== ③/3 instance vs GT 評估(重用 GT 快取)→ $ROOT/d1d2.csv ====="
"$S3" "${scenes[@]}" --iou 0.25 --root "$ROOT"

echo ""; echo "完成 → $REPO/data/eval/$ROOT/d1d2.csv"
