#!/bin/bash
# run_all_fast.sh — captures_fast 全場景 visual hull(FK 排手臂 + soft carving allow_miss=2)。
# 兩步:① arm_silhouette.py(py3.10 pyrender,FK 手臂剪影,跳過已完成)
#       ② run_scene.py(webots_visual_hull torch,voxel space carving → srp_hull_fast)。
# 自帶 oomd 隔離 scope(不黑屏)。輸出 data/eval/srp_hull_fast/<scene>/hull.npz。
#
# 用法:  ./srp/stage1_hull/run_all_fast.sh              # 全 10 組
#        ./srp/stage1_hull/run_all_fast.sh occ3 stack3  # 指定組
# env:   ALLOW_MISS(預設 2)  VOXEL(預設 0.005)

set -u
if [ -z "${_HULL_SCOPED:-}" ] && command -v systemd-run >/dev/null 2>&1; then
    exec systemd-run --user --scope -p ManagedOOMPreference=omit \
        --setenv=_HULL_SCOPED=1 --setenv=ALLOW_MISS="${ALLOW_MISS:-}" --setenv=VOXEL="${VOXEL:-}" \
        -- "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$(dirname "$SCRIPT_DIR")")"
export CAPTURES_ROOT="$REPO/data/captures_fast"
export SAM_ROOT="$REPO/data/eval/sam_only_fast"
export ARM_MASK_ROOT="$REPO/data/eval/srp_arm_masks"
ALLOW_MISS="${ALLOW_MISS:-2}"
VOXEL="${VOXEL:-0.005}"

groups=("$@")
[ "${#groups[@]}" -eq 0 ] && groups=(n1 n3 n4 n5 occ3 occ4 occ5 stack3 stack4 stack5)

# 收集場景名(給 run_scene)
scenes=()
for g in "${groups[@]}"; do
    for d in "$CAPTURES_ROOT/multi_$g/${g}_scene"*; do
        [ -d "$d" ] && scenes+=("$(basename "$d")")
    done
done
echo "共 ${#scenes[@]} 場景  組: ${groups[*]}  allow_miss=$ALLOW_MISS  voxel=$VOXEL"
[ "${#scenes[@]}" -eq 0 ] && { echo "無場景"; exit 1; }

echo ""
echo "===== 步驟 1/2: FK 手臂剪影(pyrender, 跳過已完成) ====="
"$SCRIPT_DIR/arm_silhouette.py" "${groups[@]}" 2>&1 \
    | grep -viE 'warning|futurewarning|weights_only|torch.load|state_dict'

echo ""
echo "===== 步驟 2/2: voxel space carving → srp_hull_fast ====="
"$SCRIPT_DIR/run_scene.py" "${scenes[@]}" --voxel "$VOXEL" --allow-miss "$ALLOW_MISS" --root srp_hull_fast

echo ""
echo "完成 → $REPO/data/eval/srp_hull_fast/<scene>/hull.npz"
