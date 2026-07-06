#!/bin/bash
# run_all_fast.sh — captures_fast 全場景 Stage 2 實例關聯。
# 讀 srp_hull_fast 的 hull.npz + sam_only_fast label(排手臂) → 跨視角 voxel 關聯 →
# 寫 srp_hull_fast/<scene>/instances.{npz,json}。自帶 oomd 隔離 scope。
#
# 用法:  ./srp/stage2_instances/run_all_fast.sh              # 全 10 組
#        ./srp/stage2_instances/run_all_fast.sh occ3 stack3  # 指定組
# env:   AGREE_FRAC(預設 0.5) 其餘用 associate.py 預設

set -u
if [ -z "${_ASSOC_SCOPED:-}" ] && command -v systemd-run >/dev/null 2>&1; then
    exec systemd-run --user --scope -p ManagedOOMPreference=omit \
        --setenv=_ASSOC_SCOPED=1 --setenv=AGREE_FRAC="${AGREE_FRAC:-}" -- "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$(dirname "$SCRIPT_DIR")")"
export CAPTURES_ROOT="$REPO/data/captures_fast"
export SAM_ROOT="$REPO/data/eval/sam_only_fast"
export ARM_MASK_ROOT="$REPO/data/eval/srp_arm_masks"
AGREE_FRAC="${AGREE_FRAC:-0.5}"

groups=("$@")
[ "${#groups[@]}" -eq 0 ] && groups=(n1 n3 n4 n5 occ3 occ4 occ5 stack3 stack4 stack5)

# 從 srp_hull_fast 收集有 hull 的場景
scenes=()
for g in "${groups[@]}"; do
    for d in "$REPO/data/eval/srp_hull_fast/${g}_scene"*; do
        [ -f "$d/hull.npz" ] && scenes+=("$(basename "$d")")
    done
done
echo "Stage 2 關聯: ${#scenes[@]} 場景  agree_frac=$AGREE_FRAC → srp_hull_fast/<scene>/instances.{npz,json}"
[ "${#scenes[@]}" -eq 0 ] && { echo "無 hull 場景(先跑 Stage 1)"; exit 1; }

"$SCRIPT_DIR/associate.py" "${scenes[@]}" --agree-frac "$AGREE_FRAC" --root srp_hull_fast

echo "完成 → $REPO/data/eval/srp_hull_fast/<scene>/instances.{npz,json}"
