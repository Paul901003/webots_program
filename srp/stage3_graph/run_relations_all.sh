#!/bin/bash
# run_relations_all.sh — captures_fast 全場景產物體關係 GT。
# gt_relations.py 讀 data/labels/<scene>/{amodal,actual} + scene_plan
#   → data/labels/<scene>/relations.json(on=全局 + blocks_access=每視角逐條 + 方向評估時算)。
# 純 CPU(mask decode + mesh);自帶 oomd 隔離 scope(不黑屏)。n1 單物體無關係(0),含著無妨。
#
# 用法:  ./srp/stage3_graph/run_relations_all.sh              # 全 10 組
#        ./srp/stage3_graph/run_relations_all.sh occ3 stack3  # 指定組

set -u
if [ -z "${_REL_SCOPED:-}" ] && command -v systemd-run >/dev/null 2>&1; then
    exec systemd-run --user --scope -p ManagedOOMPreference=omit --setenv=_REL_SCOPED=1 -- "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$(dirname "$SCRIPT_DIR")")"
VP=/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
GT="$SCRIPT_DIR/gt_relations.py"

groups=("$@")
[ "${#groups[@]}" -eq 0 ] && groups=(n1 n3 n4 n5 occ3 occ4 occ5 stack3 stack4 stack5)

# 收集有 amodal + actual 的場景
scenes=()
for g in "${groups[@]}"; do
    for d in "$REPO/data/labels/${g}_scene"*/amodal; do
        [ -d "$d" ] || continue
        s="$(basename "$(dirname "$d")")"
        [ -f "$REPO/data/labels/$s/actual/annotations.json" ] && scenes+=("$s")
    done
done
echo "物體關係 GT: ${#scenes[@]} 場景  組: ${groups[*]}"
[ "${#scenes[@]}" -eq 0 ] && { echo "無場景(需 labels/<scene>/{amodal,actual})"; exit 1; }

start=$(date +%s)
"$VP" "$GT" "${scenes[@]}" 2>&1 \
    | grep -viE 'warning|futurewarning|weights_only|torch.load|state_dict|column_stack'
echo ""
echo "完成(耗時 $(( $(date +%s) - start ))s)→ data/labels/<scene>/relations.json"
