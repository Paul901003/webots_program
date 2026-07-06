#!/bin/bash
# run_capture_all.sh — fast arm-move 拍攝「全部」場景(n1/n3/n4/n5/occ/stack),
# 每場景走 planned_paths_multi_all_validated.json 的 34 視角(EXEC_COUNT=all),與 GT/多相機對齊比較。
#
# 用法:
#   ./run_capture_all.sh                 # 全部 367 場景(n1+n3/4/5+occ+stack)
#   ./run_capture_all.sh occ stack       # 只拍 occ + stack
#   ./run_capture_all.sh n3 occ3         # 只拍 n3 + occ3(前綴比對)
#   FORCE=1 ./run_capture_all.sh         # 重拍(不跳過已有 scene_manifest 的場景)
#   START=100 ./run_capture_all.sh       # 從全域第 100 個場景開始
# env:
#   ARMMOVE_ROOT   輸出根(預設 data/captures_fast)
#   EXEC_X_OFFSET  規劃路徑 x 偏移 tag(預設 0.35;此模式讀 all_validated 檔,tag 不影響檔名)
#   WB_PORT        多開 webots 時給不同 port
#
# 註:此為「replay 已規劃路徑」的拍攝,不需 planning bridge(A-4 已把路徑算好)。
#    realtime 模式(非 --mode=fast),否則 realsense 每視角被強制終止、來不及寫 depth/pose。

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
WORLD="$REPO_ROOT/worlds/ycb_supervisor_four_view_capture_multi_fast.wbt"
PLANS_DIR="$REPO_ROOT/data/scene_plans"
PATHS_FILE="$REPO_ROOT/data/viewpoints/planned_paths_multi_all_validated.json"
WEBOTS="${WEBOTS:-webots}"
WEBOTS_OPTS="--batch --minimize --stdout --stderr"

export EXEC_COUNT="all"
export EXEC_X_OFFSET="${EXEC_X_OFFSET:-0.35}"
export ARMMOVE_ROOT="${ARMMOVE_ROOT:-$REPO_ROOT/data/captures_fast}"
START="${START:-1}"
FORCE="${FORCE:-0}"
WB_PORT="${WB_PORT:-}"
[ -n "$WB_PORT" ] && WEBOTS_OPTS="$WEBOTS_OPTS --port=$WB_PORT"

[ -f "$WORLD" ]      || { echo "找不到 world: $WORLD"; exit 1; }
[ -f "$PATHS_FILE" ] || { echo "★缺 $PATHS_FILE(先跑 A-4 plan_viewpoint_paths.py --all-validated)"; exit 1; }

PLANS=(single_scene_plan multi_scene_plan occ_scene_plan stack_scene_plan)
FILTERS=("$@")   # 空 = 全部

# 1) 收集所有場景名(依 plan 順序:n1 → n3/4/5 → occ → stack)
ALL_SCENES=()
for p in "${PLANS[@]}"; do
    f="$PLANS_DIR/$p.json"
    [ -f "$f" ] || continue
    while read -r n; do
        [ -n "$n" ] && ALL_SCENES+=("$n")
    done < <(python3 -c "
import json
for s in json.load(open('$f')).get('scenes', []):
    print(s.get('scene_name', ''))
")
done

# 2) 套用前綴 filter(無參數則全收)
SCENES=()
for n in "${ALL_SCENES[@]}"; do
    if [ "${#FILTERS[@]}" -eq 0 ]; then
        SCENES+=("$n"); continue
    fi
    for t in "${FILTERS[@]}"; do
        case "$n" in ${t}*) SCENES+=("$n"); break;; esac
    done
done

TOTAL="${#SCENES[@]}"
[ "$TOTAL" -eq 0 ] && { echo "沒有符合的場景(filter: ${FILTERS[*]:-<全部>})"; exit 1; }
echo "共 $TOTAL 場景待拍  EXEC_COUNT=all(34視角)  輸出根: $ARMMOVE_ROOT"
[ "${#FILTERS[@]}" -gt 0 ] && echo "  filter: ${FILTERS[*]}"

# 3) 逐場景拍(已有 scene_manifest 者跳過,除非 FORCE=1)
done_n=0; fail_n=0; skip_n=0; idx=0
for name in "${SCENES[@]}"; do
    idx=$((idx + 1))
    [ "$idx" -lt "$START" ] && continue
    group="${name%%_*}"                              # n3 / occ3 / stack3 / n1
    scene_dir="$ARMMOVE_ROOT/multi_$group/$name"
    if [ "$FORCE" != "1" ] && [ -f "$scene_dir/scene_manifest.json" ]; then
        printf '[#%d/%d] %-18s 已完成,跳過\n' "$idx" "$TOTAL" "$name"
        skip_n=$((skip_n + 1)); continue
    fi
    printf '\n--- [#%d/%d] %s ---\n' "$idx" "$TOTAL" "$name"
    if CAPTURE_SCENE="$name" "$WEBOTS" $WEBOTS_OPTS "$WORLD"; then
        done_n=$((done_n + 1))
    else
        echo "  [錯誤] $name 拍攝失敗(webots 回傳非 0)"
        fail_n=$((fail_n + 1))
    fi
done

echo ""
echo "全部結束。成功 $done_n,失敗 $fail_n,跳過 $skip_n(共 $TOTAL)。輸出: $ARMMOVE_ROOT"
[ "$fail_n" -eq 0 ]
