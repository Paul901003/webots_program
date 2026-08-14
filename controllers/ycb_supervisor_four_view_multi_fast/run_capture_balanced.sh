#!/bin/bash
# run_capture_balanced.sh — 拍平衡資料集 nb / occb / stkb(12 視角,fast 主線)。
#
# 與舊 run_capture_all.sh / run_capture_multi.sh 完全獨立,不動舊 n/occ/stack 資料:
#   · 只讀新 plan:nb/occb/stkb_scene_plan.json(舊 multi/occ/stack plan 不碰)
#   · 前綴 nb3/occb3/stkb3… → controller 自動寫 data/captures_fast/multi_{前綴}/(不撞舊 n3/occ3/stack3)
#   · 12 視角:EXEC_COUNT=12 + EXEC_X_OFFSET=0.35 → 讀 planned_paths_multi_n12_x+035.json,
#     視角名 view_el{el}_az{az} 與 srp 的 selected_view_names(12) 一致。
#   · 從不刪任何目錄;已有 scene_manifest.json 的場景自動跳過(FORCE=1 才重拍)。
#
# 用法:
#   ./run_capture_balanced.sh                 # 拍全部 nb+occb+stkb 所有 N
#   ./run_capture_balanced.sh nb3 occb3       # 只拍前綴 nb3、occb3
#   ./run_capture_balanced.sh stkb            # 拍所有 stkb*(stkb3-6)
#   START=50 ./run_capture_balanced.sh nb4    # 從第 50 個場景續拍
#   FORCE=1  ./run_capture_balanced.sh occb6  # 重拍(忽略 scene_manifest)
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
WORLD="$REPO_ROOT/worlds/ycb_supervisor_four_view_capture_multi_fast.wbt"
PLANS_DIR="$REPO_ROOT/data/scene_plans"
CAPTURES_ROOT="$REPO_ROOT/data/captures_fast"     # controller 預設輸出根(不設 ARMMOVE_ROOT)
WEBOTS="${WEBOTS:-webots}"
# 不可用 --mode=fast(realsense 來不及寫 depth/pose);realtime + minimize 全自動批次。
WEBOTS_OPTS="--batch --minimize --stdout --stderr"
[ -n "${WB_PORT:-}" ] && WEBOTS_OPTS="$WEBOTS_OPTS --port=$WB_PORT"

# 12 視角(A-3):讀 planned_paths_multi_n12_x+035.json,命名與 srp 對齊
export EXEC_COUNT="${EXEC_COUNT:-12}"
export EXEC_X_OFFSET="${EXEC_X_OFFSET:-0.35}"

START="${START:-1}"
FORCE="${FORCE:-0}"
PLANS=(nb_scene_plan occb_scene_plan stkb_scene_plan)
FILTERS=("$@")            # 空 = 全部;否則只收前綴符合者(nb3 / occb / stkb6 …)

[ -f "$WORLD" ] || { echo "找不到 world: $WORLD"; exit 1; }
PATHS_FILE="$REPO_ROOT/data/viewpoints/planned_paths_multi_n12_x+035.json"
[ -f "$PATHS_FILE" ] || { echo "★缺 $PATHS_FILE(先跑 A-4 plan_viewpoint_paths.py 產 n12)"; exit 1; }

# 1) 依 plan 順序收集場景名
ALL_SCENES=()
for p in "${PLANS[@]}"; do
    f="$PLANS_DIR/$p.json"
    [ -f "$f" ] || { echo "略過:找不到 $f"; continue; }
    while read -r n; do
        [ -n "$n" ] && ALL_SCENES+=("$n")
    done < <(python3 -c "
import json
for s in json.load(open('$f')).get('scenes', []):
    print(s.get('scene_name', ''))
")
done

# 2) 前綴 filter
SCENES=()
for n in "${ALL_SCENES[@]}"; do
    if [ "${#FILTERS[@]}" -eq 0 ]; then SCENES+=("$n"); continue; fi
    for t in "${FILTERS[@]}"; do
        case "$n" in ${t}*) SCENES+=("$n"); break;; esac
    done
done

TOTAL="${#SCENES[@]}"
[ "$TOTAL" -eq 0 ] && { echo "沒有符合的場景(filter: ${FILTERS[*]:-<全部>})"; exit 1; }
echo "共 $TOTAL 場景待拍  EXEC_COUNT=$EXEC_COUNT(12視角)  輸出根: $CAPTURES_ROOT"
[ "${#FILTERS[@]}" -gt 0 ] && echo "  filter: ${FILTERS[*]}"

# 3) 逐場景拍(scene_manifest.json 存在=已完成 → 跳過,除非 FORCE=1)
done_n=0; fail_n=0; skip_n=0; idx=0
for name in "${SCENES[@]}"; do
    idx=$((idx + 1))
    [ "$idx" -lt "$START" ] && continue
    group="${name%%_*}"                                  # nb3 / occb3 / stkb6
    scene_dir="$CAPTURES_ROOT/multi_$group/$name"
    if [ "$FORCE" != "1" ] && [ -f "$scene_dir/scene_manifest.json" ]; then
        printf '[#%d/%d] %-20s 已完成,跳過\n' "$idx" "$TOTAL" "$name"
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
echo "全部結束。成功 $done_n,失敗 $fail_n,跳過 $skip_n(共 $TOTAL)。輸出: $CAPTURES_ROOT"
[ "$fail_n" -eq 0 ]
