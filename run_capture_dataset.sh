#!/bin/bash
# run_capture_dataset.sh — 多視角 RGB+depth 擷取：軌跡規劃(手臂) + 相機＋夾爪順移(multicam)，涵蓋全部場景組。
#
# proto 的 RangeFinder noise 設 0（IntelRealsenseD455.proto）→ depth 無雜訊(注意:無雜訊
# ≠ ground truth,只是乾淨的模擬深度)。
# 手臂(軌跡規劃)受硬體限制拍 12 個規劃視角；相機＋夾爪順移拍 validated 全 39 視角(之後用
# extract_subset.py 抽任意子集;12 是 39 的子集,故仍可與手臂逐視角對應比較)。
#
# 場景組：n3 n4 n5（多物體）+ occ3 occ4 occ5（遮擋）+ stack3 stack4 stack5（堆疊）= 303 場景。
# 每場景啟動一次 Webots（supervisor 拍完即 simulationQuit），可中斷續跑（已達應有張數的場景自動跳過）。
#
# 輸出根：
#   軌跡規劃(手臂)     → data/captures_armmove/multi_<組>/<場景>/   (每場景 12 視角 el/az;與舊 view_NN 分開)
#   相機＋夾爪順移      → data/captures_multicam/multi_<組>/<場景>/  (每場景 39 視角;multicam 純運動學)
#
# 用法：
#   ./run_capture_dataset.sh both         # 兩種方法都拍（預設）
#   ./run_capture_dataset.sh arm          # 只拍軌跡規劃(手臂移動)
#   ./run_capture_dataset.sh multicam     # 只拍相機＋夾爪順移
#   SCENE_GROUPS="occ3 stack3" ./run_capture_dataset.sh both   # 只拍指定組
#   FORCE=1 ./run_capture_dataset.sh arm  # 不跳過已完成，全部重拍
#   CAM_VPFILE=... CAM_EXPECT=N           # 覆寫相機視角來源/應有張數(預設 validated 39)

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT" || exit 1

WEBOTS="${WEBOTS:-webots}"
# 只用 --minimize（不可 --mode=fast：fast 會讓 realsense 控制器來不及寫 depth/pose）。
WEBOTS_OPTS="--batch --minimize --stdout --stderr"
# Webots 用 PATH 的 python3 跑 controller。把 webots_visual_hull 擺最前面，讓腳本自帶
# 正確環境(realsense 要 numpy/cv2)，不必先 activate 也不受啟動終端機影響。
VENV_BIN="${VENV_BIN:-/home/cho/.pyenv/versions/webots_visual_hull/bin}"
[ -x "$VENV_BIN/python3" ] && export PATH="$VENV_BIN:$PATH"
PY="${PY:-$VENV_BIN/python3}"

# 手臂受硬體限制只拍規劃好的 12 視角；相機瞬移拍 validated 全 39 視角(之後可用
# extract_subset.py 抽任意子集；12 是 39 的子集,故仍可與手臂逐視角對應比較)。
EXEC_COUNT=12                                                    # 手臂視角數(planned_paths_multi_n12)
X_OFF=0.35
ARM_EXPECT="${ARM_EXPECT:-12}"                                   # 手臂每場景應有張數
ARM_ROOT="${ARM_ROOT:-$REPO_ROOT/data/captures_armmove}"         # 手臂輸出根(el/az;與 view_NN 舊資料分開)
CAM_VPFILE="${CAM_VPFILE:-validated_viewpoints_multi_latest.json}"  # 相機視角來源(全 39)
CAM_EXPECT="${CAM_EXPECT:-39}"                                   # 相機每場景應有張數
ARM_WORLD="$REPO_ROOT/worlds/ycb_supervisor_four_view_capture_multi.wbt"
CAM_WORLD="$REPO_ROOT/worlds/ycb_multicam_capture.wbt"           # 相機＋夾爪順移(multicam,含手臂+夾爪)

# 注意：不可用 GROUPS（bash 保留特殊變數，賦值無效會變成使用者群組 id）。
METHOD="${1:-both}"                                              # arm | multicam | both
SCENE_GROUPS="${SCENE_GROUPS:-n3 n4 n5 occ3 occ4 occ5 stack3 stack4 stack5}"
FORCE="${FORCE:-0}"

for w in "$ARM_WORLD" "$CAM_WORLD"; do
    [ -f "$w" ] || { echo "找不到 world: $w"; exit 1; }
done

# 列出某組（n3/occ3/stack3...）所有場景名（跨 multi/occ/stack plan）
scenes_of() {
    "$PY" - "$1" <<'PYEOF'
import glob, json, os, sys
grp = sys.argv[1]
plans = ["data/scene_plans/multi_scene_plan.json",
         "data/scene_plans/occ_scene_plan.json",
         "data/scene_plans/stack_scene_plan.json"]
out = set()
for p in plans:
    if not os.path.exists(p):
        continue
    for s in json.load(open(p, encoding="utf-8")).get("scenes", []):
        n = s.get("scene_name", "")
        if n.split("_")[0] == grp:
            out.add(n)
print("\n".join(sorted(out)))
PYEOF
}

# 該場景在指定根目錄是否已完成（>=$2 張 RGB）
done_in() {  # $1 = 場景目錄, $2 = 應有張數
    [ -d "$1" ] || return 1
    local n
    n=$(find "$1" -maxdepth 1 -name 'view_el*.png' ! -name '*_depth.png' 2>/dev/null | wc -l)
    [ "$n" -ge "$2" ]
}

run_arm() {  # $1 = 場景名
    local g="${1%%_*}"
    if done_in "$ARM_ROOT/multi_$g/$1" "$ARM_EXPECT" && [ "$FORCE" != 1 ]; then
        echo "  [跳過-arm] $1（已 $ARM_EXPECT 張）"; return 0
    fi
    EXEC_COUNT=$EXEC_COUNT EXEC_X_OFFSET=$X_OFF CAPTURE_SCENE="$1" ARMMOVE_ROOT="$ARM_ROOT" \
        REALSENSE_SYNC_SAVE=1 "$WEBOTS" $WEBOTS_OPTS "$ARM_WORLD"
}

run_cam() {  # $1 = 場景名（相機＋夾爪順移 multicam → captures_multicam）
    local g="${1%%_*}"
    if done_in "data/captures_multicam/multi_$g/$1" "$CAM_EXPECT" && [ "$FORCE" != 1 ]; then
        echo "  [跳過-cam] $1（已 $CAM_EXPECT 張）"; return 0
    fi
    MULTICAM_VIEWPOINTS="$CAM_VPFILE" MULTICAM_ROOT="$REPO_ROOT/data/captures_multicam" \
        MULTICAM_SCENE="$1" REALSENSE_SYNC_SAVE=1 "$WEBOTS" $WEBOTS_OPTS "$CAM_WORLD"
}

ok=0; fail=0
for g in $SCENE_GROUPS; do
    mapfile -t scenes < <(scenes_of "$g")
    [ "${#scenes[@]}" -eq 0 ] && { echo "組 $g 無場景，略過"; continue; }
    if [ "${SCENE_LIMIT:-0}" -gt 0 ]; then scenes=("${scenes[@]:0:$SCENE_LIMIT}"); fi   # 每組只取前 N 場景(測試用)
    echo "========== 組 $g：$((${#scenes[@]})) 場景  (方法=$METHOD) =========="
    i=0
    for sc in "${scenes[@]}"; do
        i=$((i + 1))
        printf '\n--- [%s %d/%d] %s ---\n' "$g" "$i" "${#scenes[@]}" "$sc"
        if [ "$METHOD" = "arm" ] || [ "$METHOD" = "both" ]; then
            run_arm "$sc" && ok=$((ok + 1)) || { echo "  [錯誤-arm] $sc"; fail=$((fail + 1)); }
        fi
        if [ "$METHOD" = "multicam" ] || [ "$METHOD" = "both" ]; then
            run_cam "$sc" && ok=$((ok + 1)) || { echo "  [錯誤-cam] $sc"; fail=$((fail + 1)); }
        fi
    done
done

echo ""
echo "全部結束。成功 $ok 個（場景×方法），失敗 $fail 個。"
[ "$fail" -eq 0 ]
