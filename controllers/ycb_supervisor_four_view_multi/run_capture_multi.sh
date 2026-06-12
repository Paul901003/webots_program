#!/bin/bash
# run_capture_multi.sh
#
# 用新的 multi_scene_plan.json 重新拍攝多物體場景：每個場景啟動一次 Webots，
# 透過 CAPTURE_ARGS 環境變數逐場景傳參（supervisor 跑完該場景即 simulationQuit）。
#
# 用法:
#   ./run_capture_multi.sh 3 4 5        # 拍 n3 + n4 + n5
#   ./run_capture_multi.sh 5            # 只拍 n5
#   START=10 ./run_capture_multi.sh 4   # n4 從第 10 個場景開始（中斷續拍）
#
# 預設「先清空該組舊 captures 再拍」；設 KEEP=1 可保留舊資料只覆寫同名場景。

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
WORLD="$REPO_ROOT/worlds/ycb_supervisor_four_view_capture_multi.wbt"
PLAN="$REPO_ROOT/data/scene_plans/multi_scene_plan.json"
SINGLE_PLAN="$REPO_ROOT/data/scene_plans/single_scene_plan.json"   # n1 用
CAPTURES_DIR="$REPO_ROOT/data/captures"
WEBOTS="${WEBOTS:-webots}"
# 注意：不可用 --mode=fast。fast 會讓 realsense 控制器每視角被 forced-termination，
# 來不及寫出 depth.npy / pose.json（只剩舊檔）。realtime 才能完整寫出所有產物。
WEBOTS_OPTS="--batch --minimize --stdout --stderr"
START="${START:-1}"          # 起始場景編號（續拍用）
KEEP="${KEEP:-0}"            # 1 = 不清空舊 captures
WB_PORT="${WB_PORT:-}"       # 多開時給每個實例不同 port（避免搶同一 port 互相干擾）
[ -n "$WB_PORT" ] && WEBOTS_OPTS="$WEBOTS_OPTS --port=$WB_PORT"

if [ "$#" -eq 0 ]; then
    echo "用法: $0 <group...>   例如: $0 3 4 5   或   $0 1"
    echo "  group 接受 1 / 3 / 4 / 5（1=單物體，讀 single_scene_plan.json）"
    exit 1
fi
[ -f "$WORLD" ] || { echo "找不到 world: $WORLD"; exit 1; }

scene_count() {  # $1 = group number；n1 讀單物體 plan，其餘讀多物體 plan
    local plan="$PLAN"; [ "$1" = "1" ] && plan="$SINGLE_PLAN"
    python3 -c "import json;d=json.load(open('$plan'));print(sum(1 for s in d['scenes'] if s['scene_name'].startswith('n$1_')))"
}

total_done=0
total_fail=0
for g in "$@"; do
    case "$g" in
        1|3|4|5) ;;
        *) echo "略過不支援的組: n$g（只支援 1/3/4/5）"; continue ;;
    esac

    count="$(scene_count "$g")"
    if [ -z "$count" ] || [ "$count" -eq 0 ]; then
        echo "n$g: plan 內找不到場景，略過"
        continue
    fi

    out_dir="$CAPTURES_DIR/multi_n$g"
    if [ "$KEEP" != "1" ] && [ "$START" -eq 1 ]; then
        echo "清空舊 captures: $out_dir"
        rm -rf "$out_dir"
    fi

    echo "========== n$g：共 $count 場景（從第 $START 個開始）=========="
    for i in $(seq "$START" "$count"); do
        printf '\n--- [n%s %d/%d] n%s_scene%04d ---\n' "$g" "$i" "$count" "$g" "$i"
        if CAPTURE_ARGS="--$g --$i" "$WEBOTS" $WEBOTS_OPTS "$WORLD"; then
            total_done=$((total_done + 1))
        else
            echo "  [錯誤] n$g scene $i 拍攝失敗（webots 回傳非 0）"
            total_fail=$((total_fail + 1))
        fi
    done
done

echo ""
echo "全部結束。成功 $total_done 個場景，失敗 $total_fail 個。"
[ "$total_fail" -eq 0 ]
