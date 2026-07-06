#!/bin/bash
# run_generate_labels_multi.sh
#
# 批次為 captures 產生 GT 標籤（pyrender 渲染 segmentation mask）。
# 逐場景呼叫 generate_labels.py，輸出到 data/labels/<場景>/{actual,planned}/。
#
# 資料來源預設 data/captures_armmove（手臂 el/az 資料集；可用 CAPTURES_ROOT 覆寫）。
#
# 用法（群組吃完整名稱，也相容純數字→n<數字>）:
#   ./run_generate_labels_multi.sh n3 n4 n5              # 多物體
#   ./run_generate_labels_multi.sh occ3 stack3          # 遮擋/堆疊
#   ./run_generate_labels_multi.sh 3 4 5                # 等同 n3 n4 n5(相容舊寫法)
#   CAPTURES_ROOT=.../data/captures_multicam ./run_generate_labels_multi.sh n3   # 換來源
#
# 變數: MODE(both|actual|planned, 預設 both)、START(起始場景序, 預設 1)、KEEP=1(不清空舊標籤)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PY="/home/cho/.pyenv/versions/3.10.10/bin/python3"
GEN="$SCRIPT_DIR/generate_labels.py"
CAPTURES_ROOT="${CAPTURES_ROOT:-$REPO_ROOT/data/captures_armmove}"
LABELS="$REPO_ROOT/data/labels"
MODE="${MODE:-both}"
START="${START:-1}"
KEEP="${KEEP:-0}"
NO_ARM="${NO_ARM:-0}"            # 1 = 不渲手臂(相機瞬移 captures_multicam 無手臂,用這個)
GEN_ARGS=""
[ "$NO_ARM" = 1 ] && GEN_ARGS="--no-arm"

if [ "$#" -eq 0 ]; then
    echo "用法: $0 <group...>   例如: $0 n3 n4 n5 / occ3 stack3 / 3 4 5"
    exit 1
fi
[ -x "$PY" ]  || { echo "找不到 python: $PY"; exit 1; }
[ -f "$GEN" ] || { echo "找不到 generate_labels.py: $GEN"; exit 1; }

total_done=0
total_fail=0
for arg in "$@"; do
    # 純數字 → n<數字>（相容舊寫法）；否則當完整群組名(n3/occ3/stack3...)
    case "$arg" in
        ''|*[!0-9]*) g="$arg" ;;
        *)           g="n$arg" ;;
    esac

    cap_dir="$CAPTURES_ROOT/multi_$g"
    [ -d "$cap_dir" ] || { echo "找不到 captures: $cap_dir，略過 $g"; continue; }
    mapfile -t scene_dirs < <(ls -d "$cap_dir"/${g}_scene* 2>/dev/null | sort)
    count="${#scene_dirs[@]}"
    [ "$count" -gt 0 ] || { echo "$g: captures 內無場景，略過"; continue; }

    if [ "$KEEP" != "1" ] && [ "$START" -eq 1 ]; then
        echo "清空舊標籤: $LABELS/${g}_scene*"
        find "$LABELS" -maxdepth 1 -type d -name "${g}_scene*" -exec rm -rf {} +
    fi

    echo "========== $g：共 $count 場景（從第 $START 個開始，mode=$MODE，來源 $(basename "$CAPTURES_ROOT")）=========="
    idx=0
    for sd in "${scene_dirs[@]}"; do
        idx=$((idx + 1))
        [ "$idx" -ge "$START" ] || continue
        scene="$(basename "$sd")"
        M="$sd/scene_manifest.json"
        if [ ! -f "$M" ]; then
            echo "  [略過] 找不到 manifest: $M"
            continue
        fi
        printf '\n--- [%s %d/%d] %s ---\n' "$g" "$idx" "$count" "$scene"
        if "$PY" "$GEN" --manifest "$M" --output "$LABELS" --mode "$MODE" $GEN_ARGS; then
            total_done=$((total_done + 1))
        else
            echo "  [錯誤] $scene 標籤生成失敗"
            total_fail=$((total_fail + 1))
        fi
    done
done

echo ""
echo "標籤生成結束。成功 $total_done 個場景，失敗 $total_fail 個。"
[ "$total_fail" -eq 0 ]
