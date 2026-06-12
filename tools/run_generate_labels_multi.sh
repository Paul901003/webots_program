#!/bin/bash
# run_generate_labels_multi.sh
#
# 批次為多物體 captures 產生 GT 標籤（pyrender 渲染 segmentation mask）。
# 逐場景呼叫 generate_labels.py，輸出到 data/labels/n{N}_scene{XXXX}/{actual,planned}/。
#
# 用法:
#   ./run_generate_labels_multi.sh 3 4 5      # n3 + n4 + n5
#   ./run_generate_labels_multi.sh 5          # 只做 n5
#   START=10 KEEP=1 ./run_generate_labels_multi.sh 4   # n4 從第 10 個續做、不清空
#
# 變數: MODE(both|actual|planned, 預設 both)、START(起始場景)、KEEP=1(不清空舊標籤)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PY="/home/cho/.pyenv/versions/3.10.10/bin/python3"
GEN="$SCRIPT_DIR/generate_labels.py"
CAPTURES="$REPO_ROOT/data/captures"
LABELS="$REPO_ROOT/data/labels"
MODE="${MODE:-both}"
START="${START:-1}"
KEEP="${KEEP:-0}"

if [ "$#" -eq 0 ]; then
    echo "用法: $0 <group...>   例如: $0 3 4 5"
    exit 1
fi
[ -x "$PY" ]   || { echo "找不到 python: $PY"; exit 1; }
[ -f "$GEN" ]  || { echo "找不到 generate_labels.py: $GEN"; exit 1; }

total_done=0
total_fail=0
for g in "$@"; do
    case "$g" in 1|3|4|5) ;; *) echo "略過不支援的組: n$g（只支援 1/3/4/5）"; continue ;; esac

    cap_dir="$CAPTURES/multi_n$g"
    [ -d "$cap_dir" ] || { echo "找不到 captures: $cap_dir，略過 n$g"; continue; }
    count="$(ls -d "$cap_dir"/n${g}_scene* 2>/dev/null | wc -l)"
    [ "$count" -gt 0 ] || { echo "n$g: captures 內無場景，略過"; continue; }

    if [ "$KEEP" != "1" ] && [ "$START" -eq 1 ]; then
        echo "清空舊標籤: $LABELS/n${g}_scene*"
        find "$LABELS" -maxdepth 1 -type d -name "n${g}_scene*" -exec rm -rf {} +
    fi

    echo "========== n$g：共 $count 場景（從第 $START 個開始，mode=$MODE）=========="
    for i in $(seq "$START" "$count"); do
        scene="$(printf 'n%s_scene%04d' "$g" "$i")"
        M="$cap_dir/$scene/scene_manifest.json"
        if [ ! -f "$M" ]; then
            echo "  [略過] 找不到 manifest: $M"
            continue
        fi
        printf '\n--- [n%s %d/%d] %s ---\n' "$g" "$i" "$count" "$scene"
        if "$PY" "$GEN" --manifest "$M" --output "$LABELS" --mode "$MODE"; then
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
