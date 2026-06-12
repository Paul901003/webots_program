#!/bin/bash
set -e

PYTHON="/home/cho/.pyenv/versions/webots_visual_hull/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LABELS_ROOT="$PROJECT_DIR/data/labels"
EVALUATE="$SCRIPT_DIR/evaluate_masks.py"

total=0
done_count=0
failed=0

# 統計總數
for mode in actual planned; do
    for ann in "$LABELS_ROOT"/*/  "$LABELS_ROOT"/*/"$mode"/annotations.json; do
        [ -f "$ann" ] && total=$((total + 1))
    done
done
total=$(find "$LABELS_ROOT" -path "*/actual/annotations.json" -o -path "*/planned/annotations.json" | wc -l)

echo "共找到 $total 個 annotations.json"
echo ""

for mode in actual planned; do
    for ann in $(find "$LABELS_ROOT" -path "*/${mode}/annotations.json" | sort); do
        done_count=$((done_count + 1))
        scene=$(basename "$(dirname "$(dirname "$ann")")")
        echo "=== [$done_count/$total] $scene / $mode ==="
        if $PYTHON "$EVALUATE" --labels "$ann"; then
            echo "  完成"
        else
            echo "  [錯誤] $ann"
            failed=$((failed + 1))
        fi
    done
done

echo ""
echo "全部完成。成功: $((total - failed))  失敗: $failed"
