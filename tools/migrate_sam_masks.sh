#!/bin/bash
# migrate_sam_masks.sh
# 把各場景的 data/captures/multi_n{N}/<scene>/sam_masks/ 內檔案
# 搬到 data/eval/grounded_sam_0.25_0.25_0.8/multi_n{N}/<scene>/，並刪除空的 sam_masks。
#
# 用法:
#   ./tools/migrate_sam_masks.sh           # 實際搬移
#   DRYRUN=1 ./tools/migrate_sam_masks.sh  # 只列出不搬移

set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_ROOT="$REPO/data/captures"
DST_ROOT="$REPO/data/eval/grounded_sam_0.25_0.25_0.8"
DRYRUN="${DRYRUN:-0}"

moved=0
empty=0
for sam in "$SRC_ROOT"/multi_n*/*/sam_masks; do
    [ -d "$sam" ] || continue
    scene_dir="$(dirname "$sam")"                       # .../multi_nN/<scene>
    scene="$(basename "$scene_dir")"                    # n3_scene0001
    group="$(basename "$(dirname "$scene_dir")")"       # multi_n3
    dst="$DST_ROOT/$group/$scene"

    n=$(find "$sam" -maxdepth 1 -type f | wc -l)
    if [ "$n" -eq 0 ]; then
        [ "$DRYRUN" = "1" ] || rmdir "$sam" 2>/dev/null
        empty=$((empty + 1))
        continue
    fi

    if [ "$DRYRUN" = "1" ]; then
        echo "[DRY] $group/$scene : $n 檔 → $dst"
        continue
    fi

    mkdir -p "$dst"
    mv "$sam"/* "$dst"/
    rmdir "$sam" 2>/dev/null
    echo "[MOVE] $group/$scene : $n 檔 → $dst"
    moved=$((moved + 1))
done

echo ""
if [ "$DRYRUN" = "1" ]; then
    echo "(dry-run，未實際搬移)"
else
    echo "完成。搬移 $moved 個場景，清除 $empty 個空目錄。"
fi
