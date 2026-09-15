#!/bin/bash
# run_mobilesamv2_all.sh — 對 captures_fast 全場景跑 MobileSAMv2 遮罩(對照 sam_only),按組平行。
# 輸出結構同 sam_only → data/eval/mobilesamv2_fast/<場景>/<view>/(masks/ + overlay.png + meta.txt)。
# mobilesamv2_seg.py 內建跳過已完成(overlay.png 存在);自我隔離進 systemd scope 防 oomd 黑屏。
#
# 用法:
#   ./mobilesamv2/run_mobilesamv2_all.sh                # 全 10 組
#   ./mobilesamv2/run_mobilesamv2_all.sh occ3 stack3    # 指定組
#   JOBS=4 ./mobilesamv2/run_mobilesamv2_all.sh         # 併發組數(預設 3;單行程 GPU ~2.5GB)
#   FORCE=1 ./mobilesamv2/run_mobilesamv2_all.sh        # 重做
# env: JOBS(預設3) CAPTURES_ROOT(預設 data/captures_fast) FORCE

set -u

# ── 自我保護:重啟進獨立 systemd scope(脫離終端 cgroup + oomd 永不殺),避免平行吃記憶體→黑屏 ──
if [ -z "${_MV2_SCOPED:-}" ] && command -v systemd-run >/dev/null 2>&1; then
    exec systemd-run --user --scope -p ManagedOOMPreference=omit \
        --setenv=_MV2_SCOPED=1 \
        --setenv=JOBS="${JOBS:-}" --setenv=CAPTURES_ROOT="${CAPTURES_ROOT:-}" --setenv=FORCE="${FORCE:-}" \
        --setenv=MV2_ENCODER="${MV2_ENCODER:-}" --setenv=SAM_OUT_ROOT="${SAM_OUT_ROOT:-}" \
        -- "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SEG="$SCRIPT_DIR/mobilesamv2_seg.py"
export CAPTURES_ROOT="${CAPTURES_ROOT:-$REPO_ROOT/data/captures_fast}"
export MV2_ENCODER="${MV2_ENCODER:-efficientvit_l2}"   # 編碼器:efficientvit_l2(預設) / tiny_vit / sam_vit_h
case "$MV2_ENCODER" in
    efficientvit_l2) _out="mobilesamv2_fast";;          # 保留原名(向後相容)
    *)               _out="mobilesamv2_${MV2_ENCODER}_fast";;
esac
export SAM_OUT_ROOT="${SAM_OUT_ROOT:-$REPO_ROOT/data/eval/$_out}"
export FORCE="${FORCE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
JOBS="${JOBS:-3}"

[ -x "$SEG" ] || { echo "找不到可執行的 $SEG"; exit 1; }
[ -d "$CAPTURES_ROOT" ] || { echo "找不到 captures: $CAPTURES_ROOT"; exit 1; }

groups=("$@")
# 預設全 10 組;大組(view 多)排前面,xargs 自然負載平衡
[ "${#groups[@]}" -eq 0 ] && groups=(n1 n3 n4 n5 occ3 occ4 occ5 stack3 stack4 stack5)

echo "MobileSAMv2 遮罩 (encoder=$MV2_ENCODER)  組: ${groups[*]}  併發 $JOBS"
echo "  來源: $CAPTURES_ROOT"
echo "  輸出: $SAM_OUT_ROOT/<場景>/<view>/  (跳過已完成; FORCE=$FORCE)"

start=$(date +%s)
# 每組一個 mobilesamv2_seg 行程;xargs -P 同時跑 JOBS 組。各自載 YOLO+efficientvit+decoder(~2.5GB GPU)。
printf '%s\n' "${groups[@]}" | xargs -P "$JOBS" -I{} "$SEG" {}
rc=$?

echo ""
echo "MobileSAMv2 全部結束(耗時 $(( $(date +%s) - start ))s, rc=$rc)。輸出 → $SAM_OUT_ROOT"
exit "$rc"
