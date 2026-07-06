#!/bin/bash
# run_sam_only_all.sh — 對 captures_fast 全部場景跑 sam_only(SAM 全自動遮罩),按「組」平行。
# sam_only 內建跳過已完成(該 view 的 overlay.png 存在就跳);自我隔離進 systemd scope 防 oomd 黑屏。
#
# 用法:
#   ./sam_only/run_sam_only_all.sh                  # 全部 10 組
#   ./sam_only/run_sam_only_all.sh occ3 stack3      # 指定組
#   JOBS=2 ./sam_only/run_sam_only_all.sh           # 併發組數(預設 3)
#   FORCE=1 ./sam_only/run_sam_only_all.sh          # 重做(忽略已存在)
# env: JOBS(預設3) CAPTURES_ROOT(預設 data/captures_fast) FORCE

set -u

# ── 自我保護:重啟進獨立 systemd scope(脫離終端 cgroup + oomd 永不殺),避免平行吃記憶體→黑屏 ──
if [ -z "${_SAM_SCOPED:-}" ] && command -v systemd-run >/dev/null 2>&1; then
    exec systemd-run --user --scope -p ManagedOOMPreference=omit \
        --setenv=_SAM_SCOPED=1 \
        --setenv=JOBS="${JOBS:-}" --setenv=CAPTURES_ROOT="${CAPTURES_ROOT:-}" --setenv=FORCE="${FORCE:-}" \
        --setenv=SAM_OUT_ROOT="${SAM_OUT_ROOT:-}" \
        -- "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SAM="$SCRIPT_DIR/sam_only.py"
export CAPTURES_ROOT="${CAPTURES_ROOT:-$REPO_ROOT/data/captures_fast}"
export SAM_OUT_ROOT="${SAM_OUT_ROOT:-$REPO_ROOT/data/eval/sam_only_fast}"   # 另存新根,不動舊 sam_only/
export FORCE="${FORCE:-0}"
# 每個 SAM 行程峰值 ~4GB GPU;11.7GB GPU → JOBS=2(8GB)安全,JOBS=3 會 OOM。要全序列設 JOBS=1。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # 減少碎片
JOBS="${JOBS:-2}"

[ -x "$SAM" ] || { echo "找不到可執行的 $SAM"; exit 1; }
[ -d "$CAPTURES_ROOT" ] || { echo "找不到 captures: $CAPTURES_ROOT"; exit 1; }

groups=("$@")
# 預設全 10 組;大組(view 多)排前面,xargs 先啟動、自然負載平衡
[ "${#groups[@]}" -eq 0 ] && groups=(n1 n3 n4 n5 occ3 occ4 occ5 stack3 stack4 stack5)

echo "SAM (sam_only) 全遮罩  組: ${groups[*]}  併發 $JOBS"
echo "  來源: $CAPTURES_ROOT"
echo "  輸出: $SAM_OUT_ROOT/<場景>/<view>/  (跳過已完成; FORCE=$FORCE)"

start=$(date +%s)
# 每組一個 sam_only 行程;xargs -P 同時跑 JOBS 組。各自載一份 SAM(~2.5GB GPU)。
printf '%s\n' "${groups[@]}" | xargs -P "$JOBS" -I{} "$SAM" {}
rc=$?

echo ""
echo "SAM 全部結束(耗時 $(( $(date +%s) - start ))s, rc=$rc)。輸出 → $SAM_OUT_ROOT"
exit "$rc"
