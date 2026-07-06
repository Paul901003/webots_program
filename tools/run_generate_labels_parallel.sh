#!/bin/bash
# run_generate_labels_parallel.sh — 平行生成 GT 標籤。
# 渲染本來就用 GPU(pyrender EGL → NVIDIA);瓶頸在 CPU 端(每場景重載 mesh+抽遮罩+寫檔),
# 故用 xargs -P 同時跑多個場景吃滿 GPU/CPU。逐場景呼叫 generate_labels.py。
#
# 用法:
#   ./run_generate_labels_parallel.sh                     # captures_fast 下全部場景
#   ./run_generate_labels_parallel.sh n3 occ3 stack3      # 只這些前綴
#   JOBS=6 ./run_generate_labels_parallel.sh              # 併發 6(預設 4)
#   MODE=both ./run_generate_labels_parallel.sh           # actual+planned(預設 actual)
#   FORCE=1 ./run_generate_labels_parallel.sh             # 重做(不跳過已完成)
# env: JOBS(併發,預設4) MODE(actual|planned|both,預設actual)
#      CAPTURES_ROOT(預設 data/captures_fast) LABELS(預設 data/labels) FORCE

set -u

# ── 自我保護:把自己重啟進「獨立 systemd scope」(脫離終端 cgroup + 豁免 oomd) ──
# 平行 GT 會拉高記憶體壓力,systemd-oomd 會連終端 cgroup 一起殺 → 黑屏。
# 放進自己的 scope 後:①被殺也不波及終端/桌面 ②oomd 不因壓力殺它。_GT_SCOPED 防無限重啟。
if [ -z "${_GT_SCOPED:-}" ] && command -v systemd-run >/dev/null 2>&1; then
    exec systemd-run --user --scope -p ManagedOOMPreference=omit \
        --setenv=_GT_SCOPED=1 \
        --setenv=JOBS="${JOBS:-}" --setenv=MODE="${MODE:-}" \
        --setenv=CAPTURES_ROOT="${CAPTURES_ROOT:-}" --setenv=LABELS="${LABELS:-}" \
        --setenv=FORCE="${FORCE:-}" \
        -- "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PY="/home/cho/.pyenv/versions/3.10.10/bin/python3"
GEN="$SCRIPT_DIR/generate_labels.py"
CAPTURES_ROOT="${CAPTURES_ROOT:-$REPO_ROOT/data/captures_fast}"
LABELS="${LABELS:-$REPO_ROOT/data/labels}"
MODE="${MODE:-actual}"
JOBS="${JOBS:-4}"
FORCE="${FORCE:-0}"
FILTERS=("$@")

[ -x "$PY" ]  || { echo "找不到 python: $PY"; exit 1; }
[ -f "$GEN" ] || { echo "找不到 $GEN"; exit 1; }
[ -d "$CAPTURES_ROOT" ] || { echo "找不到 captures: $CAPTURES_ROOT"; exit 1; }

is_done() {  # $1=場景名;依 MODE 檢查對應 annotations.json
    case "$MODE" in
        actual)  [ -f "$LABELS/$1/actual/annotations.json" ];;
        planned) [ -f "$LABELS/$1/planned/annotations.json" ];;
        both)    [ -f "$LABELS/$1/actual/annotations.json" ] && [ -f "$LABELS/$1/planned/annotations.json" ];;
        *)       return 1;;
    esac
}

# 收集待處理 manifest(套用前綴 filter + 跳過已完成)
manifests=()
skipped=0
while IFS= read -r M; do
    scene="$(basename "$(dirname "$M")")"
    if [ "${#FILTERS[@]}" -gt 0 ]; then
        keep=0
        for t in "${FILTERS[@]}"; do case "$scene" in ${t}*) keep=1; break;; esac; done
        [ "$keep" = 1 ] || continue
    fi
    if [ "$FORCE" != "1" ] && is_done "$scene"; then
        skipped=$((skipped + 1)); continue
    fi
    manifests+=("$M")
done < <(find "$CAPTURES_ROOT" -name scene_manifest.json 2>/dev/null | sort)

TOTAL="${#manifests[@]}"
echo "待生成 GT: $TOTAL 場景  (已完成跳過 $skipped)  併發 $JOBS  mode=$MODE"
echo "  來源: $CAPTURES_ROOT   輸出: $LABELS"
[ "$TOTAL" -eq 0 ] && { echo "沒有待處理場景(FORCE=1 可強制重做)"; exit 0; }

start=$(date +%s)
# 每個 manifest 一個 generate_labels;xargs -P 併發。--no-arm 不設(fast 拍攝有真手臂)。
printf '%s\0' "${manifests[@]}" | xargs -0 -P "$JOBS" -I{} \
    "$PY" "$GEN" --manifest "{}" --output "$LABELS" --mode "$MODE"
rc=$?
elapsed=$(( $(date +%s) - start ))

echo ""
echo "全部結束(耗時 ${elapsed}s)。已處理 $TOTAL 場景 → $LABELS"
exit "$rc"
