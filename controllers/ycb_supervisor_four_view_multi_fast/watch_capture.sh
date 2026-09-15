#!/bin/bash
# watch_capture.sh — 監控平衡拍攝(run_capture_balanced.sh)。
# driver 停止(完成/中斷)或停滯(webots 卡住)就 exit → 通知 assistant 來查+續跑。
# 用法(背景掛):watch_capture.sh   (每 5 分查一次;停滯門檻 15 分)
set -u
cd "$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")" || exit 1
PY=/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
CHK=srp/scene_gen/check_capture.py
INTERVAL="${WATCH_INTERVAL:-300}"     # 5 分查一次
STALL="${WATCH_STALL_MIN:-15}"        # 15 分沒新檔=卡住
MAX_SEC="${WATCH_MAX_SEC:-1380}"      # ~23 分後心跳 exit(趕在 harness 30 分殺之前,好重掛)

_start=$(date +%s)
while true; do
    sleep "$INTERVAL"
    if [ $(( $(date +%s) - _start )) -ge "$MAX_SEC" ]; then
        echo "===== 心跳 $(date +%H:%M):driver 仍在跑,重掛看門狗 ====="
        "$PY" "$CHK" 2>&1 | grep -viE "warn|pyembree" | grep -E "^\[|合計|停滯|續跑"
        exit 7   # 7 = 心跳(非事件),assistant 收到就重掛
    fi
    if ! pgrep -f "run_capture_balanced.sh" >/dev/null 2>&1; then
        echo "===== driver 已停止(完成或中斷)$(date +%H:%M) ====="
        "$PY" "$CHK" 2>&1 | grep -viE "warn|pyembree"
        exit 0
    fi
    if ! "$PY" "$CHK" --stall-min "$STALL" >/tmp/_watch_chk.out 2>&1; then
        echo "===== 偵測到停滯 $(date +%H:%M) ====="
        grep -viE "warn|pyembree" /tmp/_watch_chk.out
        echo "→ webots 狀態:"; pgrep -af webots | head
        exit 2
    fi
done
