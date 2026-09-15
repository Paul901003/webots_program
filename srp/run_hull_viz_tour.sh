#!/usr/bin/env bash
# run_hull_viz_tour.sh — 依「n3 n4 n5 occ stack」順序，逐場景用 hull_viz.wbt 可視化某方法的實例分離結果。
# 關掉 Webots 視窗 → 自動開下一個場景。Ctrl+C 中止整趟。
#
# 用法:
#   ./srp/run_hull_viz_tour.sh [方法root] [從哪個場景續跑]
#     方法root  預設 srp_hull_cg_solid；可填 srp_hull_semvote / srp_hull_semcluster /
#               srp_hull_sempaper / srp_hull_cg(表面殼) / srp_hull_v12(基礎hull)
#     續跑      給場景名(如 occ3_scene0005)→ 從該場景開始，前面略過(方便中斷後接續)
#
# 例:
#   ./srp/run_hull_viz_tour.sh srp_hull_cg_solid              # cg(實心)整趟
#   ./srp/run_hull_viz_tour.sh srp_hull_semvote stack3_scene0001   # 命名法，從 stack3 開始
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

ROOT="${1:-srp_hull_cg_solid}"           # 要看的方法
RESUME="${2:-}"                          # 從哪個場景續跑(空=從頭)
SHOW_GT=1                                # 1=同時擺真實 YCB 對照
WORLD="worlds/hull_viz.wbt"

# Ctrl+C 要中止「整個 tour」。預設按 Ctrl+C 只把 SIGINT 傳給前景的 webots,
# webots 結束後 bash 迴圈仍會跑下一場 → 這裡攔 SIGINT/SIGTERM 直接結束整支腳本。
_ABORT=0
trap '_ABORT=1; echo; echo "⏹ 已中止整個 tour"; exit 130' INT TERM

# cg 原生是表面殼 → 自動加 surface token；其餘實心方法不用
EXTRA=""
[ "$ROOT" = "srp_hull_cg" ] && EXTRA="surface"

# 場景組順序:n3 n4 n5 → occ3/4/5 → stack3/4/5（不含單物體 n1）
# 注意:變數名不可用 GROUPS(bash 保留變數=使用者群組id，賦值會被忽略)
SCENE_GROUPS=(n3 n4 n5 occ3 occ4 occ5 stack3 stack4 stack5)

# 依序蒐集「該方法真的有 instances.npz」的場景
SCENES=()
for g in "${SCENE_GROUPS[@]}"; do
  while IFS= read -r p; do
    [ -n "$p" ] && SCENES+=("$(basename "$(dirname "$p")")")
  done < <(ls -d "data/eval/$ROOT/${g}_scene"*/instances.npz 2>/dev/null | sort)
done

N=${#SCENES[@]}
if [ "$N" -eq 0 ]; then
  echo "✗ data/eval/$ROOT 下找不到任何 instances.npz。方法名對嗎？"
  echo "  可選: srp_hull_cg_solid / srp_hull_cg / srp_hull_semvote / srp_hull_semcluster / srp_hull_sempaper / srp_hull_v12"
  exit 1
fi

echo "== hull_viz tour =="
echo "   方法 : $ROOT ${EXTRA:+(surface 表面殼)}"
echo "   場景 : $N 個（順序 n3→n4→n5→occ→stack）"
echo "   操作 : 看完一個場景就關掉 Webots 視窗 → 自動開下一個；Ctrl+C 中止整趟"
[ -n "$RESUME" ] && echo "   續跑 : 從 $RESUME 開始"
echo

started=0
[ -z "$RESUME" ] && started=1
for i in "${!SCENES[@]}"; do
  sc="${SCENES[$i]}"
  # 續跑:還沒遇到指定場景前一律略過
  if [ "$started" -eq 0 ]; then
    if [ "$sc" = "$RESUME" ]; then started=1; else continue; fi
  fi
  printf '── [%d/%d] %s  (root=%s)\n' "$((i+1))" "$N" "$sc" "$ROOT"
  SRP_VIZ_ARGS="$sc $SHOW_GT $ROOT $EXTRA" webots --batch "$WORLD"
  # webots 為前景執行，關掉視窗才回到這裡 → 繼續下一場
  [ "$_ABORT" = 1 ] && break        # 保險:若已收到 Ctrl+C 就不再開下一場
done

echo
echo "✓ tour 結束（$ROOT，共 $N 場）"
