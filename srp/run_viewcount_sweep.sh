#!/bin/bash
# run_viewcount_sweep.sh — 視角數 sweep:6/8/10/12/34 各跑完整 Stage1→Stage2→eval,全場景組。
#
# 單一變因=視角數。挑選一律走 A-3(select_counts --only-latest → selected_n{N});
# allow_miss 一律 20%(--miss-frac 0.2,含 34=全視角);Stage1/Stage2 用同一批視角。
# 每個 N 存獨立 root data/eval/srp_hull_v{N}/{hull,instances,d1d2.csv}。最後彙總各 N 對照表。
# 共用(與 N 無關,缺才補):A-3 挑選檔、arm 剪影 srp_arm_masks、GT amodal、GT 快取 gt_hull_cache_fast。
# 序列跑(單行程 GPU,不並行避免 OOM);自帶 oomd 隔離 scope(不黑屏)。
#
# 用法:  ./srp/run_viewcount_sweep.sh                 # 全 10 組, N=6 8 10 12 34
#        ./srp/run_viewcount_sweep.sh occ3 stack3     # 指定組
# env:   COUNTS(預設 "6 8 10 12 34")  MISS_FRAC(預設 0.2)  IOU(預設 0.25)  AGREE(預設 0.5)  VOXEL(預設 0.005)

set -u
# ── 自我保護:重啟進獨立 systemd scope(脫離終端 cgroup + oomd 永不殺)──
if [ -z "${_SWEEP_SCOPED:-}" ] && command -v systemd-run >/dev/null 2>&1; then
    exec systemd-run --user --scope -p ManagedOOMPreference=omit \
        --setenv=_SWEEP_SCOPED=1 --setenv=COUNTS="${COUNTS:-}" --setenv=MISS_FRAC="${MISS_FRAC:-}" \
        --setenv=IOU="${IOU:-}" --setenv=AGREE="${AGREE:-}" --setenv=VOXEL="${VOXEL:-}" \
        -- "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
export CAPTURES_ROOT="$REPO/data/captures_fast"
export SAM_ROOT="$REPO/data/eval/sam_only_fast"
export ARM_MASK_ROOT="$REPO/data/eval/srp_arm_masks"
export GT_CACHE="$REPO/data/eval/gt_hull_cache_fast"

S1="$REPO/srp/stage1_hull/run_scene.py"
S2="$REPO/srp/stage2_instances/associate.py"
S3="$REPO/srp/stage2_instances/eval.py"
ARM="$REPO/srp/stage1_hull/arm_silhouette.py"
AMODAL="$REPO/tools/generate_amodal_masks.py"
SELDIR="$REPO/controllers/ycb_viewpoint_validator"
VP_PY="/home/cho/.pyenv/versions/webots_visual_hull/bin/python3"

COUNTS="${COUNTS:-6 8 10 12 34}"
MISS_FRAC="${MISS_FRAC:-0.2}"
IOU="${IOU:-0.25}"
AGREE="${AGREE:-0.5}"
VOXEL="${VOXEL:-0.005}"

groups=("$@")
[ "${#groups[@]}" -eq 0 ] && groups=(n1 n3 n4 n5 occ3 occ4 occ5 stack3 stack4 stack5)

# 收集場景名
scenes=()
for g in "${groups[@]}"; do
    for d in "$CAPTURES_ROOT/multi_$g/${g}_scene"*; do
        [ -d "$d" ] && scenes+=("$(basename "$d")")
    done
done
echo "視角數 sweep: COUNTS=[$COUNTS]  miss_frac=$MISS_FRAC  ${#scenes[@]} 場景  組: ${groups[*]}"
[ "${#scenes[@]}" -eq 0 ] && { echo "無場景"; exit 1; }

# ── 前置 0:確保 A-3 挑選檔存在(非 34 的 N)──
need_sel=()
for N in $COUNTS; do
    [ "$N" = 34 ] && continue
    [ -f "$REPO/data/viewpoints/selected_viewpoints_multi_n${N}_x+035.json" ] || need_sel+=("$N")
done
if [ "${#need_sel[@]}" -gt 0 ]; then
    echo ""; echo "===== 前置: A-3 挑選 selected_n{${need_sel[*]}} (--only-latest) ====="
    ( cd "$SELDIR" && /usr/bin/python3 select_counts.py --multi --x-offset 0.35 --only-latest --counts "${need_sel[@]}" ) \
        | tail -8
fi

# ── 前置 1:FK 手臂剪影(缺才補,全 group)──
echo ""; echo "===== 前置: FK 手臂剪影(跳過已完成) ====="
"$ARM" "${groups[@]}" 2>&1 | grep -viE 'warning|futurewarning|weights_only|torch.load|state_dict' | tail -5

# ── 前置 2:GT amodal(eval 需要,缺才補)──
need_amodal=()
for s in "${scenes[@]}"; do
    [ -f "$REPO/data/labels/$s/amodal/annotations.json" ] || need_amodal+=("$s")
done
if [ "${#need_amodal[@]}" -gt 0 ]; then
    echo ""; echo "===== 前置: GT amodal(缺 ${#need_amodal[@]} 場景要產) ====="
    "$AMODAL" "${need_amodal[@]}" 2>&1 | grep -viE 'warning|futurewarning|weights_only|torch.load|state_dict' | tail -5
fi

# ── 主迴圈:每個視角數各跑 Stage1 → Stage2 → eval ──
roots=()
for N in $COUNTS; do
    ROOT="srp_hull_v${N}"
    roots+=("$ROOT")
    if [ "$N" = 34 ]; then NV=(); else NV=(--num-views "$N"); fi
    echo ""; echo "########## 視角數 N=$N  →  data/eval/$ROOT/ ##########"

    echo "----- ①/3 Stage1 雕殼(miss_frac=$MISS_FRAC) -----"
    "$S1" "${scenes[@]}" --voxel "$VOXEL" --miss-frac "$MISS_FRAC" "${NV[@]}" --root "$ROOT"

    echo "----- ②/3 Stage2 關聯(同一批視角) -----"
    "$S2" "${scenes[@]}" --agree-frac "$AGREE" "${NV[@]}" --root "$ROOT"

    echo "----- ③/3 eval vs GT → $ROOT/d1d2.csv -----"
    "$S3" "${scenes[@]}" --iou "$IOU" --root "$ROOT"
done

# ── 彙總:各視角數對照表 ──
echo ""; echo "===== 彙總: 各視角數對照 ====="
"$VP_PY" - "$REPO" "$IOU" "${roots[@]}" <<'PY'
import csv, sys
from pathlib import Path
repo, iou, *roots = sys.argv[1], sys.argv[2], *sys.argv[3:]
out = Path(repo) / "data" / "eval" / "viewcount_sweep_summary.csv"
rows = []
print(f"\n{'N視角':>6}{'場景':>6}{'recall':>9}{'precision':>11}{'mIoU':>8}{'found/GT':>12}{'幻影':>7}")
print("-" * 59)
for r in roots:
    N = r.replace("srp_hull_v", "")
    csvf = Path(repo) / "data" / "eval" / r / "d1d2.csv"
    if not csvf.is_file():
        print(f"{N:>6}  (無 d1d2.csv,略過)"); continue
    d = list(csv.DictReader(open(csvf)))
    if not d: continue
    n = len(d)
    rec = sum(float(x["recall"]) for x in d) / n
    prec = sum(float(x["precision"]) for x in d) / n
    miou = sum(float(x["mean_iou"]) for x in d) / n
    tg = sum(int(x["n_gt"]) for x in d); tf = sum(int(x["found_t"]) for x in d)
    tph = sum(int(x["phantom"]) for x in d)
    print(f"{N:>6}{n:>6}{rec:>9.3f}{prec:>11.3f}{miou:>8.3f}{f'{tf}/{tg}':>12}{tph:>7}")
    rows.append({"n_views": N, "n_scenes": n, "recall": round(rec, 3),
                 "precision": round(prec, 3), "mean_iou": round(miou, 3),
                 "found": tf, "gt": tg, "phantom": tph})
if rows:
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n→ {out}")
PY

echo ""; echo "完成。各 N 明細: data/eval/srp_hull_v{N}/d1d2.csv;對照表: data/eval/viewcount_sweep_summary.csv"
