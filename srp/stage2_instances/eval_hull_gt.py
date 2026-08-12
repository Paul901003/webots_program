#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""eval_hull_gt.py — 讀 build_hull_gt/build_gt 存的結果算指標(完全不重算)。

主判準(2D 重投影,取代 3D IoU 判找到):
  recall    = 找到的 GT 物體數 / 全放置物體數  (全遮擋 unocc=0 物體不排除,永遠命中不了=計為漏;只標註其數量)
  precision = 淨命中 hull 數 / hull 總數
3D IoU(輔助,只報命中 hull 的平均;配對沿用 2D 命中):
  hull-vs-hull = hull solid occ vs 命中GT 的 amodal-hull occ
  mesh         = hull solid occ vs 命中GT 的 mesh occ
輸出 per-scene csv + 分組彙總(n1/n3/n4/n5/occ3-5/stack3-5) csv。
用法: ./eval_hull_gt.py --root <method> [scene|group|(空=全部)]
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "data" / "eval"
GT_OUT = EVAL / "gt_reproj"


def iou3(a, b):
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else 0.0


def decide_hits(hulls, gt_eval, hit_iou):
    """從已存的 per_gt 平均重投影 IoU 用可調門檻重判命中+去重(不依賴 build 時的門檻)。"""
    avg = {k: {g: hulls[k]["per_gt"].get(g, {}).get("avg", 0.0) for g in gt_eval} for k in hulls}
    hit = {k: (max(avg[k], key=avg[k].get) if avg[k] and max(avg[k].values()) > hit_iou else None)
           for k in hulls}
    byg = defaultdict(list)
    for k in hulls:
        if hit[k]:
            byg[hit[k]].append(k)
    redundant = set()
    for g, ks in byg.items():
        for k in sorted(ks, key=lambda k: -avg[k][g])[1:]:
            redundant.add(k)
    return hit, redundant


def process(scene, root, hit_iou):
    hp = EVAL / root / scene / "hull_gt.json"
    if not hp.is_file():
        return None
    hj = json.loads(hp.read_text())
    hz = np.load(EVAL / root / scene / "hull_gt.npz")
    gj = json.loads((GT_OUT / scene / "gt.json").read_text())
    gz = np.load(GT_OUT / scene / "gt.npz")
    unocc = gj["unoccluded_views"]
    # ★ recall 分母 = 全部放置物體(不排除全遮擋)。無 ≥90% 可見視角(unocc=0)的物體無評估資料 → 永遠命中不了 → 計為漏。
    #   計為漏(沒找到就是沒找到);只標數量供參,不從分母拿掉。
    placed = list(gj["gt_objects"])
    gt_occluded = [g for g in placed if len(unocc.get(g, [])) == 0]

    hulls = hj["hulls"]
    n_hull = len(hulls)
    hit, redundant = decide_hits(hulls, placed, hit_iou)   # 用本次門檻重判(全遮擋物 avg=0 自然命中不了)
    found = set(); net_hits = 0
    iou_hull = []; iou_mesh = []
    rows = []
    for k, info in hulls.items():
        g = hit[k]; red = k in redundant
        ih = im = ""
        if g and not red and g in placed:
            found.add(g); net_hits += 1
            hocc = hz[f"hull_{k}"]
            am = gz.get(f"amodalhull_{g}"); me = gz.get(f"meshocc_{g}")
            if am is not None:
                ih = iou3(hocc, am); iou_hull.append(ih)
            if me is not None:
                im = iou3(hocc, me); iou_mesh.append(im)
        rows.append({"scene": scene, "hull": k, "hit": g or "", "redundant": "Y" if red else "",
                     "iou3d_hull": round(ih, 4) if ih != "" else "",
                     "iou3d_mesh": round(im, 4) if im != "" else ""})
    n_gt = len(placed)                          # 分母 = 全放置物體
    recall = len(found) / n_gt if n_gt else 0.0
    precision = net_hits / n_hull if n_hull else 0.0
    return {"scene": scene, "n_gt": n_gt, "n_gt_occluded": len(gt_occluded), "gt_occluded": ";".join(gt_occluded),
            "n_hull": n_hull, "found": len(found), "net_hits": net_hits,
            "recall": round(recall, 3), "precision": round(precision, 3),
            "iou3d_hull_mean": round(float(np.mean(iou_hull)), 3) if iou_hull else 0.0,
            "iou3d_mesh_mean": round(float(np.mean(iou_mesh)), 3) if iou_mesh else 0.0}, rows


def grp(s):
    return re.match(r"[a-z]+\d*", s).group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--root", required=True)
    ap.add_argument("--hit-iou", type=float, default=0.8, dest="hit_iou")
    args = ap.parse_args()
    base = EVAL / args.root
    th = f"{args.hit_iou:g}"
    if not args.targets:
        scenes = sorted(p.parent.name for p in base.glob("*_scene*/hull_gt.json"))
    else:
        scenes = []
        for a in args.targets:
            scenes.append(a) if "scene" in a else scenes.extend(
                p.parent.name for p in base.glob(f"{a}_scene*/hull_gt.json"))
        scenes = sorted(set(scenes))

    summ = []; detail = []
    for sc in scenes:
        r = process(sc, args.root, args.hit_iou)
        if r:
            summ.append(r[0]); detail.extend(r[1])
    if not summ:
        print("無結果"); return

    # per-scene csv
    with open(base / f"hull_gt_scenes_iou{th}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
    with open(base / f"hull_gt_detail_iou{th}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0].keys())); w.writeheader(); w.writerows(detail)

    # 分組彙總(micro)
    by = defaultdict(list)
    for r in summ:
        by[grp(r["scene"])].append(r)
    def agg(rows):
        fg = sum(r["found"] for r in rows); gg = sum(r["n_gt"] for r in rows)
        nh = sum(r["net_hits"] for r in rows); hh = sum(r["n_hull"] for r in rows)
        ih = [r["iou3d_hull_mean"] for r in rows if r["iou3d_hull_mean"] > 0]
        im = [r["iou3d_mesh_mean"] for r in rows if r["iou3d_mesh_mean"] > 0]
        return dict(n_scene=len(rows), sum_gt=gg, sum_found=fg, sum_hull=hh,
                    recall=round(fg / gg, 3) if gg else 0, precision=round(nh / hh, 3) if hh else 0,
                    iou3d_hull=round(np.mean(ih), 3) if ih else 0, iou3d_mesh=round(np.mean(im), 3) if im else 0)
    order = ["n1", "n3", "n4", "n5", "occ3", "occ4", "occ5", "stack3", "stack4", "stack5"]
    grows = []
    print(f"\n=== {args.root}  2D 重投影評估 (IoU>{th}命中; 分母=全放置物體) ===")
    print(f"{'組':>8}{'場':>5}{'GT':>5}{'找到':>5}{'recall':>8}{'prec':>7}{'3Dhull':>8}{'3Dmesh':>8}")
    for g in order + ["全體"]:
        rows = summ if g == "全體" else by.get(g, [])
        if not rows:
            continue
        a = agg(rows); a2 = {"group": g, **a}; grows.append(a2)
        print(f"{g:>8}{a['n_scene']:>5}{a['sum_gt']:>5}{a['sum_found']:>5}"
              f"{a['recall']:>8}{a['precision']:>7}{a['iou3d_hull']:>8}{a['iou3d_mesh']:>8}")
    with open(base / f"hull_gt_summary_iou{th}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(grows[0].keys())); w.writeheader(); w.writerows(grows)
    tocc = sum(r["n_gt_occluded"] for r in summ)
    print(f"\n→ {base}/hull_gt_*_iou{th}.csv  (其中無≥90%可見視角的 GT 共 {tocc} 個(已計入分母、計為漏))")


if __name__ == "__main__":
    main()
