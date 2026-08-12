#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""missed_cause.py — 全 10 組漏物成因分類表。

讀 data/eval/srp_hull_semcluster_clip/missed_objects_iou{IOU}.csv 的每個漏物,
用和建置一致的分群(複用 cluster_purity_eval.cluster_scene)+ hull_gt 重投影,
標成成因並列出所有原始信號(不藏數字,label 只是推導)。

成因(依序判定):
  語意牆      : 漏物「主群」含另一物 ≥2 遮罩且 ≥25%(語意把它和別物併同群)
  無hull/被吸收 : 最貼它的 hull 重投影 IoU < 0.05(沒重建出來)
  遮擋        : 無遮擋視角 ≤ 4(底層被壓、hull 約束不足)
  hull過估/融合 : best hull 重投影面積 ≥ 1.5× modal(hull 吞鄰居)
  hull過小/碎裂 : best hull 重投影面積 ≤ 0.5× modal
  無遮罩/未框到 : SAM 沒有任何遮罩歸屬此物
  其他        : 都不是

輸出 data/eval/_diag/cluster_purity/missed_cause_iou{IOU}.csv + 終端彙總。
用法: ./missed_cause.py [--iou 0.6]
"""
import sys, json, csv, argparse
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_purity_eval import cluster_scene, EVAL, ROOT, load_modal   # noqa

OUTDIR = EVAL / "_diag" / "cluster_purity"
P_CNT, P_FRAC = 2, 0.25


def analyze(scene, g, cached):
    ref, clof, gtof, F, gj, modal = cached
    unocc = gj["unoccluded_views"]
    g_refs = [r for r in ref if gtof[r] == g]
    row = {"scene": scene, "group": scene.split("_")[0], "object": g,
           "n_masks": len(g_refs), "unocc": len(unocc.get(g, []))}
    if not g_refs:
        row["label"] = "無遮罩/未框到"; return row
    home = Counter(clof[r] for r in g_refs).most_common(1)[0][0]
    members = [r for r in ref if clof[r] == home]
    comp = Counter(gtof[r] for r in members if gtof[r] is not None)
    others = [(o, c) for o, c in comp.items() if o != g]
    partner, pc = (max(others, key=lambda x: x[1]) if others else (None, 0))
    labeled = sum(comp.values())
    p_frac = pc / labeled if labeled else 0.0
    idx = {r: i for i, r in enumerate(ref)}
    cos_gp = None
    if partner:
        Fg = F[[idx[r] for r in ref if gtof[r] == g]]
        Fp = F[[idx[r] for r in ref if gtof[r] == partner]]
        cos_gp = round(float((Fg @ Fp.T).mean()), 3)
    # over-cover
    over, best_iou, best_hull = None, 0.0, None
    hj = json.loads((EVAL / ROOT / scene / "hull_gt.json").read_text())
    for k, info in hj["hulls"].items():
        a = info["per_gt"].get(g, {}).get("avg", 0)
        if a > best_iou:
            best_iou, best_hull = a, k
    hz = np.load(EVAL / ROOT / scene / "hull_gt.npz")
    if best_hull is not None:
        rr = []
        for vn in unocc.get(g, []):
            rk = f"reproj_{best_hull}__{vn}"
            if rk in hz.files and (g, vn) in modal:
                ma = int(modal[(g, vn)].sum())
                if ma > 0:
                    rr.append(int(hz[rk].sum()) / ma)
        if rr:
            over = float(np.mean(rr))
    sem_wall = partner is not None and pc >= P_CNT and p_frac >= P_FRAC
    if sem_wall:
        label = "語意牆"
    elif best_iou < 0.05:
        label = "無hull/被吸收"
    elif len(unocc.get(g, [])) <= 4:
        label = "遮擋"
    elif over is not None and over >= 1.5:
        label = "hull過估/融合"
    elif over is not None and over <= 0.5:
        label = "hull過小/碎裂"
    else:
        label = "其他"
    row.update({"label": label, "partner": partner, "partner_cnt": pc,
                "p_frac": round(p_frac, 2), "cos_partner": cos_gp,
                "best_hull": best_hull, "best_iou": round(best_iou, 3),
                "overcover": round(over, 2) if over is not None else None})
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--iou", default="0.6"); a = ap.parse_args()
    src = EVAL / ROOT / f"missed_objects_iou{a.iou}.csv"
    miss = defaultdict(list)
    for r in csv.DictReader(open(src)):
        miss[r["scene"]].append(r["missed_object"])
    rows = []
    for i, sc in enumerate(sorted(miss)):
        try:
            cached = cluster_scene(sc)
        except Exception:
            cached = None
        for g in miss[sc]:
            if cached is None:
                rows.append({"scene": sc, "group": sc.split("_")[0], "object": g, "label": "(分群失敗略過)"})
                continue
            try:
                rows.append(analyze(sc, g, cached))
            except Exception as e:
                rows.append({"scene": sc, "group": sc.split("_")[0], "object": g, "label": f"(err:{e})"})
        if (i + 1) % 30 == 0:
            print(f"  ...{i+1}/{len(miss)} 場", flush=True)
    keys = ["scene", "group", "object", "label", "n_masks", "unocc", "partner", "partner_cnt",
            "p_frac", "cos_partner", "best_hull", "best_iou", "overcover"]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"missed_cause_iou{a.iou}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n→ {out}  ({len(rows)} 漏物)")
    print("\n【成因統計】")
    for lab, c in Counter(r["label"] for r in rows).most_common():
        print(f"  {lab:<16} {c}")
    print("\n【組 × 成因】")
    bg = defaultdict(Counter)
    for r in rows:
        bg[r["group"]][r["label"]] += 1
    labs = [l for l, _ in Counter(r["label"] for r in rows).most_common()]
    print(f"  {'組':<7}" + "".join(f"{l[:8]:>10}" for l in labs))
    for g in sorted(bg):
        print(f"  {g:<7}" + "".join(f"{bg[g][l]:>10}" for l in labs))


if __name__ == "__main__":
    main()
