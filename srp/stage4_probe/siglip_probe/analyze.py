#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""analyze.py — 讀 manifest + 三組特徵,算「同物 vs 異物」遮罩 cos 分佈與可分性(AUC)。

配對只在「同一場景內」(跨場不同實例不配對)。每組(clip_b32/siglip_b32/siglip_b16):
  同一場景兩兩遮罩 cos:
    intra   = 同 GT 物體(同物)          —— 希望高
    inter   = 不同 GT 物體(異物)        —— 希望低
    overcut = 同物且同視角(SAM 過切碎片) —— CLIP 失敗核心:與 inter 分不開
    xview   = 同物且跨視角
  指標:各分佈 mean±std + AUC(用 cos 當分數區分):
    auc_intra   = P(同物 cos > 異物 cos)
    auc_overcut = P(過切碎片 cos > 異物 cos)   ← 最關鍵:能不能把過切碎片認回同物
輸出 result.csv + hist.png(三組 intra/inter 分佈疊圖)。
用法: ./analyze.py   (需 webots_visual_hull:numpy+scipy+matplotlib)
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data" / "eval" / "_diag" / "siglip_probe"
GROUPS = [("clip_b32", "CLIP-B32"), ("siglip_b32", "SigLIP2-B32"), ("siglip_b16", "SigLIP2-B16")]


def auc(pos, neg):
    """AUC = P(pos>neg),tie 折半(rankdata 平均秩)。"""
    n1, n2 = len(pos), len(neg)
    if n1 == 0 or n2 == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def collect(feats, items):
    """回傳 dict of cos 陣列:intra/inter/overcut/xview。"""
    by_scene = defaultdict(list)
    for i, it in enumerate(items):
        if np.isfinite(feats[i, 0]):
            by_scene[it["scene"]].append(i)
    acc = {k: [] for k in ("intra", "inter", "overcut", "xview")}
    for sc, ids in by_scene.items():
        ids = np.array(ids)
        if len(ids) < 2:
            continue
        F = feats[ids]
        S = F @ F.T
        names = np.array([items[i]["gt_name"] for i in ids])
        views = np.array([items[i]["view"] for i in ids])
        iu = np.triu_indices(len(ids), k=1)
        cos = S[iu]
        same_obj = names[iu[0]] == names[iu[1]]
        same_view = views[iu[0]] == views[iu[1]]
        acc["intra"].append(cos[same_obj])
        acc["inter"].append(cos[~same_obj])
        acc["overcut"].append(cos[same_obj & same_view])
        acc["xview"].append(cos[same_obj & ~same_view])
    return {k: (np.concatenate(v) if v else np.array([])) for k, v in acc.items()}


def main():
    man = json.loads((OUT / "manifest.json").read_text())
    items = man["items"]
    print(f"manifest: {man['meta']['n_scenes']} 場 / {len(items)} 遮罩 "
          f"(來源 {man['meta']['mask_source']}, GT {man['meta']['gt']})\n")

    rows = []
    dists = {}
    for tag, label in GROUPS:
        fp = OUT / f"{tag}.npy"
        if not fp.is_file():
            print(f"[skip] {label}: 無 {fp.name}(先跑對應抽特徵腳本)")
            continue
        feats = np.load(fp)
        d = collect(feats, items)
        dists[tag] = d
        row = {"group": label,
               "n_intra": len(d["intra"]), "n_inter": len(d["inter"]),
               "n_overcut": len(d["overcut"]), "n_xview": len(d["xview"]),
               "intra_mean": round(float(d["intra"].mean()), 4) if len(d["intra"]) else None,
               "intra_std": round(float(d["intra"].std()), 4) if len(d["intra"]) else None,
               "inter_mean": round(float(d["inter"].mean()), 4) if len(d["inter"]) else None,
               "inter_std": round(float(d["inter"].std()), 4) if len(d["inter"]) else None,
               "overcut_mean": round(float(d["overcut"].mean()), 4) if len(d["overcut"]) else None,
               "xview_mean": round(float(d["xview"].mean()), 4) if len(d["xview"]) else None,
               "auc_intra": round(auc(d["intra"], d["inter"]), 4),
               "auc_overcut": round(auc(d["overcut"], d["inter"]), 4),
               "auc_xview": round(auc(d["xview"], d["inter"]), 4)}
        rows.append(row)

    if not rows:
        print("三組特徵都不存在,先跑 feat_clip.py / feat_siglip.py"); return

    # 主控台表
    print(f"{'組':<14}{'同物µ':>8}{'異物µ':>8}{'過切µ':>8}{'跨視µ':>8}"
          f"{'AUC同物':>9}{'AUC過切':>9}{'AUC跨視':>9}")
    for r in rows:
        print(f"{r['group']:<14}{r['intra_mean']:>8}{r['inter_mean']:>8}{r['overcut_mean']:>8}"
              f"{r['xview_mean']:>8}{r['auc_intra']:>9}{r['auc_overcut']:>9}{r['auc_xview']:>9}")
    print("\n判讀:AUC過切(把 SAM 過切碎片認回同物、與異物分開)是關鍵。")
    print("     0.5=完全分不開(等同 CLIP 失敗);越接近 1 越能分。三組同基準,差距=配方/patch 效果。")

    with open(OUT / "result.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # 直方圖:每組一列,intra(綠) vs inter(紅) 疊,標 overcut 分佈(藍虛線)
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(7, 2.6 * n), squeeze=False)
    bins = np.linspace(-0.2, 1.0, 61)
    for ax, (tag, label) in zip(axes[:, 0], [g for g in GROUPS if g[0] in dists]):
        d = dists[tag]
        ax.hist(d["inter"], bins=bins, density=True, alpha=0.5, color="#d64545", label="inter (diff obj)")
        ax.hist(d["intra"], bins=bins, density=True, alpha=0.5, color="#3ca03c", label="intra (same obj)")
        if len(d["overcut"]):
            ax.hist(d["overcut"], bins=bins, density=True, histtype="step",
                    color="#3060c0", lw=1.6, label="overcut frag")
        rr = next(r for r in rows if r["group"] == label)
        ax.set_title(f"{label}   AUC_intra={rr['auc_intra']}   AUC_overcut={rr['auc_overcut']}", fontsize=10)
        ax.set_xlabel("cosine"); ax.set_ylabel("density"); ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT / "hist.png", dpi=110)
    print(f"\n→ {OUT/'result.csv'}\n→ {OUT/'hist.png'}")


if __name__ == "__main__":
    main()
