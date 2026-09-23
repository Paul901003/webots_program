#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""gtmask_cluster_dump.py — 把 gtmask_cluster_compare 的分群【逐場攤開來看】:每群裝了哪些物體的幾個視角。

配合 RESULT_gtmask_cluster_compare.md 的彙總數字,這支給的是可目視核對的細節:
  - 每場、每種特徵:分出幾群、每群的組成(物體 × 該物體進來的視角數)
  - 標出兩種錯法:過切(同一物體散在 >1 群)、混群(同一群含 >1 物體)
門檻用各特徵在「排除 GEX」下的最佳值(見 RESULT_gtmask_cluster_compare.md),逐場一致。

★ 基準同 gtmask_cluster_compare:GT 完美遮罩、60 場 stack3/4/5、12 視角 A-3 selected。
  【不可】與 srp_hull_cluster_mv2_noarm(SAM 遮罩、303 場)相比。

拿什麼:data/eval/gt_mask_feats_v2/*.npz。排除項:GEX 標註但不濾除(逐場可見其影響)。
用法  : ./gtmask_cluster_dump.py [--scenes stack3_scene0001 ...]  輸出 RESULT_gtmask_cluster_dump.md
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import mask_clip_cluster as MC   # noqa: E402  只讀 F_BG 快取

FEAT = REPO / "data" / "eval" / "gt_mask_feats_v2"
GEX = {"skillet_lid", "windex_bottle", "colored_wood_blocks", "dice"}
# 各特徵在「排除 GEX」下的最佳門檻(來源:RESULT_gtmask_cluster_compare.md)
BEST = {"clip_debias": 0.35, "clip_raw": 0.15, "dino_cw": 0.40, "dino_ce": 0.40}


def l2(F):
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


def feats(z):
    bg = MC.F_BG.astype(np.float64)
    c = z["clip"].astype(np.float64)
    return {"clip_debias": l2(c - (c @ bg)[:, None] * bg[None, :]),
            "clip_raw": l2(c),
            "dino_cw": l2(z["dino_cw"].astype(np.float64)),
            "dino_ce": l2(z["dino_ce"].astype(np.float64))}


def short(n):
    """070-b_colored_wood_blocks -> colored_wood_blocks(前綴只是 YCB 編號)。"""
    return n.split("_", 1)[-1]


def main():
    want = None
    if "--scenes" in sys.argv:
        want = set(sys.argv[sys.argv.index("--scenes") + 1:])
    md = ["# GT 遮罩語意分群:逐場分群結果(可目視核對)\n",
          "- 建檔 2026-09-23;程式 `srp/stage4_probe/gtmask_cluster_dump.py`;可復現。",
          "- 資料 `data/eval/gt_mask_feats_v2/`(60 場 stack3/4/5、12 視角 A-3 selected)。",
          "- 門檻用各特徵「排除 GEX」下的最佳值:" +
          "、".join(f"{k}={v:.2f}" for k, v in BEST.items()) + "。",
          "- 群組成寫法 `物體×n` = 該物體有 n 個視角被分進這一群(每物體最多 12)。",
          "- **過切** = 同一物體散在 >1 群;**混群** = 同一群含 >1 物體。GEX 物體以 `*` 標註。\n",
          "> ⚠ GT 完美遮罩下的結果,**不可**與 `srp_hull_cluster_mv2_noarm`(MobileSAMv2 遮罩、303 場)相比。\n"]
    agg = {v: defaultdict(lambda: [0, 0]) for v in BEST}   # 物體 -> [被過切場數, 被混群場數]
    for p in sorted(FEAT.glob("*.npz")):
        if want and p.stem not in want:
            continue
        z = np.load(p); lab = np.array([short(n) for n in z["names"]])
        F = feats(z); nobj = len(set(lab.tolist()))
        md.append(f"### {p.stem} — {nobj} 物體 × 12 視角 = {len(lab)} 筆\n")
        md.append("| 特徵 | 群數 | 過切 | 混群 | 各群組成 |")
        md.append("|---|---|---|---|---|")
        for vn, thr in BEST.items():
            cl = fcluster(linkage(pdist(F[vn], "cosine"), "average"), t=thr, criterion="distance")
            comp = defaultdict(Counter)
            for c, o in zip(cl, lab):
                comp[int(c)][o] += 1
            spread = {o for o in set(lab.tolist())
                      if sum(1 for cc in comp.values() if o in cc) > 1}     # 過切
            mixed = {c for c, cc in comp.items() if len(cc) > 1}            # 混群
            mixobj = {o for c in mixed for o in comp[c]}
            for o in spread:
                agg[vn][o][0] += 1
            for o in mixobj:
                agg[vn][o][1] += 1
            parts = []
            for c in sorted(comp):
                s = " + ".join(f"{o}{'*' if o in GEX else ''}×{k}" for o, k in comp[c].most_common())
                parts.append(f"**C{c}**: {s}" if c in mixed else f"C{c}: {s}")
            md.append(f"| {vn} | {len(comp)} | {len(spread)} | {len(mixed)} | " + " ｜ ".join(parts) + " |")
        md.append("")
    md.append("## 逐物體彙總:被過切 / 被混群的場次(60 場)\n")
    objs = sorted({o for v in agg for o in agg[v]},
                  key=lambda o: -(agg["clip_debias"][o][0] + agg["clip_debias"][o][1]))
    md.append("| 物體 | clip_debias 過切/混群 | clip_raw 過切/混群 | dino_cw 過切/混群 | dino_ce 過切/混群 | GEX |")
    md.append("|---|---|---|---|---|---|")
    for o in objs:
        cells = " | ".join(f"{agg[v][o][0]} / {agg[v][o][1]}" for v in BEST)
        md.append(f"| {o} | {cells} | {'是' if o in GEX else '否'} |")
    md.append("")
    out = HERE / "RESULT_gtmask_cluster_dump.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[存檔] {out}  ({len(md)} 行)")


if __name__ == "__main__":
    main()
