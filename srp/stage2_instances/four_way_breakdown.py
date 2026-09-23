#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""four_way_breakdown.py — per-object 四分帳(歸對/洩漏/漏標/過切),分 n/occ/stack + 逐 stack on 對。落地存檔可復現。

★ 為什麼(2026-09-23):只看 leak 會藏住「漏標(unassigned)~20%、歸對只~80%」的大洞。
  每個物體的真實表面 voxel 一定落在三類之一,correct+leak+unassigned=100%;另加 fragment 量過切:
    - 歸對 correct(o)     = o 的 voxel 落在「主導=o」instance 的比例。
    - 洩漏 leak(o)        = 落在「主導≠o」instance 的比例(= 合併/污染)。
    - 漏標 unassigned(o)  = 預測 label 0(沒歸到任何群)的比例。
    - 過切 fragment(o)    = correct 裡落在「最大正確 instance 以外」的比例(同物被切散)。
  判準:合併看洩漏、過切看 fragment、分離看歸對、漏看 unassigned——四欄一起看。

基準/一致:同前景 hull 表面 voxel + gtlabel 完美標籤(同 stack_leak_nosep);排 GEX;stack 平均只算有參與 on 關係的物體。

用法: ./four_way_breakdown.py --inst-root srp_hull_divB_t50_reNNcSd_am1 --gt-root srp_hull_gtlabel_am1
輸出: RESULT_four_way_<inst-root>.md + four_way_<inst-root>_perpair.csv(同目錄)。
"""
import sys
import csv
import glob
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "io"))
from stack_leak_nosep import scene_stats, on_pairs, grp_of, GEX, EVAL  # noqa: E402

KEYS = ("correct", "leak", "unassigned", "fragment")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst-root", required=True)
    ap.add_argument("--gt-root", default="srp_hull_gtlabel_am1")
    a = ap.parse_args()
    scenes = sorted(Path(p).parent.name for p in glob.glob(str(EVAL / a.inst_root / "*_scene*" / "instances.npz")))
    scenes = [s for s in scenes if not s.startswith("n1_")]
    G = {g: {k: [] for k in KEYS} for g in ("n", "occ", "stack", "all")}
    pairs = []
    for sc in scenes:
        st = scene_stats(a.inst_root, a.gt_root, sc)
        if st is None:
            continue
        g = grp_of(sc); onobjs = set()
        for (u, l) in on_pairs(sc):
            if u in GEX or l in GEX:
                continue
            iu = st["name2idx"].get(u); il = st["name2idx"].get(l)
            if iu is None or il is None:
                continue
            onobjs |= {iu, il}
            pairs.append((sc, u, l, st["per"][iu], st["per"][il]))
        for o in st["objs"]:
            if st["per"][o]["name"] in GEX:
                continue
            if g == "stack" and o not in onobjs:
                continue
            p = st["per"][o]
            for k in KEYS:
                G[g][k].append(p[f"{k}_frac"]); G["all"][k].append(p[f"{k}_frac"])

    md = [f"# 四分帳(歸對/洩漏/漏標/過切):{a.inst_root}\n",
          f"- 建檔 2026-09-23;程式 `four_way_breakdown.py`;gt=`{a.gt_root}`;排 GEX;303 多物;可復現(重跑本檔)。",
          "- 每物體 voxel:correct+leak+unassigned=100%;fragment=同物被切散(過切)。合併看洩漏、過切看fragment、分離看歸對、漏看unassigned。\n",
          "## 分組(每物體平均%)\n",
          "| 組 | 物體數 | 歸對% | 洩漏%(合併) | 漏標% | 過切% |", "|---|---|---|---|---|---|"]
    for g in ("n", "occ", "stack", "all"):
        n = len(G[g]["correct"])
        if n == 0:
            continue
        md.append(f"| {g} | {n} | {np.mean(G[g]['correct'])*100:.1f} | {np.mean(G[g]['leak'])*100:.1f} | "
                  f"{np.mean(G[g]['unassigned'])*100:.1f} | {np.mean(G[g]['fragment'])*100:.1f} |")
    md.append("\n## stack 逐 on 對(上/下物 各自 歸對/洩漏/漏標/過切 %)\n")
    md.append("| 場景 | 上→下 | 物 | 歸對 | 洩漏 | 漏標 | 過切 |")
    md.append("|---|---|---|---|---|---|---|")
    for sc, u, l, pu, pl in sorted(pairs):
        for nm, p in ((u, pu), (l, pl)):
            md.append(f"| {sc.replace('_scene','#')} | {u}→{l} | {nm} | {p['correct_frac']*100:.0f} | "
                      f"{p['leak_frac']*100:.0f} | {p['unassigned_frac']*100:.0f} | {p['fragment_frac']*100:.0f} |")
    out_md = HERE / f"RESULT_four_way_{a.inst_root}.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    out_csv = HERE / f"four_way_{a.inst_root}_perpair.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "upper", "lower", "object", "correct", "leak", "unassigned", "fragment"])
        for sc, u, l, pu, pl in sorted(pairs):
            for nm, p in ((u, pu), (l, pl)):
                w.writerow([sc, u, l, nm] + [f"{p[f'{k}_frac']:.4f}" for k in KEYS])
    print(f"[存檔] {out_md}\n[存檔] {out_csv}\n場景={len(scenes)} on對={len(pairs)}")
    print("\n" + "\n".join(md[:14]))


if __name__ == "__main__":
    main()
