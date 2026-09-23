#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""four_way_compare.py — 對多條 inst-root 一次算四分帳(歸對/洩漏/漏標/過切)group 摘要,出一張合併比較表。

復用 four_way_breakdown 的定義(scene_stats;correct/leak/unassigned/fragment;排 GEX;stack 只算 on 物體)。
預設 = 表A 的 12 條 divB(中心/fp × 併最近/drop/侵蝕 × 瘦/胖);瘦配 gtlabel_am1、胖配 gtlabel_am1fp。
用法: ./four_way_compare.py   輸出 RESULT_four_way_compare_divB.md
"""
import sys
import glob
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "io"))
from stack_leak_nosep import scene_stats, on_pairs, grp_of, GEX, EVAL  # noqa: E402

KEYS = ("correct", "leak", "unassigned", "fragment")
# (顯示名, inst-root, gt-root)
METHODS = []
for hull, gt in [("am1", "srp_hull_gtlabel_am1"), ("am1fp", "srp_hull_gtlabel_am1fp")]:
    hn = "瘦" if hull == "am1" else "胖"
    for vote, vc in [("中心", "c"), ("fp", "fp")]:
        for rea, rc in [("併最近", "S"), ("drop", "Sd"), ("侵蝕", "Se")]:
            METHODS.append((f"{hn}{vote}{rea}", f"srp_hull_divB_t50_reNN{vc}{rc}_{hull}", gt))


def group_summary(inst_root, gt_root):
    scenes = sorted(Path(p).parent.name for p in glob.glob(str(EVAL / inst_root / "*_scene*" / "instances.npz")))
    scenes = [s for s in scenes if not s.startswith("n1_")]
    G = {g: {k: [] for k in KEYS} for g in ("n", "occ", "stack", "all")}
    for sc in scenes:
        st = scene_stats(inst_root, gt_root, sc)
        if st is None:
            continue
        g = grp_of(sc); onobjs = set()
        for (u, l) in on_pairs(sc):
            if u in GEX or l in GEX:
                continue
            iu = st["name2idx"].get(u); il = st["name2idx"].get(l)
            if iu is not None and il is not None:
                onobjs |= {iu, il}
        for o in st["objs"]:
            if st["per"][o]["name"] in GEX:
                continue
            if g == "stack" and o not in onobjs:
                continue
            for k in KEYS:
                G[g][k].append(st["per"][o][f"{k}_frac"]); G["all"][k].append(st["per"][o][f"{k}_frac"])
    return {g: {k: (np.mean(v) * 100 if v else 0.0) for k, v in d.items()} for g, d in G.items()}


def main():
    res = {}
    for name, root, gt in METHODS:
        if not (EVAL / root).is_dir():
            print(f"[skip] 缺 {root}"); continue
        res[name] = group_summary(root, gt)
        print(f"  算完 {name}", flush=True)
    md = ["# 表A 12 方法四分帳合併比較(divB;歸對/洩漏/漏標/過切 %)\n",
          "- 建檔 2026-09-23;`four_way_compare.py`;瘦配 gtlabel_am1、胖配 gtlabel_am1fp;排 GEX;303 多物;可復現。",
          "- 每物體 voxel:correct+leak+unassigned=100%;fragment=過切。stack 只算有 on 關係的物體。\n"]
    for g in ("all", "stack", "occ", "n"):
        md.append(f"## {g} 組\n")
        md.append("| 方法 | 歸對% | 洩漏%(合併) | 漏標% | 過切% |")
        md.append("|---|---|---|---|---|")
        for name, root, gt in METHODS:
            if name not in res:
                continue
            d = res[name][g]
            md.append(f"| {name} | {d['correct']:.1f} | {d['leak']:.1f} | {d['unassigned']:.1f} | {d['fragment']:.1f} |")
        md.append("")
    out = HERE / "RESULT_four_way_compare_divB.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[存檔] {out}\n\n" + "\n".join(md))


if __name__ == "__main__":
    main()
