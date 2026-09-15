#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""analyze_group_zoverlap.py — 只讀 reproj_labels.npz,測「群對 z(高度)重疊」能否分同物體 vs 上下相疊異物。
每群 z 範圍 = 該群語意表面 voxel(own_group==g)的 z(grid 索引)min/max，用全部 voxel(不排鬼影,訊號須推論可算)。
群對 z重疊 = z 區間交集長 / 較短區間長(0~1;上下錯開→0,同高度→1)。
群物體 = 群 voxel 多數 gt_obj(排鬼影,只給貼標)。同物體對=物體同;相疊異物對=物體異且在 relations.json 的 "on"。
用法: ./analyze_group_zoverlap.py --inst-root srp_hull_semcluster_surf_am1photo
"""
import argparse, sys, json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
REPO = Path(__file__).resolve().parents[2]; EVAL = REPO / "data" / "eval"
sys.path.insert(0, str(REPO / "srp" / "io")); from labels import label_dir

def auc(s, t):
    s, t = np.array(s), np.array(t)
    if len(s) == 0 or len(t) == 0: return float("nan")
    r = rankdata(np.concatenate([s, t])); U = r[:len(s)].sum() - len(s) * (len(s) + 1) / 2
    return U / (len(s) * len(t))

def zov(a0, a1, b0, b1):
    inter = max(0, min(a1, b1) - max(a0, b0)); short = min(a1 - a0, b1 - b0)
    return inter / short if short > 0 else (1.0 if inter > 0 else 0.0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inst-root", default="srp_hull_semcluster_surf_am1photo")
    args = ap.parse_args(); root = EVAL / args.inst_root
    same = []; stk = []; skip_grp = 0; tot_grp = 0
    for f in sorted(root.glob("stack*_scene*/reproj_labels.npz")):
        sc = f.parent.name
        z = np.load(f, allow_pickle=True)
        coords = z["coords"].astype(int); og = z["own_group"].astype(int); gt = z["gt_obj"].astype(int)
        onames = json.loads(str(z["build_meta"]))["onames"]
        # "on" 關係的物體對(名字→index)
        rel = json.loads((label_dir(sc) / "relations.json").read_text())["relations"]
        on_pairs = set()
        for r in rel:
            if r["type"] == "on" and r["x"] in onames and r["y"] in onames:
                on_pairs.add(frozenset((onames.index(r["x"]), onames.index(r["y"]))))
        groups = [int(g) for g in np.unique(og) if g > 0]
        zr = {}; gobj = {}
        for g in groups:
            tot_grp += 1
            zs = coords[og == g, 2]
            if len(zs) < 5: skip_grp += 1; continue      # 排除 <5 voxel 群
            zr[g] = (zs.min(), zs.max())
            o = gt[(og == g) & (gt >= 0)]
            gobj[g] = Counter(o.tolist()).most_common(1)[0][0] if len(o) else None
        gs = [g for g in groups if g in zr and gobj[g] is not None]
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                A, B = gs[i], gs[j]; v = zov(*zr[A], *zr[B])
                if gobj[A] == gobj[B]: same.append(v)
                elif frozenset((gobj[A], gobj[B])) in on_pairs: stk.append(v)
    print(f"排除 <5voxel 群: {skip_grp}/{tot_grp}")
    print(f"\n列:同物體群對 / 相疊異物群對(z重疊 0~1,上下錯開→0)")
    print(f"欄:對數 n | 中位z重疊 | 平均")
    for nm, x in [("同物體對", same), ("相疊異物對", stk)]:
        p = np.median(x) if x else float("nan")
        print(f"  {nm}: n={len(x)} | 中位 {p:.3f} | 平均 {np.mean(x) if x else float('nan'):.3f}")
    print(f"\nAUC(同物體 z重疊 > 相疊異物) = {auc(same, stk):.3f}  (0.5=沒用, >0.8=可分)")

if __name__ == "__main__": main()
