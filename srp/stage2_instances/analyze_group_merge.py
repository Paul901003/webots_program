#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""analyze_group_merge.py — 只讀 reproj_labels.npz,用「群對交叉投影比例」合併過切群,評估收斂效果。
邊:兩群 max(A→B,B→A) 交叉比例 > τ → 連通;連通元件=合併後物體。
評估(GT 只拿來評,不進合併):合併後物體數 vs GT物體數、錯合(元件含≥2 GT物體)、GT物體被切成幾個元件。
用法: ./analyze_group_merge.py --inst-root srp_hull_semcluster_surf_am1photo [--groups stack]
"""
import argparse, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; EVAL = REPO / "data" / "eval"

def scene_data(f):
    z = np.load(f); og = z["own_group"].astype(int); gt = z["gt_obj"].astype(int); vis = z["vis"]; pg = z["proj_group"].astype(int)
    groups = [int(g) for g in np.unique(og) if g > 0]
    gobj = {}; vv = {}
    for g in groups:
        sel = og == g; objs = gt[sel]; objs = objs[objs >= 0]
        gobj[g] = Counter(objs.tolist()).most_common(1)[0][0] if len(objs) else None
        vv[g] = int(vis[sel].sum())
    sig = {}
    idxs = {g: np.where(og == g)[0] for g in groups}
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            A, B = groups[a], groups[b]
            if vv[A] == 0 or vv[B] == 0: continue
            fab = int(((pg[idxs[A]] == B) & vis[idxs[A]]).sum()) / vv[A]
            fba = int(((pg[idxs[B]] == A) & vis[idxs[B]]).sum()) / vv[B]
            sig[(A, B)] = max(fab, fba)
    return groups, gobj, sig

def union_find(groups, edges):
    par = {g: g for g in groups}
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in edges: par[find(a)] = find(b)
    comp = defaultdict(list)
    for g in groups: comp[find(g)].append(g)
    return list(comp.values())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inst-root", default="srp_hull_semcluster_surf_am1photo")
    ap.add_argument("--groups", default="all")
    args = ap.parse_args(); root = EVAL / args.inst_root
    files = sorted(root.glob("*_scene*/reproj_labels.npz"))
    if args.groups != "all":
        pref = tuple(args.groups.split(",")); files = [f for f in files if f.parent.name.startswith(pref)]
    scenes = [(f, *scene_data(f)) for f in files]
    print(f"{args.inst_root} | {len(scenes)} 場 | 群對交叉合併掃描\n")
    print(f"{'τ':>6}{'過切比':>7}{'物體recall%':>11}{'1:1乾淨%':>9}{'錯合元件%':>10}{'GT被切>1%':>10}")
    from collections import Counter
    for tau in [999, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0]:
        tot_comp = tot_gt = 0; wrong_comp = tot_valid_comp = 0
        gt_total = 0; gt_split = 0; recall_hit = 0; clean = 0
        for f, groups, gobj, sig in scenes:
            valid = [g for g in groups if gobj[g] is not None]
            if not valid: continue
            edges = [(a, b) for (a, b), s in sig.items() if s > tau and gobj[a] is not None and gobj[b] is not None]
            comps = [c for c in union_find(valid, edges) if c]
            ngt = len(set(gobj[g] for g in valid)); tot_comp += len(comps); tot_gt += ngt
            comp_objs = [ [gobj[g] for g in c] for c in comps ]
            comp_maj = [Counter(co).most_common(1)[0][0] for co in comp_objs]
            for co in comp_objs:
                tot_valid_comp += 1
                if len(set(co)) >= 2: wrong_comp += 1
            majcnt = Counter(comp_maj)                       # 每物體當幾個元件的多數
            obj2comp = defaultdict(set)
            for ci, c in enumerate(comps):
                for g in c: obj2comp[gobj[g]].add(ci)
            for o in set(gobj[g] for g in valid):
                gt_total += 1
                if majcnt.get(o, 0) >= 1: recall_hit += 1     # 有元件以它為多數 = 沒被吃掉
                if len(obj2comp[o]) > 1: gt_split += 1
                # 乾淨 = 剛好一個元件、且該元件純(只含此物體)
                cs = [ci for ci, cm in enumerate(comp_maj) if cm == o]
                if len(cs) == 1 and len(set(comp_objs[cs[0]])) == 1 and len(obj2comp[o]) == 1:
                    clean += 1
        tag = "不合(∞)" if tau == 999 else f"{tau}"
        print(f"{tag:>6}{tot_comp/max(tot_gt,1):>7.2f}{recall_hit/max(gt_total,1)*100:>10.1f}%"
              f"{clean/max(gt_total,1)*100:>8.1f}%{wrong_comp/max(tot_valid_comp,1)*100:>9.1f}%{gt_split/max(gt_total,1)*100:>9.1f}%")
    print("\n物體recall=有元件以它為多數(沒被錯併吃掉)的GT物體比例;1:1乾淨=剛好對一個純元件;")
    print("錯合元件=元件混≥2 GT物體;GT被切>1=同物體散在多元件。τ=不合(∞)=完全不合併的基準。")

if __name__ == "__main__": main()
