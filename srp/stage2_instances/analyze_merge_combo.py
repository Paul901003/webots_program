#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""analyze_merge_combo.py — 測「交叉>τc 且 z重疊>τz」合併,對真疊物(relations "on")與分開物的效果。
讀 reproj_labels.npz(coords/own_group/gt_obj/vis/proj_group)+ relations.json。僅堆疊 60 場。
每真疊物(在 "on" 關係)/分開物 分 clean(1:1純)/lost(被併掉)/mixed(在混別物元件)/split(還被切多塊)。
用法: ./analyze_merge_combo.py --tc 0.1
"""
import argparse, sys, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; EVAL = REPO / "data" / "eval"
sys.path.insert(0, str(REPO / "srp" / "io")); from labels import label_dir

def uf(groups, edges):
    par = {g: g for g in groups}
    def fd(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in edges: par[fd(a)] = fd(b)
    c = defaultdict(list)
    for g in groups: c[fd(g)].append(g)
    return list(c.values())

def zov(a0, a1, b0, b1):
    inter = max(0, min(a1, b1) - max(a0, b0)); short = min(a1 - a0, b1 - b0)
    return inter / short if short > 0 else (1.0 if inter > 0 else 0.0)

def classify(comps, gobj, valid, stacked_objs):
    comp_objs = [[gobj[g] for g in c] for c in comps]; comp_maj = [Counter(co).most_common(1)[0][0] for co in comp_objs]
    obj2comp = defaultdict(set); majcnt = Counter(comp_maj)
    for ci, c in enumerate(comps):
        for g in c: obj2comp[gobj[g]].add(ci)
    out = {"stacked": Counter(), "separate": Counter()}
    for o in set(gobj[g] for g in valid):
        cs = [ci for ci, cm in enumerate(comp_maj) if cm == o]
        if majcnt.get(o, 0) == 0: cat = "lost"
        elif len(cs) == 1 and len(set(comp_objs[cs[0]])) == 1 and len(obj2comp[o]) == 1: cat = "clean"
        elif len(obj2comp[o]) > 1: cat = "split"
        else: cat = "mixed"
        out["stacked" if o in stacked_objs else "separate"][cat] += 1
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inst-root", default="srp_hull_semcluster_surf_am1photo")
    ap.add_argument("--tc", type=float, default=0.1); args = ap.parse_args(); root = EVAL / args.inst_root
    variants = [("只交叉", None), ("交叉+z(τz=0.1)", 0.1), ("交叉+z(τz=0.3)", 0.3), ("交叉+z(τz=0.5)", 0.5)]
    agg = {v[0]: {"stacked": Counter(), "separate": Counter()} for v in variants}
    for f in sorted(root.glob("stack*_scene*/reproj_labels.npz")):
        sc = f.parent.name; z = np.load(f, allow_pickle=True)
        coords = z["coords"].astype(int); og = z["own_group"].astype(int); gt = z["gt_obj"].astype(int); vis = z["vis"]; pg = z["proj_group"].astype(int)
        onames = json.loads(str(z["build_meta"]))["onames"]
        rel = json.loads((label_dir(sc) / "relations.json").read_text())["relations"]
        stacked_objs = set()
        for r in rel:
            if r["type"] == "on":
                for nm in (r["x"], r["y"]):
                    if nm in onames: stacked_objs.add(onames.index(nm))
        groups = [int(g) for g in np.unique(og) if g > 0]; idx = {g: np.where(og == g)[0] for g in groups}
        vv = {g: int(vis[idx[g]].sum()) for g in groups}; zr = {}; gobj = {}
        for g in groups:
            if len(idx[g]) < 5: continue
            zz = coords[idx[g], 2]; zr[g] = (zz.min(), zz.max()); o = gt[(og == g) & (gt >= 0)]
            gobj[g] = Counter(o.tolist()).most_common(1)[0][0] if len(o) else None
        valid = [g for g in groups if g in zr and gobj[g] is not None and vv[g] > 0]
        pair = {}
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                A, B = valid[i], valid[j]
                cx = max(int(((pg[idx[A]] == B) & vis[idx[A]]).sum()) / vv[A], int(((pg[idx[B]] == A) & vis[idx[B]]).sum()) / vv[B])
                pair[(A, B)] = (cx, zov(*zr[A], *zr[B]))
        for name, tz in variants:
            edges = [(a, b) for (a, b), (cx, zo) in pair.items() if cx > args.tc and (tz is None or zo > tz)]
            comps = uf(valid, edges)
            r = classify(comps, gobj, valid, stacked_objs)
            for k in ("stacked", "separate"):
                for c, n in r[k].items(): agg[name][k][c] += n
    print(f"τc={args.tc} | 真疊物(on)與分開物,各合併規則的結果分類 (%)")
    for grp in ("stacked", "separate"):
        lab = "真疊物" if grp == "stacked" else "分開物"
        print(f"\n[{lab}]  (clean=乾淨1:1, lost=被併掉, mixed=在混別物元件, split=還被切多塊)")
        print(f"{'規則':<16}{'clean':>7}{'lost':>7}{'mixed':>7}{'split':>7}{'總數':>6}")
        for name, _ in variants:
            d = agg[name][grp]; tot = sum(d.values())
            print(f"{name:<16}{d['clean']/tot*100:>6.0f}%{d['lost']/tot*100:>6.0f}%{d['mixed']/tot*100:>6.0f}%{d['split']/tot*100:>6.0f}%{tot:>6}")

if __name__ == "__main__": main()
