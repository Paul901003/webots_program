#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""analyze_group_normal.py — 測「兩相鄰群交界處表面法向量夾角」分同物體過切 vs 相疊異物。
法向量 = voxel_normals(hull occupancy);交界 = A、B 的表面 voxel 在 ≤2 voxel 內相鄰的對;
訊號 = 交界處法向量夾角中位數(小=平滑=同物)。同/異物體用 gt_obj,相疊用 relations "on"。
用法: ./analyze_group_normal.py --inst-root srp_hull_semcluster_surf_am1photo --hull-root srp_hull_mv2_v12_am1_photo
"""
import argparse, sys, json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata
REPO = Path(__file__).resolve().parents[2]; EVAL = REPO / "data" / "eval"
sys.path.insert(0, str(REPO / "srp" / "io")); sys.path.insert(0, str(REPO / "srp" / "stage1_hull")); sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
from labels import label_dir
from photo_carve_warp import voxel_normals

def auc(s, t):
    s, t = np.array(s), np.array(t)
    if len(s) == 0 or len(t) == 0: return float("nan")
    r = rankdata(np.concatenate([s, t])); U = r[:len(s)].sum() - len(s) * (len(s) + 1) / 2
    return U / (len(s) * len(t))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inst-root", default="srp_hull_semcluster_surf_am1photo")
    ap.add_argument("--hull-root", default="srp_hull_mv2_v12_am1_photo"); args = ap.parse_args(); root = EVAL / args.inst_root
    same = []; stk = []
    for f in sorted(root.glob("stack*_scene*/reproj_labels.npz")):
        sc = f.parent.name; z = np.load(f, allow_pickle=True)
        coords = z["coords"].astype(int); og = z["own_group"].astype(int); gt = z["gt_obj"].astype(int)
        onames = json.loads(str(z["build_meta"]))["onames"]
        occ = np.load(EVAL / args.hull_root / sc / "hull.npz")["occupancy"]
        nrm = voxel_normals(occ)                       # (X,Y,Z,3)
        rel = json.loads((label_dir(sc) / "relations.json").read_text())["relations"]
        onp = set()
        for r in rel:
            if r["type"] == "on" and r["x"] in onames and r["y"] in onames:
                onp.add(frozenset((onames.index(r["x"]), onames.index(r["y"]))))
        groups = [int(g) for g in np.unique(og) if g > 0]; idx = {}; gobj = {}
        for g in groups:
            ii = np.where(og == g)[0]
            if len(ii) < 5: continue
            idx[g] = ii; o = gt[(og == g) & (gt >= 0)]
            gobj[g] = Counter(o.tolist()).most_common(1)[0][0] if len(o) else None
        gs = [g for g in groups if g in idx and gobj[g] is not None]
        trees = {g: cKDTree(coords[idx[g]]) for g in gs}
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                A, B = gs[i], gs[j]; ca = coords[idx[A]]; cb = coords[idx[B]]
                d, nb = trees[B].query(ca, distance_upper_bound=2.0)  # A 每點找最近 B 點(≤2)
                bd = d < 1e9
                if bd.sum() < 3: continue                # 交界太小,略
                na = nrm[ca[bd, 0], ca[bd, 1], ca[bd, 2]]
                nbv = nrm[cb[nb[bd], 0], cb[nb[bd], 1], cb[nb[bd], 2]]
                dot = np.clip((na * nbv).sum(1) / (np.linalg.norm(na, axis=1) * np.linalg.norm(nbv, axis=1) + 1e-9), -1, 1)
                ang = np.degrees(np.arccos(dot))          # 交界法向量夾角(度)
                med = float(np.median(ang))
                if gobj[A] == gobj[B]: same.append(med)
                elif frozenset((gobj[A], gobj[B])) in onp: stk.append(med)
    print("列:同物體群對(過切,交界應平滑→夾角小)/ 相疊異物群對(交界應有階躍→夾角大)")
    for nm, x in [("同物體對", same), ("相疊異物對", stk)]:
        p = np.percentile(x, [25, 50, 75]) if x else [np.nan] * 3
        print(f"  {nm}: n={len(x)} | 交界夾角(度) 25/50/75% = {p[0]:.1f}/{p[1]:.1f}/{p[2]:.1f}")
    print(f"\nAUC(相疊夾角 > 同物體夾角) = {auc(stk, same):.3f}  (>0.8=可分;對照 z重疊 0.805)")

if __name__ == "__main__": main()
