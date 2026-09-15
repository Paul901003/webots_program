#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""analyze_group_cross.py — 只讀 reproj_labels.npz,測「群對交叉投影比例」能否分同物體/異物體群對。
交叉(A→B)= A 的可見 voxel-view 中,footprint 落在 B 群 SAM 遮罩(proj_group==B)的比例。
群對訊號 = max(A→B, B→A)。群的 GT 物體 = 其 voxel 多數 gt_obj(排除鬼影)。
標籤:同物體(obj_A==obj_B)/ 異物體(obj_A!=obj_B)。分「全部」與「僅堆疊 stack」出中位+AUC。
用法: ./analyze_group_cross.py --inst-root srp_hull_semcluster_surf_am1photo
"""
import argparse, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
REPO = Path(__file__).resolve().parents[2]; EVAL = REPO / "data" / "eval"

def auc(s, t):
    s, t = np.array(s), np.array(t)
    if len(s) == 0 or len(t) == 0: return float("nan")
    r = rankdata(np.concatenate([s, t])); U = r[:len(s)].sum() - len(s) * (len(s) + 1) / 2
    return U / (len(s) * len(t))

def scene_pairs(f):
    z = np.load(f)
    og = z["own_group"].astype(int); gt = z["gt_obj"].astype(int); vis = z["vis"]; pg = z["proj_group"].astype(int)
    groups = [g for g in np.unique(og) if g > 0]
    gobj = {}; vv = {}
    for g in groups:
        sel = og == g
        objs = gt[sel]; objs = objs[objs >= 0]
        gobj[g] = Counter(objs.tolist()).most_common(1)[0][0] if len(objs) else None
        vv[g] = int(vis[sel].sum())                       # 可見 voxel-view 數
    cross = defaultdict(int)                               # (A,B)->A的voxel-view投到B群數
    for g in groups:
        sel = np.where(og == g)[0]
        sub_vis = vis[sel]; sub_pg = pg[sel]
        for h in groups:
            if h == g: continue
            cross[(g, h)] = int(((sub_pg == h) & sub_vis).sum())
    rows = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            A, B = groups[i], groups[j]
            if gobj[A] is None or gobj[B] is None or vv[A] == 0 or vv[B] == 0: continue
            fab = cross[(A, B)] / vv[A]; fba = cross[(B, A)] / vv[B]
            sig = max(fab, fba)
            same = gobj[A] == gobj[B]
            rows.append((same, sig))
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inst-root", default="srp_hull_semcluster_surf_am1photo")
    args = ap.parse_args(); root = EVAL / args.inst_root
    allrows = []; stkrows = []
    for f in sorted(root.glob("*_scene*/reproj_labels.npz")):
        r = scene_pairs(f); allrows += r
        if f.parent.name.startswith("stack"): stkrows += r
    for title, rows in [("全部 367 場", allrows), ("僅堆疊 stack3/4/5", stkrows)]:
        same = [s for sm, s in rows if sm]; diff = [s for sm, s in rows if not sm]
        print(f"\n===== {title}:群對交叉投影比例 (同物體群對{len(same)} / 異物體群對{len(diff)}) =====")
        for nm, x in [("同物體(該合)", same), ("異物體(該分)", diff)]:
            if x:
                p = np.percentile(x, [25, 50, 75, 90])
                print(f"  {nm}: 25/50/75/90% = {p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f}/{p[3]:.3f}  mean={np.mean(x):.3f}")
        print(f"  AUC(同物體交叉 > 異物體) = {auc(same, diff):.3f}")

if __name__ == "__main__": main()
