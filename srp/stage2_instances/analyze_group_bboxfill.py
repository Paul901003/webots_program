#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""analyze_group_bboxfill.py — 測「群對合併後 bbox 填充度」分同物體 vs 上下相疊異物。
每對群(A,B):合併 surface coords → 各軸 min/max bbox → 填充度 = occupancy∩bbox / bbox體素數。
occupancy 讀 hull-root 的 hull.npz(實心,同 grid)。同/異物體用 gt_obj,相疊用 relations.json "on"。
用法: ./analyze_group_bboxfill.py --inst-root srp_hull_semcluster_surf_am1photo --hull-root srp_hull_mv2_v12_am1_photo
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

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inst-root", default="srp_hull_semcluster_surf_am1photo")
    ap.add_argument("--hull-root", default="srp_hull_mv2_v12_am1_photo")
    args = ap.parse_args(); root = EVAL / args.inst_root
    same = []; stk = []; skip = 0; tot = 0
    for f in sorted(root.glob("stack*_scene*/reproj_labels.npz")):
        sc = f.parent.name; z = np.load(f, allow_pickle=True)
        coords = z["coords"].astype(int); og = z["own_group"].astype(int); gt = z["gt_obj"].astype(int)
        onames = json.loads(str(z["build_meta"]))["onames"]
        hp = EVAL / args.hull_root / sc / "hull.npz"
        if not hp.is_file(): continue
        occ = np.load(hp)["occupancy"]
        rel = json.loads((label_dir(sc) / "relations.json").read_text())["relations"]
        onp = set()
        for r in rel:
            if r["type"] == "on" and r["x"] in onames and r["y"] in onames:
                onp.add(frozenset((onames.index(r["x"]), onames.index(r["y"]))))
        groups = [int(g) for g in np.unique(og) if g > 0]; idx = {}; gobj = {}
        for g in groups:
            tot += 1; ii = np.where(og == g)[0]
            if len(ii) < 5: skip += 1; continue
            idx[g] = ii; o = gt[(og == g) & (gt >= 0)]
            gobj[g] = Counter(o.tolist()).most_common(1)[0][0] if len(o) else None
        gs = [g for g in groups if g in idx and gobj[g] is not None]
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                A, B = gs[i], gs[j]; c = coords[np.concatenate([idx[A], idx[B]])]
                lo = c.min(0); hi = c.max(0)
                bboxvol = int(np.prod(hi - lo + 1))
                occin = int(occ[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1].sum())
                fill = occin / bboxvol if bboxvol > 0 else 0
                if gobj[A] == gobj[B]: same.append(fill)
                elif frozenset((gobj[A], gobj[B])) in onp: stk.append(fill)
    print(f"排除 <5voxel 群: {skip}/{tot}")
    print(f"\n列:同物體群對 / 相疊異物群對(填充度 0~1,越高=合起來越填滿bbox)")
    for nm, x in [("同物體對", same), ("相疊異物對", stk)]:
        p = np.percentile(x, [25, 50, 75]) if x else [np.nan] * 3
        print(f"  {nm}: n={len(x)} | 25/50/75% = {p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f} | 平均 {np.mean(x) if x else float('nan'):.3f}")
    print(f"\nAUC(同物體填充 > 相疊) = {auc(same, stk):.3f}  (對照:z重疊 0.805、交叉 0.680)")

if __name__ == "__main__": main()
