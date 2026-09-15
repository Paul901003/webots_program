#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""carve_mixed_solid.py — 實心 hull 重投影,把「混群」的表面 voxel 雕掉,看實心 hull 變怎樣。

混群 voxel = 表面 voxel 跨視角被多數決歸給 ≥2 個語意群 且 主群佔比 < min_dom(沒有明確主群)。
這些多在物體交界;雕掉後看實心 hull 是否斷開成乾淨 per-object 連通塊。
輸出 instances.npz(連通元件)供 Webots;印雕前/後連通元件數與大小。

用法: SAM_ROOT=$PWD/data/eval/mobilesamv2_fast \
  ./carve_mixed_solid.py stack3_scene0001 --hull-root srp_hull_warp_carved --sem-root semdonut_warp_carved --min-dom 1.0
"""
import argparse, os, sys, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io")); sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import camera as cam, masks as MK, viewpoints as VP
import cg_associate as CG
from voxel_sem_cluster_donut import donut_masks

SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data/eval/mobilesamv2_fast")))
CAP = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data/captures_fast")))
EVAL = REPO / "data" / "eval"


def comps(occ):
    lab, n = ndimage.label(occ, ndimage.generate_binary_structure(3, 1))
    sizes = sorted((int((lab == i).sum()) for i in range(1, n + 1)), reverse=True)
    return n, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--hull-root", default="srp_hull_warp_carved")
    ap.add_argument("--sem-root", default="semdonut_warp_carved")
    ap.add_argument("--min-dom", type=float, default=1.0, help="主群佔比<此的表面voxel算混群→雕(1.0=任何混群都雕)")
    ap.add_argument("--dilate", type=int, default=0, help="混群區膨脹幾層再雕(切斷薄接縫用)")
    ap.add_argument("--max-iter", type=int, default=1, dest="max_iter",
                    help="迭代次數:雕→露新面→重判→再雕,直到無混群或到此上限(1=只雕一層)")
    args = ap.parse_args()
    sc = args.scene

    z = np.load(EVAL / args.hull_root / sc / "hull.npz")
    occ = z["occupancy"].copy(); gm = z["grid_min"]; vs = float(z["voxel_size"])
    mc = json.loads((EVAL / args.sem_root / sc / "instances.json").read_text()).get("mask_clusters", {})
    sdir = CAP / f"multi_{sc.split('_')[0]}" / sc
    views = sorted(VP.selected_view_names(12))
    n0, s0 = comps(occ)
    print(f"{sc}: 實心 {int(occ.sum())} vox / {n0} 塊 {s0[:4]}")

    def mixed_surface(o):
        """回傳當前 occ 的混群表面 voxel grid(bool)。"""
        surf = o & ~ndimage.binary_erosion(o, ndimage.generate_binary_structure(3, 1))
        vidx = np.array(np.nonzero(surf)).T; Pw = gm + (vidx + 0.5) * vs
        vgc = defaultdict(Counter)
        for vn in views:
            vd = SAM_ROOT / sc / vn; pf = sdir / f"{vn}_pose.json"
            if not (vd.is_dir() and pf.is_file()): continue
            km = MK.kept_object_masks(vd); names = [n for _, n in km]; ms0 = [m for m, _ in km]
            if not ms0: continue
            ms = donut_masks(ms0); C, Rb = cam.load_pose(pf); H, W = ms0[0].shape
            va = CG.zbuffer_visible(Pw, C, Rb, W, H, vs).reshape(H, W)
            vmc = defaultdict(Counter)
            for mi, m in enumerate(ms):
                ys, xs = np.where(m); vv = va[ys, xs]
                for v in vv[vv >= 0]: vmc[int(v)][mi] += 1
            cl = mc.get(vn, {})
            for v, cnt in vmc.items():
                g = cl.get(names[cnt.most_common(1)[0][0]])
                if g is not None: vgc[v][g] += 1
        ml = [v for v, c in vgc.items() if len(c) >= 2 and max(c.values()) / sum(c.values()) < args.min_dom]
        grid = np.zeros(o.shape, bool)
        for v in ml:
            i, j, k = vidx[v]; grid[i, j, k] = True
        if args.dilate > 0:
            grid = ndimage.binary_dilation(grid, iterations=args.dilate) & o
        return grid, len(ml)

    carved = occ
    for it in range(args.max_iter):
        mg, nmix = mixed_surface(carved)
        if nmix == 0:
            print(f"  iter {it+1}: 無混群 → 收斂"); break
        carved = carved & ~mg
        n1, s1 = comps(carved)
        print(f"  iter {it+1}: 雕 {nmix} 混群 → {int(carved.sum())} vox / {n1} 塊 {s1[:4]}")

    lab = ndimage.label(carved, ndimage.generate_binary_structure(3, 1))[0].astype(np.int32)
    out = EVAL / f"{args.hull_root}_mixedcarve" / sc; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "instances.npz", labels=lab, grid_min=gm, voxel_size=np.float64(vs),
                        build_meta=json.dumps({"src": "carve_mixed_solid", "min_dom": args.min_dom}))
    print(f"→ Webots root: {args.hull_root}_mixedcarve")


if __name__ == "__main__":
    main()
