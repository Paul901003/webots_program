#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""比較 hull 變體對 GT 實心 mesh 的品質:真實voxel(TP)/切掉真mesh(FN)/憑空過估(FP)。
變體:am1、am0、am0+光度雕、am0+語意重疊雕、am0+光度雕+語意重疊雕。60場堆疊。"""
import numpy as np, sys, json, cv2
from collections import defaultdict, Counter
from pathlib import Path
from scipy import ndimage
sys.path.insert(0, "srp/io"); sys.path.insert(0, "srp/stage2_instances"); sys.path.insert(0, "srp/stage1_hull")
import camera as cam, masks as MK, viewpoints as VP
import cg_associate as CG
import eval_mesh as EM
from voxel_sem_cluster_donut import donut_masks
from photo_carve_b import load_views
from photo_carve_warp import prep_views, carve_warp
MV2 = Path("data/eval/mobilesamv2_fast"); CAP = Path("data/captures_fast")
SEM = Path("data/eval/semdonut_warp_carved")
scenes = [p.name for g in ("stack3", "stack4", "stack5")
          for p in sorted(Path("data/eval/srp_hull_mv2_v12_am0").glob(f"{g}_scene*")) if (p / "hull.npz").is_file()]

def surface_of(o):
    return o & ~ndimage.binary_erosion(o, ndimage.generate_binary_structure(3, 1))

def mixed_carve(occ, gm, vs, mc, sc, max_iter=6, min_dom=1.0):
    """迭代雕掉混群表面 voxel(跨視角被歸≥2群且主群佔比<min_dom)。"""
    sdir = CAP / f"multi_{sc.split('_')[0]}" / sc
    views_meta = []
    for vn in sorted(VP.selected_view_names(12)):
        vd = MV2 / sc / vn; pf = sdir / f"{vn}_pose.json"
        if not (vd.is_dir() and pf.is_file()): continue
        km = MK.kept_object_masks(vd); names = [n for _, n in km]; ms0 = [m for m, _ in km]
        if not ms0: continue
        ms = donut_masks(ms0); C, Rb = cam.load_pose(pf)
        views_meta.append((vn, names, ms, C, Rb, ms0[0].shape))
    carved = occ.copy()
    for _ in range(max_iter):
        surf = surface_of(carved)
        Pw = gm + (np.array(np.nonzero(surf)).T + 0.5) * vs
        sidx = np.array(np.nonzero(surf)).T
        vgc = defaultdict(Counter)
        for vn, names, ms, C, Rb, (H, W) in views_meta:
            va = CG.zbuffer_visible(Pw, C, Rb, W, H, vs).reshape(H, W)
            vmc = defaultdict(Counter)
            for mi, m in enumerate(ms):
                ys, xs = np.where(m); vv = va[ys, xs]
                for v in vv[vv >= 0]: vmc[int(v)][mi] += 1
            cl = mc.get(vn, {})
            for v, cnt in vmc.items():
                g = cl.get(names[cnt.most_common(1)[0][0]])
                if g is not None: vgc[v][g] += 1
        mixed = [i for i, c in vgc.items() if len(c) >= 2 and max(c.values()) / sum(c.values()) < min_dom]
        if not mixed: break
        mg = np.zeros_like(carved)
        for i in mixed: mg[tuple(sidx[i])] = True
        carved = carved & ~mg
    return carved

def qual(occ, union_gt):
    tp = int((occ & union_gt).sum()); fn = int((~occ & union_gt).sum()); fp = int((occ & ~union_gt).sum())
    G = int(union_gt.sum())
    return dict(cover=tp / G, cut=fn / G, ghost=fp / max(occ.sum(), 1), tp=tp, fn=fn, fp=fp)

agg = defaultdict(lambda: defaultdict(list))
for i, sc in enumerate(scenes):
    try:
        z = np.load(f"data/eval/srp_hull_mv2_v12_am1/{sc}/hull.npz")
        occ0 = z["occupancy"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
        gt = EM.solid_mesh_occ(sc, gm, vs, occ0.shape)
        if not gt: print(f"  [skip] {sc} 無GT mesh", flush=True); continue
        ugt = np.zeros(occ0.shape, bool)
        for g in gt.values(): ugt |= g
        views = prep_views(load_views(sc, 12))
        occ_photo = carve_warp(occ0, gm, vs, views, 0.1, 4, 2, 3, 2, 15, 20.0)
        mc = json.loads((SEM / sc / "instances.json").read_text()).get("mask_clusters", {})
        occ_mix = mixed_carve(occ0, gm, vs, mc, sc)
        occ_pm = mixed_carve(occ_photo, gm, vs, mc, sc)
        variants = {"am1": occ0, "am1+光度雕": occ_photo, "am1+語意重疊雕": occ_mix, "am1+光度+語意雕": occ_pm}
        am0p = Path(f"data/eval/srp_hull_mv2_v12_am0/{sc}/hull.npz")
        if am0p.is_file(): variants["am0(對照)"] = np.load(am0p)["occupancy"]
        for name, o in variants.items():
            q = qual(o, ugt)
            for k, v in q.items(): agg[name][k].append(v)
        print(f"  [{i+1}/{len(scenes)}] {sc} ok", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"  [err] {sc}: {e}", flush=True)

print(f"\n===== hull 品質 vs GT 實心 mesh, {len(scenes)}場堆疊 =====")
print(f"{'變體':<18}{'覆蓋(真mesh被蓋)':>16}{'切掉(真mesh缺)':>15}{'過估(ghost)':>13}")
order = ["am1", "am1+光度雕", "am1+語意重疊雕", "am1+光度+語意雕", "am0(對照)"]
for name in order:
    if name not in agg: continue
    a = agg[name]
    print(f"{name:<18}{np.mean(a['cover'])*100:>14.1f}%{np.mean(a['cut'])*100:>14.1f}%{np.mean(a['ghost'])*100:>12.1f}%")
print(f"\n(覆蓋越高越好=少切真mesh;切掉=真mesh被雕缺的洞;過估=hull憑空多出、不在真mesh內)")
