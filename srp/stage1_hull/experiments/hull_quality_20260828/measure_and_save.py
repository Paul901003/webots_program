#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""重跑 hull 品質(am1 基底),留下永久佐證:
(1) per_scene_quality.csv — 60 場每場每變體的覆蓋/切掉/過估。
(2) 2 個範例場的『語意重疊雕』hull + 表面上色 instances → data/eval/demo_mixedcarve_destroys/
    供 Webots 目視「重疊雕挖穿 hull」的證據。"""
import sys, json, csv
import numpy as np
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
HERE = Path("srp/stage1_hull/experiments/hull_quality_20260828")
DEMO = Path("data/eval/demo_mixedcarve_destroys")
scenes = [p.name for g in ("stack3", "stack4", "stack5")
          for p in sorted(Path("data/eval/srp_hull_mv2_v12_am1").glob(f"{g}_scene*")) if (p / "hull.npz").is_file()]
EXAMPLES = {"stack3_scene0001", "stack5_scene0001"}

def surface(o): return o & ~ndimage.binary_erosion(o, ndimage.generate_binary_structure(3, 1))

def mixed_carve2(occ, gm, vs, mc, sc, max_iter=6, min_dom=1.0):
    sdir = CAP / f"multi_{sc.split('_')[0]}" / sc; vm = []
    for vn in sorted(VP.selected_view_names(12)):
        vd = MV2 / sc / vn; pf = sdir / f"{vn}_pose.json"
        if not (vd.is_dir() and pf.is_file()): continue
        km = MK.kept_object_masks(vd); names = [n for _, n in km]; ms0 = [m for m, _ in km]
        if not ms0: continue
        ms = donut_masks(ms0); C, Rb = cam.load_pose(pf); vm.append((vn, names, ms, C, Rb, ms0[0].shape))
    carved = occ.copy()
    for _ in range(max_iter):
        surf = surface(carved); Pw = gm + (np.array(np.nonzero(surf)).T + 0.5) * vs
        sidx = np.array(np.nonzero(surf)).T; vgc = defaultdict(Counter)
        for vn, names, ms, C, Rb, (H, W) in vm:
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

def qual(occ, ugt):
    tp = int((occ & ugt).sum()); fn = int((~occ & ugt).sum()); fp = int((occ & ~ugt).sum()); G = int(ugt.sum())
    return dict(cover=tp / G, cut=fn / G, ghost=fp / max(int(occ.sum()), 1), tp=tp, fn=fn, fp=fp, hull_vox=int(occ.sum()), gt_vox=G)

def save_demo(sc, name, occ, gm, vs):
    d = DEMO / f"{sc}__{name}"; d.mkdir(parents=True, exist_ok=True)
    lab, n = ndimage.label(occ, ndimage.generate_binary_structure(3, 1)); lab = lab.astype(np.int32)
    surf = surface(occ); lab_surf = (lab * surf).astype(np.int32)
    np.savez_compressed(d / "hull.npz", occupancy=occ, surface=surf, observed=occ,
                        grid_min=gm, voxel_size=np.float64(vs),
                        build_meta=json.dumps({"demo": name, "scene": sc}))
    np.savez_compressed(d / "instances.npz", labels=lab_surf, occupancy=occ, grid_min=gm,
                        voxel_size=np.float64(vs), build_meta=json.dumps({"demo": name}))

rows = []; agg = defaultdict(lambda: defaultdict(list))
for i, sc in enumerate(scenes):
    try:
        z = np.load(f"data/eval/srp_hull_mv2_v12_am1/{sc}/hull.npz")
        occ0 = z["occupancy"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
        gt = EM.solid_mesh_occ(sc, gm, vs, occ0.shape)
        if not gt: continue
        ugt = np.zeros(occ0.shape, bool)
        for g in gt.values(): ugt |= g
        views = prep_views(load_views(sc, 12))
        occ_p = carve_warp(occ0, gm, vs, views, 0.1, 4, 2, 3, 2, 15, 20.0)
        mc = json.loads((SEM / sc / "instances.json").read_text()).get("mask_clusters", {})
        occ_m = mixed_carve2(occ0, gm, vs, mc, sc)
        occ_pm = mixed_carve2(occ_p, gm, vs, mc, sc)
        variants = {"am1": occ0, "am1+photo": occ_p, "am1+mixed": occ_m, "am1+photo+mixed": occ_pm}
        for name, o in variants.items():
            q = qual(o, ugt); agg[name]["cover"].append(q["cover"]); agg[name]["cut"].append(q["cut"]); agg[name]["ghost"].append(q["ghost"])
            rows.append({"scene": sc, "variant": name, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in q.items()}})
        if sc in EXAMPLES:
            for name, o in [("am1", occ0), ("am1_photo", occ_p), ("am1_mixedcarve_DESTROYED", occ_m)]:
                save_demo(sc, name, o, gm, vs)
        print(f"  [{i+1}/{len(scenes)}] {sc} ok", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"  [err] {sc}: {e}", flush=True)

with open(HERE / "per_scene_quality.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["scene", "variant", "cover", "cut", "ghost", "tp", "fn", "fp", "hull_vox", "gt_vox"])
    w.writeheader(); w.writerows(rows)
print(f"\n寫 per_scene_quality.csv ({len(rows)} 列)")
print(f"{'變體':<18}{'覆蓋':>8}{'切掉':>8}{'過估':>8}")
for name in ["am1", "am1+photo", "am1+mixed", "am1+photo+mixed"]:
    a = agg[name]; print(f"{name:<18}{np.mean(a['cover'])*100:>7.1f}%{np.mean(a['cut'])*100:>7.1f}%{np.mean(a['ghost'])*100:>7.1f}%")
print(f"範例 demo hull 存於 {DEMO}")
