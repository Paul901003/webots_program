#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""遮罩層級 nnratio 分佈:每個遮罩=它蓋到的 hull 表面 voxel 集(am0+光度雕)。
兩兩遮罩算 nnratio(雙向max, 2.5cm)+CLIP,依 GT 標同物體/相觸異物,出中位+AUC。
= 群層級 nnratio_clip.py 的遮罩版(單位換成單遮罩,不先 CLIP 分群)。"""
import numpy as np, sys, json, cv2
from collections import defaultdict
from pathlib import Path
from pycocotools import mask as cocomask
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import rankdata
sys.path.insert(0, "srp/io"); sys.path.insert(0, "srp/stage2_instances"); sys.path.insert(0, "srp/stage1_hull")
import camera as cam, masks as MK, viewpoints as VP, labels as L
import cg_associate as CG
import mask_clip_cluster as MC
from voxel_sem_cluster_donut import donut_masks, donut_feats
from photo_carve_b import load_views
from photo_carve_warp import prep_views, carve_warp
MV2 = Path("data/eval/mobilesamv2_fast"); CAP = Path("data/captures_fast")
DNN = 0.025; MINPTS = 8
FBG = MC.F_BG.astype(np.float64)
scenes = [p.name for g in ("stack3", "stack4", "stack5")
          for p in sorted(Path("data/eval/srp_hull_mv2_v12_am0").glob(f"{g}_scene*")) if (p / "hull.npz").is_file()]

def debias(f):
    f = np.asarray(f, np.float64); f = f - (f @ FBG) * FBG
    return f / (np.linalg.norm(f) + 1e-9)

def gt(sc):
    ann = json.loads((L.label_dir(sc) / "actual" / "annotations.json").read_text())
    cat = {c["id"]: c["name"] for c in ann["categories"]}; id2v = {im["id"]: Path(im["file_name"]).stem for im in ann["images"]}
    modal = defaultdict(dict)
    for a in ann["annotations"]:
        if a["category_id"] == 1: continue
        s = a["segmentation"]; c = s["counts"].encode() if isinstance(s["counts"], str) else s["counts"]
        modal[id2v[a["image_id"]]][cat[a["category_id"]]] = cocomask.decode({"size": s["size"], "counts": c}).astype(bool)
    pos = {o["name"]: np.array(o["position_m"][:2]) for o in ann["images"][0]["objects"]}
    return modal, pos

def scene_masks(sc, modal):
    z = np.load(f"data/eval/srp_hull_mv2_v12_am0/{sc}/hull.npz")
    occ0 = z["occupancy"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
    views = prep_views(load_views(sc, 12))
    occ = carve_warp(occ0, gm, vs, views, 0.1, 4, 2, 3, 2, 15, 20.0)   # am0+光度雕
    surf = occ & ~ndimage.binary_erosion(occ, ndimage.generate_binary_structure(3, 1))
    Pw = gm + (np.array(np.nonzero(surf)).T + 0.5) * vs
    sdir = CAP / f"multi_{sc.split('_')[0]}" / sc
    items = []   # (coords Nx3, gt_name or None, clipvec)
    for vn in sorted(VP.selected_view_names(12)):
        vd = MV2 / sc / vn; pf = sdir / f"{vn}_pose.json"
        if not (vd.is_dir() and pf.is_file()): continue
        km = MK.kept_object_masks(vd); names = [n for _, n in km]; ms0 = [m for m, _ in km]
        if not ms0: continue
        ms = donut_masks(ms0); C, Rb = cam.load_pose(pf); H, W = ms0[0].shape
        va = CG.zbuffer_visible(Pw, C, Rb, W, H, vs).reshape(H, W)
        rgb = cv2.cvtColor(cv2.imread(str(sdir / f"{vn}.png")), cv2.COLOR_BGR2RGB)
        fmap = donut_feats(vd, rgb, ms, names); gm_v = modal.get(vn, {})
        for mi, nm in enumerate(names):
            m = ms[mi]; ys, xs = np.where(m); vv = va[ys, xs]
            pidx = np.unique(vv[vv >= 0]); f = fmap.get(nm)
            if len(pidx) < MINPTS or f is None: continue
            a = int(m.sum()); best = None; bc = 0.5
            for on, om in gm_v.items():
                cov = (m & om).sum() / max(a, 1)
                if cov > bc: bc = cov; best = on
            items.append((Pw[pidx], best, debias(f)))
    return items

def auc(s, t):
    s, t = np.array(s), np.array(t)
    if len(s) == 0 or len(t) == 0: return float("nan")
    r = rankdata(np.concatenate([s, t])); U = r[:len(s)].sum() - len(s) * (len(s) + 1) / 2
    return U / (len(s) * len(t))

G_s, G_t, S_s, S_t = [], [], [], []
for i, sc in enumerate(scenes):
    try:
        modal, pos = gt(sc); items = scene_masks(sc, modal)
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                (cA, oA, fA), (cB, oB, fB) = items[a], items[b]
                if oA is None or oB is None: continue
                nn = max((cKDTree(cB).query(cA)[0] < DNN).mean(), (cKDTree(cA).query(cB)[0] < DNN).mean())
                sem = (float(fA @ fB) + 1) / 2
                if oA == oB:
                    G_s.append(nn); S_s.append(sem)
                else:
                    hd = np.linalg.norm(pos[oA] - pos[oB]) if oA in pos and oB in pos else 9
                    if hd < 0.05: G_t.append(nn); S_t.append(sem)
        print(f"  [{i+1}/{len(scenes)}] {sc} ok (masks有GT對: 同{len(G_s)} 相觸{len(G_t)})", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"  [err] {sc}: {e}", flush=True)

np.savez("/tmp/claude-1000/-home-cho-webots-program/338e7fee-7000-4d61-a82a-92a2c5d8b46c/scratchpad/mask_nn_scores.npz",
         G_s=np.array(G_s), G_t=np.array(G_t), S_s=np.array(S_s), S_t=np.array(S_t))
print(f"\n===== 遮罩層級 nnratio+CLIP, {len(scenes)}場堆疊 (am0+光度雕) =====")
print(f"{'訊號':<14}{'同物體中位':>10}{'相觸異物中位':>12}{'AUC':>8}")
for name, sa, to in [("nnratio(相鄰)", G_s, G_t), ("CLIP(語意)", S_s, S_t)]:
    print(f"{name:<14}{np.median(sa):>10.3f}{np.median(to):>12.3f}{auc(sa, to):>8.3f}")
print(f"(同物體遮罩對={len(G_s)}, 相觸異物遮罩對={len(G_t)})")
print(f"\n對照群層級(先CLIP分群): nnratio AUC 0.658 / CLIP 0.575")
