#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""ConceptGraphs 式關聯實測:nnratio(相鄰)+ CLIP 在 60 場堆疊,分同/異物體群對。
φ_geo=nnratio(A的voxel在B的δ_nn內比例,取雙向max);φ_sem=(cos+1)/2;φ=φ_geo+φ_sem(等權)。
拆開量 nnratio單獨 / CLIP單獨 / 合併 各自 AUC。"""
import numpy as np, sys, json
from collections import defaultdict, Counter
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
DNN = 0.025   # 2.5cm(ConceptGraphs)
FBG = MC.F_BG.astype(np.float64)
scenes = [p.name for g in ("stack3", "stack4", "stack5")
          for p in sorted(Path("data/eval/srp_hull_mv2_v12_am0").glob(f"{g}_scene*")) if (p / "hull.npz").is_file()]

def debias(F):
    F = np.atleast_2d(F).astype(np.float64); F = F - (F @ FBG)[:, None] * FBG[None, :]
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)

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

def scene_groups(sc, modal):
    hp = Path(f"data/eval/srp_hull_mv2_v12_am0/{sc}/hull.npz")
    ij = Path(f"data/eval/semdonut_warp_carved/{sc}/instances.json")
    if not (hp.is_file() and ij.is_file()): return {}
    mc = json.loads(ij.read_text()).get("mask_clusters", {})
    z = np.load(hp); occ0 = z["occupancy"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
    views = prep_views(load_views(sc, 12))
    occ = carve_warp(occ0, gm, vs, views, 0.1, 4, 2, 3, 2, 15, 20.0)   # ★am0+光度雕
    surf = occ & ~ndimage.binary_erosion(occ, ndimage.generate_binary_structure(3, 1))
    Pw = gm + (np.array(np.nonzero(surf)).T + 0.5) * vs
    sdir = CAP / f"multi_{sc.split('_')[0]}" / sc
    gvox = defaultdict(set); gobj = defaultdict(Counter); gfeat = defaultdict(list)
    for vn in sorted(VP.selected_view_names(12)):
        vd = MV2 / sc / vn; pf = sdir / f"{vn}_pose.json"
        if not (vd.is_dir() and pf.is_file()): continue
        km = MK.kept_object_masks(vd); names = [n for _, n in km]; ms0 = [m for m, _ in km]
        if not ms0: continue
        ms = donut_masks(ms0); C, Rb = cam.load_pose(pf); H, W = ms0[0].shape
        va = CG.zbuffer_visible(Pw, C, Rb, W, H, vs).reshape(H, W)
        vmc = defaultdict(Counter)
        for mi, m in enumerate(ms):
            ys, xs = np.where(m); vv = va[ys, xs]
            for v in vv[vv >= 0]: vmc[int(v)][mi] += 1
        mvx = defaultdict(set)
        for v, cnt in vmc.items(): mvx[cnt.most_common(1)[0][0]].add(v)
        import cv2
        rgb = cv2.cvtColor(cv2.imread(str(sdir / f"{vn}.png")), cv2.COLOR_BGR2RGB)
        fmap = donut_feats(vd, rgb, ms, names)
        cl = mc.get(vn, {}); gm_v = modal.get(vn, {})
        for mi, nm in enumerate(names):
            cid = cl.get(nm)
            if cid is None: continue
            gvox[cid] |= mvx.get(mi, set())
            f = fmap.get(nm)
            if f is not None: gfeat[cid].append(f)
            m = ms[mi]; a = int(m.sum()); best = None; bc = 0.5
            for on, om in gm_v.items():
                cov = (m & om).sum() / max(a, 1)
                if cov > bc: bc = cov; best = on
            if best: gobj[cid][best] += 1
    out = {}
    for c in gvox:
        if not (gobj[c] and gvox[c] and gfeat[c]): continue
        coords = Pw[sorted(gvox[c])] if False else gm + (np.array([np.unravel_index(0,(1,))]) )  # placeholder
    # 用 voxel local idx → 世界座標:重建 Pw 對應(gvox 存的是 Pw 的 index)
    res = {}
    for c in gvox:
        if not (gobj[c] and gvox[c] and gfeat[c]): continue
        idxs = np.array(sorted(gvox[c]))
        coords = Pw[idxs]
        clipf = debias(np.mean(gfeat[c], axis=0)).ravel()
        res[c] = (coords, gobj[c].most_common(1)[0][0], clipf)
    return res

def auc(same, other):
    s, t = np.array(same), np.array(other)
    if len(s) == 0 or len(t) == 0: return float("nan")
    r = rankdata(np.concatenate([s, t])); U = r[:len(s)].sum() - len(s) * (len(s) + 1) / 2
    return U / (len(s) * len(t))

G_same, G_touch = [], []   # nnratio
S_same, S_touch = [], []   # clip
P_same, P_touch = [], []   # 合併
for i, sc in enumerate(scenes):
    try:
        modal, pos = gt(sc); gd = scene_groups(sc, modal); cids = list(gd)
        for a in range(len(cids)):
            for b in range(a + 1, len(cids)):
                (cA, oA, fA), (cB, oB, fB) = gd[cids[a]], gd[cids[b]]
                r1 = (cKDTree(cB).query(cA)[0] < DNN).mean()
                r2 = (cKDTree(cA).query(cB)[0] < DNN).mean()
                nn = max(r1, r2)
                sem = (float(fA @ fB) + 1) / 2
                phi = nn + sem
                if oA == oB:
                    G_same.append(nn); S_same.append(sem); P_same.append(phi)
                else:
                    hd = np.linalg.norm(pos[oA] - pos[oB]) if oA in pos and oB in pos else 9
                    if hd < 0.05:
                        G_touch.append(nn); S_touch.append(sem); P_touch.append(phi)
        print(f"  [{i+1}/{len(scenes)}] {sc} ok", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"  [err] {sc}: {e}", flush=True)

print(f"\n===== ConceptGraphs 式 nnratio+CLIP, {len(scenes)}場堆疊 (該合=同物體 該分=相觸異物) =====")
print(f"{'訊號':<14}{'同物體中位':>10}{'異物體中位':>10}{'AUC':>8}")
for name, sa, to in [("nnratio(相鄰)", G_same, G_touch), ("CLIP(語意)", S_same, S_touch), ("合併 φ", P_same, P_touch)]:
    print(f"{name:<14}{np.median(sa):>10.3f}{np.median(to):>10.3f}{auc(sa, to):>8.3f}")
print(f"(n 同物體對={len(G_same)}, 相觸異物對={len(G_touch)})")
