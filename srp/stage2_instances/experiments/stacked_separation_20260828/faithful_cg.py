#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""忠實 ConceptGraphs:每視角每遮罩=一個偵測,逐一 nnratio+CLIP 關聯成物體(不先 CLIP 分群)。
am0+光度雕。量最終物體數 vs GT、過切、錯併(相觸物被合)。φ=φ_geo(nnratio)+φ_sem(CLIP),δ_sim=1.1。"""
import numpy as np, sys, json, cv2
from collections import defaultdict, Counter
from pathlib import Path
from pycocotools import mask as cocomask
from scipy import ndimage
from scipy.spatial import cKDTree
sys.path.insert(0, "srp/io"); sys.path.insert(0, "srp/stage2_instances"); sys.path.insert(0, "srp/stage1_hull")
import camera as cam, masks as MK, viewpoints as VP, labels as L
import cg_associate as CG
import mask_clip_cluster as MC
from voxel_sem_cluster_donut import donut_masks, donut_feats
from photo_carve_b import load_views
from photo_carve_warp import prep_views, carve_warp
MV2 = Path("data/eval/mobilesamv2_fast"); CAP = Path("data/captures_fast")
DNN = 0.025; DSIM = 1.1; MINPTS = 8
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
    ngt = len(ann["images"][0].get("objects", []))
    pos = {o["name"]: np.array(o["position_m"][:2]) for o in ann["images"][0].get("objects", [])}
    return modal, ngt, pos

def run_scene(sc):
    modal, ngt, pos = gt(sc)
    z = np.load(f"data/eval/srp_hull_mv2_v12_am0/{sc}/hull.npz")
    occ0 = z["occupancy"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
    views = prep_views(load_views(sc, 12))
    occ = carve_warp(occ0, gm, vs, views, 0.1, 4, 2, 3, 2, 15, 20.0)
    surf = occ & ~ndimage.binary_erosion(occ, ndimage.generate_binary_structure(3, 1))
    Pw = gm + (np.array(np.nonzero(surf)).T + 0.5) * vs
    sdir = CAP / f"multi_{sc.split('_')[0]}" / sc
    objs = []   # 每物體: dict(idx=set voxel, coords=array, clip=vec, n=int, gt=Counter)
    for vn in sorted(VP.selected_view_names(12)):
        vd = MV2 / sc / vn; pf = sdir / f"{vn}_pose.json"
        if not (vd.is_dir() and pf.is_file()): continue
        km = MK.kept_object_masks(vd); names = [n for _, n in km]; ms0 = [m for m, _ in km]
        if not ms0: continue
        ms = donut_masks(ms0); C, Rb = cam.load_pose(pf); H, W = ms0[0].shape
        va = CG.zbuffer_visible(Pw, C, Rb, W, H, vs).reshape(H, W)
        rgb = cv2.cvtColor(cv2.imread(str(sdir / f"{vn}.png")), cv2.COLOR_BGR2RGB)
        fmap = donut_feats(vd, rgb, ms, names)
        gm_v = modal.get(vn, {})
        # 本視角每遮罩的 3D 點(voxel idx)
        for mi, nm in enumerate(names):
            m = ms[mi]; ys, xs = np.where(m); vv = va[ys, xs]
            pidx = set(int(v) for v in vv[vv >= 0])
            f = fmap.get(nm)
            if len(pidx) < MINPTS or f is None: continue
            pts = Pw[list(pidx)]; clip = debias(f)
            # gt of this mask
            a = int(m.sum()); best = None; bc = 0.5
            for on, om in gm_v.items():
                cov = (m & om).sum() / max(a, 1)
                if cov > bc: bc = cov; best = on
            # 關聯到既有物體(nnratio 新→物 + CLIP)
            bo, bphi = None, -9
            for o in objs:
                geo = (cKDTree(o["coords"]).query(pts)[0] < DNN).mean()
                sem = (float(clip @ o["clip"]) + 1) / 2
                phi = geo + sem
                if phi > bphi: bphi, bo = phi, o
            if bo is not None and bphi > DSIM:
                bo["idx"] |= pidx; bo["coords"] = Pw[list(bo["idx"])]
                bo["clip"] = (bo["clip"] * bo["n"] + clip) / (bo["n"] + 1)
                bo["clip"] /= (np.linalg.norm(bo["clip"]) + 1e-9); bo["n"] += 1
                if best: bo["gt"][best] += 1
            else:
                objs.append({"idx": set(pidx), "coords": pts, "clip": clip, "n": 1,
                             "gt": Counter([best] if best else [])})
    return objs, ngt

n_obj, n_gt, over, wrongmerge, pure = [], [], [], [], []
for i, sc in enumerate(scenes):
    try:
        objs, ngt = run_scene(sc)
        objs = [o for o in objs if o["gt"]]        # 有對到 GT 的
        n_obj.append(len(objs)); n_gt.append(ngt)
        if ngt: over.append(len(objs) / ngt)
        # 錯併:一個物體覆蓋 ≥2 個 GT(各≥30% 票)
        wm = 0
        for o in objs:
            tot = sum(o["gt"].values())
            big = [k for k, v in o["gt"].items() if v / tot >= 0.3]
            if len(big) >= 2: wm += 1
            pure.append(o["gt"].most_common(1)[0][1] / tot)
        wrongmerge.append(wm)
        print(f"  [{i+1}/{len(scenes)}] {sc}: {len(objs)}物體/GT{ngt} 錯併{wm}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"  [err] {sc}: {e}", flush=True)

print(f"\n===== 忠實 ConceptGraphs(mask層級,am0+光度雕), {len(n_obj)}場堆疊 =====")
print(f"平均 GT 物體數: {np.mean(n_gt):.1f}")
print(f"平均關聯出物體數: {np.mean(n_obj):.1f}  (過切比 物體/GT = {np.mean(over):.2f})")
print(f"物體純度(最大GT票佔比)中位: {np.median(pure):.2f}  (1.0=乾淨屬一物)")
print(f"錯併場數(有物體混≥2個GT): {sum(1 for w in wrongmerge if w>0)}/{len(wrongmerge)},平均每場錯併物體 {np.mean(wrongmerge):.2f}")
