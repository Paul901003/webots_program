#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""store_reproj_labels.py — 算並存每 voxel 的重投影標籤(資料產品,非快取;每次從源頭重算後覆蓋存)。
之後的分析直接讀這兩個標籤(mesh 物體 + 每視角投到哪),不再重投影。

每個語意 voxel(instances.npz labels>0)存:
- coords(N,3 網格索引)、own_group(N,自己語意群 label)、gt_obj(N,所屬 GT mesh 物體 index;-1=鬼影/不在任何 mesh)。
- 每視角(V=12,sorted selected_view_names):
    vis(N,V bool)         中心投影是否 zbuffer 最前(可見)。
    in_own(N,V bool)      voxel 8 角 footprint bbox 是否含「自己群」遮罩像素。
    proj_obj(N,V int16)   footprint bbox 覆蓋最多的 GT modal 物體 index;-1=footprint 全在物體遮罩外。
    proj_group(N,V int16) footprint bbox 覆蓋最多的「語意群 SAM 遮罩」的群 id;-1=無(供群對交叉分析)。
輸出 data/eval/<inst-root>/<scene>/reproj_labels.npz(含 build_meta:view 順序、物體名、來源 root)。
用法: SAM_ROOT=$PWD/data/eval/mobilesamv2_fast CAPTURES_ROOT=$PWD/data/captures_fast \
  ./cache_reproj_labels.py [scene|group|(空=舊367)] --inst-root srp_hull_semcluster_surf_am1photo [--force]
"""
import argparse, os, sys, json
from pathlib import Path
import numpy as np, cv2
from pycocotools import mask as cocomask
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io")); sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import camera as cam, viewpoints as VP, labels as L
import cg_associate as CG
import eval_mesh as EM
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data/eval/mobilesamv2_fast")))
CAP = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data/captures_fast")))
EVAL = REPO / "data" / "eval"

def integral(b): return np.pad(b.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
def bsum(II, y0, y1, x0, x1): return II[y1 + 1, x1 + 1] - II[y0, x1 + 1] - II[y1 + 1, x0] + II[y0, x0]

def modal_masks(sc, vn):
    ann = json.loads((L.label_dir(sc) / "actual" / "annotations.json").read_text())
    cat = {c["id"]: c["name"] for c in ann["categories"]}; id2v = {im["id"]: Path(im["file_name"]).stem for im in ann["images"]}
    out = {}
    for a in ann["annotations"]:
        if a["category_id"] == 1 or id2v[a["image_id"]] != vn: continue
        s = a["segmentation"]; c = s["counts"].encode() if isinstance(s["counts"], str) else s["counts"]
        out[cat[a["category_id"]]] = cocomask.decode({"size": s["size"], "counts": c}).astype(bool)
    return out

def build(sc, inst_root, nviews):
    outp = EVAL / inst_root / sc / "reproj_labels.npz"
    ip = EVAL / inst_root / sc / "instances.npz"
    if not ip.is_file(): return None
    z = np.load(ip); labels = z["labels"]; gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = labels.shape
    ij = json.loads((EVAL / inst_root / sc / "instances.json").read_text())
    inst_masks = {it["instance"]: it.get("masks", {}) for it in ij["instances"]}
    vidx = np.argwhere(labels > 0); lab = labels[labels > 0].astype(np.int16); N = len(vidx)
    if N == 0: return None
    Pw = gm + (vidx + 0.5) * vs
    gt = EM.solid_mesh_occ(sc, gm, vs, shape)
    if not gt: return None
    onames = list(gt.keys()); gt_obj = np.full(N, -1, np.int16)
    for oi, on in enumerate(onames):
        ins = gt[on][vidx[:, 0], vidx[:, 1], vidx[:, 2]]; gt_obj[ins] = oi
    corn = np.array([[dx, dy, dz] for dx in (-.5, .5) for dy in (-.5, .5) for dz in (-.5, .5)]) * vs
    views = sorted(VP.selected_view_names(nviews)); V = len(views)
    VIS = np.zeros((N, V), bool); INOWN = np.zeros((N, V), bool); PROJ = np.full((N, V), -1, np.int16)
    PROJG = np.full((N, V), -1, np.int16)   # footprint 落在哪個語意群的 SAM 遮罩(argmax;-1=無)
    sdir = CAP / f"multi_{sc.split('_')[0]}" / sc
    for vj, vn in enumerate(views):
        vd = SAM_ROOT / sc / vn; pf = sdir / f"{vn}_pose.json"
        if not (vd.is_dir() and pf.is_file()): continue
        am = next(iter((vd / "masks").glob("mask_*.png")), None)
        if am is None: continue
        H, W = cv2.imread(str(am), 0).shape
        C, Rb = cam.load_pose(pf); Rwc, t = cam.pose_to_w2c(C, Rb); K = cam.intrinsics(W, H)
        X = Pw @ Rwc.T + t; zc = np.clip(X[:, 2], 1e-6, None); ok = X[:, 2] > 1e-6
        px = np.round(K[0, 0] * X[:, 0] / zc + K[0, 2]).astype(int); py = np.round(K[1, 1] * X[:, 1] / zc + K[1, 2]).astype(int)
        inb = ok & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        va = CG.zbuffer_visible(Pw, C, Rb, W, H, vs).reshape(H, W)
        ii = np.where(inb)[0]; VIS[ii[va[py[ii], px[ii]] == ii], vj] = True
        Xc = (Pw[:, None, :] + corn[None]) @ Rwc.T + t; zcn = np.clip(Xc[:, :, 2], 1e-6, None)
        cpx = K[0, 0] * Xc[:, :, 0] / zcn + K[0, 2]; cpy = K[1, 1] * Xc[:, :, 1] / zcn + K[1, 2]
        x0 = np.clip(np.floor(cpx.min(1)), 0, W - 1).astype(int); x1 = np.clip(np.ceil(cpx.max(1)), 0, W - 1).astype(int)
        y0 = np.clip(np.floor(cpy.min(1)), 0, H - 1).astype(int); y1 = np.clip(np.ceil(cpy.max(1)), 0, H - 1).astype(int)
        gii = {}
        for k, mv in inst_masks.items():
            names = mv.get(vn, [])
            if not names: continue
            u = np.zeros((H, W), bool)
            for nm in names:
                m = cv2.imread(str(vd / "masks" / nm), 0)
                if m is not None: u |= (m > 127)
            gii[k] = integral(u)
        mm = modal_masks(sc, vn); mii = [(onames.index(on), integral(mm[on])) for on in mm]
        giil = list(gii.items())
        for i in np.where(VIS[:, vj])[0]:
            GII = gii.get(int(lab[i]))
            if GII is not None and bsum(GII, y0[i], y1[i], x0[i], x1[i]) > 0: INOWN[i, vj] = True
            best = -1; bestv = 0
            for oi, II in mii:
                s = bsum(II, y0[i], y1[i], x0[i], x1[i])
                if s > bestv: bestv = s; best = oi
            PROJ[i, vj] = best
            gb = -1; gbv = 0                          # footprint 落在哪個語意群 SAM 遮罩(argmax)
            for gk, II in giil:
                s = bsum(II, y0[i], y1[i], x0[i], x1[i])
                if s > gbv: gbv = s; gb = gk
            PROJG[i, vj] = gb
    meta = {"script": "cache_reproj_labels.py", "inst_root": inst_root, "sam_root": SAM_ROOT.name,
            "views": views, "onames": onames, "nviews": nviews}
    np.savez_compressed(outp, coords=vidx.astype(np.int16), own_group=lab, gt_obj=gt_obj,
                        vis=VIS, in_own=INOWN, proj_obj=PROJ, proj_group=PROJG,
                        build_meta=json.dumps(meta, ensure_ascii=False))
    return N

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("targets", nargs="*")
    ap.add_argument("--inst-root", default="srp_hull_semcluster_surf_am1photo")
    ap.add_argument("--nviews", type=int, default=12)
    args = ap.parse_args(); root = EVAL / args.inst_root
    if not args.targets:
        groups = ("n1", "n3", "n4", "n5", "occ3", "occ4", "occ5", "stack3", "stack4", "stack5")
        scenes = sorted(p.name for g in groups for p in root.glob(f"{g}_scene*"))
    else:
        scenes = []
        for a in args.targets: scenes += [a] if "scene" in a else sorted(p.name for p in root.glob(f"{a}_scene*"))
    done = 0
    for i, sc in enumerate(scenes):
        try:
            r = build(sc, args.inst_root, args.nviews)
            if isinstance(r, int): done += 1
            if (i + 1) % 40 == 0: print(f"  {i+1}/{len(scenes)} (stored {done})", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}", flush=True)
    print(f"完成:建 {done} 場 reproj_labels.npz 於 data/eval/{args.inst_root}/<scene>/")

if __name__ == "__main__": main()
