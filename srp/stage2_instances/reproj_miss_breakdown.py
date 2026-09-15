#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""reproj_miss_breakdown.py — 把「整個 footprint 落在自己群語意遮罩外」的旗標事件(voxel×視角)拆三類。

怎麼生 / 設置:
- instances(--inst-root)labels + per-instance per-view 群遮罩;GT modal(labels/<sc>/actual);
  GT 實心 mesh(eval_mesh.solid_mesh_occ)給每 voxel 貼 GT 物體(mesh 含它者)+ 真mesh/鬼影。
- 每 voxel 8 角投影 → 像素 bbox(footprint);zbuffer(中心)判可見。
  可見視角中,若 footprint bbox 完全不含「自己群遮罩」像素 → 旗標事件,再分:
    (b) 同物體別群 = footprint 落在自己 GT 物體的 modal 上;
    (a) 別的物體   = 落在別 GT 物體 modal 上(且非自己物體);
    (c) 真鬼影區   = 落在所有物體 modal 外。
  分「真mesh voxel / 鬼影 voxel」各自統計 a/b/c 旗標事件數。
用法: SAM_ROOT=$PWD/data/eval/mobilesamv2_fast CAPTURES_ROOT=$PWD/data/captures_fast \
  ./reproj_miss_breakdown.py [scene|group|(空=舊367)] --inst-root srp_hull_semcluster_surf_am1photo
"""
import argparse, os, sys, json
from collections import defaultdict
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

def integral(binmask):
    return np.pad(binmask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))

def bbox_sum(II, y0, y1, x0, x1):   # 含端點
    return II[y1 + 1, x0 * 0 + x1 + 1] if False else II[y1 + 1, x1 + 1] - II[y0, x1 + 1] - II[y1 + 1, x0] + II[y0, x0]

def modal_masks(sc, vn):
    ann = json.loads((L.label_dir(sc) / "actual" / "annotations.json").read_text())
    cat = {c["id"]: c["name"] for c in ann["categories"]}; id2v = {im["id"]: Path(im["file_name"]).stem for im in ann["images"]}
    out = {}
    for a in ann["annotations"]:
        if a["category_id"] == 1 or id2v[a["image_id"]] != vn: continue
        s = a["segmentation"]; c = s["counts"].encode() if isinstance(s["counts"], str) else s["counts"]
        out[cat[a["category_id"]]] = cocomask.decode({"size": s["size"], "counts": c}).astype(bool)
    return out

def scene_break(sc, inst_root, nviews=12):
    ip = EVAL / inst_root / sc / "instances.npz"
    if not ip.is_file(): return None
    z = np.load(ip); labels = z["labels"]; gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = labels.shape
    ij = json.loads((EVAL / inst_root / sc / "instances.json").read_text())
    inst_masks = {it["instance"]: it.get("masks", {}) for it in ij["instances"]}
    vidx = np.argwhere(labels > 0); lab = labels[labels > 0]; N = len(vidx)
    if N == 0: return None
    Pw = gm + (vidx + 0.5) * vs
    gt = EM.solid_mesh_occ(sc, gm, vs, shape)
    if not gt: return None
    onames = list(gt.keys()); ugt = np.zeros(shape, bool)
    objid = np.full(N, -1, int)                       # 每 voxel 的 GT 物體 index(-1=鬼影)
    for oi, on in enumerate(onames):
        g = gt[on]; ugt |= g
        inside = g[vidx[:, 0], vidx[:, 1], vidx[:, 2]]
        objid[inside] = oi
    real = objid >= 0
    corn = np.array([[dx, dy, dz] for dx in (-.5, .5) for dy in (-.5, .5) for dz in (-.5, .5)]) * vs
    tally = {"real": defaultdict(int), "ghost": defaultdict(int)}   # a/b/c 旗標事件數
    sdir = CAP / f"multi_{sc.split('_')[0]}" / sc
    for vn in sorted(VP.selected_view_names(nviews)):
        vd = SAM_ROOT / sc / vn; pf = sdir / f"{vn}_pose.json"
        if not (vd.is_dir() and pf.is_file()): continue
        anymask = next(iter((vd / "masks").glob("mask_*.png")), None)
        if anymask is None: continue
        H, W = cv2.imread(str(anymask), 0).shape
        C, Rb = cam.load_pose(pf); Rwc, t = cam.pose_to_w2c(C, Rb); K = cam.intrinsics(W, H)
        X = Pw @ Rwc.T + t; zc = np.clip(X[:, 2], 1e-6, None); ok = X[:, 2] > 1e-6
        px = np.round(K[0, 0] * X[:, 0] / zc + K[0, 2]).astype(int); py = np.round(K[1, 1] * X[:, 1] / zc + K[1, 2]).astype(int)
        inb = ok & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        va = CG.zbuffer_visible(Pw, C, Rb, W, H, vs).reshape(H, W)
        vis = np.zeros(N, bool); ii = np.where(inb)[0]; vis[ii] = va[py[ii], px[ii]] == ii
        Xc = (Pw[:, None, :] + corn[None]) @ Rwc.T + t; zcn = np.clip(Xc[:, :, 2], 1e-6, None)
        cpx = K[0, 0] * Xc[:, :, 0] / zcn + K[0, 2]; cpy = K[1, 1] * Xc[:, :, 1] / zcn + K[1, 2]
        x0 = np.clip(np.floor(cpx.min(1)), 0, W - 1).astype(int); x1 = np.clip(np.ceil(cpx.max(1)), 0, W - 1).astype(int)
        y0 = np.clip(np.floor(cpy.min(1)), 0, H - 1).astype(int); y1 = np.clip(np.ceil(cpy.max(1)), 0, H - 1).astype(int)
        # 群遮罩積分圖
        gii = {}
        for k, mv in inst_masks.items():
            names = mv.get(vn, [])
            if not names: continue
            u = np.zeros((H, W), bool)
            for nm in names:
                m = cv2.imread(str(vd / "masks" / nm), 0)
                if m is not None: u |= (m > 127)
            gii[k] = integral(u)
        # GT modal 積分圖(每物體 + 有哪些)
        mm = modal_masks(sc, vn); mii = {on: integral(mm[on]) for on in mm}
        oidx = {on: oi for oi, on in enumerate(onames)}
        for i in np.where(vis)[0]:
            GII = gii.get(int(lab[i]))
            if GII is not None and bbox_sum(GII, y0[i], y1[i], x0[i], x1[i]) > 0:
                continue                                   # 自己群遮罩有覆蓋 → 不算旗標
            # 旗標事件:footprint 落在哪
            own = objid[i]; own_name = onames[own] if own >= 0 else None
            on_own = own_name is not None and own_name in mii and bbox_sum(mii[own_name], y0[i], y1[i], x0[i], x1[i]) > 0
            on_other = any(on != own_name and bbox_sum(II, y0[i], y1[i], x0[i], x1[i]) > 0 for on, II in mii.items())
            bucket = "b_same_object" if on_own else ("a_diff_object" if on_other else "c_ghost_region")
            tally["real" if real[i] else "ghost"][bucket] += 1
    return tally

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("targets", nargs="*")
    ap.add_argument("--inst-root", default="srp_hull_semcluster_surf_am1photo"); ap.add_argument("--nviews", type=int, default=12)
    args = ap.parse_args(); root = EVAL / args.inst_root
    if not args.targets:
        groups = ("n1", "n3", "n4", "n5", "occ3", "occ4", "occ5", "stack3", "stack4", "stack5")
        scenes = sorted(p.name for g in groups for p in root.glob(f"{g}_scene*"))
    else:
        scenes = []
        for a in args.targets: scenes += [a] if "scene" in a else sorted(p.name for p in root.glob(f"{a}_scene*"))
    agg = {"real": defaultdict(int), "ghost": defaultdict(int)}
    for i, sc in enumerate(scenes):
        try:
            t = scene_break(sc, args.inst_root, args.nviews)
            if t is None: continue
            for grp in ("real", "ghost"):
                for k, v in t[grp].items(): agg[grp][k] += v
            if (i + 1) % 40 == 0: print(f"  {i+1}/{len(scenes)}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}", flush=True)
    print(f"\n===== {len(scenes)}場 旗標事件拆解 (inst={args.inst_root}) =====")
    for grp in ("real", "ghost"):
        d = agg[grp]; tot = sum(d.values())
        lab = "真mesh voxel" if grp == "real" else "鬼影 voxel"
        print(f"\n[{lab}] 旗標事件共 {tot}")
        for b, nm in [("b_same_object", "(b)同物體別群"), ("a_diff_object", "(a)別的物體"), ("c_ghost_region", "(c)真鬼影區(全遮罩外)")]:
            print(f"   {nm:<22}: {d[b]:>8} ({d[b]/max(tot,1)*100:4.1f}%)")

if __name__ == "__main__": main()
