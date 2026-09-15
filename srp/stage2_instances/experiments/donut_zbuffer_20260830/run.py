#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""donut_zbuffer — 含入遮罩對(A內含B)的 hull-zbuffer 表面深度:內圈B vs 外環A 深度差,分同/異物體。

假設:B 是前面別的物體(異物遮擋)→ 子較近、深度差(d_ring-d_child)為正、邊界有跳變;
      B 是同物體一塊(CLIP 過切)→ 深度連續、深度差≈0、無跳變。
zbuffer 由 hull 幾何算(非感測器深度,合免深度)。

每視角:投 hull 表面 voxel 建 zbuffer(每 pixel 最近相機 z);找含入對(|B∩A|/area(B)≥0.8, area(A)>area(B));
d_child=B內有效深度中位, d_ring=環(A\B)內深度中位, 深度差=d_ring-d_child;邊界跳變=B膨脹帶 子側vs環側深度中位差。
GT:B/A 各自 modal GT 覆蓋最多物體;同物體對=同物體。
env:SAM_ROOT CAPTURES_ROOT。用法: ./run.py stack4_scene0001 stack5_scene0001 stack3_scene0005
"""
import sys, json, csv
import numpy as np, cv2
from collections import defaultdict
from pathlib import Path
from pycocotools import mask as cocomask
from scipy.stats import rankdata
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO/"srp/io")); sys.path.insert(0, str(REPO/"srp/stage2_instances"))
import camera as cam, masks as MK, viewpoints as VP, labels as L
EVAL = REPO/"data"/"eval"; MV2 = EVAL/"mobilesamv2_fast"; CAP = REPO/"data"/"captures_fast"
ARM = EVAL/"srp_arm_masks"; HULL = "srp_hull_mv2_v12_am1_photo"
HERE = Path(__file__).resolve().parent
GEX = {"skillet_lid", "windex_bottle", "colored_wood_blocks", "dice"}


def surface(occ):
    s = np.zeros_like(occ)
    s[1:-1, 1:-1, 1:-1] = occ[1:-1, 1:-1, 1:-1] & ~(
        occ[:-2, 1:-1, 1:-1] & occ[2:, 1:-1, 1:-1] & occ[1:-1, :-2, 1:-1] &
        occ[1:-1, 2:, 1:-1] & occ[1:-1, 1:-1, :-2] & occ[1:-1, 1:-1, 2:])
    return s


def modal(sc):
    ann = json.loads((L.label_dir(sc)/"actual"/"annotations.json").read_text())
    cat = {c["id"]: c["name"] for c in ann["categories"]}
    id2v = {im["id"]: Path(im["file_name"]).stem for im in ann["images"]}
    md = defaultdict(dict)
    for a in ann["annotations"]:
        if a["category_id"] == 1: continue
        s = a["segmentation"]; c = s["counts"].encode() if isinstance(s["counts"], str) else s["counts"]
        md[id2v[a["image_id"]]][cat[a["category_id"]]] = cocomask.decode({"size": s["size"], "counts": c}).astype(bool)
    return md


def auc(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) == 0 or len(b) == 0: return float("nan")
    r = rankdata(np.concatenate([a, b])); U = r[:len(a)].sum()-len(a)*(len(a)+1)/2
    return U/(len(a)*len(b))


def objof(mask, gmv):
    a = int(mask.sum()); best = None; bc = 0.3
    for on, om in gmv.items():
        cov = (mask & om).sum()/max(a, 1)
        if cov > bc: bc = cov; best = on
    return best


def run(scenes):
    rows = []
    for sc in scenes:
        z = np.load(EVAL/HULL/sc/"hull.npz"); occ = z["occupancy"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
        surf = surface(occ); sidx = np.argwhere(surf); Pw = gm+(sidx+0.5)*vs
        md = modal(sc); sdir = CAP/f"multi_{sc.split('_')[0]}"/sc
        for vn in sorted(VP.selected_view_names(12)):
            vd = MV2/sc/vn; pf = sdir/f"{vn}_pose.json"
            if not (vd.is_dir() and pf.is_file()): continue
            km = MK.kept_object_masks(vd); ms0 = [m for m, _ in km]
            ap = ARM/sc/f"{vn}_arm.png"; arm = (cv2.imread(str(ap), 0) > 127) if ap.is_file() else None
            if arm is not None:
                ms0 = [m for m in ms0 if (m & arm).sum()/max(int(m.sum()), 1) < 0.5]
            if len(ms0) < 2: continue
            H, W = ms0[0].shape
            C, Rb = cam.load_pose(pf); Rwc, t = cam.pose_to_w2c(C, Rb); K = cam.intrinsics(W, H)
            X = Pw@Rwc.T+t; zc = X[:, 2]; ok = zc > 1e-6; zz = np.where(ok, zc, 1.0)
            px = np.round(K[0, 0]*X[:, 0]/zz+K[0, 2]).astype(int); py = np.round(K[1, 1]*X[:, 1]/zz+K[1, 2]).astype(int)
            inb = ok & (px >= 0) & (px < W) & (py >= 0) & (py < H); ii = np.where(inb)[0]
            depth = np.full((H, W), np.inf)  # zbuffer:最近相機 z
            for a in ii:
                yy, xx = py[a], px[a]
                if zc[a] < depth[yy, xx]: depth[yy, xx] = zc[a]
            valid = np.isfinite(depth)
            gmv = md.get(vn, {})
            areas = [int(m.sum()) for m in ms0]
            for i in range(len(ms0)):
                for j in range(len(ms0)):
                    if i == j: continue
                    B, A = ms0[i], ms0[j]      # 子 B、母 A
                    ab = areas[i]
                    if ab == 0 or areas[j] <= ab: continue
                    if (B & A).sum()/ab < 0.8: continue     # B 含入 A
                    ring = A & ~B
                    bv = B & valid; rv = ring & valid
                    if bv.sum() < 20 or rv.sum() < 20: continue
                    d_child = float(np.median(depth[bv])); d_ring = float(np.median(depth[rv]))
                    # 邊界跳變:B 膨脹帶
                    Bd = cv2.dilate(B.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                    inner = B & valid; outer = (Bd & ~B) & ring & valid
                    bj = (float(np.median(depth[inner]))-float(np.median(depth[outer]))) if (inner.sum() > 10 and outer.sum() > 10) else np.nan
                    ob = objof(B, gmv); oa = objof(ring, gmv)
                    if ob is None or oa is None: continue
                    nb = ob.split("_", 1)[-1]; na = oa.split("_", 1)[-1]
                    if nb in GEX or na in GEX: continue
                    rows.append([sc, vn, nb, na, int(ob == oa), round((d_ring-d_child)*1000, 1),
                                 round(bj*1000, 1) if np.isfinite(bj) else "", round(d_child*1000, 1), round(d_ring*1000, 1)])
    return rows


if __name__ == "__main__":
    rows = run(sys.argv[1:])
    with open(HERE/"pairs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene", "view", "childobj", "parentobj", "same", "d_ring-d_child_mm", "boundary_jump_mm", "d_child_mm", "d_ring_mm"]); w.writerows(rows)
    A = np.array(rows, dtype=object)
    if len(A) == 0:
        print("無含入對(0)"); sys.exit()
    same = A[:, 4].astype(int) == 1
    print(f"含入對 n={len(A)}(同物體 {same.sum()} / 異物體 {(~same).sum()})")
    dd = A[:, 5].astype(float)
    print(f"深度差(d_ring-d_child, mm, 正=子在前):")
    print(f"  同物體: 中位 {np.median(dd[same]):+.1f} | 異物體: 中位 {np.median(dd[~same]):+.1f} | AUC(異>同)={auc(dd[same],dd[~same]):.3f}")
    bjmask = np.array([r != "" for r in A[:, 6]])
    if bjmask.any():
        bj = np.array([float(x) for x in A[bjmask, 6]]); sm = same[bjmask]
        print(f"邊界跳變(內-外, mm, 負=子較近):")
        print(f"  同物體: 中位 {np.median(bj[sm]):+.1f} | 異物體: 中位 {np.median(bj[~sm]):+.1f} | AUC(同>異)={auc(bj[~sm],bj[sm]):.3f}")
    print(f"pairs.csv → {HERE}")
