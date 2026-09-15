#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""boundary_continuity — 相鄰遮罩「共享邊界帶兩側」前表面深度跳變 vs 舊「整非重疊區中位差」。

假設:同一物體被切兩群→交界處表面連續→邊界跳變小(即使整區中位因斜面退縮而差,救高物誤拆);
      相疊兩物→交界處有深度階梯→邊界跳變大。
邊界帶:A側=dilate(B,k)∩A、B側=dilate(A,k)∩B(各靠近對方 ~k px 的自己像素)。
前表面=hull 表面 voxel splat 後每 pixel min-z。同時輸出 region 中位差對照 + 高物誤拆比較。
env:SAM_ROOT CAPTURES_ROOT。用法: ./boundary_continuity.py <scenes...>
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
ARM = EVAL/"srp_arm_masks"; HULL = "srp_hull_mv2_v12_am1_photo"; HERE = Path(__file__).resolve().parent
GEX = {"skillet_lid", "windex_bottle", "colored_wood_blocks", "dice"}
TALL = {"sugar_box", "pitcher_base", "bleach_cleanser", "cracker_box", "master_chef_can", "mustard_bottle",
        "fork", "knife", "spatula", "flat_screwdriver", "wood_block"}
KB = 7   # 邊界帶膨脹核(≈3px 兩側)
MINPX = 15


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


def surface(occ):
    s = np.zeros_like(occ)
    s[1:-1, 1:-1, 1:-1] = occ[1:-1, 1:-1, 1:-1] & ~(
        occ[:-2, 1:-1, 1:-1] & occ[2:, 1:-1, 1:-1] & occ[1:-1, :-2, 1:-1] &
        occ[1:-1, 2:, 1:-1] & occ[1:-1, 1:-1, :-2] & occ[1:-1, 1:-1, 2:])
    return s


def auc(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    r = rankdata(np.concatenate([pos, neg])); U = r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2
    return U/(len(pos)*len(neg))


def objof(mask, gmv):
    a = int(mask.sum()); best = None; bc = 0.3
    for on, om in gmv.items():
        cov = (mask & om).sum()/max(a, 1)
        if cov > bc: bc = cov; best = on
    return best


def run(scenes):
    R = 3; offs = [(dx, dy) for dx in range(-R, R+1) for dy in range(-R, R+1) if dx*dx+dy*dy <= R*R]
    ker = np.ones((KB, KB), np.uint8)
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
            dnear = np.full((H, W), np.inf)
            yy = py[ii]; xx = px[ii]; zv = zc[ii]
            for dx, dy in offs:
                np.minimum.at(dnear, (np.clip(yy+dy, 0, H-1), np.clip(xx+dx, 0, W-1)), zv)
            vld = np.isfinite(dnear)
            gmv = md.get(vn, {}); objs = [objof(m, gmv) for m in ms0]
            mu = [m.astype(np.uint8) for m in ms0]
            for i in range(len(ms0)):
                for j in range(i+1, len(ms0)):
                    if objs[i] is None or objs[j] is None: continue
                    ni = objs[i].split("_", 1)[-1]; nj = objs[j].split("_", 1)[-1]
                    if ni in GEX or nj in GEX: continue
                    if (cv2.dilate(mu[i], ker).astype(bool) & ms0[j]).sum() < 30: continue  # 相鄰
                    # 邊界帶
                    aside = (cv2.dilate(mu[j], ker).astype(bool) & ms0[i]) & ~ms0[j] & vld
                    bside = (cv2.dilate(mu[i], ker).astype(bool) & ms0[j]) & ~ms0[i] & vld
                    if aside.sum() < MINPX or bside.sum() < MINPX: continue
                    da = dnear[aside]*1000; db = dnear[bside]*1000
                    bjump = abs(np.median(da)-np.median(db))
                    sa = np.percentile(da, 75)-np.percentile(da, 25); sb = np.percentile(db, 75)-np.percentile(db, 25)
                    bjump_rel = bjump/((sa+sb)/2+1e-6)
                    # 對照:整非重疊區中位差(絕對+相對)
                    ao = ms0[i] & ~ms0[j] & vld; bo = ms0[j] & ~ms0[i] & vld
                    if ao.sum() >= 50 and bo.sum() >= 50:
                        ra = dnear[ao]*1000; rb = dnear[bo]*1000
                        reg = abs(np.median(ra)-np.median(rb))
                        rsa = np.percentile(ra, 75)-np.percentile(ra, 25); rsb = np.percentile(rb, 75)-np.percentile(rb, 25)
                        reg_rel = reg/((rsa+rsb)/2+1e-6)
                    else:
                        reg = -1; reg_rel = -1
                    tall = int(ni in TALL or nj in TALL)
                    rows.append([sc, vn, ni, nj, int(objs[i] == objs[j]), round(bjump, 1), round(bjump_rel, 2),
                                 round(reg, 1), round(reg_rel, 2), tall])
    return rows


if __name__ == "__main__":
    rows = run(sys.argv[1:])
    with open(HERE/"boundary_pairs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene", "view", "objA", "objB", "same", "boundary_jump_mm", "boundary_rel", "region_diff_mm", "region_rel", "tall"]); w.writerows(rows)
    A = np.array(rows, dtype=object)
    if len(A) == 0:
        print("無相鄰對"); sys.exit()
    same = A[:, 4].astype(int) == 1
    bj = A[:, 5].astype(float); bjr = A[:, 6].astype(float); rg = A[:, 7].astype(float); rgr = A[:, 8].astype(float)
    tall = A[:, 9].astype(int) == 1; hasreg = rg >= 0
    print(f"相鄰對 n={len(A)}(同物 {int(same.sum())}/異物 {int((~same).sum())}); 高物涉入 {int(tall.sum())}")
    print(f"\n{'訊號':<18}{'AUC(異>同)':>12}{'同物中位':>10}{'異物中位':>10}")
    for nm, v, msk in [("邊界跳變 絕對", bj, np.ones(len(A), bool)), ("邊界跳變 相對", bjr, np.ones(len(A), bool)),
                       ("整區中位差 絕對", rg, hasreg), ("整區中位差 相對", rgr, hasreg)]:
        a = auc(v[msk & ~same], v[msk & same])
        print(f"{nm:<18}{a:>12.3f}{np.median(v[msk&same]):>10.1f}{np.median(v[msk&~same]):>10.1f}")
    print(f"\n高物「同物」對的誤拆比較(相對值門檻>1=誤判異物):")
    st = same & tall
    print(f"  邊界跳變 相對>1:  高物誤拆 {(bjr[st]>1).mean()*100:.0f}% (n={int(st.sum())})")
    print(f"  整區中位差 相對>1:高物誤拆 {(rgr[st&hasreg]>1).mean()*100:.0f}% (n={int((st&hasreg).sum())})")
    print(f"\nboundary_pairs.csv → {HERE}")
