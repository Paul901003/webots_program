#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""plane_step — 穩健平面擬合去斜率,量相鄰群交界深度階梯(解長物誤拆)。

長物斜面=一個平面(固定斜率)、頂面=常數平面;相疊兩物=兩平面有階梯。
每群非重疊區前表面點 (x,y)->depth 用「最小平方 + MAD 去離群(2輪)」擬平面 z=ax+by+c;
兩平面外插到共享邊界中心 → step=|planeA(pt)-planeB(pt)| = 去斜率後真階梯。相對版=step/擬合殘差。
對照整區中位差/邊界跳變。看高物誤拆。
env:SAM_ROOT CAPTURES_ROOT。用法: ./plane_step.py <scenes...>
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
KB = 7; MINPX = 15


def robust_plane(xs, ys, zs, iters=2):
    """最小平方擬 z=ax+by+c,MAD 去離群 2 輪。回 (a,b,c, 殘差std_mm)。"""
    keep = np.ones(len(zs), bool)
    coef = None
    for _ in range(iters+1):
        if keep.sum() < 5: break
        Aa = np.c_[xs[keep], ys[keep], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(Aa, zs[keep], rcond=None)
        pred_all = coef[0]*xs+coef[1]*ys+coef[2]
        resid = zs-pred_all
        m = np.median(resid[keep]); mad = np.median(np.abs(resid[keep]-m))+1e-6
        keep = np.abs(resid-m) < 2.5*1.4826*mad
    if coef is None: return None
    pred = coef[0]*xs+coef[1]*ys+coef[2]
    r = zs[keep]-pred[keep] if keep.sum() > 3 else zs-pred
    return coef[0], coef[1], coef[2], float(np.std(r)*1000)


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
    ker = np.ones((KB, KB), np.uint8); rows = []
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
            dnear = np.full((H, W), np.inf); yy = py[ii]; xx = px[ii]; zv = zc[ii]
            for dx, dy in offs:
                np.minimum.at(dnear, (np.clip(yy+dy, 0, H-1), np.clip(xx+dx, 0, W-1)), zv)
            vld = np.isfinite(dnear)
            gmv = md.get(vn, {}); objs = [objof(m, gmv) for m in ms0]; mu = [m.astype(np.uint8) for m in ms0]
            for i in range(len(ms0)):
                for j in range(i+1, len(ms0)):
                    if objs[i] is None or objs[j] is None: continue
                    ni = objs[i].split("_", 1)[-1]; nj = objs[j].split("_", 1)[-1]
                    if ni in GEX or nj in GEX: continue
                    bzone = cv2.dilate(mu[i], ker).astype(bool) & cv2.dilate(mu[j], ker).astype(bool)
                    if (cv2.dilate(mu[i], ker).astype(bool) & ms0[j]).sum() < 30: continue
                    ao = ms0[i] & ~ms0[j] & vld; bo = ms0[j] & ~ms0[i] & vld
                    if ao.sum() < 50 or bo.sum() < 50: continue
                    ay, ax_ = np.where(ao); by, bx = np.where(bo)
                    pa = robust_plane(ax_.astype(float), ay.astype(float), dnear[ao])
                    pb = robust_plane(bx.astype(float), by.astype(float), dnear[bo])
                    if pa is None or pb is None: continue
                    # 邊界中心點外插兩平面
                    bz = bzone & (ao | bo | vld)
                    byy, bxx = np.where(bzone)
                    if len(bxx) < 5: continue
                    cx, cy = bxx.mean(), byy.mean()
                    za = (pa[0]*cx+pa[1]*cy+pa[2]); zb = (pb[0]*cx+pb[1]*cy+pb[2])
                    step = abs(za-zb)*1000
                    step_rel = step/((pa[3]+pb[3])/2+1e-6)          # ÷擬合殘差
                    bnd_depth = (za+zb)/2                            # 邊界絕對深度(m)
                    step_reldepth = abs(za-zb)/(bnd_depth+1e-6)      # ÷相機距離
                    # ÷「兩群深度去離群後半全距」(max-min)/2
                    allz = np.concatenate([dnear[ao], dnear[bo]])*1000
                    mz = np.median(allz); madz = np.median(np.abs(allz-mz))+1e-6
                    inl = allz[np.abs(allz-mz) < 2.5*1.4826*madz]
                    halfrange = (inl.max()-inl.min())/2 if len(inl) > 3 else (allz.max()-allz.min())/2
                    step_relrange = step/(halfrange+1e-6)
                    # 對照 整區中位差
                    reg = abs(np.median(dnear[ao])-np.median(dnear[bo]))*1000
                    tall = int(ni in TALL or nj in TALL)
                    rows.append([sc, vn, ni, nj, int(objs[i] == objs[j]), round(step, 1), round(step_rel, 2),
                                 round(step_reldepth*1000, 2), round(step_relrange, 2), round(reg, 1), tall])
    return rows


if __name__ == "__main__":
    rows = run(sys.argv[1:])
    with open(HERE/"plane_step_pairs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene", "view", "objA", "objB", "same", "plane_step_mm", "plane_step_rel", "plane_step_reldepth", "plane_step_relrange", "region_diff_mm", "tall"]); w.writerows(rows)
    A = np.array(rows, dtype=object)
    if len(A) == 0:
        print("無相鄰對"); sys.exit()
    same = A[:, 4].astype(int) == 1
    ps = A[:, 5].astype(float); psr = A[:, 6].astype(float); psd = A[:, 7].astype(float); psrr = A[:, 8].astype(float); rg = A[:, 9].astype(float); tall = A[:, 10].astype(int) == 1
    print(f"相鄰對 n={len(A)}(同物 {int(same.sum())}/異物 {int((~same).sum())}); 高物涉入 {int(tall.sum())}")
    print(f"\n{'訊號':<22}{'AUC(異>同)':>12}{'同物中位':>10}{'異物中位':>10}")
    for nm, v in [("平面階梯 絕對mm", ps), ("平面階梯 ÷擬合殘差", psr), ("平面階梯 ÷相機距離‰", psd), ("平面階梯 ÷半全距", psrr), ("整區中位差(對照)", rg)]:
        print(f"{nm:<20}{auc(v[~same], v[same]):>12.3f}{np.median(v[same]):>10.2f}{np.median(v[~same]):>10.2f}")
    st = same & tall
    print(f"\n高物「同物」誤拆:")
    print(f"  平面階梯 絕對>50mm: {(ps[st]>50).mean()*100:.0f}% | 整區>50mm: {(rg[st]>50).mean()*100:.0f}% (n={int(st.sum())})")
    print(f"\nplane_step_pairs.csv → {HERE}")
