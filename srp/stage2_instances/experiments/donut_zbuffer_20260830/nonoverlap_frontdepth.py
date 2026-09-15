#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""nonoverlap_frontdepth — 每對遮罩「非重疊區」前(最近)表面深度差,分同/異物體。

修正前錯:整遮罩取中位會被重疊區(前物表面)洗掉後物的遠深度。改比 A\B 與 B\A 非重疊區。
前表面=hull 表面 voxel splat 後每 pixel min-z。每對:|median(前深度 A-only) − median(B-only)|。
GT modal 判每遮罩物體,同物對(過切,該合)/異物對(相鄰,該分)。排 GLOBAL_EXCLUDE。
env:SAM_ROOT CAPTURES_ROOT。用法: ./nonoverlap_frontdepth.py stack4_scene0001 ...
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
MINPX = 50


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
    R = 3; offs = [(dx, dy) for dx in range(-R, R+1) for dy in range(-R, R+1) if dx*dx+dy*dy <= R*R]
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
            for i in range(len(ms0)):
                for j in range(i+1, len(ms0)):
                    if objs[i] is None or objs[j] is None: continue
                    ni = objs[i].split("_", 1)[-1]; nj = objs[j].split("_", 1)[-1]
                    if ni in GEX or nj in GEX: continue
                    aonly = ms0[i] & ~ms0[j] & vld; bonly = ms0[j] & ~ms0[i] & vld
                    if aonly.sum() < MINPX or bonly.sum() < MINPX: continue
                    # 只看 2D 相鄰對(膨脹相交),遠離的異物太好分沒意義
                    adj = (cv2.dilate(ms0[i].astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ms0[j]).sum() > 30
                    if not adj: continue
                    da = dnear[aonly]*1000; db = dnear[bonly]*1000
                    dd = abs(np.median(da)-np.median(db))                       # 絕對 mm
                    # 相對:差 / 區內深度散度(IQR 當 spread,類似 signal/noise)
                    sa = np.percentile(da, 75)-np.percentile(da, 25); sb = np.percentile(db, 75)-np.percentile(db, 25)
                    drel = dd/((sa+sb)/2+1e-6)
                    rows.append([sc, vn, ni, nj, int(objs[i] == objs[j]), round(dd, 1), round(drel, 2)])
    return rows


if __name__ == "__main__":
    rows = run(sys.argv[1:])
    with open(HERE/"nonoverlap_pairs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene", "view", "objA", "objB", "same", "front_depth_diff_mm", "rel_diff"]); w.writerows(rows)
    A = np.array(rows, dtype=object)
    if len(A) == 0:
        print("無相鄰對"); sys.exit()
    same = A[:, 4].astype(int) == 1; d = A[:, 5].astype(float); dr = A[:, 6].astype(float)
    print(f"2D 相鄰遮罩對 n={len(A)}(同物 {int(same.sum())}(過切該合) / 異物 {int((~same).sum())}(相鄰該分))")
    print(f"\n[絕對] 非重疊區前表面深度差(mm):")
    print(f"  同物: 中位{np.median(d[same]):.0f} 75%{np.percentile(d[same],75):.0f} | 異物: 中位{np.median(d[~same]):.0f} 75%{np.percentile(d[~same],75):.0f} | AUC(異>同)={auc(d[same],d[~same]):.3f}")
    for thr in [20, 30, 50, 80]:
        print(f"    門檻{thr}mm判異物: 異物正確{(d[~same]>thr).mean()*100:.0f}% / 同物誤判{(d[same]>thr).mean()*100:.0f}%")
    print(f"\n[相對] 差 / 區內深度散度:")
    print(f"  同物: 中位{np.median(dr[same]):.2f} 75%{np.percentile(dr[same],75):.2f} | 異物: 中位{np.median(dr[~same]):.2f} 75%{np.percentile(dr[~same],75):.2f} | AUC(異>同)={auc(dr[same],dr[~same]):.3f}")
    for thr in [1, 2, 3, 5]:
        print(f"    門檻{thr}判異物: 異物正確{(dr[~same]>thr).mean()*100:.0f}% / 同物誤判{(dr[same]>thr).mean()*100:.0f}%")
    print(f"\nnonoverlap_pairs.csv → {HERE}")
