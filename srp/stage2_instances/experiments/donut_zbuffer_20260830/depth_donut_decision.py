#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""depth_donut_decision — 父子遮罩最深表面深度差,當 donut 挖洞決策判準。

使用者定義:父子(含入)遮罩若深度差不多→同物→不挖洞;深度有差→子是前面別的物→挖洞。
測三種深度統計:中位/平均/最大。

每視角:全 occupancy 投影,每 pixel 取最深(max-z)= hull 背面深度圖。
含入對:子 C ⊂ 母 P(|C∩P|/area(C)≥0.8, area(P)>area(C));母環=P\C。
子/環各要 ≥50 有效 pixel。對 中位/平均/最大 各算 |深度(子)−深度(環)|。
GT modal 判子/母環物體 → 同物對(該不挖)/異物對(該挖)。排除 GLOBAL_EXCLUDE。
env:SAM_ROOT CAPTURES_ROOT。用法: ./depth_donut_decision.py stack4_scene0001 ...
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
        vidx = np.argwhere(occ); Pw = gm+(vidx+0.5)*vs
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
            dfar = np.full((H, W), -np.inf)
            np.maximum.at(dfar, (py[ii], px[ii]), zc[ii])
            vld = np.isfinite(dfar)
            gmv = md.get(vn, {}); areas = [int(m.sum()) for m in ms0]
            for i in range(len(ms0)):
                for j in range(len(ms0)):
                    if i == j: continue
                    Cm, Pm = ms0[i], ms0[j]; ac = areas[i]
                    if ac == 0 or areas[j] <= ac: continue
                    if (Cm & Pm).sum()/ac < 0.8: continue
                    ring = Pm & ~Cm
                    cv_ = Cm & vld; rv = ring & vld
                    if cv_.sum() < MINPX or rv.sum() < MINPX: continue
                    dc = dfar[cv_]*1000; dr = dfar[rv]*1000
                    d_med = abs(np.median(dc)-np.median(dr))
                    d_mean = abs(dc.mean()-dr.mean())
                    d_max = abs(dc.max()-dr.max())
                    oc = objof(Cm, gmv); orr = objof(ring, gmv)
                    if oc is None or orr is None: continue
                    nc = oc.split("_", 1)[-1]; nr = orr.split("_", 1)[-1]
                    if nc in GEX or nr in GEX: continue
                    rows.append([sc, vn, nc, nr, int(oc == orr), round(d_med, 1), round(d_mean, 1), round(d_max, 1)])
    return rows


if __name__ == "__main__":
    rows = run(sys.argv[1:])
    with open(HERE/"depth_donut_pairs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene", "view", "childobj", "ringobj", "same", "d_median_mm", "d_mean_mm", "d_max_mm"]); w.writerows(rows)
    A = np.array(rows, dtype=object)
    if len(A) == 0:
        print("無含入對"); sys.exit()
    same = A[:, 4].astype(int) == 1
    print(f"含入對 n={len(A)}(同物 {int(same.sum())}(該不挖) / 異物 {int((~same).sum())}(該挖))")
    for col, nm in [(5, "中位深度差"), (6, "平均深度差"), (7, "最大深度差")]:
        d = A[:, col].astype(float); s = d[same]; t = d[~same]
        a = auc(s, t)  # P(異物 |差| > 同物)
        print(f"\n{nm}(mm, |子−環|):")
        print(f"  同物: 中位{np.median(s):.0f} 75%{np.percentile(s,75):.0f} | 異物: 中位{np.median(t):.0f} 75%{np.percentile(t,75):.0f} | AUC(異>同)={a:.3f}")
        for thr in [20, 40, 60, 100]:
            print(f"    門檻{thr}mm挖洞: 異物正確挖{(t>thr).mean()*100:.0f}% / 同物誤挖{(s>thr).mean()*100:.0f}%")
    print(f"\ndepth_donut_pairs.csv → {HERE}")
