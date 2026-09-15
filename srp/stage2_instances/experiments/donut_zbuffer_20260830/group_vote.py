#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""group_vote — 語意群對 × 各視角平面階梯投票(異物票),看能否據以合併。

每視角:同 gid 遮罩聯集=群遮罩;相鄰群對算絕對平面階梯(穩健平面外插到邊界),>THR 記一票異物。
彙整每對群:相鄰視角數 n_adj、異物票 n_diff、frac=n_diff/n_adj。GT 標該對同物(過切該合)/異物(該分)。
看 same-object 群對是否 frac 低、diff-object 是否 frac 高 → 可否用 frac 門檻合併。
env:SAM_ROOT CAPTURES_ROOT。用法: ./group_vote.py <scenes...> [--thr 40]
"""
import sys, json, argparse
import numpy as np, cv2
from collections import defaultdict
from pathlib import Path
from pycocotools import mask as cocomask
from scipy.stats import rankdata
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO/"srp/io")); sys.path.insert(0, str(REPO/"srp/stage2_instances"))
import camera as cam, masks as MK, viewpoints as VP, labels as L
EVAL = REPO/"data"/"eval"; MV2 = EVAL/"mobilesamv2_fast"; CAP = REPO/"data"/"captures_fast"
INST = "srp_hull_semcluster_surf_am1photo"; HULL = "srp_hull_mv2_v12_am1_photo"
HERE = Path(__file__).resolve().parent
GEX = {"skillet_lid", "windex_bottle", "colored_wood_blocks", "dice"}
KB = 7; MINPX = 15


def robust_plane(xs, ys, zs, iters=2):
    keep = np.ones(len(zs), bool); coef = None
    for _ in range(iters+1):
        if keep.sum() < 5: break
        Aa = np.c_[xs[keep], ys[keep], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(Aa, zs[keep], rcond=None)
        resid = zs-(coef[0]*xs+coef[1]*ys+coef[2])
        m = np.median(resid[keep]); mad = np.median(np.abs(resid[keep]-m))+1e-6
        keep = np.abs(resid-m) < 2.5*1.4826*mad
    return coef


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


def objof(mask, gmv):
    a = int(mask.sum()); best = None; bc = 0.3
    for on, om in gmv.items():
        cov = (mask & om).sum()/max(a, 1)
        if cov > bc: bc = cov; best = on
    return best


def auc(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    r = rankdata(np.concatenate([pos, neg])); U = r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2
    return U/(len(pos)*len(neg))


def process(sc, THR, R, offs, ker):
    mc = json.loads((EVAL/INST/sc/"instances.json").read_text()).get("mask_clusters", {})
    z = np.load(EVAL/HULL/sc/"hull.npz"); occ = z["occupancy"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
    surf = surface(occ); sidx = np.argwhere(surf); Pw = gm+(sidx+0.5)*vs
    md = modal(sc); sdir = CAP/f"multi_{sc.split('_')[0]}"/sc
    grp_obj = defaultdict(lambda: defaultdict(int))   # gid -> {gt_obj: count}(多數決)
    steps = defaultdict(list)   # (a,b) -> 各相鄰視角 step(mm)
    for vn in sorted(VP.selected_view_names(12)):
        cl = mc.get(vn, {}); pf = sdir/f"{vn}_pose.json"
        if not cl or not pf.is_file(): continue
        km = MK.kept_object_masks(MV2/sc/vn); name2m = {n: m for m, n in km}
        gmask = defaultdict(lambda: None)
        for name, gid in cl.items():
            m = name2m.get(name)
            if m is None: continue
            gmask[gid] = m if gmask[gid] is None else (gmask[gid] | m)
        gids = [g for g in gmask if gmask[g] is not None and int(gmask[g].sum()) > 30]
        if len(gids) < 2: continue
        H, W = gmask[gids[0]].shape
        gmv = md.get(vn, {})
        for g in gids:
            o = objof(gmask[g], gmv)
            if o: grp_obj[g][o] += 1
        C, Rb = cam.load_pose(pf); Rwc, t = cam.pose_to_w2c(C, Rb); K = cam.intrinsics(W, H)
        X = Pw@Rwc.T+t; zc = X[:, 2]; ok = zc > 1e-6; zz = np.where(ok, zc, 1.0)
        px = np.round(K[0, 0]*X[:, 0]/zz+K[0, 2]).astype(int); py = np.round(K[1, 1]*X[:, 1]/zz+K[1, 2]).astype(int)
        inb = ok & (px >= 0) & (px < W) & (py >= 0) & (py < H); ii = np.where(inb)[0]
        dnear = np.full((H, W), np.inf); yy = py[ii]; xx = px[ii]; zv = zc[ii]
        for dx, dy in offs:
            np.minimum.at(dnear, (np.clip(yy+dy, 0, H-1), np.clip(xx+dx, 0, W-1)), zv)
        vld = np.isfinite(dnear)
        mu = {g: gmask[g].astype(np.uint8) for g in gids}
        for ai in range(len(gids)):
            for bi in range(ai+1, len(gids)):
                a, b = gids[ai], gids[bi]
                if (cv2.dilate(mu[a], ker).astype(bool) & gmask[b]).sum() < 30: continue
                ao = gmask[a] & ~gmask[b] & vld; bo = gmask[b] & ~gmask[a] & vld
                if ao.sum() < 50 or bo.sum() < 50: continue
                bzone = cv2.dilate(mu[a], ker).astype(bool) & cv2.dilate(mu[b], ker).astype(bool)
                byy, bxx = np.where(bzone)
                if len(bxx) < 5: continue
                ay, ax_ = np.where(ao); by, bx = np.where(bo)
                pa = robust_plane(ax_.astype(float), ay.astype(float), dnear[ao])
                pb = robust_plane(bx.astype(float), by.astype(float), dnear[bo])
                if pa is None or pb is None: continue
                cx, cy = bxx.mean(), byy.mean()
                step = abs((pa[0]*cx+pa[1]*cy+pa[2])-(pb[0]*cx+pb[1]*cy+pb[2]))*1000
                steps[(min(a, b), max(a, b))].append(step)
    # 群對 GT 標籤
    gid_obj = {g: max(c, key=c.get) if c else None for g, c in grp_obj.items()}
    recs = []
    for key, sl in steps.items():
        oa = gid_obj.get(key[0]); ob = gid_obj.get(key[1])
        if oa is None or ob is None: continue
        na = oa.split("_", 1)[-1]; nb = ob.split("_", 1)[-1]
        if na in GEX or nb in GEX: continue
        sl = np.array(sl); n_adj = len(sl); n_diff = int((sl > THR).sum())
        n_cont = int((sl <= THR).sum())            # 連續視角數
        recs.append((sc, key, n_adj, n_diff, n_cont, float(sl.min()), float(np.median(sl)), int(oa == ob), na, nb))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--thr", type=float, default=40.0)
    a = ap.parse_args()
    R = 3; offs = [(dx, dy) for dx in range(-R, R+1) for dy in range(-R, R+1) if dx*dx+dy*dy <= R*R]
    ker = np.ones((KB, KB), np.uint8)
    import csv
    allrecs = []
    for sc in a.scenes:
        try:
            allrecs += process(sc, a.thr, R, offs, ker)
        except Exception as e:
            print(f"[fail] {sc}: {e}")
    with open(HERE/"group_pairs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene", "gidA", "gidB", "n_adj", "n_diff", "n_cont", "min_step", "median_step", "same", "objA", "objB"])
        for sc_, key, nadj, ndiff, ncont, mn, med, same, na, nb in allrecs:
            w.writerow([sc_, key[0], key[1], nadj, ndiff, ncont, round(mn, 1), round(med, 1), same, na, nb])
    A = allrecs
    same = np.array([r[7] for r in A]) == 1
    ncont = np.array([r[4] for r in A]); nadj = np.array([r[2] for r in A]); mn = np.array([r[5] for r in A])
    frac_diff = np.array([r[3]/max(r[2], 1) for r in A])
    print(f"group_pairs.csv 寫入 {len(A)} 對")
    print(f"\n===== 全部 群對 n={len(A)}(同物 {int(same.sum())}/異物 {int((~same).sum())}), THR={a.thr}mm =====")
    print(f"[舊 多數異物票 frac_diff] AUC={auc(frac_diff[~same],frac_diff[same]):.3f}")
    print(f"[min step] 同物中位{np.median(mn[same]):.0f} 異物中位{np.median(mn[~same]):.0f}mm | AUC(異>同)={auc(mn[~same],mn[same]):.3f}")
    print(f"[連續視角數 ncont] 同物中位{np.median(ncont[same]):.0f} 異物中位{np.median(ncont[~same]):.0f}")
    print(f"\n合併規則對照(判「同物該合」):")
    print(f"  規則A 舊-多數連續(ncont/nadj≥0.5): 同物正確合 {(ncont[same]/nadj[same]>=0.5).mean()*100:.0f}% / 異物誤合 {(ncont[~same]/nadj[~same]>=0.5).mean()*100:.0f}%")
    for k in [1, 2]:
        sc_ok = (ncont[same] >= k).mean()*100; df_bad = (ncont[~same] >= k).mean()*100
        print(f"  規則B 連續視角≥{k} 就合: 同物正確合 {sc_ok:.0f}% / 異物誤合 {df_bad:.0f}%")
    for mt in [30, 40, 50]:
        print(f"  規則C min step<{mt}mm 就合: 同物正確合 {(mn[same]<mt).mean()*100:.0f}% / 異物誤合 {(mn[~same]<mt).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
