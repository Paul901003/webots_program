#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""clean_measure — 乾淨量測:四類相鄰群對 × 各訊號,不混母體。得「堆疊分離」乾淨 AUC。

四類(GT 判):同物過切 / 相疊on(relations on) / 並排相觸(異物+3D連通) / 並排有縫(異物+不連通)。
訊號:平面階梯 median、平面階梯 min(跨視角)、z重疊(voxel 高度區間)、空間連通(0/1)。
AUC(相疊 > 同物過切)= 每訊號「能否把相疊和過切分開」的乾淨分數。
群 = srp_hull_semcluster_surf_am1photo 的 instance(未合併=CLIP群)。排 GLOBAL_EXCLUDE(兩物都非才計)。
env:SAM_ROOT CAPTURES_ROOT。用法: ./clean_measure.py <scenes...>
"""
import sys, json
import numpy as np, cv2
from collections import defaultdict, Counter
from pathlib import Path
from scipy import ndimage
from scipy.stats import rankdata
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO/"srp/io")); sys.path.insert(0, str(REPO/"srp/stage2_instances"))
import camera as cam, masks as MK, viewpoints as VP, labels as L
import eval_mesh as EM
EVAL = REPO/"data"/"eval"; MV2 = EVAL/"mobilesamv2_fast"; CAP = REPO/"data"/"captures_fast"
INST = "srp_hull_semcluster_surf_am1photo"; HULL = "srp_hull_mv2_v12_am1_photo"; HERE = Path(__file__).resolve().parent
GEX = {"skillet_lid", "windex_bottle", "colored_wood_blocks", "dice"}
KB = 7


def robust_plane(xs, ys, zs, it=2):
    keep = np.ones(len(zs), bool); coef = None
    for _ in range(it+1):
        if keep.sum() < 5: break
        A = np.c_[xs[keep], ys[keep], np.ones(keep.sum())]; coef, *_ = np.linalg.lstsq(A, zs[keep], rcond=None)
        r = zs-(coef[0]*xs+coef[1]*ys+coef[2]); m = np.median(r[keep]); mad = np.median(np.abs(r[keep]-m))+1e-6
        keep = np.abs(r-m) < 2.5*1.4826*mad
    return coef


def surface(o):
    s = np.zeros_like(o)
    s[1:-1, 1:-1, 1:-1] = o[1:-1, 1:-1, 1:-1] & ~(o[:-2, 1:-1, 1:-1] & o[2:, 1:-1, 1:-1] & o[1:-1, :-2, 1:-1] &
                                                  o[1:-1, 2:, 1:-1] & o[1:-1, 1:-1, :-2] & o[1:-1, 1:-1, 2:])
    return s


def zov(a, b):
    lo = max(a[0], b[0]); hi = min(a[1], b[1]); inter = max(0, hi-lo)
    return inter/max(min(a[1]-a[0], b[1]-b[0]), 1e-6)


def relations_on(sc):
    """回 set{frozenset(shortA,shortB)} for type==on。"""
    try:
        rel = json.loads((L.label_dir(sc)/"relations.json").read_text())
    except Exception:
        return set()
    s = set()
    for r in rel.get("relations", []):
        if r.get("type") == "on":
            x = r.get("x", "").split("_", 1)[-1]; y = r.get("y", "").split("_", 1)[-1]
            if x and y: s.add(frozenset((x, y)))
    return s


def auc(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    r = rankdata(np.concatenate([pos, neg])); U = r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2
    return U/(len(pos)*len(neg))


def process(sc, offs, ker):
    d = json.loads((EVAL/INST/sc/"instances.json").read_text())
    z = np.load(EVAL/INST/sc/"instances.npz", allow_pickle=True); labels = z["labels"]
    zz = np.load(EVAL/HULL/sc/"hull.npz"); occ = zz["occupancy"]; gm = zz["grid_min"]; vs = float(zz["voxel_size"])
    surf = surface(occ); sidx = np.argwhere(surf); Pw = gm+(sidx+0.5)*vs
    insts = [it["instance"] for it in d["instances"]]
    imasks = {it["instance"]: it["masks"] for it in d["instances"]}
    ivox = {i: np.argwhere(labels == i) for i in insts}
    zint = {i: (ivox[i][:, 2].min(), ivox[i][:, 2].max()) if len(ivox[i]) else (0, 0) for i in insts}
    # 群屬 GT 物體(mesh argmax)
    gtocc = EM.solid_mesh_occ(sc, gm, vs, labels.shape)
    iobj = {}
    for i in insts:
        m = (labels == i); ov = {k: int((m & v).sum()) for k, v in gtocc.items()}
        b = max(ov, key=ov.get) if ov else None; iobj[i] = b if (b and ov[b] > 0) else None
    onset = relations_on(sc)

    def connected(i, j):
        vi, vj = ivox[i], ivox[j]
        if len(vi) == 0 or len(vj) == 0: return False
        lo = np.minimum(vi.min(0), vj.min(0))-1; hi = np.maximum(vi.max(0), vj.max(0))+1; sh = hi-lo+1
        a = np.zeros(sh, bool); a[tuple((vi-lo).T)] = True
        b = np.zeros(sh, bool); b[tuple((vj-lo).T)] = True
        return bool((ndimage.binary_dilation(a) & b).any())
    # plane-step per view
    sdir = CAP/f"multi_{sc.split('_')[0]}"/sc; steps = defaultdict(list); adj = set()
    for vn in sorted(VP.selected_view_names(12)):
        pf = sdir/f"{vn}_pose.json"
        if not pf.is_file(): continue
        km = MK.kept_object_masks(MV2/sc/vn); n2m = {n: m for m, n in km}
        gmask = {}
        for i in insts:
            mm = None
            for nm in imasks[i].get(vn, []):
                m = n2m.get(nm)
                if m is not None: mm = m if mm is None else (mm | m)
            if mm is not None and int(mm.sum()) > 30: gmask[i] = mm
        gids = list(gmask)
        if len(gids) < 2: continue
        H, W = gmask[gids[0]].shape
        C, Rb = cam.load_pose(pf); Rwc, t = cam.pose_to_w2c(C, Rb); K = cam.intrinsics(W, H)
        X = Pw@Rwc.T+t; zc = X[:, 2]; ok = zc > 1e-6; zc2 = np.where(ok, zc, 1.0)
        px = np.round(K[0, 0]*X[:, 0]/zc2+K[0, 2]).astype(int); py = np.round(K[1, 1]*X[:, 1]/zc2+K[1, 2]).astype(int)
        inb = ok & (px >= 0) & (px < W) & (py >= 0) & (py < H); ii = np.where(inb)[0]
        dn = np.full((H, W), np.inf); yy = py[ii]; xx = px[ii]; zv = zc[ii]
        for dx, dy in offs:
            np.minimum.at(dn, (np.clip(yy+dy, 0, H-1), np.clip(xx+dx, 0, W-1)), zv)
        vld = np.isfinite(dn); mu = {i: gmask[i].astype(np.uint8) for i in gids}
        for ai in range(len(gids)):
            for bi in range(ai+1, len(gids)):
                a, b = gids[ai], gids[bi]
                if (cv2.dilate(mu[a], ker).astype(bool) & gmask[b]).sum() < 30: continue
                adj.add((min(a, b), max(a, b)))
                ao = gmask[a] & ~gmask[b] & vld; bo = gmask[b] & ~gmask[a] & vld
                if ao.sum() < 50 or bo.sum() < 50: continue
                bz = cv2.dilate(mu[a], ker).astype(bool) & cv2.dilate(mu[b], ker).astype(bool)
                byy, bxx = np.where(bz)
                if len(bxx) < 5: continue
                ay, ax = np.where(ao); by, bx = np.where(bo)
                pa = robust_plane(ax.astype(float), ay.astype(float), dn[ao]); pb = robust_plane(bx.astype(float), by.astype(float), dn[bo])
                if pa is None or pb is None: continue
                cx, cy = bxx.mean(), byy.mean()
                s = abs((pa[0]*cx+pa[1]*cy+pa[2])-(pb[0]*cx+pb[1]*cy+pb[2]))*1000
                na = np.array([-pa[0], -pa[1], 1.0]); nb = np.array([-pb[0], -pb[1], 1.0])
                na /= np.linalg.norm(na); nb /= np.linalg.norm(nb)
                kink = np.degrees(np.arccos(np.clip(na@nb, -1, 1)))   # 兩面法向夾角(kink 折角)
                steps[(min(a, b), max(a, b))].append((s, kink))
    recs = []
    for key in adj:
        i, j = key
        oi, oj = iobj[i], iobj[j]
        if oi is None or oj is None: continue
        ni = oi.split("_", 1)[-1]; nj = oj.split("_", 1)[-1]
        if ni in GEX or nj in GEX: continue
        sl = steps.get(key)
        if not sl: continue
        svals = [x[0] for x in sl]; kvals = [x[1] for x in sl]
        conn = connected(i, j); zo = zov(zint[i], zint[j])
        if oi == oj: cls = "同物過切"
        elif frozenset((ni, nj)) in onset: cls = "相疊on"
        elif conn: cls = "並排相觸"
        else: cls = "並排有縫"
        recs.append([sc, i, j, cls, round(float(np.median(svals)), 1), round(float(min(svals)), 1),
                     round(float(np.median(kvals)), 1), round(zo, 2), int(conn), ni, nj])
    return recs


def main():
    import csv
    R = 3; offs = [(dx, dy) for dx in range(-R, R+1) for dy in range(-R, R+1) if dx*dx+dy*dy <= R*R]
    ker = np.ones((KB, KB), np.uint8); rows = []
    for si, sc in enumerate(sys.argv[1:]):
        try: rows += process(sc, offs, ker)
        except Exception as e: print(f"[fail] {sc}: {e}")
        if (si+1) % 50 == 0: print(f"  ..{si+1}", flush=True)
    with open(HERE/"clean_pairs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene", "i", "j", "class", "plane_med", "plane_min", "kink_deg", "zov", "conn", "objA", "objB"]); w.writerows(rows)
    A = rows
    cls_of = {"同物過切": [], "相疊on": [], "並排相觸": [], "並排有縫": []}
    for r in A: cls_of.setdefault(r[3], []).append(r)
    print(f"\n===== 四類相鄰群對(n={len(A)}) =====")
    print(f"{'類型':<10}{'n':>5}{'平面med':>9}{'平面min':>9}{'kink度':>9}{'z重疊':>8}{'連通%':>8}")
    for c in ["同物過切", "相疊on", "並排相觸", "並排有縫"]:
        rs = cls_of.get(c, [])
        if not rs: print(f"{c:<10}{0:>5}"); continue
        pm = np.array([r[4] for r in rs]); pn = np.array([r[5] for r in rs]); kk = np.array([r[6] for r in rs]); zo = np.array([r[7] for r in rs]); cn = np.array([r[8] for r in rs])
        print(f"{c:<10}{len(rs):>5}{np.median(pm):>9.0f}{np.median(pn):>9.0f}{np.median(kk):>9.1f}{np.median(zo):>8.2f}{100*cn.mean():>7.0f}%")
    same = cls_of.get("同物過切", []); onp = cls_of.get("相疊on", [])
    if same and onp:
        print(f"\n★ 乾淨 AUC(相疊on > 同物過切):")
        for idx, nm in [(4, "平面階梯 median"), (5, "平面階梯 min"), (6, "kink 法向夾角")]:
            print(f"    {nm:<16} AUC = {auc([r[idx] for r in onp], [r[idx] for r in same]):.3f}")
        print(f"    {'z重疊(低=相疊)':<16} AUC = {auc([r[7] for r in same], [r[7] for r in onp]):.3f}")


if __name__ == "__main__":
    main()
