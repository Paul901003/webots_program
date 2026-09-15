#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""cliff_scan — 逐點掃相機深度「切面」(局部深度突跳),分相疊on vs 同物過切。

每視角:兩群共享邊界帶,沿邊界每隔幾點取一點;該點取 11x11 窗,窗內 A-pixel 前深度中位 vs B-pixel 中位 → 局部切面 |差|。
沿整條邊界取 80 分位 = 該視角該對切面強度;跨視角取中位。與 plane-step 差別=逐點掃(抓局部肩台)非中心一點。
GT 分類(同 clean_measure)。輸出各類分佈 + AUC(相疊on > 同物過切)。
env:SAM_ROOT CAPTURES_ROOT。用法: ./cliff_scan.py <scenes...>
"""
import sys, json
import numpy as np, cv2
from collections import defaultdict
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
KB = 7; WIN = 5; STEP = 3


def surface(o):
    s = np.zeros_like(o)
    s[1:-1, 1:-1, 1:-1] = o[1:-1, 1:-1, 1:-1] & ~(o[:-2, 1:-1, 1:-1] & o[2:, 1:-1, 1:-1] & o[1:-1, :-2, 1:-1] &
                                                  o[1:-1, 2:, 1:-1] & o[1:-1, 1:-1, :-2] & o[1:-1, 1:-1, 2:])
    return s


def zov(a, b):
    lo = max(a[0], b[0]); hi = min(a[1], b[1]); return max(0, hi-lo)/max(min(a[1]-a[0], b[1]-b[0]), 1e-6)


def auc(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    r = rankdata(np.concatenate([pos, neg])); U = r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2
    return U/(len(pos)*len(neg))


def objof2(mask, gtsurf):  # 用 mesh occ 判(和 clean_measure 一致改用 mesh)
    pass


def relations_on(sc):
    try: rel = json.loads((L.label_dir(sc)/"relations.json").read_text())
    except Exception: return set()
    s = set()
    for r in rel.get("relations", []):
        if r.get("type") == "on":
            x = r.get("x", "").split("_", 1)[-1]; y = r.get("y", "").split("_", 1)[-1]
            if x and y: s.add(frozenset((x, y)))
    return s


def process(sc, offs, ker):
    d = json.loads((EVAL/INST/sc/"instances.json").read_text())
    z = np.load(EVAL/INST/sc/"instances.npz", allow_pickle=True); labels = z["labels"]
    zz = np.load(EVAL/HULL/sc/"hull.npz"); occ = zz["occupancy"]; gm = zz["grid_min"]; vs = float(zz["voxel_size"])
    surf = surface(occ); sidx = np.argwhere(surf); Pw = gm+(sidx+0.5)*vs
    insts = [it["instance"] for it in d["instances"]]; imasks = {it["instance"]: it["masks"] for it in d["instances"]}
    ivox = {i: np.argwhere(labels == i) for i in insts}
    zint = {i: (ivox[i][:, 2].min(), ivox[i][:, 2].max()) if len(ivox[i]) else (0, 0) for i in insts}
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
    sdir = CAP/f"multi_{sc.split('_')[0]}"/sc; cliffs = defaultdict(list)
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
                bz = cv2.dilate(mu[a], ker).astype(bool) & cv2.dilate(mu[b], ker).astype(bool)
                if bz.sum() < 30: continue
                Am = gmask[a]; Bm = gmask[b]
                byy, bxx = np.where(bz); cs = []
                for idx in range(0, len(byy), STEP):
                    y, x = byy[idx], bxx[idx]
                    y0, y1, x0, x1 = max(y-WIN, 0), min(y+WIN+1, H), max(x-WIN, 0), min(x+WIN+1, W)
                    wA = Am[y0:y1, x0:x1] & vld[y0:y1, x0:x1] & ~Bm[y0:y1, x0:x1]
                    wB = Bm[y0:y1, x0:x1] & vld[y0:y1, x0:x1] & ~Am[y0:y1, x0:x1]
                    dsub = dn[y0:y1, x0:x1]
                    if wA.sum() >= 5 and wB.sum() >= 5:
                        cs.append(abs(np.median(dsub[wA])-np.median(dsub[wB]))*1000)
                if len(cs) >= 3:
                    cliffs[(min(a, b), max(a, b))].append(np.percentile(cs, 80))
    recs = []
    for key, cl in cliffs.items():
        i, j = key; oi, oj = iobj[i], iobj[j]
        if oi is None or oj is None: continue
        ni = oi.split("_", 1)[-1]; nj = oj.split("_", 1)[-1]
        if ni in GEX or nj in GEX: continue
        if oi == oj: c = "同物過切"
        elif frozenset((ni, nj)) in onset: c = "相疊on"
        elif connected(i, j): c = "並排相觸"
        else: c = "並排有縫"
        recs.append((c, float(np.median(cl)), zov(zint[i], zint[j]), ni, nj, sc))
    return recs


def main():
    import csv
    R = 3; offs = [(dx, dy) for dx in range(-R, R+1) for dy in range(-R, R+1) if dx*dx+dy*dy <= R*R]
    ker = np.ones((KB, KB), np.uint8); out = []
    for si, sc in enumerate(sys.argv[1:]):
        try: out += process(sc, offs, ker)
        except Exception as e: print(f"[fail] {sc}: {e}")
        if (si+1) % 50 == 0: print(f"  ..{si+1}", flush=True)
    with open(HERE/"cliff_pairs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["class", "cliff", "zov", "objA", "objB", "scene"]); w.writerows(out)
    cls = defaultdict(list)
    for r in out: cls[r[0]].append(r)
    print(f"\nn={len(out)}")
    print(f"{'類型':<10}{'n':>5}{'切面cliff中位mm':>16}")
    for c in ["同物過切", "相疊on", "並排相觸", "並排有縫"]:
        rs = cls.get(c, [])
        print(f"{c:<10}{len(rs):>5}" + (f"{np.median([r[1] for r in rs]):>16.0f}" if rs else ""))
    same = cls.get("同物過切", []); onp = cls.get("相疊on", [])
    if same and onp:
        print(f"\n★ AUC(相疊on > 同物過切):")
        print(f"    切面cliff       = {auc([r[1] for r in onp],[r[1] for r in same]):.3f}")
        print(f"    z重疊(低=相疊)  = {auc([r[2] for r in same],[r[2] for r in onp]):.3f}")
        combo = [max(r[1]/100, 1-r[2]) for r in onp]; combos = [max(r[1]/100, 1-r[2]) for r in same]
        print(f"    cliff 或 z重疊  = {auc(combo,combos):.3f}")


if __name__ == "__main__":
    main()
