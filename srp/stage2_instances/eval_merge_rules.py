#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""eval_merge_rules — 把 plane-step / span 兩訊號做成合併規則,物體級掃門檻評估。

遮罩宇宙、指標、分組套用皆與 eval_merge_methods 相同(固定 baseline 遮罩、無損重組)。
差別:分組不是從現成 root 還原,而是「訊號 + 門檻 → 相鄰群對合併邊 → union-find」。

每對相鄰 baseline 群的訊號(不用 GT):
  plane-step:兩群非重疊區前表面深度各擬合平面,外插到交界,取深度階梯 mm(跨視角中位)。
             規則:階梯 < T → 同物 → 併(T 小→保守)。
  span:各視角是否有「單一 kept MobileSAMv2 遮罩同時覆蓋 a、b 各 ≥0.6」;view-fraction。
        規則:view-frac > S → 同物 → 併(SAM 自己把兩群圈成一塊)。
  無法算 plane-step 的相鄰對(無非重疊前表面)→ 該規則下不併(保守),另計數。

用法: ./eval_merge_rules.py [scenes...]   (不給=全 303 多物場)
輸出:兩規則各自的門檻掃描表 + 對照 baseline;唯讀。
"""
import sys
import os
import json
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
sys.path.insert(0, str(REPO / "srp" / "io"))
import eval_mask_grouping as G
import eval_merge_methods as M
import camera as cam
import masks as MK
import viewpoints as VP

CAP = REPO / "data" / "captures_fast"
HULL = os.environ.get("HULL_ROOT_NAME", "srp_hull_mv2_v12_am1_photo")   # env 覆蓋:am1 vs am1_photo(無光雕);merge_div 經 RU.HULL 沿用
KB = 7
SPAN_COV = 0.6


def robust_plane(xs, ys, zs, it=2):
    keep = np.ones(len(zs), bool); coef = None
    for _ in range(it + 1):
        if keep.sum() < 5:
            break
        A = np.c_[xs[keep], ys[keep], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(A, zs[keep], rcond=None)
        r = zs - (coef[0] * xs + coef[1] * ys + coef[2])
        m = np.median(r[keep]); mad = np.median(np.abs(r[keep] - m)) + 1e-6
        keep = np.abs(r - m) < 2.5 * 1.4826 * mad
    return coef


def surface(o):
    s = np.zeros_like(o)
    s[1:-1, 1:-1, 1:-1] = o[1:-1, 1:-1, 1:-1] & ~(o[:-2, 1:-1, 1:-1] & o[2:, 1:-1, 1:-1] & o[1:-1, :-2, 1:-1] &
                                                  o[1:-1, 2:, 1:-1] & o[1:-1, 1:-1, :-2] & o[1:-1, 1:-1, 2:])
    return s


def union_find(nodes, edges):
    par = {n: n for n in nodes}
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in edges:
        if a in par and b in par:
            par[find(a)] = find(b)
    return {n: find(n) for n in nodes}


def signals(scene, views, bg, offs, ker):
    """回 adj(set), plane_med{pair}, span_frac{pair}。pair=(min,max) baseline gid。"""
    hp = G.EVAL / HULL / scene / "hull.npz"
    if not hp.is_file():
        return set(), {}, {}
    zz = np.load(hp); occ = zz["occupancy"]; gm = zz["grid_min"]; vs = float(zz["voxel_size"])
    sidx = np.argwhere(surface(occ)); Pw = gm + (sidx + 0.5) * vs
    sdir = CAP / f"multi_{scene.split('_')[0]}" / scene
    adj = set()
    steps = defaultdict(list)
    span_hit = defaultdict(int); covis = defaultdict(int)
    for vn in views:
        pf = sdir / f"{vn}_pose.json"
        if not pf.is_file():
            continue
        km = MK.kept_object_masks(G.MV2 / scene / vn)
        n2m = {n: m for m, n in km}
        allkept = [m for m, _ in km]
        gmask = {}
        for gid, pairs in bg.items():
            mm = None
            for (v, nm) in pairs:
                if v != vn:
                    continue
                m = n2m.get(nm)
                if m is not None:
                    mm = m if mm is None else (mm | m)
            if mm is not None and int(mm.sum()) > 30:
                gmask[gid] = mm
        gids = list(gmask)
        if len(gids) < 2:
            continue
        H, W = gmask[gids[0]].shape
        C, Rb = cam.load_pose(pf); Rwc, t = cam.pose_to_w2c(C, Rb); K = cam.intrinsics(W, H)
        X = Pw @ Rwc.T + t; zc = X[:, 2]; ok = zc > 1e-6; zc2 = np.where(ok, zc, 1.0)
        px = np.round(K[0, 0] * X[:, 0] / zc2 + K[0, 2]).astype(int)
        py = np.round(K[1, 1] * X[:, 1] / zc2 + K[1, 2]).astype(int)
        inb = ok & (px >= 0) & (px < W) & (py >= 0) & (py < H); ii = np.where(inb)[0]
        dn = np.full((H, W), np.inf); yy = py[ii]; xx = px[ii]; zv = zc[ii]
        for dx, dy in offs:
            np.minimum.at(dn, (np.clip(yy + dy, 0, H - 1), np.clip(xx + dx, 0, W - 1)), zv)
        vld = np.isfinite(dn)
        mu = {g: gmask[g].astype(np.uint8) for g in gids}
        area = {g: int(gmask[g].sum()) for g in gids}
        for i in range(len(gids)):
            for j in range(i + 1, len(gids)):
                a, b = gids[i], gids[j]
                if (cv2.dilate(mu[a], ker).astype(bool) & gmask[b]).sum() < 30:
                    continue
                key = (min(a, b), max(a, b))
                adj.add(key); covis[key] += 1
                # plane-step
                ao = gmask[a] & ~gmask[b] & vld; bo = gmask[b] & ~gmask[a] & vld
                if ao.sum() >= 50 and bo.sum() >= 50:
                    bz = cv2.dilate(mu[a], ker).astype(bool) & cv2.dilate(mu[b], ker).astype(bool)
                    byy, bxx = np.where(bz)
                    if len(bxx) >= 5:
                        ay, ax = np.where(ao); by_, bx = np.where(bo)
                        pa = robust_plane(ax.astype(float), ay.astype(float), dn[ao])
                        pb = robust_plane(bx.astype(float), by_.astype(float), dn[bo])
                        if pa is not None and pb is not None:
                            cx, cy = bxx.mean(), byy.mean()
                            s = abs((pa[0] * cx + pa[1] * cy + pa[2]) - (pb[0] * cx + pb[1] * cy + pb[2])) * 1000
                            steps[key].append(s)
                # span:任一 kept 遮罩同蓋 a、b 各 ≥SPAN_COV
                aA, aB = area[a], area[b]
                for m in allkept:
                    if int((m & gmask[a]).sum()) >= SPAN_COV * aA and int((m & gmask[b]).sum()) >= SPAN_COV * aB:
                        span_hit[key] += 1
                        break
    plane_med = {k: float(np.median(v)) for k, v in steps.items()}
    span_frac = {k: span_hit[k] / covis[k] for k in adj if covis[k] > 0}
    # 3D 連通閘門:候選對的 voxel 必須 3D 相鄰才可併(bbox 內判,省記憶體)
    bp = G.EVAL / M.BASE / scene / "instances.npz"
    conn3d = set()
    if bp.is_file():
        bl = np.load(bp)["labels"]
        vox = {}
        for (a, b) in adj:
            for g in (a, b):
                if g not in vox:
                    vox[g] = np.argwhere(bl == g)
        for (a, b) in adj:
            vi, vj = vox[a], vox[b]
            if len(vi) == 0 or len(vj) == 0:
                continue
            lo = np.minimum(vi.min(0), vj.min(0)) - 1
            hi = np.maximum(vi.max(0), vj.max(0)) + 1
            sh = hi - lo + 1
            aa = np.zeros(sh, bool); aa[tuple((vi - lo).T)] = True
            bb = np.zeros(sh, bool); bb[tuple((vj - lo).T)] = True
            if (ndimage.binary_dilation(aa) & bb).any():
                conn3d.add((a, b))
    return adj, plane_med, span_frac, conn3d


def eval_grouping(assign, scene_data, edge_fn):
    """edge_fn(adj,plane_med,span_frac,gids)->edges;回 (rows, tie, n_undecided)。"""
    rows = []; tie = 0
    for sc, (bg, adj, pm, sf, c3) in scene_data.items():
        mask_obj, nk, nu = assign[sc]
        if nk == 0:
            continue
        gids = list(bg)
        edges = edge_fn(adj, pm, sf, c3, gids)
        grouping = union_find(gids, edges)
        clusters = M.build_clusters(bg, grouping)
        r, t = M.eval_partition(mask_obj, clusters)
        rows += r; tie += t
    return rows, tie


def summ(rows):
    n = len(rows)
    if n == 0:
        return (0, float("nan"), float("nan"), float("nan"))
    exact = sum(1 for r in rows if r[6]) / n
    return (n, exact, float(np.mean([r[4] for r in rows])), float(np.mean([r[5] for r in rows])))


def main():
    scenes = sys.argv[1:]
    if not scenes:
        scenes = []
        for grp in G.GROUPS:
            scenes += sorted(Path(p).name for p in glob.glob(str(G.MV2 / f"{grp}_scene*")))
    views = sorted(VP.selected_view_names(12))
    R = 3; offs = [(dx, dy) for dx in range(-R, R + 1) for dy in range(-R, R + 1) if dx * dx + dy * dy <= R * R]
    ker = np.ones((KB, KB), np.uint8)
    print(f"場景數={len(scenes)},視角數={len(views)}\n", flush=True)

    print("[指派] ...", flush=True)
    assign = {sc: G.assign_scene(sc, views) for sc in scenes}
    print("[指派] 完成\n[訊號] 算 plane-step / span ...", flush=True)
    scene_data = {}
    for i, sc in enumerate(scenes):
        try:
            bg = M.baseline_groups(sc, views)
        except Exception:
            continue
        adj, pm, sf, c3 = signals(sc, views, bg, offs, ker)
        scene_data[sc] = (bg, adj, pm, sf, c3)
        if (i + 1) % 30 == 0:
            print(f"    {i+1}/{len(scenes)}", flush=True)
    print("[訊號] 完成\n", flush=True)

    # baseline(不併)對照
    base_rows, _ = eval_grouping(assign, scene_data, lambda adj, pm, sf, c3, g: [])
    print("★ 合併候選皆須通過 3D voxel 連通閘門(k in conn3d)")
    print(f"{'規則/門檻':<20}{'物體':>6}{'完全相等':>9}{'均Recall':>10}{'均Prec':>9}")
    n, e, rc, pr = summ(base_rows)
    print(f"{'baseline(不併)':<20}{n:>6}{e:>9.3f}{rc:>10.3f}{pr:>9.3f}")

    print("  -- plane-step:3D連通 且 階梯 < T(mm) 則併 --")
    for T in [5, 10, 15, 20, 30]:
        rows, _ = eval_grouping(assign, scene_data,
                                lambda adj, pm, sf, c3, g, T=T: [k for k in c3 if k in pm and pm[k] < T])
        n, e, rc, pr = summ(rows)
        print(f"{'  plane<'+str(T):<20}{n:>6}{e:>9.3f}{rc:>10.3f}{pr:>9.3f}")

    print("  -- span:3D連通 且 view-frac > S 則併 --")
    for S in [0.3, 0.5, 0.6, 0.7, 0.8]:
        rows, _ = eval_grouping(assign, scene_data,
                                lambda adj, pm, sf, c3, g, S=S: [k for k in c3 if k in sf and sf[k] > S])
        n, e, rc, pr = summ(rows)
        print(f"{'  span>'+str(S):<20}{n:>6}{e:>9.3f}{rc:>10.3f}{pr:>9.3f}")


if __name__ == "__main__":
    main()
