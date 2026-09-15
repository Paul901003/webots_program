#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""photo_carve_warp.py — 自造:以 visual hull 為基礎,借 COLMAP 技術雕掉冗餘 voxel(不跑 COLMAP)。

route B 失敗是因為:①patch 沒 warp(寬基線外觀差)②跟所有視角比 ③單點顏色噪 ④無空間正則→碎裂。
本法逐一補上(全在 voxel 上自己做,純 RGB、免深度、不跑 colmap):
  法向     = visual hull occupancy 梯度(免費給的表面法向,供單應性 warp)
  warp     = 由 (voxel位置, hull法向) 算 ref→src 平面單應性,把 ref patch 對齊到 src
  一致性   = warped-patch NCC(去亮度),只跟「視線夾角最小的 K 個鄰視角」比
  正則化   = 只雕「連通成團(≥min_cluster)」的不一致表面 voxel,不雕孤立散點
迭代逐層剝(露出新面再判),內部/無紋理保守保留(薄物先不管)。

目標:降 hull 冗餘(過估)而不碎裂 → 供下游實例分離/物體關係。
用法: ./photo_carve_warp.py stack3_scene0001 [--num-views 12] [--ncc-min 0.5] [--k-src 4]
env: CAPTURES_ROOT SAM_ROOT(+ 夾爪遮罩 gripper_masks_canonical 由 photo_carve_b.load_views 減)
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera as cam            # noqa: E402
import cg_associate as CG       # noqa: E402
from photo_carve_b import load_views, build_hull, surface_of   # noqa: E402
from photo_carve import eval_vs_gt                             # noqa: E402

OUT_ROOT = REPO / "data" / "eval" / "photo_carve_warp"


def prep_views(views):
    """每視角預存 world2cam(Rwc,t)、K、Kinv、相機中心 C、灰階。"""
    for v in views:
        Rwc, t = cam.pose_to_w2c(v["C"], v["Rb"])
        K = cam.intrinsics(v["W"], v["H"])
        v["Rwc"] = Rwc; v["t"] = t; v["K"] = K; v["Kinv"] = np.linalg.inv(K)
        v["Cw"] = np.asarray(v["C"], float)
    return views


def voxel_normals(occ):
    """occupancy 梯度 → 外向法向(佔據內高、外低,外向 = -grad)。回傳整格 (X,Y,Z,3)。"""
    f = ndimage.gaussian_filter(occ.astype(np.float32), sigma=1.0)
    gx, gy, gz = np.gradient(f)
    n = -np.stack([gx, gy, gz], axis=-1)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / (norm + 1e-9)
    return n


def bilinear(gray, u, v):
    H, W = gray.shape
    u0 = np.floor(u).astype(int); v0 = np.floor(v).astype(int)
    du = u - u0; dv = v - v0
    valid = (u0 >= 0) & (v0 >= 0) & (u0 + 1 < W) & (v0 + 1 < H)
    u0c = np.clip(u0, 0, W - 2); v0c = np.clip(v0, 0, H - 2)
    Ia = gray[v0c, u0c]; Ib = gray[v0c, u0c + 1]
    Ic = gray[v0c + 1, u0c]; Id = gray[v0c + 1, u0c + 1]
    val = Ia * (1 - du) * (1 - dv) + Ib * du * (1 - dv) + Ic * (1 - du) * dv + Id * du * dv
    return val, valid


def ncc_rows(A, B):
    """逐列 NCC(A,B 同形 (N,P));回傳 (N,) NCC 與 (N,) 是否有紋理(兩邊 std>eps)。"""
    Am = A - A.mean(1, keepdims=True); Bm = B - B.mean(1, keepdims=True)
    na = np.sqrt((Am ** 2).sum(1)); nb = np.sqrt((Bm ** 2).sum(1))
    tex = (na > 1e-2) & (nb > 1e-2)
    ncc = np.where(tex, (Am * Bm).sum(1) / (na * nb + 1e-9), np.nan)
    return ncc, tex


def candidate_normals(n0, tilt_deg):
    """對每個法向 n0 (m,3),在切平面內傾斜生成候選法向 (Nc,m,3)。tilt_deg<=0 → 只回 n0 本身(不搜)。"""
    m = len(n0)
    if tilt_deg <= 0:
        return n0[None]
    e = np.tile(np.array([0, 0, 1.0]), (m, 1))
    alt = np.abs(n0[:, 2]) > 0.9
    e[alt] = np.array([1.0, 0, 0])
    t1 = np.cross(n0, e); t1 /= (np.linalg.norm(t1, axis=1, keepdims=True) + 1e-9)
    t2 = np.cross(n0, t1); t2 /= (np.linalg.norm(t2, axis=1, keepdims=True) + 1e-9)
    d = np.tan(np.deg2rad(tilt_deg))
    offs = [(0., 0.)] + [(a, b) for a in (-d, 0, d) for b in (-d, 0, d) if not (a == 0 and b == 0)]
    cands = []
    for a, b in offs:
        c = n0 + a * t1 + b * t2
        c /= (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
        cands.append(c)
    return np.stack(cands, 0)                          # (Nc,m,3)


def carve_warp(occ, gm, vs, views, ncc_min, k_src, min_src, min_cluster, patch_r, max_iter, tilt_deg=20.0):
    occ = occ.copy()
    off = np.arange(-patch_r, patch_r + 1)
    dv_, du_ = np.meshgrid(off, off, indexing="ij")
    du_ = du_.ravel().astype(np.float64); dv_ = dv_.ravel().astype(np.float64)
    P = len(du_); V = len(views)
    normals_grid = None
    for it in range(max_iter):
        normals_grid = voxel_normals(occ)
        surf = surface_of(occ)
        sidx = np.argwhere(surf)
        if len(sidx) == 0:
            break
        Ns = len(sidx)
        X0 = gm + (sidx + 0.5) * vs                       # (Ns,3) 世界座標
        nrm = normals_grid[sidx[:, 0], sidx[:, 1], sidx[:, 2]]  # (Ns,3) 法向

        # 每視角:投影 (u,v,valid) + 可見性(zbuffer)+ facing
        pu = np.zeros((V, Ns)); pv = np.zeros((V, Ns)); pvalid = np.zeros((V, Ns), bool)
        vis = np.zeros((V, Ns), bool); facing = np.zeros((V, Ns))
        for vi, v in enumerate(views):
            X = X0 @ v["Rwc"].T + v["t"]; z = X[:, 2]; ok = z > 1e-6
            zz = np.where(ok, z, 1.0)
            u = v["K"][0, 0] * X[:, 0] / zz + v["K"][0, 2]
            w = v["K"][1, 1] * X[:, 1] / zz + v["K"][1, 2]
            inb = ok & (u >= 0) & (u < v["W"]) & (w >= 0) & (w < v["H"])
            pu[vi] = u; pv[vi] = w; pvalid[vi] = inb
            vox_at = CG.zbuffer_visible(X0, v["C"], v["Rb"], v["W"], v["H"], vs)
            seen = np.zeros(Ns, bool); uq = np.unique(vox_at[vox_at >= 0])
            seen[uq] = True
            # facing:法向對相機
            d2c = v["Cw"][None, :] - X0; d2c /= (np.linalg.norm(d2c, axis=1, keepdims=True) + 1e-9)
            facing[vi] = (nrm * d2c).sum(1)
            vis[vi] = seen & inb & (facing[vi] > 0.1)     # 可見 + 正面朝這視角

        # 每 voxel 選 ref(最正面)+ K 個鄰視角(與 ref 視線夾角最小)
        d2c_all = np.zeros((V, Ns, 3))
        for vi, v in enumerate(views):
            d = v["Cw"][None, :] - X0; d2c_all[vi] = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        face_masked = np.where(vis, facing, -9.0)
        ref = face_masked.argmax(0)                        # (Ns,) ref 視角
        has_ref = face_masked.max(0) > -1

        # 每候選法向各累積一份 NCC(Nc,Ns),最後取每 voxel 最佳法向的 NCC(法向微搜尋)
        n0_all = nrm                                        # (Ns,3) hull 法向
        Nc = len(candidate_normals(n0_all[:1], tilt_deg))   # 候選數
        ncc_sum = np.zeros((Nc, Ns)); ncc_cnt = np.zeros((Nc, Ns))
        for r in range(V):
            sel = has_ref & (ref == r)
            if not sel.any():
                continue
            vsel = np.nonzero(sel)[0]                       # 這批 voxel(ref=r)
            dref = d2c_all[r, vsel]
            align = np.full((V, len(vsel)), -9.0)
            for s in range(V):
                if s == r:
                    continue
                m = vis[s, vsel]
                align[s, m] = (d2c_all[s, vsel][m] * dref[m]).sum(1)
            order = np.argsort(-align, axis=0)[:k_src]
            ur = pu[r, vsel]; vr = pv[r, vsel]
            refpixu = np.round(ur[:, None] + du_[None, :]).astype(int)
            refpixv = np.round(vr[:, None] + dv_[None, :]).astype(int)
            gr = views[r]["gray"]
            okp = (refpixu >= 0) & (refpixu < views[r]["W"]) & (refpixv >= 0) & (refpixv < views[r]["H"])
            refpixu = np.clip(refpixu, 0, views[r]["W"] - 1); refpixv = np.clip(refpixv, 0, views[r]["H"] - 1)
            ref_patch = gr[refpixv, refpixu]                # (m,P)
            R_r = views[r]["Rwc"]; t_r = views[r]["t"]; Kinv_r = views[r]["Kinv"]
            X0r = X0[vsel] @ R_r.T + t_r                     # voxel → ref cam (m,3)
            cands_w = candidate_normals(nrm[vsel], tilt_deg)  # (Nc,m,3) 世界法向候選
            nr_c = np.einsum("cmj,jk->cmk", cands_w, R_r.T)   # → ref cam frame (Nc,m,3)
            d_r_c = (nr_c * X0r[None]).sum(-1)                # (Nc,m)
            refhom = np.stack([ur[:, None] + du_[None, :], vr[:, None] + dv_[None, :],
                               np.ones((len(vsel), P))], axis=-1)  # (m,P,3)
            for ks in range(k_src):
                svarr = order[ks]
                for s in np.unique(svarr):
                    if s == r or s < 0:
                        continue
                    mm = np.nonzero((svarr == s) & (align[s, np.arange(len(vsel))] > -1))[0]
                    if len(mm) == 0:
                        continue
                    vw = views[s]
                    R_sr = vw["Rwc"] @ R_r.T
                    t_sr = vw["t"] - vw["Rwc"] @ R_r.T @ t_r
                    rp = refhom[mm]                          # (len,P,3)
                    gi = vsel[mm]
                    for c in range(Nc):                      # ★試每個候選法向,取最佳
                        nr = nr_c[c, mm]; dr = d_r_c[c, mm]
                        Mmat = R_sr[None] + (t_sr[None, :, None] * nr[:, None, :]) / dr[:, None, None]
                        Hmat = np.einsum("ij,njk,kl->nil", vw["K"], Mmat, Kinv_r)
                        wp = np.einsum("nij,npj->npi", Hmat, rp)
                        su = wp[:, :, 0] / wp[:, :, 2]; sv = wp[:, :, 1] / wp[:, :, 2]
                        src_val, sok = bilinear(vw["gray"], su, sv)
                        good = sok & okp[mm]
                        allok = good.all(1)
                        if not allok.any():
                            continue
                        nc, tex = ncc_rows(ref_patch[mm][allok], src_val[allok])
                        valid = tex & ~np.isnan(nc)
                        gg = gi[allok][valid]
                        ncc_sum[c, gg] += nc[valid]; ncc_cnt[c, gg] += 1

        agg_c = np.where(ncc_cnt >= min_src, ncc_sum / np.maximum(ncc_cnt, 1), np.nan)
        with np.errstate(all="ignore"):
            best_agg = np.nanmax(agg_c, axis=0)              # 每 voxel 最佳法向的 NCC
        has_ev = (ncc_cnt >= min_src).any(0)                 # 至少一個候選有足夠證據
        if os.environ.get("DBG") == "1":
            v2 = int(has_ev.sum()); av = best_agg[has_ev]
            pr = np.percentile(av, [10, 50, 90]) if len(av) else [np.nan]*3
            print(f"    [dbg] Ns={Ns} Nc={Nc} 有證據:{v2}  best NCC 10/50/90%={pr[0]:.2f}/{pr[1]:.2f}/{pr[2]:.2f}")
        cand = has_ev & (best_agg < ncc_min)                # 不一致(連最佳法向都低)的表面 voxel
        # 空間正則化:只雕連通成團(≥min_cluster)者,去孤立散點
        cand_grid = np.zeros(occ.shape, bool)
        cg = sidx[cand]; cand_grid[cg[:, 0], cg[:, 1], cg[:, 2]] = True
        lbl, nl = ndimage.label(cand_grid, ndimage.generate_binary_structure(3, 1))
        if nl == 0:
            print(f"    iter {it+1}: 無不一致 → 停"); break
        sizes = ndimage.sum(np.ones_like(lbl), lbl, index=np.arange(1, nl + 1))
        keep_lbl = np.nonzero(sizes >= min_cluster)[0] + 1
        carve_grid = np.isin(lbl, keep_lbl)
        nc_total = int(carve_grid.sum())
        if nc_total == 0:
            print(f"    iter {it+1}: 不一致皆散點(<{min_cluster})→ 停"); break
        occ &= ~carve_grid
        print(f"    iter {it+1}: 雕 {nc_total} (連通團,原候選 {int(cand.sum())}) → 剩 {int(occ.sum())}")
    return occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--num-views", type=int, default=12, dest="num_views")
    ap.add_argument("--voxel", type=float, default=0.005)
    ap.add_argument("--ncc-min", type=float, default=0.5, dest="ncc_min")
    ap.add_argument("--k-src", type=int, default=4, dest="k_src")
    ap.add_argument("--min-src", type=int, default=2, dest="min_src")
    ap.add_argument("--min-cluster", type=int, default=3, dest="min_cluster")
    ap.add_argument("--patch-radius", type=int, default=2, dest="patch_r")
    ap.add_argument("--max-iter", type=int, default=15, dest="max_iter")
    ap.add_argument("--tilt-deg", type=float, default=20.0, dest="tilt_deg",
                    help="法向微搜尋傾斜角(度);0=不搜尋(只用 hull 法向)")
    args = ap.parse_args()
    for sc in args.scenes:
        print(f"\n######## {sc} ########")
        views = prep_views(load_views(sc, args.num_views))
        print(f"  {len(views)} 視角")
        occ, gm, vs = build_hull(views, args.voxel)
        print(f"  hull {int(occ.sum())} vox")
        t0 = time.time()
        carved = carve_warp(occ, gm, vs, views, args.ncc_min, args.k_src, args.min_src,
                            args.min_cluster, args.patch_r, args.max_iter, args.tilt_deg)
        print(f"  warped-NCC 雕刻: {time.time()-t0:.1f}s  ({int(occ.sum())}→{int(carved.sum())})")
        out = OUT_ROOT / sc; out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "hull_warp.npz", occupancy=carved, grid_min=gm, voxel_size=vs)
        e0 = eval_vs_gt(occ, gm, vs, sc); e1 = eval_vs_gt(carved, gm, vs, sc)
        if e0 and e1:
            print(f"  {'':6}{'體積L':>8}{'覆蓋':>8}{'冗餘':>8}{'膨脹':>8}")
            print(f"  {'雕前':6}{e0['vol_L']:8.2f}{e0['coverage']:8.3f}{e0['redundancy']:8.3f}{e0['bloat']:8.2f}")
            print(f"  {'雕後':6}{e1['vol_L']:8.2f}{e1['coverage']:8.3f}{e1['redundancy']:8.3f}{e1['bloat']:8.2f}")


if __name__ == "__main__":
    main()
