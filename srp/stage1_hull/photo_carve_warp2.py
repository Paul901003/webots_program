#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""photo_carve_warp2.py — photo_carve_warp 的改良視角選擇版(參考 COLMAP 的 view selection)。不動原檔。

改的三點(相對 carve_warp,核心單應性 warp + NCC 不變):
  ① ref(基準視角):原本 `argmax facing`(用不準的 hull 法向挑最正面)→ 改用**可見視角中 footprint
     最大(解析度最高)**那張,**完全不用法向**。
  ② 可見閘:原本 `seen & inb & facing>0.1`(用原始法向硬閘)→ 改 `seen & inb`;**原始法向不再做任何硬決策**。
  ③ src(對比視角)：原本「夾角最近 top-k、等權」→ 改**加權平均 NCC**,權重
     `w = w_tri(三角化角) × w_inc(入射角,軟) × w_res(footprint 大小)`,用所有可見 src(不再硬取 top-k)。
     - w_tri = 1-exp(-(ang/σ_tri)²):懲罰基線太小(夾角近)——正好修掉原本「挑最近」對揭發鬼影最弱的偏誤。
     - w_inc = clip(facing,0,1):入射角軟權重(用原始法向,但只當**軟**權重,對法向誤差寬容,非硬決策)。
     - w_res = footprint 半徑 / 該 voxel 各視角最大值:看得越大越可靠。
  → 法向從此只出現在單應性(且是 ±tilt 微搜尋的);你指出的「ref 凍結在爛法向」問題消失。

不照抄 COLMAP 的:雙邊加權 NCC、幾何一致性(需 per-view 深度圖)、5-sweep 深度傳播、HMM 機率抽樣——
那些要 per-view 深度圖、較重,不在本次範圍。

用法(吃現成 hull,對照 am1):
  SAM_ROOT=.. CAPTURES_ROOT=.. ./photo_carve_warp2.py <scenes...> --in-root srp_hull_mv2_v12_am1 --out-root <out>
  (--in-root 省略則自己 build_hull)。參數同 carve_warp:--ncc-min 0.1 --min-cluster 3 --patch-radius 2 --max-iter 15 --tilt-deg 20
provenance: build_meta 記 script=photo_carve_warp2.py + 改點。
"""
import argparse
import os
import sys
import json
import time
import datetime as _dt
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "stage1_hull"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
sys.path.insert(0, str(REPO / "srp" / "io"))
import cg_associate as CG                                          # noqa: E402
from photo_carve_b import load_views, build_hull, surface_of       # noqa: E402
from photo_carve_warp import (prep_views, voxel_normals, bilinear,  # noqa: E402
                              ncc_rows, mask_samples, candidate_normals)

EVAL = REPO / "data" / "eval"


def carve_warp2(occ, gm, vs, views, ncc_min, min_wsum, min_cluster, patch_r,
                max_iter, tilt_deg=20.0, sig_tri_deg=5.0, fg_patch_guard=False):
    occ = occ.copy()
    off = np.arange(-patch_r, patch_r + 1)
    dv_, du_ = np.meshgrid(off, off, indexing="ij")
    du_ = du_.ravel().astype(np.float64); dv_ = dv_.ravel().astype(np.float64)
    P = len(du_); V = len(views)
    sig_tri = np.deg2rad(sig_tri_deg)
    for it in range(max_iter):
        normals_grid = voxel_normals(occ)
        surf = surface_of(occ)
        sidx = np.argwhere(surf)
        if len(sidx) == 0:
            break
        Ns = len(sidx)
        X0 = gm + (sidx + 0.5) * vs
        nrm = normals_grid[sidx[:, 0], sidx[:, 1], sidx[:, 2]]

        pu = np.zeros((V, Ns)); pv = np.zeros((V, Ns))
        vis = np.zeros((V, Ns), bool); facing = np.zeros((V, Ns))
        res = np.zeros((V, Ns))                        # footprint 半徑(像素)= 解析度
        d2c_all = np.zeros((V, Ns, 3))
        for vi, v in enumerate(views):
            X = X0 @ v["Rwc"].T + v["t"]; z = X[:, 2]; ok = z > 1e-6
            zz = np.where(ok, z, 1.0)
            u = v["K"][0, 0] * X[:, 0] / zz + v["K"][0, 2]
            w = v["K"][1, 1] * X[:, 1] / zz + v["K"][1, 2]
            inb = ok & (u >= 0) & (u < v["W"]) & (w >= 0) & (w < v["H"])
            pu[vi] = u; pv[vi] = w
            vox_at = CG.zbuffer_visible(X0, v["C"], v["Rb"], v["W"], v["H"], vs)
            seen = np.zeros(Ns, bool); seen[np.unique(vox_at[vox_at >= 0])] = True
            vis[vi] = seen & inb                        # ★② 去掉 facing 硬閘
            d2c = v["Cw"][None, :] - X0
            d2c /= (np.linalg.norm(d2c, axis=1, keepdims=True) + 1e-9)
            d2c_all[vi] = d2c
            facing[vi] = (nrm * d2c).sum(1)             # 只當軟權重用
            res[vi] = np.where(ok, v["K"][0, 0] * (vs * 0.5) / zz, 0.0)

        # ★① ref = 可見視角中 footprint 最大(解析度最高),不用法向
        res_masked = np.where(vis, res, -1.0)
        ref = res_masked.argmax(0)
        has_ref = res_masked.max(0) > 0

        Nc = candidate_normals(nrm[:1], tilt_deg).shape[0]
        ncc_wsum = np.zeros((Nc, Ns)); wsum = np.zeros((Nc, Ns))
        for r in range(V):
            sel = has_ref & (ref == r)
            if not sel.any():
                continue
            vsel = np.nonzero(sel)[0]
            dref = d2c_all[r, vsel]
            # ★③ 每 src 的權重 w_tri × w_inc × w_res(對這批 ref==r 的 voxel)
            ang = np.full((V, len(vsel)), 0.0)
            for s in range(V):
                cosv = np.clip((d2c_all[s, vsel] * dref).sum(1), -1, 1)
                ang[s] = np.arccos(cosv)
            w_tri = 1.0 - np.exp(-(ang / sig_tri) ** 2)             # 小基線→0
            w_inc = np.clip(facing[:, vsel], 0.0, 1.0)              # 入射角軟權重
            res_ref = np.maximum(res[:, vsel].max(0), 1e-9)
            w_res = res[:, vsel] / res_ref[None, :]
            W = w_tri * w_inc * w_res                               # (V,len)
            ur = pu[r, vsel]; vr = pv[r, vsel]
            refpixu = np.round(ur[:, None] + du_[None, :]).astype(int)
            refpixv = np.round(vr[:, None] + dv_[None, :]).astype(int)
            gr = views[r]["gray"]
            okp = (refpixu >= 0) & (refpixu < views[r]["W"]) & (refpixv >= 0) & (refpixv < views[r]["H"])
            refpixu = np.clip(refpixu, 0, views[r]["W"] - 1); refpixv = np.clip(refpixv, 0, views[r]["H"] - 1)
            ref_fg = okp & views[r]["fg"][refpixv, refpixu]
            ref_patch = gr[refpixv, refpixu]
            R_r = views[r]["Rwc"]; t_r = views[r]["t"]; Kinv_r = views[r]["Kinv"]
            X0r = X0[vsel] @ R_r.T + t_r
            cands_w = candidate_normals(nrm[vsel], tilt_deg)
            nr_c = np.einsum("cmj,jk->cmk", cands_w, R_r.T)
            d_r_c = (nr_c * X0r[None]).sum(-1)
            refhom = np.stack([ur[:, None] + du_[None, :], vr[:, None] + dv_[None, :],
                               np.ones((len(vsel), P))], axis=-1)
            for s in range(V):
                if s == r:
                    continue
                mm = np.nonzero(vis[s, vsel] & (W[s] > 1e-6))[0]   # ★用所有可見 src(非 top-k)
                if len(mm) == 0:
                    continue
                vw = views[s]
                R_sr = vw["Rwc"] @ R_r.T
                t_sr = vw["t"] - vw["Rwc"] @ R_r.T @ t_r
                rp = refhom[mm]; gi = vsel[mm]; ws = W[s, mm]
                for c in range(Nc):
                    nr = nr_c[c, mm]; dr = d_r_c[c, mm]
                    Mmat = R_sr[None] + (t_sr[None, :, None] * nr[:, None, :]) / dr[:, None, None]
                    Hmat = np.einsum("ij,njk,kl->nil", vw["K"], Mmat, Kinv_r)
                    wp = np.einsum("nij,npj->npi", Hmat, rp)
                    su = wp[:, :, 0] / wp[:, :, 2]; sv = wp[:, :, 1] / wp[:, :, 2]
                    src_val, sok = bilinear(vw["gray"], su, sv)
                    good = sok & okp[mm]
                    if fg_patch_guard:
                        good &= ref_fg[mm] & mask_samples(vw["fg"], su, sv)
                    allok = good.all(1)
                    if not allok.any():
                        continue
                    nc, tex = ncc_rows(ref_patch[mm][allok], src_val[allok])
                    valid = tex & ~np.isnan(nc)
                    gg = gi[allok][valid]; wv = ws[allok][valid]
                    ncc_wsum[c, gg] += wv * nc[valid]
                    wsum[c, gg] += wv

        with np.errstate(all="ignore"):
            agg_c = np.where(wsum >= min_wsum, ncc_wsum / np.maximum(wsum, 1e-9), np.nan)
            best_agg = np.nanmax(agg_c, axis=0)
        has_ev = (wsum >= min_wsum).any(0)
        if os.environ.get("DBG") == "1":
            av = best_agg[has_ev]
            pr = np.percentile(av, [10, 50, 90]) if len(av) else [np.nan] * 3
            print(f"    [dbg] Ns={Ns} 有證據:{int(has_ev.sum())} best NCC 10/50/90%="
                  f"{pr[0]:.2f}/{pr[1]:.2f}/{pr[2]:.2f}")
        cand = has_ev & (best_agg < ncc_min)
        cand_grid = np.zeros(occ.shape, bool)
        cg = sidx[cand]; cand_grid[cg[:, 0], cg[:, 1], cg[:, 2]] = True
        lbl, nl = ndimage.label(cand_grid, ndimage.generate_binary_structure(3, 1))
        if nl == 0:
            print(f"    iter {it+1}: 無不一致 → 停"); break
        sizes = ndimage.sum(np.ones_like(lbl), lbl, index=np.arange(1, nl + 1))
        keep = np.nonzero(sizes >= min_cluster)[0] + 1
        carve_grid = np.isin(lbl, keep)
        if int(carve_grid.sum()) == 0:
            print(f"    iter {it+1}: 不一致皆散點 → 停"); break
        occ &= ~carve_grid
        print(f"    iter {it+1}: 雕 {int(carve_grid.sum())} (原候選 {int(cand.sum())}) → 剩 {int(occ.sum())}")
    return occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--num-views", type=int, default=12, dest="num_views")
    ap.add_argument("--voxel", type=float, default=0.005)
    ap.add_argument("--in-root", default=None, help="吃現成 hull.npz occupancy(省略=自己 build_hull)")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ncc-min", type=float, default=0.1, dest="ncc_min")
    ap.add_argument("--min-wsum", type=float, default=0.3, dest="min_wsum", help="有效證據需的總權重下限")
    ap.add_argument("--min-cluster", type=int, default=3, dest="min_cluster")
    ap.add_argument("--patch-radius", type=int, default=2, dest="patch_r")
    ap.add_argument("--max-iter", type=int, default=15, dest="max_iter")
    ap.add_argument("--tilt-deg", type=float, default=20.0, dest="tilt_deg")
    ap.add_argument("--sig-tri-deg", type=float, default=5.0, dest="sig_tri_deg")
    a = ap.parse_args()
    for sc in a.scenes:
        print(f"\n######## {sc} ########")
        views = prep_views(load_views(sc, a.num_views))
        print(f"  {len(views)} 視角")
        if a.in_root:
            z = np.load(EVAL / a.in_root / sc / "hull.npz")
            occ = z["occupancy"].astype(bool); gm = z["grid_min"]; vs = float(z["voxel_size"])
        else:
            occ, gm, vs = build_hull(views, a.voxel)
        print(f"  hull {int(occ.sum())} vox")
        t0 = time.time()
        carved = carve_warp2(occ, gm, vs, views, a.ncc_min, a.min_wsum, a.min_cluster,
                             a.patch_r, a.max_iter, a.tilt_deg, a.sig_tri_deg)
        dt = time.time() - t0
        out = EVAL / a.out_root / sc; out.mkdir(parents=True, exist_ok=True)
        meta = {"script": "photo_carve_warp2.py", "in_root": a.in_root,
                "carve": f"warp2_tilt{a.tilt_deg:.0f}_ncc{a.ncc_min}_sigtri{a.sig_tri_deg}_minw{a.min_wsum}"
                         f"_clu{a.min_cluster}_patch{a.patch_r}_iter{a.max_iter}",
                "changes": "ref=maxres(no-normal); vis=seen&inb(no facing-gate); src=weighted w_tri*w_inc*w_res",
                "built": _dt.datetime.now().isoformat(timespec="seconds"), "seconds": round(dt, 2)}
        np.savez_compressed(out / "hull.npz", occupancy=carved, grid_min=gm, voxel_size=vs,
                            build_meta=json.dumps(meta, ensure_ascii=False))
        print(f"  → {int(carved.sum())} vox ({dt:.1f}s) 存 {out}")


if __name__ == "__main__":
    main()
