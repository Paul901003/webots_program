#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""photo_confirm.py — confirm-then-freespace 的「第一步/確認步」:用 warped-NCC 找出高可信度的真表面 voxel,
輸出成 hull.npz(occupancy=只留確認 voxel),給 Webots 可視化(像 COLMAP 只顯示確定的點)。不動現有檔。

流程(單趟,不迭代、不雕):
  對 base hull 的表面 voxel,用 photo_carve_warp2 的視角選擇(ref=解析度最高、src 加權 w_tri×w_inc×w_res)
  算每 voxel 的 warped-NCC best_agg(法向 ±tilt 微搜尋取最佳)。
  確認集合 = best_agg ≥ ncc_high 且 加權證據 wsum ≥ min_wsum。
輸出 data/eval/<out-root>/<sc>/hull.npz:occupancy = 確認 voxel(其餘清 0);grid 不變。
可視化: SRP_VIZ_ARGS="<sc> 1 <out-root> cubes" webots worlds/hull_viz.wbt

用法: SAM_ROOT=.. CAPTURES_ROOT=.. ./photo_confirm.py <sc...> --in-root srp_hull_mv2_v12_am1 --out-root <out> [--ncc-high 0.6]
provenance: build_meta 記 script/ncc_high/來源 root。
"""
import argparse
import os
import sys
import json
import time
import datetime as _dt
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "stage1_hull"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
sys.path.insert(0, str(REPO / "srp" / "io"))
import cg_associate as CG                                          # noqa: E402
from photo_carve_b import load_views, build_hull, surface_of       # noqa: E402
from photo_carve_warp import (prep_views, voxel_normals, bilinear,  # noqa: E402
                              ncc_rows, candidate_normals)

EVAL = REPO / "data" / "eval"


def surface_confidence(occ, gm, vs, views, tilt_deg=20.0, sig_tri_deg=5.0,
                       min_wsum=0.3, patch_r=2):
    """單趟:回 (sidx, best_agg, wsum)。視角選擇同 carve_warp2(ref=解析度、src 加權)。"""
    off = np.arange(-patch_r, patch_r + 1)
    dv_, du_ = np.meshgrid(off, off, indexing="ij")
    du_ = du_.ravel().astype(np.float64); dv_ = dv_.ravel().astype(np.float64)
    P = len(du_); V = len(views)
    sig_tri = np.deg2rad(sig_tri_deg)
    normals_grid = voxel_normals(occ)
    surf = surface_of(occ)
    sidx = np.argwhere(surf); Ns = len(sidx)
    if Ns == 0:
        return sidx, np.zeros(0), np.zeros(0)
    X0 = gm + (sidx + 0.5) * vs
    nrm = normals_grid[sidx[:, 0], sidx[:, 1], sidx[:, 2]]
    pu = np.zeros((V, Ns)); pv = np.zeros((V, Ns))
    vis = np.zeros((V, Ns), bool); facing = np.zeros((V, Ns)); res = np.zeros((V, Ns))
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
        vis[vi] = seen & inb
        d2c = v["Cw"][None, :] - X0; d2c /= (np.linalg.norm(d2c, axis=1, keepdims=True) + 1e-9)
        d2c_all[vi] = d2c
        facing[vi] = (nrm * d2c).sum(1)
        res[vi] = np.where(ok, v["K"][0, 0] * (vs * 0.5) / zz, 0.0)
    res_masked = np.where(vis, res, -1.0)
    ref = res_masked.argmax(0); has_ref = res_masked.max(0) > 0
    Nc = candidate_normals(nrm[:1], tilt_deg).shape[0]
    ncc_wsum = np.zeros((Nc, Ns)); wsum = np.zeros((Nc, Ns))
    for r in range(V):
        sel = has_ref & (ref == r)
        if not sel.any():
            continue
        vsel = np.nonzero(sel)[0]; dref = d2c_all[r, vsel]
        ang = np.arccos(np.clip(np.einsum("svj,vj->sv", d2c_all[:, vsel], dref), -1, 1))
        w_tri = 1.0 - np.exp(-(ang / sig_tri) ** 2)
        w_inc = np.clip(facing[:, vsel], 0.0, 1.0)
        w_res = res[:, vsel] / np.maximum(res[:, vsel].max(0), 1e-9)[None, :]
        W = w_tri * w_inc * w_res
        ur = pu[r, vsel]; vr = pv[r, vsel]
        refpixu = np.round(ur[:, None] + du_[None, :]).astype(int)
        refpixv = np.round(vr[:, None] + dv_[None, :]).astype(int)
        gr = views[r]["gray"]
        okp = (refpixu >= 0) & (refpixu < views[r]["W"]) & (refpixv >= 0) & (refpixv < views[r]["H"])
        refpixu = np.clip(refpixu, 0, views[r]["W"] - 1); refpixv = np.clip(refpixv, 0, views[r]["H"] - 1)
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
            mm = np.nonzero(vis[s, vsel] & (W[s] > 1e-6))[0]
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
                allok = (sok & okp[mm]).all(1)
                if not allok.any():
                    continue
                nc, tex = ncc_rows(ref_patch[mm][allok], src_val[allok])
                valid = tex & ~np.isnan(nc)
                gg = gi[allok][valid]; wv = ws[allok][valid]
                ncc_wsum[c, gg] += wv * nc[valid]; wsum[c, gg] += wv
    with np.errstate(all="ignore"):
        agg_c = np.where(wsum >= min_wsum, ncc_wsum / np.maximum(wsum, 1e-9), np.nan)
        best_agg = np.nanmax(agg_c, axis=0)
    tot_w = wsum.max(0)                                   # 該 voxel 最佳法向的證據量
    return sidx, best_agg, tot_w, ref, pu, pv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--num-views", type=int, default=12, dest="num_views")
    ap.add_argument("--voxel", type=float, default=0.005)
    ap.add_argument("--in-root", default="srp_hull_mv2_v12_am1")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ncc-high", type=float, default=0.6, dest="ncc_high")
    ap.add_argument("--min-wsum", type=float, default=0.3, dest="min_wsum")
    ap.add_argument("--tilt-deg", type=float, default=20.0, dest="tilt_deg")
    a = ap.parse_args()
    for sc in a.scenes:
        print(f"\n######## {sc} ########")
        views = prep_views(load_views(sc, a.num_views))
        z = np.load(EVAL / a.in_root / sc / "hull.npz")
        occ = z["occupancy"].astype(bool); gm = z["grid_min"]; vs = float(z["voxel_size"])
        surf_n = int(surface_of(occ).sum())
        t0 = time.time()
        sidx, best_agg, tot_w, ref, pu, pv = surface_confidence(occ, gm, vs, views, a.tilt_deg,
                                                                min_wsum=a.min_wsum)
        conf = np.isfinite(best_agg) & (best_agg >= a.ncc_high)
        ci = np.nonzero(conf)[0]
        # ★ COLMAP 式上色:每確認 voxel 取它 ref 視角在投影像素的真實 RGB
        cols = np.zeros((len(ci), 3))
        for n, i in enumerate(ci):
            r = int(ref[i]); vw = views[r]
            px = int(np.clip(round(pu[r, i]), 0, vw["W"] - 1))
            py = int(np.clip(round(pv[r, i]), 0, vw["H"] - 1))
            cols[n] = vw["rgb"][py, px]
        Q = 6                                    # 每通道量化階數 → 相近色併成一個色 bin(減 obj 數)
        binned = np.round(cols * (Q - 1)) / (Q - 1)
        uniq, inv = (np.unique(binned, axis=0, return_inverse=True) if len(ci)
                     else (np.zeros((0, 3)), np.zeros(0, int)))
        inv = np.asarray(inv).ravel()
        labels = np.zeros(occ.shape, np.int32)   # label = 色 bin id(1..N);label_colors[k-1]=該 bin RGB
        cg = sidx[ci]; labels[cg[:, 0], cg[:, 1], cg[:, 2]] = inv + 1
        grid = labels > 0
        dt = time.time() - t0
        out = EVAL / a.out_root / sc; out.mkdir(parents=True, exist_ok=True)
        meta = {"script": "photo_confirm.py", "in_root": a.in_root, "ncc_high": a.ncc_high,
                "min_wsum": a.min_wsum, "note": "labels=色bin id;label_colors=各bin真實RGB(COLMAP式上色)",
                "n_confirm": int(conf.sum()), "n_color_bins": int(len(uniq)),
                "built": _dt.datetime.now().isoformat(timespec="seconds")}
        # instances.npz 供 Webots 可視化(cubes);label_colors 讓 gen_viz_objs 用真實 RGB
        np.savez_compressed(out / "instances.npz", labels=labels, grid_min=gm, voxel_size=vs,
                            occupancy=grid, label_colors=uniq.astype(np.float32),
                            build_meta=json.dumps(meta, ensure_ascii=False))
        np.savez_compressed(out / "hull.npz", occupancy=grid, grid_min=gm, voxel_size=vs,
                            surface=grid, build_meta=json.dumps(meta, ensure_ascii=False))
        haveev = int(np.isfinite(best_agg).sum())
        print(f"  表面 {surf_n} | 有證據 {haveev} | 確認(≥{a.ncc_high}) {int(conf.sum())} "
              f"({int(conf.sum())/max(surf_n,1)*100:.0f}%) ({dt:.1f}s)")
        print(f'  SRP_VIZ_ARGS="{sc} 1 {a.out_root} cubes" webots worlds/hull_viz.wbt')


if __name__ == "__main__":
    main()
