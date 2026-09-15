#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""photo_carve_b.py — 路線 B:光度一致性迭代雕刻(純 RGB,不用 COLMAP)。

概念:hull 是剪影外包絡(過估)。對每個表面體素,取「看得到它的各視角」在其投影點的顏色;
真表面點各視角顏色一致、幽靈/凹面體素各視角顏色不一致(視線穿過它打到背後不同真點)→ 雕掉。
迭代:每輪重算可見性(外層幽靈雕掉後凹面才露出、才判得出)→ 逐層剝。不動 carve.py。

分別計時:① 建 voxel hull(GPU carve)② route B 光度雕刻(CPU zbuffer + 顏色一致性)。
去夾爪:各視角減固定夾爪遮罩(data/eval/gripper_masks_canonical/<view>_gripper.png)。
評估:對 GT mesh 量雕前/後 體積・冗餘・覆蓋(photo_carve.eval_vs_gt)。

用法: ./photo_carve_b.py stack3_scene0001 [--num-views 34] [--voxel 0.005] [--color-tau 0.09]
env: CAPTURES_ROOT(captures_fast) SAM_ROOT(sam_only_fast/mobilesamv2_fast)
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera as cam            # noqa: E402
import masks as MK             # noqa: E402
import viewpoints as VP        # noqa: E402
import cg_associate as CG      # noqa: E402  (zbuffer_visible)
from carve import carve_visual_hull   # noqa: E402
from photo_carve import eval_vs_gt     # noqa: E402  (對 GT mesh 量冗餘/覆蓋)

CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures")))
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "sam_only")))
GRIP = REPO / "data" / "eval" / "gripper_masks_canonical"
OUT_ROOT = REPO / "data" / "eval" / "photo_carve_b"

BOX_MIN = np.array([0.0, -0.35, 0.0]); BOX_MAX = np.array([0.7, 0.35, 0.35]); TABLE_Z = 0.0


def load_views(scene, num_views):
    group = scene.split("_")[0]; sdir = CAPTURES / f"multi_{group}" / scene
    try:
        want = VP.selected_view_names(num_views) if num_views else None
    except FileNotFoundError:
        want = None                              # 無 A-3 挑選檔(如 34=全視角)→ 用全部拍攝視角
    views = []
    for vd in sorted((SAM_ROOT / scene).glob("view_*")):
        if want is not None and vd.name not in want:
            continue
        pf = sdir / f"{vd.name}_pose.json"; img = sdir / f"{vd.name}.png"
        if not (pf.is_file() and img.is_file()):
            continue
        km = MK.kept_object_masks(vd)
        if not km:
            continue
        fg = None
        for b, _ in km:
            fg = b if fg is None else (fg | b)
        gp = GRIP / f"{vd.name}_gripper.png"                 # 減固定夾爪遮罩
        if gp.is_file():
            g = cv2.imread(str(gp), 0) > 127
            if g.shape == fg.shape:
                fg = fg & ~g
        if not fg.any():
            continue
        C, Rb = cam.load_pose(pf)
        rgb = cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gray = rgb.mean(axis=2)                              # NCC 用灰階 patch
        H, W = fg.shape
        views.append({"rgb": rgb, "fg": fg, "gray": gray, "C": C, "Rb": Rb, "W": W, "H": H})
    return views


def build_hull(views, voxel):
    masks = [v["fg"] for v in views]
    Ks = [cam.intrinsics(v["W"], v["H"]) for v in views]
    extr = [cam.pose_to_w2c(v["C"], v["Rb"]) for v in views]
    miss = round(0.083 * len(views))
    hull = carve_visual_hull(masks, Ks, extr, BOX_MIN, BOX_MAX, voxel,
                             table_z=TABLE_Z, allow_miss=miss)
    return hull.occupancy, hull.grid_min, hull.voxel_size


def surface_of(occ):
    return occ & ~ndimage.binary_erosion(occ, ndimage.generate_binary_structure(3, 1))


def photo_carve(occ, gm, vs, views, color_tau, min_views, max_iter):
    occ = occ.copy()
    fgflat = [v["fg"].reshape(-1) for v in views]
    rgbflat = [v["rgb"].reshape(-1, 3) for v in views]
    for it in range(max_iter):
        idx = np.argwhere(occ)
        if len(idx) == 0:
            break
        P = gm + (idx + 0.5) * vs
        M = len(P)
        vsum = np.zeros((M, 3)); vsq = np.zeros((M, 3)); vseen = np.zeros(M)
        for vi, v in enumerate(views):
            vox_at = CG.zbuffer_visible(P, v["C"], v["Rb"], v["W"], v["H"], vs)  # (H*W,)
            pix = np.nonzero(vox_at >= 0)[0]
            if len(pix) == 0:
                continue
            vox = vox_at[pix]
            keep = fgflat[vi][pix]                       # 只採前景(非背景/夾爪)像素
            pix = pix[keep]; vox = vox[keep]
            if len(pix) == 0:
                continue
            col = rgbflat[vi][pix]
            csum = np.zeros((M, 3)); ccnt = np.zeros(M)
            np.add.at(csum, vox, col); np.add.at(ccnt, vox, 1)
            seen = ccnt > 0
            mean = np.zeros((M, 3)); mean[seen] = csum[seen] / ccnt[seen, None]
            vsum[seen] += mean[seen]; vsq[seen] += mean[seen] ** 2; vseen[seen] += 1
        ok = vseen >= min_views
        var = np.zeros((M, 3))
        var[ok] = vsq[ok] / vseen[ok, None] - (vsum[ok] / vseen[ok, None]) ** 2
        incons = np.sqrt(np.clip(var.sum(1), 0, None))   # 跨視角顏色 std
        surf = surface_of(occ)
        surf_local = surf[idx[:, 0], idx[:, 1], idx[:, 2]]
        carve_local = surf_local & ok & (incons > color_tau)
        nc = int(carve_local.sum())
        if nc == 0:
            break
        ci = idx[carve_local]
        occ[ci[:, 0], ci[:, 1], ci[:, 2]] = False
        print(f"    iter {it+1}: 雕 {nc} 表面體素 (剩 {int(occ.sum())})")
    return occ


def _project(P, C, Rb, W, H):
    Rwc, t = cam.pose_to_w2c(C, Rb); K = cam.intrinsics(W, H)
    X = P @ Rwc.T + t; z = X[:, 2]; ok = z > 1e-9
    zz = np.where(ok, z, 1.0)
    u = np.round(K[0, 0] * X[:, 0] / zz + K[0, 2]).astype(np.int64)
    v = np.round(K[1, 1] * X[:, 1] / zz + K[1, 2]).astype(np.int64)
    return u, v, ok


def photo_carve_ncc(occ, gm, vs, views, ncc_min, min_views, max_iter, patch_r):
    """NCC 版:每視角取投影點 K×K 灰階 patch,去均值+L2 正規化(→亮度/對比不變)。
    跨視角平均 pairwise NCC = (|Σd|²−V)/(V(V−1));NCC 低=不一致=雕。無紋理 patch(std≈0)不判(保守留)。"""
    occ = occ.copy()
    off = np.arange(-patch_r, patch_r + 1)
    dv, du = np.meshgrid(off, off, indexing="ij"); du = du.ravel(); dv = dv.ravel()
    for it in range(max_iter):
        idx = np.argwhere(occ)
        if len(idx) == 0:
            break
        P = gm + (idx + 0.5) * vs; M = len(P); Kp = len(du)
        sum_d = np.zeros((M, Kp)); cnt_tex = np.zeros(M)
        for v in views:
            vox_at = CG.zbuffer_visible(P, v["C"], v["Rb"], v["W"], v["H"], vs)
            vis = np.unique(vox_at[vox_at >= 0])             # 該視角可見(最前)體素
            if len(vis) == 0:
                continue
            uu, vv, ok = _project(P, v["C"], v["Rb"], v["W"], v["H"])
            gray = v["gray"]; fg = v["fg"]; W = v["W"]; H = v["H"]
            vis = vis[ok[vis] & fg[np.clip(vv[vis], 0, H-1), np.clip(uu[vis], 0, W-1)]]  # 前景內
            if len(vis) == 0:
                continue
            pu = np.clip(uu[vis][:, None] + du[None, :], 0, W - 1)
            pv = np.clip(vv[vis][:, None] + dv[None, :], 0, H - 1)
            patch = gray[pv, pu]                              # (nvis, K²)
            cen = patch - patch.mean(1, keepdims=True)
            norm = np.sqrt((cen ** 2).sum(1))
            tex = norm > 1e-3                                 # 有紋理才算 NCC
            desc = np.zeros_like(cen); desc[tex] = cen[tex] / norm[tex, None]
            sum_d[vis] += desc; cnt_tex[vis] += tex
        Vt = cnt_tex
        sq = (sum_d ** 2).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_ncc = np.where(Vt >= 2, (sq - Vt) / (Vt * (Vt - 1)), 1.0)  # 無/單一紋理視角→1(保守留)
        surf = surface_of(occ)
        surf_local = surf[idx[:, 0], idx[:, 1], idx[:, 2]]
        carve_local = surf_local & (Vt >= min_views) & (mean_ncc < ncc_min)
        nc = int(carve_local.sum())
        if nc == 0:
            break
        ci = idx[carve_local]; occ[ci[:, 0], ci[:, 1], ci[:, 2]] = False
        print(f"    iter {it+1}: 雕 {nc} 表面體素 (剩 {int(occ.sum())})")
    return occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--num-views", type=int, default=34, dest="num_views")
    ap.add_argument("--voxel", type=float, default=0.005)
    ap.add_argument("--metric", choices=["color", "ncc"], default="ncc")
    ap.add_argument("--color-tau", type=float, default=0.09, dest="color_tau")
    ap.add_argument("--ncc-min", type=float, default=0.3, dest="ncc_min",
                    help="NCC 低於此=不一致=雕(NCC∈[-1,1],越大越嚴)")
    ap.add_argument("--patch-radius", type=int, default=3, dest="patch_r", help="NCC patch 半徑(3→7×7)")
    ap.add_argument("--min-views", type=int, default=2, dest="min_views")
    ap.add_argument("--max-iter", type=int, default=15, dest="max_iter")
    args = ap.parse_args()

    for sc in args.scenes:
        print(f"\n######## {sc} ########")
        t0 = time.time()
        views = load_views(sc, args.num_views)
        t_load = time.time() - t0
        print(f"  載入 {len(views)} 視角: {t_load:.2f}s")

        t0 = time.time()
        occ, gm, vs = build_hull(views, args.voxel)
        t_hull = time.time() - t0
        print(f"  ① 建 voxel hull: {t_hull:.2f}s  ({int(occ.sum())} vox)")

        t0 = time.time()
        if args.metric == "ncc":
            carved = photo_carve_ncc(occ, gm, vs, views, args.ncc_min, args.min_views,
                                     args.max_iter, args.patch_r)
        else:
            carved = photo_carve(occ, gm, vs, views, args.color_tau, args.min_views, args.max_iter)
        t_b = time.time() - t0
        print(f"  ② route B 光度雕刻[{args.metric}]: {t_b:.2f}s  ({int(occ.sum())}→{int(carved.sum())} vox)")

        out = OUT_ROOT / sc; out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "hull_photoB.npz", occupancy=carved, grid_min=gm, voxel_size=vs)
        e0 = eval_vs_gt(occ, gm, vs, sc); e1 = eval_vs_gt(carved, gm, vs, sc)
        if e0 and e1:
            print(f"  {'':6}{'體積L':>8}{'覆蓋':>8}{'冗餘':>8}{'膨脹':>8}")
            print(f"  {'雕前':6}{e0['vol_L']:8.2f}{e0['coverage']:8.3f}{e0['redundancy']:8.3f}{e0['bloat']:8.2f}")
            print(f"  {'雕後':6}{e1['vol_L']:8.2f}{e1['coverage']:8.3f}{e1['redundancy']:8.3f}{e1['bloat']:8.2f}")
        print(f"  ⏱ 建hull {t_hull:.2f}s + routeB {t_b:.2f}s (載入 {t_load:.2f}s 另計)")


if __name__ == "__main__":
    main()
