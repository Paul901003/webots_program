#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""photo_carve.py — 路線 A:用現成 MVS 點雲雕掉 visual hull 的幽靈/凹面過估體素。

概念:hull 是剪影交集的「外包絡」(保證含真物,但幽靈體積+凹面填實=過估)。COLMAP MVS 的
fused.ply 是「真實表面點」。用它把「非表面、且從外部可達」的佔據體素雕掉,同時保留內部實心、
且不誤雕無紋理物(無 MVS 點處保守保留)。不動 carve.py(見 dont-overwrite 原則)。

演算法(不挖空實心的關鍵=外緣連通判定):
  supported = 佔據體素離 MVS 點 <TAU(在真實表面附近)
  passable  = 非 supported 的體素(空的 或 佔據但不在表面)
  從格子外緣 flood fill passable → 碰到 supported 就停
  carve = 外緣 flood 可達 & 非 supported & 佔據 & 附近 R 內有 MVS 點(覆蓋守衛)
    → 幽靈(凸出)+ 凹面填充被雕;內部(被 supported 包住)保留;無紋理物(無 MVS 點)保留

輸入: hull.npz(occupancy) + COLMAP fused.ply。輸出: data/eval/photo_carve/<scene>/hull_photo.npz。
評估: 對 GT mesh 量雕刻前後 體積/冗餘/recall(用 eval_mesh.solid_mesh_occ)。

用法: ./photo_carve.py stack3_scene0001 --hull-root srp_hull_vox5mm_v34 --ply-tag v34_s1280
env 依 eval_mesh(GT mesh)。
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import eval_mesh as EM          # noqa: E402

EVAL = REPO / "data" / "eval"
OUT_ROOT = EVAL / "photo_carve"


def load_ply_points(path):
    m = trimesh.load(str(path), process=False)
    pts = np.asarray(m.vertices) if hasattr(m, "vertices") else np.asarray(m)
    return pts.reshape(-1, 3)


def carve(occ, grid_min, vs, mvs_pts, tau, radius):
    """回傳 (carved_occ, stats)。"""
    occ = occ.astype(bool)
    idx = np.argwhere(occ)                         # 佔據體素索引 (M,3)
    centers = grid_min + (idx + 0.5) * vs          # 世界座標
    tree = cKDTree(mvs_pts)
    d, _ = tree.query(centers)                     # 每佔據體素→最近 MVS 點
    supported = d < tau                            # 在真實表面附近
    covered = d < radius                           # 附近有 MVS 點(有紋理/有重建)

    sup_grid = np.zeros(occ.shape, bool)           # supported 佔據體素(阻擋 flood)
    sup_grid[idx[supported, 0], idx[supported, 1], idx[supported, 2]] = True
    cov_grid = np.zeros(occ.shape, bool)
    cov_grid[idx[covered, 0], idx[covered, 1], idx[covered, 2]] = True

    # passable = 非 supported 的所有體素(空的 或 佔據但非表面)
    passable = ~sup_grid
    # 從格子外緣 flood:label passable 的連通元件,取碰到邊界的那些 = 外部可達
    lbl, n = ndimage.label(passable, ndimage.generate_binary_structure(3, 1))
    border_ids = set(np.unique(np.concatenate([
        lbl[0, :, :].ravel(), lbl[-1, :, :].ravel(),
        lbl[:, 0, :].ravel(), lbl[:, -1, :].ravel(),
        lbl[:, :, 0].ravel(), lbl[:, :, -1].ravel()])))
    border_ids.discard(0)
    exterior = np.isin(lbl, list(border_ids))      # 外部可達空間

    # carve = 佔據 & 非 supported & 外部可達 & 有 MVS 覆蓋(守衛無紋理物)
    carve_mask = occ & (~sup_grid) & exterior & cov_grid
    carved = occ & ~carve_mask

    stats = {
        "hull_vox": int(occ.sum()),
        "supported": int(sup_grid.sum()),
        "carved": int(carve_mask.sum()),
        "kept": int(carved.sum()),
        "no_mvs_kept": int((occ & ~sup_grid & exterior & ~cov_grid).sum()),  # 無紋理保守保留
    }
    return carved, stats


def eval_vs_gt(occ, grid_min, vs, scene):
    """對 GT 實心 mesh 量 recall(覆蓋)/冗餘(過估)。回傳 dict。"""
    shape = occ.shape
    gt = EM.solid_mesh_occ(scene, grid_min, vs, shape)
    if not gt:
        return None
    union_gt = np.zeros(shape, bool)
    for g in gt.values():
        union_gt |= g
    inter = (occ & union_gt).sum()
    return {
        "vol_L": float(occ.sum()) * (vs ** 3) * 1e3,
        "gt_vol_L": float(union_gt.sum()) * (vs ** 3) * 1e3,
        "coverage": float(inter) / float(union_gt.sum()) if union_gt.sum() else 0,  # 真物被 hull 蓋住比例
        "redundancy": 1 - float(inter) / float(occ.sum()) if occ.sum() else 0,      # hull 有多少不在真物內(過估)
        "bloat": float(occ.sum()) / float(union_gt.sum()) if union_gt.sum() else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--hull-root", default="srp_hull_vox5mm_v34")
    ap.add_argument("--ply-tag", default="v34_s1280", help="colmap_mvs/<scene>/<tag>/fused.ply")
    ap.add_argument("--tau", type=float, default=0.010, help="表面支撐帶(m)")
    ap.add_argument("--radius", type=float, default=0.030, help="MVS 覆蓋守衛半徑(m)")
    args = ap.parse_args()

    for sc in args.scenes:
        hp = EVAL / args.hull_root / sc / "hull.npz"
        ply = EVAL / "colmap_mvs" / sc / args.ply_tag / "fused.ply"
        if not hp.is_file():
            print(f"[skip] {sc}: 無 {hp}"); continue
        if not ply.is_file():
            print(f"[skip] {sc}: 無 {ply}"); continue
        z = np.load(hp)
        occ = z["occupancy"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
        mvs = load_ply_points(ply)
        carved, st = carve(occ, gm, vs, mvs, args.tau, args.radius)

        e0 = eval_vs_gt(occ, gm, vs, sc)
        e1 = eval_vs_gt(carved, gm, vs, sc)
        out = OUT_ROOT / sc; out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "hull_photo.npz", occupancy=carved,
                            grid_min=gm, voxel_size=vs)
        print(f"\n[{sc}] MVS點 {len(mvs)}  hull {st['hull_vox']} vox "
              f"→ supported {st['supported']} / 雕 {st['carved']} / 留 {st['kept']} "
              f"(無紋理保守留 {st['no_mvs_kept']})")
        if e0 and e1:
            print(f"  {'':8}{'體積L':>8}{'覆蓋':>8}{'冗餘':>8}{'膨脹':>8}")
            print(f"  {'雕前':8}{e0['vol_L']:8.2f}{e0['coverage']:8.3f}{e0['redundancy']:8.3f}{e0['bloat']:8.2f}")
            print(f"  {'雕後':8}{e1['vol_L']:8.2f}{e1['coverage']:8.3f}{e1['redundancy']:8.3f}{e1['bloat']:8.2f}")
            print(f"  → 冗餘 {e0['redundancy']:.3f}→{e1['redundancy']:.3f} "
                  f"(過估降 {(e0['redundancy']-e1['redundancy'])*100:.1f}pp);"
                  f"覆蓋 {e0['coverage']:.3f}→{e1['coverage']:.3f} "
                  f"(掉 {(e0['coverage']-e1['coverage'])*100:.1f}pp)")


if __name__ == "__main__":
    main()
