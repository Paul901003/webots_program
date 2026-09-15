#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""photo_carve_warp_onhull.py — 對「現成 hull root」做 warped-NCC 光雕(不自己 build_hull)。

photo_carve_warp.py 內建 build_hull(中心版),無法對 footprint hull(am1_fp)光雕。
本檔複用其核心 carve_warp,但 occupancy 改讀 --in-root 的 hull.npz(如 srp_hull_mv2_v12_am1_fp),
逐場計時、存到 --out-root。不改原檔。

用法: SAM_ROOT=.. CAPTURES_ROOT=.. \
  ./photo_carve_warp_onhull.py <scenes...> --in-root srp_hull_mv2_v12_am1_fp \
     --out-root srp_hull_mv2_v12_am1_fp_photo --ncc-min 0.1
參數預設對齊 am1_photo(warp_tilt20_ncc0.1_k4_min2_clu3_patch2_iter15)。
"""
import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import photo_carve_warp as PW           # noqa: E402  carve_warp / prep_views
from photo_carve_b import load_views    # noqa: E402
from photo_carve import eval_vs_gt      # noqa: E402
EVAL = REPO / "data" / "eval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--in-root", default="srp_hull_mv2_v12_am1_fp")
    ap.add_argument("--out-root", default="srp_hull_mv2_v12_am1_fp_photo")
    ap.add_argument("--num-views", type=int, default=12, dest="num_views")
    ap.add_argument("--ncc-min", type=float, default=0.1, dest="ncc_min")
    ap.add_argument("--k-src", type=int, default=4, dest="k_src")
    ap.add_argument("--min-src", type=int, default=2, dest="min_src")
    ap.add_argument("--min-cluster", type=int, default=3, dest="min_cluster")
    ap.add_argument("--patch-radius", type=int, default=2, dest="patch_r")
    ap.add_argument("--max-iter", type=int, default=15, dest="max_iter")
    ap.add_argument("--tilt-deg", type=float, default=20.0, dest="tilt_deg")
    a = ap.parse_args()
    for sc in a.scenes:
        hp = EVAL / a.in_root / sc / "hull.npz"
        if not hp.is_file():
            print(f"[skip] {sc}: 無 {a.in_root}/hull.npz"); continue
        z = np.load(hp); occ = z["occupancy"].astype(bool); gm = z["grid_min"]; vs = float(z["voxel_size"])
        views = PW.prep_views(load_views(sc, a.num_views))
        t0 = time.time()
        carved = PW.carve_warp(occ, gm, vs, views, a.ncc_min, a.k_src, a.min_src,
                               a.min_cluster, a.patch_r, a.max_iter, a.tilt_deg)
        dt = time.time() - t0
        out = EVAL / a.out_root / sc; out.mkdir(parents=True, exist_ok=True)
        meta = {"script": "photo_carve_warp_onhull.py", "src": a.in_root, "min_comp": 0,
                "carve": f"warp_tilt{int(a.tilt_deg)}_ncc{a.ncc_min}_k{a.k_src}_min{a.min_src}_clu{a.min_cluster}_patch{a.patch_r}_iter{a.max_iter}",
                "built": _dt.datetime.now().isoformat(timespec="seconds"), "seconds": round(dt, 2)}
        np.savez_compressed(out / "hull.npz", occupancy=carved, grid_min=gm, voxel_size=vs,
                            build_meta=json.dumps(meta, ensure_ascii=False))
        print(f"[{sc}] 光雕 {dt:.1f}s  {int(occ.sum())}→{int(carved.sum())} (雕掉 {int(occ.sum())-int(carved.sum())})", flush=True)


if __name__ == "__main__":
    main()
