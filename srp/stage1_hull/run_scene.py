#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""run_scene.py — 對真實拍攝場景跑 Stage 1 visual hull(class-agnostic 全前景)。

輸入:既有拍攝資料
  遮罩 data/eval/sam_only/<scene>/<view>/masks/mask_*.png(每視角前景=遮罩聯集,排除地板)
  位姿 data/captures/multi_<g>/<scene>/<view>_pose.json
輸出:data/eval/srp_hull/<scene>/hull.npz(occupancy / observed / grid_min / voxel_size)
      + 印體積、連通元件數。

雕刻直接呼叫 srp.stage1_hull.carve.carve_visual_hull(規格實作,含 observed)。
不用深度、不用 GT。需 webots_visual_hull(numpy/cv2/scipy)。

用法: ./srp/stage1_hull/run_scene.py n1_scene0001 [--voxel 0.005] [--no-table]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera as cam                      # noqa: E402
import masks as M                         # noqa: E402
from carve import carve_visual_hull       # noqa: E402

CAPTURES = REPO / "data" / "captures"
SAM_ROOT = REPO / "data" / "eval" / "sam_only"
OUT_ROOT = REPO / "data" / "eval" / "srp_hull"

# 工作空間 AABB(世界,公尺)+ 封底。0.7×0.7×0.35,中心 x=0.35(對齊現有拍攝資料)。
BOX_MIN = np.array([0.0, -0.35, 0.0])
BOX_MAX = np.array([0.7, 0.35, 0.35])
TABLE_Z = 0.0


def view_foreground(view_dir):
    """該視角前景 = 保留(非地板)遮罩的聯集。"""
    km = M.kept_object_masks(view_dir)
    if not km:
        return None
    fg = None
    for b, _ in km:
        fg = b if fg is None else (fg | b)
    return fg if fg is not None and fg.any() else None


def load_scene(scene):
    """回傳 (masks[], Ks[], extr[])。"""
    group = scene.split("_")[0]
    sdir = CAPTURES / f"multi_{group}" / scene
    sam = SAM_ROOT / scene
    masks, Ks, extr = [], [], []
    for vdir in sorted(sam.glob("view_*")):
        pose = sdir / f"{vdir.name}_pose.json"
        if not pose.is_file():
            continue
        fg = view_foreground(vdir)
        if fg is None:
            continue
        H, W = fg.shape
        C, R_body = cam.load_pose(pose)
        masks.append(fg)
        Ks.append(cam.intrinsics(W, H))
        extr.append(cam.pose_to_w2c(C, R_body))
    return masks, Ks, extr


def _suf(tag):
    return f"_{tag}" if tag else ""


def process(scene, voxel, use_table, allow_miss, out_root=OUT_ROOT, tag=""):
    masks, Ks, extr = load_scene(scene)
    if len(masks) < 2:
        print(f"[skip] {scene}: 有效視角 < 2")
        return None
    hull = carve_visual_hull(masks, Ks, extr, BOX_MIN, BOX_MAX, voxel,
                             table_z=(TABLE_Z if use_table else None),
                             allow_miss=allow_miss)
    n_comp = ndimage.label(hull.occupancy,
                           ndimage.generate_binary_structure(3, 1))[1]
    out_dir = out_root / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"hull{_suf(tag)}.npz",
                        occupancy=hull.occupancy, observed=hull.observed,
                        grid_min=hull.grid_min, voxel_size=hull.voxel_size)
    print(f"[{scene}] 視角{len(masks)} voxel{voxel} → 佔據 {int(hull.occupancy.sum())} vox "
          f"({hull.volume()*1e3:.2f} L) 連通元件 {n_comp} | "
          f"observed {int(hull.observed.sum())}/{hull.observed.size}")
    return hull


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--voxel", type=float, default=0.005)
    ap.add_argument("--no-table", action="store_true", help="不封底")
    ap.add_argument("--allow-miss", type=int, default=0, dest="allow_miss",
                    help="soft carving 容忍漏檢視角數(0=硬交集,證實最佳;>0 hull 會膨脹)")
    ap.add_argument("--root", default="srp_hull",
                    help="輸出根目錄名(data/eval/<root>/)")
    ap.add_argument("--tag", default="",
                    help="檔名後綴(如 am1):輸出 hull_<tag>.npz,分開儲存不同設定")
    args = ap.parse_args()
    out_root = REPO / "data" / "eval" / args.root
    for sc in args.scenes:
        try:
            process(sc, args.voxel, not args.no_table, args.allow_miss, out_root, args.tag)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")


if __name__ == "__main__":
    main()
