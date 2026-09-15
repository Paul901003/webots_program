#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""add_surface_mask.py — 對 hull.npz 加 `surface` mask:表面 voxel = occupied 且至少一個
6-鄰居(面相鄰)是空的;內部 voxel(6 面鄰居全滿)挖掉。不動原 occupancy。

下游可用 z["surface"] 只取表面 voxel(例如把遮罩 CLIP 特徵只投影到表面)。
用法: ./add_surface_mask.py [scene|group|(空=全部)] [--root srp_hull_v12]
"""
import argparse
import glob
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure

REPO = Path(__file__).resolve().parents[2]
_ST6 = generate_binary_structure(3, 1)   # 6-鄰居(只面相鄰,不含對角)


def surface_of(occ):
    """occ(bool 3D)→ 表面 voxel bool:occupied 且非「6 面鄰居全滿」。"""
    occ = occ.astype(bool)
    return occ & ~binary_erosion(occ, structure=_ST6)


def add(npz_path):
    z = dict(np.load(npz_path))
    occ = z["occupancy"].astype(bool)
    surf = surface_of(occ)
    z["surface"] = surf.astype(np.uint8)
    np.savez(npz_path, **z)
    return int(occ.sum()), int(surf.sum())


def resolve(base, targets):
    if not targets:
        return sorted(glob.glob(str(base / "*_scene*/hull.npz")))
    out = []
    for t in targets:
        if "scene" in t:
            out.append(str(base / t / "hull.npz"))
        else:
            out += sorted(glob.glob(str(base / f"{t}_scene*/hull.npz")))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--root", default="srp_hull_v12")
    args = ap.parse_args()
    base = REPO / "data" / "eval" / args.root
    paths = resolve(base, args.targets)
    n = 0
    for p in paths:
        if not Path(p).is_file():
            continue
        no, ns = add(p)
        n += 1
        if len(paths) <= 3:
            print(f"{Path(p).parent.name}: occupied {no} → surface {ns} ({ns/max(no,1)*100:.0f}%)")
    print(f"完成 {n} 個 hull.npz 加 surface mask（6-鄰居）")


if __name__ == "__main__":
    main()
