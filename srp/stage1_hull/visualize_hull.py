#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""visualize_hull.py — 把 Stage 1 的 hull.npz 視覺化驗證。

輸出(到 data/eval/srp_hull/<scene>/):
  hull.obj         marching cubes → 世界座標三角網格(可丟 Webots/MeshLab)
  hull_render.png  matplotlib 離線渲染:依 6-鄰接連通元件上色的多視角圖(headless)

用法: ./srp/stage1_hull/visualize_hull.py n3_scene0001 [--max-dim 64]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import mcubes
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "data" / "eval" / "srp_hull"


def export_obj(occ, grid_min, vs, path):
    """marching cubes → 世界座標 .obj。array 座標 v → 世界 grid_min+(v+0.5)*vs。"""
    verts, faces = mcubes.marching_cubes(occ.astype(np.float32), 0.5)
    if len(verts) == 0:
        print(f"  [warn] {path.name}: 空 hull"); return 0
    world = grid_min + (verts + 0.5) * vs
    lines = [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in world]
    lines += [f"f {a+1} {b+1} {c+1}" for a, b, c in faces.astype(int)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(verts)


def maxpool(a, f):
    """以 f 為塊做 max-pool 下採樣(bool)。"""
    if f <= 1:
        return a
    pad = [(0, (-s) % f) for s in a.shape]
    a = np.pad(a, pad)
    nx, ny, nz = (s // f for s in a.shape)
    return a.reshape(nx, f, ny, f, nz, f).max(axis=(1, 3, 5))


def render(occ, grid_min, vs, path, max_dim):
    lab, n = ndimage.label(occ, ndimage.generate_binary_structure(3, 1))
    f = max(1, int(np.ceil(max(occ.shape) / max_dim)))
    occ_d = maxpool(occ, f)
    lab_d = maxpool(lab.astype(np.int16) > 0, f)  # 佔據遮罩(下採樣)
    # 下採樣後重新標元件以上色(避免 label 值在 maxpool 後失真)
    lab_s, n_s = ndimage.label(occ_d, ndimage.generate_binary_structure(3, 1))
    cmap = plt.get_cmap("tab10")
    colors = np.zeros(occ_d.shape + (4,), dtype=float)
    for c in range(1, n_s + 1):
        colors[lab_s == c] = cmap((c - 1) % 10)
    fig = plt.figure(figsize=(12, 4.5))
    for i, (el, az) in enumerate([(25, -60), (25, 30), (80, -90)]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.voxels(occ_d, facecolors=colors, edgecolor=None)
        ax.view_init(elev=el, azim=az)
        ax.set_box_aspect(occ_d.shape)
        ax.set_title(f"view {i+1}  components={n}")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.suptitle(f"{path.parent.name}  occupied {int(occ.sum())} vox  "
                 f"voxel={vs} m  components={n}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return n


def process(scene, max_dim):
    d = OUT_ROOT / scene / "hull.npz"
    if not d.is_file():
        print(f"[skip] {scene}: 找不到 {d}(先跑 run_scene.py)"); return
    z = np.load(d)
    occ = z["occupancy"]; grid_min = z["grid_min"]; vs = float(z["voxel_size"])
    nv = export_obj(occ, grid_min, vs, OUT_ROOT / scene / "hull.obj")
    n = render(occ, grid_min, vs, OUT_ROOT / scene / "hull_render.png", max_dim)
    print(f"[{scene}] obj 頂點{nv} 連通元件{n} → {OUT_ROOT/scene}/hull.obj, hull_render.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--max-dim", type=int, default=64, help="渲染下採樣後最大軸格數")
    args = ap.parse_args()
    for sc in args.scenes:
        try:
            process(sc, args.max_dim)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")


if __name__ == "__main__":
    main()
