#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""gen_normals_viz — 表面法向箭頭(線段版,方向用實際端點;每群一個 shape 供上色)。

對某場景某 root:取 labels==k(完整可見表面 voxel,鎖定集合)每 voxel:
  法向 = np.gradient(occupancy) 三軸(正確軸序);朝外 = -grad/‖grad‖。已驗證 cosine 0.89。
  每支箭頭 = 線段 base=p → tip=p+n*L。方向由端點決定,不碰任何 primitive 軸慣例。
  依語意群分組,每群一個 shape(color + points + lineIndex),控制器每群一個 IndexedLineSet
  + Material.emissiveColor 上色(VRML 線段標準上色法)。
輸出 controllers/srp_normal_viz/normals.json。
用法: ./gen_normals_viz.py <scene> [--root ...] [--len-vox 3]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "data" / "eval"
HULL = "srp_hull_mv2_v12_am1_photo"
OUT = REPO / "controllers" / "srp_normal_viz" / "normals.json"
MAXN = 1200
PAL = [[0.95, 0.20, 0.20], [0.20, 0.60, 0.95], [0.20, 0.85, 0.35], [0.98, 0.80, 0.15],
       [0.80, 0.30, 0.90], [0.15, 0.88, 0.88], [0.98, 0.55, 0.12], [0.60, 0.40, 0.95],
       [0.50, 0.85, 0.20], [0.95, 0.35, 0.62], [0.35, 0.35, 0.98], [0.72, 0.72, 0.20]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--root", default="srp_hull_semcluster_surf_am1photo")
    ap.add_argument("--len-vox", type=float, default=3.0)
    a = ap.parse_args()
    zz = np.load(EVAL / HULL / a.scene / "hull.npz")
    occ = zz["occupancy"].astype(float); gm = zz["grid_min"]; vs = float(zz["voxel_size"])
    labels = np.load(EVAL / a.root / a.scene / "instances.npz")["labels"]
    g0, g1, g2 = np.gradient(occ)
    idx = np.argwhere(labels > 0)
    grad = np.stack([g0[labels > 0], g1[labels > 0], g2[labels > 0]], 1)
    mag = np.linalg.norm(grad, axis=1); keep = mag > 1e-6
    idx = idx[keep]; grad = grad[keep]; mag = mag[keep]; lab = labels[labels > 0][keep]
    if len(idx) > MAXN:
        sel = np.linspace(0, len(idx) - 1, MAXN).astype(int)
        idx = idx[sel]; grad = grad[sel]; mag = mag[sel]; lab = lab[sel]
    p = gm + (idx + 0.5) * vs
    n = -grad / mag[:, None]
    L = a.len_vox * vs
    tip = p + n * L
    per = defaultdict(lambda: {"points": [], "index": []})
    for i in range(len(idx)):
        s = per[int(lab[i])]; b = len(s["points"])
        s["points"].append([round(float(x), 5) for x in p[i]])
        s["points"].append([round(float(x), 5) for x in tip[i]])
        s["index"] += [b, b + 1, -1]
    shapes = [{"color": PAL[(k - 1) % len(PAL)], "points": v["points"], "index": v["index"]}
              for k, v in sorted(per.items())]
    OUT.write_text(json.dumps({"vs": vs, "scene": a.scene, "root": a.root, "shapes": shapes}))
    print(f"{a.scene}: {len(idx)} 箭頭 / {len(shapes)} 群 (線段長 {L*100:.1f}cm) → {OUT}")


if __name__ == "__main__":
    main()
