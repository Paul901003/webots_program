#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""gen_colmap_viz.py — 為 srp_colmap_viz 產「COLMAP MVS 點雲 + 真實 YCB GT」manifest。

仿 gen_viz_objs.py,只是把 hull mesh 換成 COLMAP dense 點雲:
讀 data/eval/colmap_mvs/<scene>/fused.ply → 降採樣 → 寫 ascii ply(colmap_points.ply)供 controller 用 PointSet 顯示。
GT 沿用(actual annotations 擺真實 YCB mesh,半透明對照)。輸出 manifest.json。

用法: ./gen_colmap_viz.py <scene> --out <objs_dir> [--no-gt] [--max-points 20000]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
MVS_ROOT = REPO / "data" / "eval" / "colmap_mvs"
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "srp" / "io")); from labels import LABELS  # data/labels 分層(類別/數量/場景)
ASSETS = REPO / "urdfs" / "ycb_assets"
GEO = json.loads((REPO / "controllers" / "ycb_supervisor" / "ycb_geometries.json").read_text())


def ycb_center(name):
    c = GEO.get(name, {}).get("center", {"x": 0, "y": 0, "z": 0})
    return np.array([c["x"], c["y"], c["z"]], dtype=float)


def axis_angle_to_mat(axis, angle):
    axis = np.asarray(axis, float); n = np.linalg.norm(axis)
    if n < 1e-9 or abs(angle) < 1e-12:
        return np.eye(3)
    x, y, z = axis / n; c, s = np.cos(angle), np.sin(angle); C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def gt_objects(scene):
    ann = LABELS / scene / "actual" / "annotations.json"
    if not ann.is_file():
        return []
    d = json.loads(ann.read_text())
    return d["images"][0].get("objects", []) if d.get("images") else []


def ycb_mesh_path(name):
    for fn in ("textured.obj", "nontextured.ply", "nontextured.stl"):
        p = ASSETS / name / "google_16k" / fn
        if p.is_file():
            return str(p.resolve())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-gt", action="store_true")
    ap.add_argument("--max-points", type=int, default=20000)
    ap.add_argument("--tag", default="", help="讀 <scene>/<tag>/fused.ply(sweep 各組,如 v12_s640)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ── COLMAP 點雲 → 降採樣 → ascii ply ──
    ply = (MVS_ROOT / args.scene / args.tag / "fused.ply") if args.tag else (MVS_ROOT / args.scene / "fused.ply")
    n_pts = 0
    if not ply.is_file():
        print(f"[gen_colmap_viz] 無 {ply} → 只顯示 GT(先跑 colmap_mvs.py 產點雲)")
    else:
        pc = trimesh.load(str(ply))
        pts = np.asarray(pc.vertices, dtype=float)
        cols = (np.asarray(pc.colors)[:, :3] if getattr(pc, "colors", None) is not None
                and len(pc.colors) == len(pts) else np.full((len(pts), 3), 200, np.uint8))
        if len(pts) > args.max_points:
            idx = np.random.default_rng(0).choice(len(pts), args.max_points, replace=False)
            pts, cols = pts[idx], cols[idx]
        n_pts = len(pts)
        with open(out / "colmap_points.ply", "w") as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {n_pts}\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for (x, y, z), (r, g, b) in zip(pts, cols):
                f.write(f"{x:.5f} {y:.5f} {z:.5f} {int(r)} {int(g)} {int(b)}\n")

    # ── GT(沿用 gen_viz_objs 邏輯)──
    ycb_items = []
    if not args.no_gt:
        for o in gt_objects(args.scene):
            name = o["name"]; mesh = ycb_mesh_path(name)
            if not mesh:
                continue
            aa = o.get("rotation_axis_angle", [0, 1, 0, 0])
            R = axis_angle_to_mat(aa[:3], aa[3]) if len(aa) == 4 else np.eye(3)
            tr = (np.asarray(o["position_m"], float) - R @ ycb_center(name)).tolist()
            ycb_items.append({"mesh": mesh, "translation": [float(x) for x in tr],
                              "rotation": [float(x) for x in aa],
                              "color": [0.6, 0.6, 0.6], "transparency": 0.55,
                              "name": f"gt_{name}"})

    mani = {"source": args.scene, "n_points": n_pts,
            "pointcloud": "colmap_points.ply" if n_pts else None,
            "ycb": len(ycb_items), "ycb_items": ycb_items}
    (out / "manifest.json").write_text(json.dumps(mani, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[gen_colmap_viz] {args.scene}: COLMAP 點雲 {n_pts} 點 + 真實YCB {len(ycb_items)} → {out}")


if __name__ == "__main__":
    main()
