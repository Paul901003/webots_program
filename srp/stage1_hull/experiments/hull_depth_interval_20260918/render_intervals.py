#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""Render conservative per-pixel depth intervals from an existing voxel hull.

Each occupied voxel is rendered as a 3-D cube, not just its center.  For every
pixel touched by a cube, `z_enter` is the nearest cube face bound and `z_exit`
is the farthest.  These are geometric MVS search bounds, not estimated depth.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[4]
os.environ.setdefault("SAM_ROOT", str(REPO / "data" / "eval" / "mobilesamv2_fast"))
os.environ.setdefault("CAPTURES_ROOT", str(REPO / "data" / "captures_fast"))
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage1_hull"))
import camera as cam  # noqa: E402
import run_scene as RS  # noqa: E402


DEFAULT_HULL_ROOT = "srp_hull_mv2_v12_am1"
DEFAULT_OUT_ROOT = "hull_depth_interval_mv2_v12_am1"


def cube_depth_extent(Rwc, voxel_size):
    """Half extent of an axis-aligned world voxel along camera z."""
    return 0.5 * voxel_size * np.abs(np.asarray(Rwc)[2]).sum()


def render_intervals(occupancy, grid_min, voxel_size, K, Rwc, t, shape):
    """Return conservative (enter, exit) camera-z maps for occupied voxels.

    A projected cube is conservatively rasterized by its image-space bounding
    rectangle.  This may widen an interval by at most one voxel footprint, but
    never creates a too-narrow search range that could exclude a true surface.
    """
    H, W = shape
    idx = np.argwhere(occupancy)
    enter = np.full((H, W), np.inf, np.float32)
    exit_ = np.full((H, W), -np.inf, np.float32)
    if len(idx) == 0:
        return enter, exit_

    centers = np.asarray(grid_min, float) + (idx + 0.5) * float(voxel_size)
    X = centers @ np.asarray(Rwc).T + np.asarray(t)
    z = X[:, 2]
    front = z > 1e-6
    X = X[front]
    z = z[front]
    if len(z) == 0:
        return enter, exit_

    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    u = fx * X[:, 0] / z + cx
    v = fy * X[:, 1] / z + cy
    half_diag = np.sqrt(3.0) * float(voxel_size) * 0.5
    # A square with this radius contains every projected cube corner.  The
    # camera-z extent is exact for an axis-aligned world cube under Rwc.
    radius = np.ceil(np.maximum(fx, fy) * half_diag / np.maximum(z - half_diag, 1e-6)).astype(int) + 1
    z_half = cube_depth_extent(Rwc, voxel_size)
    z_lo = z - z_half
    z_hi = z + z_half

    for ui, vi, ri, lo, hi in zip(np.rint(u).astype(int), np.rint(v).astype(int), radius, z_lo, z_hi):
        x0, x1 = max(0, ui - ri), min(W, ui + ri + 1)
        y0, y1 = max(0, vi - ri), min(H, vi + ri + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        enter[y0:y1, x0:x1] = np.minimum(enter[y0:y1, x0:x1], lo)
        exit_[y0:y1, x0:x1] = np.maximum(exit_[y0:y1, x0:x1], hi)
    return enter, exit_


def save_preview(out_dir, view, fg, enter, exit_):
    coverage = np.isfinite(enter)
    thickness = np.zeros_like(enter, np.float32)
    thickness[coverage] = exit_[coverage] - enter[coverage]
    cv2.imwrite(str(out_dir / f"{view}_coverage.png"), (coverage.astype(np.uint8) * 255))
    # Green: SAM foreground supported by the hull.  Red: foreground that has
    # no hull ray interval.  Blue: hull interval outside the current foreground.
    overlap = np.zeros((*fg.shape, 3), np.uint8)
    overlap[fg & coverage] = (0, 210, 0)
    overlap[fg & ~coverage] = (0, 0, 230)
    overlap[coverage & ~fg] = (220, 90, 0)
    cv2.imwrite(str(out_dir / f"{view}_overlap.png"), overlap)
    max_mm = max(10.0, float(np.percentile(thickness[coverage], 99)) * 1000.0) if coverage.any() else 10.0
    scaled = np.clip(thickness * 1000.0 / max_mm * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{view}_thickness.png"), cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO))
    fg_count = int(fg.sum())
    covered_fg = int((coverage & fg).sum())
    return {
        "view": view,
        "foreground_pixels": fg_count,
        "interval_pixels": int(coverage.sum()),
        "foreground_covered_pixels": covered_fg,
        "foreground_uncovered_pixels": int((fg & ~coverage).sum()),
        "interval_outside_foreground_pixels": int((coverage & ~fg).sum()),
        "foreground_coverage": (covered_fg / fg_count) if fg_count else 0.0,
        "thickness_mm_p50": float(np.median(thickness[coverage]) * 1000.0) if coverage.any() else None,
        "thickness_mm_p95": float(np.percentile(thickness[coverage], 95) * 1000.0) if coverage.any() else None,
    }


def process(scene, hull_root, out_root):
    hull_path = REPO / "data" / "eval" / hull_root / scene / "hull.npz"
    data = np.load(hull_path, allow_pickle=False)
    occ = data["occupancy"].astype(bool)
    gm = data["grid_min"].astype(float)
    vs = float(data["voxel_size"])
    meta = json.loads(str(data["build_meta"]))
    if meta.get("sam_root") != "mobilesamv2_fast" or meta.get("n_views") != 12:
        raise ValueError(f"{hull_path}: expected mobilesamv2_fast / 12 views, got {meta.get('sam_root')} / {meta.get('n_views')}")
    if RS.SAM_ROOT.name != "mobilesamv2_fast" or RS.CAPTURES.name != "captures_fast":
        raise ValueError(f"interval input must be mobilesamv2_fast/captures_fast, got {RS.SAM_ROOT}/{RS.CAPTURES}")

    masks, Ks, extr, names = RS.load_scene(scene, num_views=12)
    if names != meta.get("views"):
        raise ValueError("selected 12 views differ from the hull build metadata")
    out_dir = REPO / "data" / "eval" / out_root / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for fg, K, (Rwc, t), name in zip(masks, Ks, extr, names):
        enter, exit_ = render_intervals(occ, gm, vs, K, Rwc, t, fg.shape)
        np.savez_compressed(out_dir / f"{name}_interval.npz", z_enter=enter, z_exit=exit_, foreground=fg)
        summaries.append(save_preview(out_dir, name, fg, enter, exit_))
    result = {
        "scene": scene,
        "hull_path": str(hull_path.relative_to(REPO)),
        "sam_root": meta["sam_root"],
        "n_views": len(names),
        "voxel_size_m": vs,
        "views": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    cov = [x["foreground_coverage"] for x in summaries]
    print(f"[{scene}] interval foreground coverage mean/min={np.mean(cov):.4f}/{np.min(cov):.4f} -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--hull-root", default=DEFAULT_HULL_ROOT)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    args = ap.parse_args()
    for scene in args.scenes:
        process(scene, args.hull_root, args.out_root)


if __name__ == "__main__":
    main()
