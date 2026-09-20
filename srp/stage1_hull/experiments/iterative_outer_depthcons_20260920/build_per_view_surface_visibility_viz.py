#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""Write one red/gray center and footprint visible-surface view per camera.

The files are ``instances_{c|fp}_{view}.npz`` under one visualization root.
They are intentionally per-view, not a union over cameras.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
os.environ.setdefault("SAM_ROOT", str(REPO / "data" / "eval" / "mobilesamv2_fast"))
os.environ.setdefault("CAPTURES_ROOT", str(REPO / "data" / "captures_fast"))
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage1_hull"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
sys.path.insert(0, str(REPO / "srp" / "stage1_hull" / "experiments" / "hull_depth_interval_20260918"))

import camera as CAM  # noqa: E402
import cg_associate as CG  # noqa: E402
import interval_candidates as IC  # noqa: E402
import run_scene as RS  # noqa: E402


EVAL = REPO / "data" / "eval"


def center_visible(points, Rwc, t, K, width, height):
    """Indices owning the nearest projected center at an image pixel."""
    X = points @ Rwc.T + t
    z = X[:, 2]
    positive = z > 1e-9
    safe_z = np.where(positive, z, 1.0)
    u = np.rint(K[0, 0] * X[:, 0] / safe_z + K[0, 2]).astype(np.int64)
    v = np.rint(K[1, 1] * X[:, 1] / safe_z + K[1, 2]).astype(np.int64)
    valid = positive & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    ids = np.flatnonzero(valid)
    pixel = v[ids] * width + u[ids]
    order = np.argsort(z[ids], kind="stable")
    _, first = np.unique(pixel[order], return_index=True)
    return ids[order][first]


def save_view(out, tag, occupancy, surface_indices, visible_ids, grid_min, voxel_size, meta):
    labels = np.zeros(occupancy.shape, np.int16)
    selected = surface_indices[visible_ids]
    labels[tuple(selected.T)] = 1  # gen_viz_objs label 1 is red.
    np.savez_compressed(out / f"instances_{tag}.npz", occupancy=occupancy, labels=labels,
                        grid_min=grid_min, voxel_size=np.float64(voxel_size))
    (out / f"summary_{tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def build(scene, hull_root, out_root):
    source = np.load(EVAL / hull_root / scene / "hull.npz", allow_pickle=False)
    occupancy = source["occupancy"].astype(bool)
    grid_min = source["grid_min"].astype(float)
    voxel_size = float(source["voxel_size"])
    indices = np.argwhere(IC.outer_surface(occupancy))
    points = grid_min + (indices + 0.5) * voxel_size
    masks, Ks, extrinsics, names = RS.load_scene(scene, num_views=12)
    capture_dir = RS.CAPTURES / f"multi_{scene.split('_')[0]}" / scene
    out = EVAL / out_root / scene
    out.mkdir(parents=True, exist_ok=True)

    for name, foreground, K, (Rwc, t) in zip(names, masks, Ks, extrinsics):
        height, width = foreground.shape
        c_ids = center_visible(points, Rwc, t, K, width, height)
        C, Rb = CAM.load_pose(capture_dir / f"{name}_pose.json")
        fp_ids = np.unique(CG.zbuffer_visible(points, C, Rb, width, height, voxel_size))
        fp_ids = fp_ids[fp_ids >= 0]
        common = {
            "source_hull": hull_root,
            "camera_view": name,
            "occupied_voxels": int(occupancy.sum()),
            "geometric_surface_candidates": int(len(indices)),
            "red_means": "visible from this camera only",
            "gray_means": "occupied but not visible from this camera",
        }
        save_view(out, f"c_{name}", occupancy, indices, c_ids, grid_min, voxel_size,
                  {**common, "rule": "one center pixel wins z-buffer", "red_voxels": int(len(c_ids))})
        save_view(out, f"fp_{name}", occupancy, indices, fp_ids, grid_min, voxel_size,
                  {**common, "rule": "one projected voxel-footprint pixel wins z-buffer", "red_voxels": int(len(fp_ids))})
        print(f"[{name}] c red={len(c_ids)} fp red={len(fp_ids)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="+")
    parser.add_argument("--hull-root", default="srp_hull_mv2_v12_am1")
    parser.add_argument("--out-root", default="srp_hull_mv2_v12_am1_surface_visibility_perview")
    args = parser.parse_args()
    for scene in args.scenes:
        build(scene, args.hull_root, args.out_root)


if __name__ == "__main__":
    main()
