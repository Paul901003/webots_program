#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""Create red/gray Webots inputs for center versus footprint visible surfaces.

Both outputs use the same input occupancy.  Red means a geometric outer voxel
is visible in at least one of the 12 cameras; gray means occupied but not red.
Only the per-view visibility test changes:
``center`` tests one projected voxel center, while ``footprint`` uses the
existing projected-voxel-footprint z-buffer.
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
    """Local point indices owning one nearest center-projected image pixel."""
    camera_points = points @ Rwc.T + t
    depth = camera_points[:, 2]
    positive = depth > 1e-9
    safe_depth = np.where(positive, depth, 1.0)
    u = np.rint(K[0, 0] * camera_points[:, 0] / safe_depth + K[0, 2]).astype(np.int64)
    v = np.rint(K[1, 1] * camera_points[:, 1] / safe_depth + K[1, 2]).astype(np.int64)
    valid = positive & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    local = np.flatnonzero(valid)
    if not len(local):
        return local
    pixel = v[local] * width + u[local]
    order = np.argsort(depth[local], kind="stable")
    _, first = np.unique(pixel[order], return_index=True)
    return local[order][first]


def build(scene, hull_root, out_prefix):
    source_path = EVAL / hull_root / scene / "hull.npz"
    source = np.load(source_path, allow_pickle=False)
    occupancy = source["occupancy"].astype(bool)
    grid_min = source["grid_min"].astype(float)
    voxel_size = float(source["voxel_size"])
    geometric_surface = IC.outer_surface(occupancy)
    surface_indices = np.argwhere(geometric_surface)
    points = grid_min + (surface_indices + 0.5) * voxel_size

    masks, Ks, extrinsics, names = RS.load_scene(scene, num_views=12)
    group = scene.split("_")[0]
    capture_dir = RS.CAPTURES / f"multi_{group}" / scene
    center = np.zeros(len(points), bool)
    footprint = np.zeros(len(points), bool)
    per_view = []
    for name, foreground, K, (Rwc, t) in zip(names, masks, Ks, extrinsics):
        height, width = foreground.shape
        center_ids = center_visible(points, Rwc, t, K, width, height)
        center[center_ids] = True
        C, Rb = CAM.load_pose(capture_dir / f"{name}_pose.json")
        footprint_ids = np.unique(CG.zbuffer_visible(points, C, Rb, width, height, voxel_size))
        footprint_ids = footprint_ids[footprint_ids >= 0]
        footprint[footprint_ids] = True
        per_view.append({"view": name, "center": int(len(center_ids)), "footprint": int(len(footprint_ids))})

    results = {"c": center, "fp": footprint}
    for tag, visible in results.items():
        labels = np.zeros(occupancy.shape, np.int16)
        selected = surface_indices[visible]
        labels[tuple(selected.T)] = 1  # PALETTE[0] in gen_viz_objs.py is red.
        out = EVAL / f"{out_prefix}_{tag}" / scene
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "source_hull": hull_root,
            "surface_rule": "center_zbuffer" if tag == "c" else "footprint_zbuffer",
            "surface_candidate_rule": "occupied with at least one empty 6-neighbor",
            "views": names,
            "occupied_voxels": int(occupancy.sum()),
            "geometric_surface_voxels": int(geometric_surface.sum()),
            "red_visible_surface_voxels": int(visible.sum()),
            "gray_occupied_voxels": int(occupancy.sum() - visible.sum()),
            "per_view": per_view,
        }
        np.savez_compressed(out / "instances.npz", occupancy=occupancy, labels=labels,
                            grid_min=grid_min, voxel_size=np.float64(voxel_size))
        np.savez_compressed(out / "hull.npz", occupancy=occupancy, surface=geometric_surface,
                            grid_min=grid_min, voxel_size=np.float64(voxel_size),
                            build_meta=json.dumps(meta))
        (out / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[{scene} {tag}] red={int(visible.sum())} gray={int(occupancy.sum() - visible.sum())} -> {out}")

    print(json.dumps({
        "center_only": int((center & ~footprint).sum()),
        "footprint_only": int((footprint & ~center).sum()),
        "both": int((center & footprint).sum()),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="+")
    parser.add_argument("--hull-root", default="srp_hull_mv2_v12_am1")
    parser.add_argument("--out-prefix", default="srp_hull_mv2_v12_am1_surface_visibility")
    args = parser.parse_args()
    for scene in args.scenes:
        build(scene, args.hull_root, args.out_prefix)


if __name__ == "__main__":
    main()
