#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""Score prepared hull-interval ray candidates with the established NCC test.

This is an isolated comparison with `photo_ray_search_20260917`: it preserves
the old two-source foreground-patch NCC thresholds and changes only candidate
depths from 0--3 voxels to the conservative per-pixel hull interval.  A hull
cell is removed only from camera to a deeper, better-supported candidate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[4]
os.environ.setdefault("SAM_ROOT", str(REPO / "data" / "eval" / "mobilesamv2_fast"))
os.environ.setdefault("CAPTURES_ROOT", str(REPO / "data" / "captures_fast"))
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage1_hull"))
sys.path.insert(0, str(REPO / "srp" / "stage1_hull" / "experiments" / "photo_ray_search_20260917"))
import run_scene as RS  # noqa: E402
import photo_carve_warp as warp  # noqa: E402
import ray_search  # noqa: E402


EVAL = REPO / "data" / "eval"


def load_views(scene):
    """Load RGB together with the exact Stage-1 foreground/pose contract."""
    masks, Ks, extrinsics, names = RS.load_scene(scene, num_views=12)
    views = []
    group = scene.split("_")[0]
    scene_dir = RS.CAPTURES / f"multi_{group}" / scene
    for fg, K, (Rwc, t), name in zip(masks, Ks, extrinsics, names):
        image = cv2.imread(str(scene_dir / f"{name}.png"), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(scene_dir / f"{name}.png")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        views.append({
            "fg": fg, "gray": rgb.mean(axis=2), "W": fg.shape[1], "H": fg.shape[0],
            "K": K, "Kinv": np.linalg.inv(K), "Rwc": Rwc, "t": t,
            "Cw": -Rwc.T @ t,
        })
    return views, names


def removal_rays(occupancy, grid_min, voxel_size, surface_indices, refs, layers,
                 selected_layer, centers):
    """Mark occupied cells strictly before each selected coarse candidate."""
    remove = np.zeros_like(occupancy, bool)
    shape = np.asarray(occupancy.shape)
    selected = np.flatnonzero(selected_layer > 0)
    for row in selected:
        endpoint = layers[row, selected_layer[row]]
        if (endpoint < 0).any() or refs[row] < 0:
            continue
        origin = grid_min + (surface_indices[row] + 0.5) * voxel_size
        direction = origin - centers[refs[row]]
        direction /= np.linalg.norm(direction) + 1e-9
        target = grid_min + (endpoint + 0.5) * voxel_size
        length = float(np.dot(target - origin, direction))
        for distance in np.arange(0.0, max(0.0, length - 0.25 * voxel_size), voxel_size):
            cell = np.floor((origin + direction * distance - grid_min) / voxel_size).astype(int)
            if ((cell >= 0) & (cell < shape)).all() and occupancy[tuple(cell)]:
                remove[tuple(cell)] = True
    return remove


def connected_keep(mask, minimum):
    labels, count = ndimage.label(mask, ndimage.generate_binary_structure(3, 1))
    if count == 0:
        return mask
    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, count + 1))
    return np.isin(labels, np.flatnonzero(sizes >= minimum) + 1)


def reference_support(endpoint_indices, endpoint_refs, shape):
    """Count distinct reference cameras supporting each endpoint within 1 voxel."""
    count = np.zeros(shape, np.uint8)
    for ref in np.unique(endpoint_refs):
        support = np.zeros(shape, bool)
        cells = endpoint_indices[endpoint_refs == ref]
        support[tuple(cells.T)] = True
        count += ndimage.binary_dilation(
            support, ndimage.generate_binary_structure(3, 3)
        )
    return count


def run_scene(scene, candidate_root, out_root, ncc_reject, ncc_accept, ncc_margin,
              min_cluster, min_ref_support):
    source = EVAL / "srp_hull_mv2_v12_am1" / scene / "hull.npz"
    candidates_path = EVAL / candidate_root / scene / "candidates.npz"
    hull = np.load(source, allow_pickle=False)
    candidates = np.load(candidates_path, allow_pickle=False)
    occupancy = hull["occupancy"].astype(bool)
    grid_min = hull["grid_min"].astype(float)
    voxel_size = float(hull["voxel_size"])
    indices = candidates["surface_indices"].astype(np.int32)
    refs = candidates["reference_views"].astype(np.int16)
    layers = candidates["layers"].astype(np.int32)

    views, names = load_views(scene)
    if list(candidates["view_names"]) != names:
        raise ValueError("candidate and scoring view order differ")
    if len(views) != 12:
        raise ValueError(f"expected 12 views, got {len(views)}")

    normals = warp.voxel_normals(occupancy)[tuple(indices.T)]
    valid = (layers >= 0).all(2)
    rows, columns = np.nonzero(valid)
    candidate_indices = layers[rows, columns]
    points = grid_min + (candidate_indices + 0.5) * voxel_size
    t0 = time.time()
    scores_flat = ray_search._score_points(points, normals[rows], views)
    elapsed = time.time() - t0
    scores = np.full(valid.shape, np.nan, np.float32)
    scores[rows, columns] = scores_flat

    outer = scores[:, 0]
    best_layer = np.nanargmax(np.where(np.isfinite(scores), scores, -np.inf), axis=1)
    best = scores[np.arange(len(indices)), best_layer]
    move = (
        (best_layer > 0) & np.isfinite(outer) & np.isfinite(best)
        & (best >= ncc_accept) & (best >= outer + ncc_margin)
        & (outer < ncc_reject)
    )
    endpoint = layers[np.arange(len(indices)), best_layer]
    supported = np.zeros(len(indices), np.uint8)
    if move.any():
        support = reference_support(endpoint[move], refs[move], occupancy.shape)
        supported[move] = support[tuple(endpoint[move].T)]
        move &= supported >= min_ref_support
    selected_layer = np.where(move, best_layer, 0)
    centers = np.asarray([view["Cw"] for view in views])
    raw_remove = removal_rays(occupancy, grid_min, voxel_size, indices, refs, layers,
                              selected_layer, centers)
    remove = connected_keep(raw_remove, min_cluster)
    carved = occupancy & ~remove

    target = EVAL / out_root / scene
    target.mkdir(parents=True, exist_ok=True)
    meta = {
        "src": "srp_hull_mv2_v12_am1", "candidate_root": candidate_root,
        "method": "interval_ray_search_second_ncc_fgpatch",
        "views": names, "num_views": 12, "allow_miss": 1,
        "ncc_reject": ncc_reject, "ncc_accept": ncc_accept,
        "ncc_margin": ncc_margin, "min_cluster": min_cluster,
        "min_reference_support": min_ref_support,
        "coarse_stride_voxels": int(candidates["coarse_stride_voxels"]),
        "max_step_voxels": int(candidates["max_step_voxels"]),
        "ncc_seconds": elapsed,
        "surface_voxels": int(len(indices)), "candidate_points": int(len(points)),
        "scored_points": int(np.isfinite(scores_flat).sum()),
        "deeper_winners": int(move.sum()),
        "winners_with_two_references": int((supported >= 2).sum()),
        "raw_removed": int(raw_remove.sum()),
        "removed": int(remove.sum()),
    }
    np.savez_compressed(
        target / "hull.npz", occupancy=carved,
        surface=ray_search.surface(carved), observed=hull["observed"],
        grid_min=grid_min, voxel_size=np.float64(voxel_size),
        build_meta=json.dumps(meta),
    )
    print(f"[{scene}] scored {len(points)} candidates in {elapsed:.1f}s; "
          f"winners {int(move.sum())}; removed {int(remove.sum())}; "
          f"{int(occupancy.sum())}->{int(carved.sum())}")
    (target / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="+")
    parser.add_argument("--candidate-root", default="hull_interval_candidates_mv2_v12_am1")
    parser.add_argument("--out-root", default="srp_hull_mv2_v12_am1_intervalray_s3")
    parser.add_argument("--ncc-reject", type=float, default=0.10)
    parser.add_argument("--ncc-accept", type=float, default=0.25)
    parser.add_argument("--ncc-margin", type=float, default=0.10)
    parser.add_argument("--min-cluster", type=int, default=3)
    parser.add_argument("--min-ref-support", type=int, default=2,
                        help="distinct reference cameras agreeing within one voxel")
    args = parser.parse_args()
    for scene in args.scenes:
        run_scene(scene, args.candidate_root, args.out_root, args.ncc_reject,
                  args.ncc_accept, args.ncc_margin, args.min_cluster, args.min_ref_support)


if __name__ == "__main__":
    main()
