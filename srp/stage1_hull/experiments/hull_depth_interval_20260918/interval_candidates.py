#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""Prepare coarse occupied ray candidates within rendered hull intervals.

This does no photometric matching and never changes a hull.  It establishes
the finite candidate set that a later coarse-to-fine NCC stage may score.
Each current outer voxel selects its most front-facing reference camera.  Its
ray may move inward only while it stays in that reference pixel's rendered
hull interval and lands in an occupied voxel.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[4]
os.environ.setdefault("SAM_ROOT", str(REPO / "data" / "eval" / "mobilesamv2_fast"))
os.environ.setdefault("CAPTURES_ROOT", str(REPO / "data" / "captures_fast"))
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage1_hull"))
import run_scene as RS  # noqa: E402


EVAL = REPO / "data" / "eval"


def outer_surface(occupancy):
    return occupancy & ~ndimage.binary_erosion(
        occupancy, ndimage.generate_binary_structure(3, 1)
    )


def hull_normals(occupancy):
    smooth = ndimage.gaussian_filter(occupancy.astype(np.float32), sigma=1.0)
    gradient = np.gradient(smooth)
    normal = -np.stack(gradient, axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-9
    return normal


def choose_references(points, normals, centers):
    """Index of the most front-facing camera, or -1 if none faces the voxel."""
    facing = np.empty((len(centers), len(points)), np.float32)
    for vi, center in enumerate(centers):
        to_camera = center[None] - points
        to_camera /= np.linalg.norm(to_camera, axis=1, keepdims=True) + 1e-9
        facing[vi] = (normals * to_camera).sum(1)
    refs = facing.argmax(0).astype(np.int16)
    refs[facing.max(0) <= 0.1] = -1
    return refs


def build(scene, hull_root, interval_root, out_root, stride, max_step):
    hull_path = EVAL / hull_root / scene / "hull.npz"
    hull = np.load(hull_path, allow_pickle=False)
    occupancy = hull["occupancy"].astype(bool)
    grid_min = hull["grid_min"].astype(float)
    voxel_size = float(hull["voxel_size"])
    meta = json.loads(str(hull["build_meta"]))
    masks, Ks, extrinsics, names = RS.load_scene(scene, num_views=12)
    if names != meta.get("views"):
        raise ValueError("selected views differ from hull build metadata")

    interval_dir = EVAL / interval_root / scene
    interval_maps = []
    for name, foreground in zip(names, masks):
        path = interval_dir / f"{name}_interval.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        interval = np.load(path)
        if interval["foreground"].shape != foreground.shape:
            raise ValueError(f"{path}: foreground shape differs from current input")
        interval_maps.append(interval["z_exit"])

    surface = outer_surface(occupancy)
    indices = np.argwhere(surface)
    points = grid_min + (indices + 0.5) * voxel_size
    normals = hull_normals(occupancy)[tuple(indices.T)]
    centers = np.asarray([-Rwc.T @ t for Rwc, t in extrinsics])
    refs = choose_references(points, normals, centers)

    width = max_step // stride + 1
    layers = np.full((len(indices), width, 3), -1, np.int32)
    layers[:, 0] = indices
    allowed_steps = np.zeros(len(indices), np.int16)
    shape = np.asarray(occupancy.shape)
    for vi, (K, (Rwc, t), name) in enumerate(zip(Ks, extrinsics, names)):
        rows = np.flatnonzero(refs == vi)
        if len(rows) == 0:
            continue
        p = points[rows]
        X = p @ Rwc.T + t
        z = X[:, 2]
        u = np.rint(K[0, 0] * X[:, 0] / z + K[0, 2]).astype(int)
        v = np.rint(K[1, 1] * X[:, 1] / z + K[1, 2]).astype(int)
        z_exit = np.full(len(rows), -np.inf, np.float32)
        in_image = (u >= 0) & (u < interval_maps[vi].shape[1]) & (v >= 0) & (v < interval_maps[vi].shape[0])
        z_exit[in_image] = interval_maps[vi][v[in_image], u[in_image]]
        ray = p - centers[vi][None]
        ray /= np.linalg.norm(ray, axis=1, keepdims=True) + 1e-9
        dz_per_m = ray @ Rwc[2]
        depth_limit = np.floor((z_exit - z) / (np.maximum(dz_per_m, 1e-6) * voxel_size)).astype(int)
        depth_limit = np.clip(depth_limit, 0, max_step)
        allowed_steps[rows] = depth_limit.astype(np.int16)

        for column, step in enumerate(range(stride, max_step + 1, stride), 1):
            active = depth_limit >= step
            if not active.any():
                continue
            candidate = np.floor((p[active] + ray[active] * (step * voxel_size) - grid_min) / voxel_size).astype(np.int32)
            inside = ((candidate >= 0) & (candidate < shape)).all(1)
            candidate_rows = rows[active][inside]
            candidate = candidate[inside]
            occupied = occupancy[tuple(candidate.T)]
            layers[candidate_rows[occupied], column] = candidate[occupied]

    coarse_valid = (layers >= 0).all(2)
    output = EVAL / out_root / scene
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "candidates.npz", surface_indices=indices, reference_views=refs,
        allowed_steps=allowed_steps, layers=layers, grid_min=grid_min,
        voxel_size=np.float64(voxel_size), view_names=np.asarray(names),
        coarse_stride_voxels=np.int32(stride), max_step_voxels=np.int32(max_step),
    )
    count = coarse_valid.sum(1)
    stats = {
        "scene": scene, "source_hull": hull_root, "interval_root": interval_root,
        "coarse_stride_voxels": stride, "max_step_voxels": max_step,
        "surface_voxels": int(len(indices)),
        "referenceable_surface_voxels": int((refs >= 0).sum()),
        "surface_with_deeper_interval": int((allowed_steps >= stride).sum()),
        "coarse_candidates_total": int(coarse_valid.sum()),
        "coarse_candidates_per_surface_p50": float(np.median(count)),
        "coarse_candidates_per_surface_p95": float(np.percentile(count, 95)),
        "max_allowed_depth_mm_p50": float(np.median(allowed_steps) * voxel_size * 1000.0),
        "max_allowed_depth_mm_p95": float(np.percentile(allowed_steps, 95) * voxel_size * 1000.0),
    }
    (output / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="+")
    parser.add_argument("--hull-root", default="srp_hull_mv2_v12_am1")
    parser.add_argument("--interval-root", default="hull_depth_interval_mv2_v12_am1")
    parser.add_argument("--out-root", default="hull_interval_candidates_mv2_v12_am1")
    parser.add_argument("--stride", type=int, default=3, help="coarse candidate spacing in voxels")
    parser.add_argument("--max-step", type=int, default=60, help="maximum inward distance in voxels")
    args = parser.parse_args()
    if args.stride < 1 or args.max_step < args.stride:
        raise ValueError("need 1 <= stride <= max-step")
    for scene in args.scenes:
        build(scene, args.hull_root, args.interval_root, args.out_root, args.stride, args.max_step)


if __name__ == "__main__":
    main()
