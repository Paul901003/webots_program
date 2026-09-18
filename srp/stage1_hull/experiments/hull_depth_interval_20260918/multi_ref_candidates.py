#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""Prepare independent coarse depth candidates for every usable reference view.

Unlike interval_candidates.py, an outer voxel is not assigned to only one
camera.  Every front-facing view whose projected pixel is foreground and has a
hull interval receives its own ray candidate row.  These rows are the sparse
per-view depth domains needed for later reprojection-consistency checks.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_scene as RS  # noqa: E402
import interval_candidates as IC  # noqa: E402


EVAL = REPO / "data" / "eval"


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

    intervals = []
    for name, foreground in zip(names, masks):
        path = EVAL / interval_root / scene / f"{name}_interval.npz"
        data = np.load(path)
        if data["foreground"].shape != foreground.shape or not np.array_equal(data["foreground"], foreground):
            raise ValueError(f"{path}: foreground differs from current Stage-1 input")
        intervals.append(data["z_exit"])

    surface = IC.outer_surface(occupancy)
    surface_indices = np.argwhere(surface).astype(np.int32)
    points = grid_min + (surface_indices + 0.5) * voxel_size
    normals = IC.hull_normals(occupancy)[tuple(surface_indices.T)]
    centers = np.asarray([-Rwc.T @ t for Rwc, t in extrinsics])
    shape = np.asarray(occupancy.shape)
    width = max_step // stride + 1
    all_surface_rows, all_refs, all_limits, all_layers = [], [], [], []

    for ref, (K, (Rwc, t), foreground, z_exit) in enumerate(zip(Ks, extrinsics, masks, intervals)):
        X = points @ Rwc.T + t
        z = X[:, 2]
        u = np.rint(K[0, 0] * X[:, 0] / z + K[0, 2]).astype(int)
        v = np.rint(K[1, 1] * X[:, 1] / z + K[1, 2]).astype(int)
        to_camera = centers[ref][None] - points
        to_camera /= np.linalg.norm(to_camera, axis=1, keepdims=True) + 1e-9
        facing = (normals * to_camera).sum(1)
        in_image = (z > 1e-6) & (u >= 0) & (u < foreground.shape[1]) & (v >= 0) & (v < foreground.shape[0])
        usable = np.zeros(len(points), bool)
        usable[in_image] = foreground[v[in_image], u[in_image]] & np.isfinite(z_exit[v[in_image], u[in_image]])
        eligible = np.flatnonzero((facing > 0.1) & usable)
        # A reference depth map represents the first current hull surface on
        # each pixel ray. Back shell cells at the same pixel cannot be an
        # independent reference-depth hypothesis.
        pixel = v[eligible] * foreground.shape[1] + u[eligible]
        order = np.lexsort((z[eligible], pixel))
        ordered = eligible[order]
        ordered_pixel = pixel[order]
        rows = ordered[np.r_[True, ordered_pixel[1:] != ordered_pixel[:-1]]]
        if len(rows) == 0:
            continue
        p = points[rows]
        ray = p - centers[ref][None]
        ray /= np.linalg.norm(ray, axis=1, keepdims=True) + 1e-9
        dz_per_m = ray @ Rwc[2]
        limit_z = z_exit[v[rows], u[rows]]
        limit = np.floor((limit_z - z[rows]) / (np.maximum(dz_per_m, 1e-6) * voxel_size)).astype(int)
        limit = np.clip(limit, 0, max_step).astype(np.int16)
        layers = np.full((len(rows), width, 3), -1, np.int32)
        layers[:, 0] = surface_indices[rows]
        for column, step in enumerate(range(stride, max_step + 1, stride), 1):
            active = limit >= step
            if not active.any():
                continue
            candidate = np.floor((p[active] + ray[active] * (step * voxel_size) - grid_min) / voxel_size).astype(np.int32)
            inside = ((candidate >= 0) & (candidate < shape)).all(1)
            target_rows = np.flatnonzero(active)[inside]
            candidate = candidate[inside]
            keep = occupancy[tuple(candidate.T)]
            layers[target_rows[keep], column] = candidate[keep]
        all_surface_rows.append(rows.astype(np.int32))
        all_refs.append(np.full(len(rows), ref, np.int16))
        all_limits.append(limit)
        all_layers.append(layers)

    if not all_layers:
        raise RuntimeError(f"{scene}: no usable reference rays")
    source_rows = np.concatenate(all_surface_rows)
    refs = np.concatenate(all_refs)
    limits = np.concatenate(all_limits)
    layers = np.concatenate(all_layers)
    valid = (layers >= 0).all(2)
    ref_count = np.bincount(source_rows, minlength=len(surface_indices))
    output = EVAL / out_root / scene
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "multi_ref_candidates.npz", surface_indices=surface_indices,
        source_surface_rows=source_rows, reference_views=refs, allowed_steps=limits,
        layers=layers, grid_min=grid_min, voxel_size=np.float64(voxel_size),
        view_names=np.asarray(names), coarse_stride_voxels=np.int32(stride),
        max_step_voxels=np.int32(max_step),
    )
    result = {
        "scene": scene, "source_hull": hull_root, "interval_root": interval_root,
        "surface_voxels": int(len(surface_indices)), "reference_ray_rows": int(len(layers)),
        "reference_rows_per_surface_p50": float(np.median(ref_count)),
        "reference_rows_per_surface_p95": float(np.percentile(ref_count, 95)),
        "coarse_candidates_total": int(valid.sum()),
        "coarse_candidates_per_reference_p50": float(np.median(valid.sum(1))),
        "coarse_candidates_per_reference_p95": float(np.percentile(valid.sum(1), 95)),
        "coarse_stride_voxels": stride, "max_step_voxels": max_step,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="+")
    parser.add_argument("--hull-root", default="srp_hull_mv2_v12_am1")
    parser.add_argument("--interval-root", default="hull_depth_interval_mv2_v12_am1")
    parser.add_argument("--out-root", default="hull_multi_ref_candidates_mv2_v12_am1")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-step", type=int, default=60)
    args = parser.parse_args()
    if args.stride < 1 or args.max_step < args.stride:
        raise ValueError("need 1 <= stride <= max-step")
    for scene in args.scenes:
        build(scene, args.hull_root, args.interval_root, args.out_root, args.stride, args.max_step)


if __name__ == "__main__":
    main()
