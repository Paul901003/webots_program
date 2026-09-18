#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""Offline-only anchor free-space diagnostic.

A hull candidate becomes a surface anchor only when 7x7 and 11x11 NCC choose
exactly the same occupied voxel on one reference ray.  Each anchor says that
occupied voxels strictly between its reference camera and itself should be
free space.  Votes are counted per distinct camera.  This script only writes
anchors, vote masks, and offline GT diagnostics; it never writes a new hull.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[4]
os.environ.setdefault("SAM_ROOT", str(REPO / "data" / "eval" / "mobilesamv2_fast"))
os.environ.setdefault("CAPTURES_ROOT", str(REPO / "data" / "captures_fast"))
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage1_hull"))
sys.path.insert(0, str(REPO / "srp" / "stage1_hull" / "experiments" / "hull_depth_interval_20260918"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import eval_mesh as EM  # noqa: E402
import interval_candidates as IC  # noqa: E402
from depth_consistency import forced_reference_scores  # noqa: E402
from run_interval_ray_search import load_views  # noqa: E402


EVAL = REPO / "data" / "eval"


def gt_surface_tree(scene, total_samples):
    """Sample the GT mesh surface only for offline anchor evaluation."""
    np.random.seed(0)
    clouds = []
    objects = EM.gt_objects(scene)
    per_object = max(1, total_samples // max(1, len(objects)))
    for obj in objects:
        mesh = EM.load_mesh(obj["name"])
        if mesh is None:
            continue
        aa = obj.get("rotation_axis_angle", [0, 1, 0, 0])
        rotation = EM.aa_to_mat(aa[:3], aa[3])
        vertices = (mesh.vertices - EM.ycb_center(obj["name"])) @ rotation.T
        vertices += np.asarray(obj["position_m"], float)
        world_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
        points, _ = trimesh.sample.sample_surface(world_mesh, per_object)
        clouds.append(points)
    if not clouds:
        raise RuntimeError(f"{scene}: no GT mesh surface samples")
    return cKDTree(np.vstack(clouds))


def cells_before_anchor(camera, anchor_index, grid_min, voxel_size, shape):
    """Exact grid traversal from camera to anchor center, excluding anchor cell."""
    start = (np.asarray(camera, float) - grid_min) / voxel_size
    end = np.asarray(anchor_index, float) + 0.5
    direction = end - start
    lo, hi = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if start[axis] < 0 or start[axis] >= shape[axis]:
                return np.empty((0, 3), np.int32)
            continue
        first = (0.0 - start[axis]) / direction[axis]
        last = (float(shape[axis]) - start[axis]) / direction[axis]
        lo = max(lo, min(first, last))
        hi = min(hi, max(first, last))
    if lo > hi or lo >= 1.0:
        return np.empty((0, 3), np.int32)

    # Move a tiny amount inside the grid before flooring an entry boundary.
    point = start + direction * min(1.0, lo + 1e-9)
    cell = np.floor(point).astype(np.int32)
    target = np.asarray(anchor_index, np.int32)
    step = np.sign(direction).astype(np.int32)
    delta = np.full(3, np.inf)
    nonzero = direction != 0
    delta[nonzero] = 1.0 / np.abs(direction[nonzero])
    next_boundary = cell.astype(float) + (step > 0)
    t_max = np.full(3, np.inf)
    t_max[nonzero] = (next_boundary[nonzero] - start[nonzero]) / direction[nonzero]

    traversed = []
    while ((cell >= 0) & (cell < shape)).all() and not np.array_equal(cell, target):
        traversed.append(cell.copy())
        next_t = t_max.min()
        if next_t > 1.0 + 1e-9:
            break
        axes = np.isclose(t_max, next_t, atol=1e-12, rtol=0.0)
        cell[axes] += step[axes]
        t_max[axes] += delta[axes]
    return np.asarray(traversed, np.int32) if traversed else np.empty((0, 3), np.int32)


def build_votes(occupancy, anchor_indices, anchor_refs, centers, grid_min, voxel_size):
    """Return per-camera free-space votes and anchors protected from deletion."""
    shape = np.asarray(occupancy.shape)
    votes = np.zeros(occupancy.shape, np.uint8)
    anchors = np.zeros_like(occupancy, bool)
    anchors[tuple(anchor_indices.T)] = True
    for ref in np.unique(anchor_refs):
        support = np.zeros_like(occupancy, bool)
        for index in anchor_indices[anchor_refs == ref]:
            cells = cells_before_anchor(centers[ref], index, grid_min, voxel_size, shape)
            if len(cells):
                support[tuple(cells.T)] = True
        votes += support
    return votes, anchors


def combined_gt(scene, grid_min, voxel_size, shape):
    parts = EM.solid_mesh_occ(scene, grid_min, voxel_size, shape)
    gt = np.zeros(shape, bool)
    for part in parts.values():
        gt |= part
    if not gt.any():
        raise RuntimeError(f"{scene}: no GT solid mesh occupancy")
    return gt


def proposal_metrics(occupancy, gt, votes, anchors, minimum_votes):
    raw = occupancy & (votes >= minimum_votes)
    remove = raw & ~anchors
    real = occupancy & gt
    ghost = occupancy & ~gt
    removed_real = int((remove & gt).sum())
    removed_ghost = int((remove & ~gt).sum())
    remaining = occupancy & ~remove
    remaining_real = int((remaining & gt).sum())
    return {
        "minimum_views": minimum_votes,
        "raw_proposal_voxels": int(raw.sum()),
        "protected_anchor_conflicts": int((raw & anchors).sum()),
        "proposed_remove_voxels": int(remove.sum()),
        "proposed_remove_ghost_voxels": removed_ghost,
        "proposed_remove_real_voxels": removed_real,
        "removed_is_ghost_precision": removed_ghost / int(remove.sum()) if remove.any() else None,
        "ghost_removed_recall": removed_ghost / int(ghost.sum()) if ghost.any() else None,
        "real_voxels_removed_fraction": removed_real / int(real.sum()) if real.any() else None,
        "prospective_coverage": remaining_real / int(gt.sum()),
        "prospective_ghost_fraction": 1 - remaining_real / int(remaining.sum()) if remaining.any() else None,
    }


def process(scene, candidate_root, out_root, batch_size, device, surface_tol_mm, gt_samples, vote_levels):
    candidate_path = EVAL / candidate_root / scene / "multi_ref_candidates.npz"
    hull_path = EVAL / "srp_hull_mv2_v12_am1" / scene / "hull.npz"
    candidates = np.load(candidate_path, allow_pickle=False)
    hull = np.load(hull_path, allow_pickle=False)
    occupancy = hull["occupancy"].astype(bool)
    grid_min = hull["grid_min"].astype(float)
    voxel_size = float(hull["voxel_size"])
    surface_indices = candidates["surface_indices"].astype(np.int32)
    source_rows = candidates["source_surface_rows"].astype(np.int32)
    refs = candidates["reference_views"].astype(np.int16)
    layers = candidates["layers"].astype(np.int32)
    views, names = load_views(scene)
    if list(candidates["view_names"]) != names:
        raise ValueError("candidate and scoring view order differ")

    valid = (layers >= 0).all(2)
    rows, columns = np.nonzero(valid)
    point_indices = layers[rows, columns]
    points = grid_min + (point_indices + 0.5) * voxel_size
    normal_grid = IC.hull_normals(occupancy)
    normals = normal_grid[tuple(surface_indices[source_rows[rows]].T)]
    scores = {}
    for radius in (3, 5):
        started = time.time()
        flat = forced_reference_scores(points, normals, refs[rows], views, patch_radius=radius,
                                      batch_size=batch_size, device=device)
        score = np.full(valid.shape, np.nan, np.float32)
        score[rows, columns] = flat
        scores[radius] = score
        print(f"{scene}: {2 * radius + 1}x{2 * radius + 1} NCC in {time.time() - started:.2f}s", flush=True)

    score7, score11 = scores[3], scores[5]
    has7 = np.isfinite(score7).any(1)
    has11 = np.isfinite(score11).any(1)
    choice7 = np.argmax(np.where(np.isfinite(score7), score7, -np.inf), axis=1)
    choice11 = np.argmax(np.where(np.isfinite(score11), score11, -np.inf), axis=1)
    same = has7 & has11 & np.all(layers[np.arange(len(layers)), choice7] ==
                                  layers[np.arange(len(layers)), choice11], axis=1)
    anchor_rows = np.flatnonzero(same)
    anchor_indices = layers[anchor_rows, choice11[anchor_rows]]
    anchor_refs = refs[anchor_rows]
    unique_anchor_indices = np.unique(anchor_indices, axis=0)
    anchor_points = grid_min + (unique_anchor_indices + 0.5) * voxel_size
    surface_distance, _ = gt_surface_tree(scene, gt_samples).query(anchor_points)
    anchor_surface_like = surface_distance <= surface_tol_mm / 1000.0

    centers = np.asarray([view["Cw"] for view in views])
    votes, anchors = build_votes(occupancy, anchor_indices, anchor_refs, centers, grid_min, voxel_size)
    gt = combined_gt(scene, grid_min, voxel_size, occupancy.shape)
    sweep = [proposal_metrics(occupancy, gt, votes, anchors, level) for level in vote_levels]
    for result in sweep:
        result["scene"] = scene
    out_scene = EVAL / out_root / scene
    out_scene.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_scene / "anchor_free_space.npz", anchors=anchors, free_space_votes=votes,
                        anchor_rows=anchor_rows, anchor_indices=anchor_indices, anchor_refs=anchor_refs,
                        grid_min=grid_min, voxel_size=np.float64(voxel_size))
    with (out_scene / "vote_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)
    summary = {
        "scene": scene,
        "device": device,
        "source_hull": "srp_hull_mv2_v12_am1",
        "anchor_rule": "7x7 and 11x11 select the exact same candidate voxel",
        "anchor_rows": int(len(anchor_rows)),
        "unique_anchor_voxels": int(len(unique_anchor_indices)),
        "anchor_surface_like_rate": float(anchor_surface_like.mean()) if len(anchor_surface_like) else None,
        "anchor_surface_tolerance_mm": surface_tol_mm,
        "votes_max": int(votes.max()),
        "baseline_voxels": int(occupancy.sum()),
        "baseline_ghost_fraction": float((occupancy & ~gt).sum() / occupancy.sum()),
    }
    (out_scene / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    for result in sweep:
        print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="+")
    parser.add_argument("--candidate-root", default="hull_multi_ref_candidates_mv2_v12_am1")
    parser.add_argument("--out-root", default="anchor_free_space_mv2_v12_am1")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--surface-tol-mm", type=float, default=12.0)
    parser.add_argument("--gt-samples", type=int, default=400000)
    parser.add_argument("--vote-levels", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()
    if min(args.vote_levels) < 1:
        raise ValueError("vote levels must be positive")
    for scene in args.scenes:
        process(scene, args.candidate_root, args.out_root, args.batch_size, args.device,
                args.surface_tol_mm, args.gt_samples, sorted(set(args.vote_levels)))


if __name__ == "__main__":
    main()
