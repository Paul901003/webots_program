#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""Offline visible-outer-voxel diagnostic.

A reference ray may remove only its current z-buffer-frontmost hull voxel V0,
not every voxel between camera and a deeper NCC anchor. V0 is proposed only if
7x7 and 11x11 NCC agree on the same deeper candidate. This script never writes
a new hull; GT is used only after proposal construction.
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

REPO = Path(__file__).resolve().parents[4]
os.environ.setdefault("SAM_ROOT", str(REPO / "data" / "eval" / "mobilesamv2_fast"))
os.environ.setdefault("CAPTURES_ROOT", str(REPO / "data" / "captures_fast"))
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage1_hull"))
sys.path.insert(0, str(REPO / "srp" / "stage1_hull" / "experiments" / "hull_depth_interval_20260918"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import interval_candidates as IC  # noqa: E402
from depth_consistency import forced_reference_scores  # noqa: E402
from run_interval_ray_search import load_views  # noqa: E402
from anchor_free_space import combined_gt, gt_surface_tree, proposal_metrics  # noqa: E402


EVAL = REPO / "data" / "eval"


def vote_visible_outer(shape, outer_indices, outer_refs):
    """Count at most one deletion vote per camera for every visible V0 voxel."""
    votes = np.zeros(shape, np.uint8)
    for ref in np.unique(outer_refs):
        cells = np.unique(outer_indices[outer_refs == ref], axis=0)
        votes[tuple(cells.T)] += 1
    return votes


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
    points = grid_min + (layers[rows, columns] + 0.5) * voxel_size
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
    agreed = has7 & has11 & np.all(layers[np.arange(len(layers)), choice7] ==
                                   layers[np.arange(len(layers)), choice11], axis=1)
    anchor_rows = np.flatnonzero(agreed)
    anchor_indices = layers[anchor_rows, choice11[anchor_rows]]
    anchors = np.zeros_like(occupancy, bool)
    anchors[tuple(anchor_indices.T)] = True

    # Only a deeper agreed candidate means the currently visible V0 is challenged.
    deeper_rows = np.flatnonzero(agreed & (choice11 > 0))
    outer_indices = surface_indices[source_rows[deeper_rows]]
    outer_refs = refs[deeper_rows]
    votes = vote_visible_outer(occupancy.shape, outer_indices, outer_refs)
    gt = combined_gt(scene, grid_min, voxel_size, occupancy.shape)
    sweep = [proposal_metrics(occupancy, gt, votes, anchors, level) for level in vote_levels]
    for result in sweep:
        result["scene"] = scene

    unique_anchors = np.unique(anchor_indices, axis=0)
    anchor_points = grid_min + (unique_anchors + 0.5) * voxel_size
    distance, _ = gt_surface_tree(scene, gt_samples).query(anchor_points)
    out_scene = EVAL / out_root / scene
    out_scene.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_scene / "visible_outer_votes.npz", visible_outer_votes=votes,
                        protected_anchors=anchors, deeper_rows=deeper_rows,
                        visible_outer_indices=outer_indices, visible_outer_refs=outer_refs,
                        grid_min=grid_min, voxel_size=np.float64(voxel_size))
    with (out_scene / "vote_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)
    summary = {
        "scene": scene,
        "device": device,
        "source_hull": "srp_hull_mv2_v12_am1",
        "rule": "only visible V0, when 7x7 and 11x11 agree on the same deeper candidate",
        "agreed_anchor_rows": int(len(anchor_rows)),
        "deeper_anchor_rows": int(len(deeper_rows)),
        "unique_visible_outer_voxels": int(len(np.unique(outer_indices, axis=0))),
        "anchor_surface_like_rate": float((distance <= surface_tol_mm / 1000.0).mean()),
        "anchor_surface_tolerance_mm": surface_tol_mm,
        "votes_max": int(votes.max()),
    }
    (out_scene / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    for result in sweep:
        print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="+")
    parser.add_argument("--candidate-root", default="hull_multi_ref_candidates_mv2_v12_am1")
    parser.add_argument("--out-root", default="visible_outer_votes_mv2_v12_am1")
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
