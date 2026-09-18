#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""Offline-only test: can NCC window size rank mesh-surface candidates first?

No voxel is deleted.  Given the fixed multi-reference hull candidate rays, the
script scores the same candidates with several patch radii.  GT mesh samples
label candidates only for evaluation; they never affect NCC or depth choice.
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
import interval_candidates as IC  # noqa: E402
import eval_mesh as EM  # noqa: E402
from depth_consistency import forced_reference_scores  # noqa: E402
from run_interval_ray_search import load_views  # noqa: E402


EVAL = REPO / "data" / "eval"


def gt_surface_tree(scene, total_samples):
    """World-coordinate GT mesh surface samples, used only for offline labels."""
    np.random.seed(0)
    clouds = []
    objects = EM.gt_objects(scene)
    per_object = max(1, total_samples // max(1, len(objects)))
    for obj in objects:
        mesh = EM.load_mesh(obj["name"])
        if mesh is None:
            continue
        aa = obj.get("rotation_axis_angle", [0, 1, 0, 0])
        R = EM.aa_to_mat(aa[:3], aa[3])
        vertices = (mesh.vertices - EM.ycb_center(obj["name"])) @ R.T + np.asarray(obj["position_m"], float)
        world_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
        points, _ = trimesh.sample.sample_surface(world_mesh, per_object)
        clouds.append(points)
    if not clouds:
        raise RuntimeError(f"{scene}: no GT mesh surface samples")
    return cKDTree(np.vstack(clouds))


def summarize_rows(scores, truth):
    """Return ranking statistics for rows containing true and ghost candidates."""
    finite = np.isfinite(scores)
    true_scores = np.where(finite & truth, scores, -np.inf)
    ghost_scores = np.where(finite & ~truth, scores, -np.inf)
    has_true = np.isfinite(true_scores).any(1)
    has_ghost = np.isfinite(ghost_scores).any(1)
    both = has_true & has_ghost
    chosen = np.argmax(np.where(finite, scores, -np.inf), axis=1)
    chosen_true = truth[np.arange(len(truth)), chosen] & finite[np.arange(len(truth)), chosen]
    true_best = true_scores.max(1)
    ghost_best = ghost_scores.max(1)
    return {
        "rows": int(len(scores)),
        "rows_with_score": int(finite.any(1).sum()),
        "rows_with_true_and_ghost": int(both.sum()),
        "winner_is_surface_rate": float(chosen_true[finite.any(1)].mean()) if finite.any(1).any() else None,
        "surface_beats_ghost_rate": float((true_best[both] > ghost_best[both]).mean()) if both.any() else None,
        "surface_beats_ghost_margin_010_rate": float((true_best[both] >= ghost_best[both] + 0.10).mean()) if both.any() else None,
        "true_best_ncc_median": float(np.median(true_best[has_true])) if has_true.any() else None,
        "ghost_best_ncc_median": float(np.median(ghost_best[has_ghost])) if has_ghost.any() else None,
    }


def process(scene, candidate_root, out_dir, radii, total_samples, surface_tol_mm, batch_size, device):
    candidate_path = EVAL / candidate_root / scene / "multi_ref_candidates.npz"
    source_path = EVAL / "srp_hull_mv2_v12_am1" / scene / "hull.npz"
    candidates = np.load(candidate_path, allow_pickle=False)
    hull = np.load(source_path, allow_pickle=False)
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
    tree = gt_surface_tree(scene, total_samples)
    distance, _ = tree.query(points)
    truth_flat = distance <= surface_tol_mm / 1000.0
    truth = np.zeros(valid.shape, bool)
    truth[rows, columns] = truth_flat

    out_scene = out_dir / scene
    out_scene.mkdir(parents=True, exist_ok=True)
    results = []
    scores_by_radius = {}
    for radius in radii:
        t0 = time.time()
        flat_scores = forced_reference_scores(points, normals, refs[rows], views,
                                              patch_radius=radius, batch_size=batch_size, device=device)
        scores = np.full(valid.shape, np.nan, np.float32)
        scores[rows, columns] = flat_scores
        scores_by_radius[radius] = scores
        summary = summarize_rows(scores, truth)
        summary.update({
            "scene": scene, "patch_radius": radius, "patch_width": 2 * radius + 1,
            "candidate_points": int(len(points)), "surface_candidates": int(truth_flat.sum()),
            "surface_tolerance_mm": surface_tol_mm, "ncc_seconds": time.time() - t0,
        })
        results.append(summary)
        print(summary, flush=True)
        fields = list(results[0])
        with (out_scene / "window_discriminability.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)

    np.savez_compressed(out_scene / "candidate_truth.npz", rows=rows, columns=columns,
                        distance_to_gt_surface_m=distance, is_surface=truth_flat)
    if 3 in scores_by_radius and 5 in scores_by_radius:
        score7, score11 = scores_by_radius[3], scores_by_radius[5]
        valid7, valid11 = np.isfinite(score7).any(1), np.isfinite(score11).any(1)
        choice7 = np.argmax(np.where(np.isfinite(score7), score7, -np.inf), axis=1)
        choice11 = np.argmax(np.where(np.isfinite(score11), score11, -np.inf), axis=1)
        both = valid7 & valid11
        same = both & np.all(layers[np.arange(len(layers)), choice7] ==
                              layers[np.arange(len(layers)), choice11], axis=1)
        chosen_true = truth[np.arange(len(truth)), choice11]
        agreement = {
            "scene": scene,
            "scales": "7x7_and_11x11",
            "rows_with_both_scores": int(both.sum()),
            "same_depth_rows": int(same.sum()),
            "same_depth_rate": float(same[both].mean()) if both.any() else None,
            "same_depth_winner_is_surface_rate": float(chosen_true[same].mean()) if same.any() else None,
            "same_depth_winner_is_ghost_rate": float((~chosen_true[same]).mean()) if same.any() else None,
        }
        (out_scene / "multiscale_agreement.json").write_text(json.dumps(agreement, indent=2), encoding="utf-8")
        print(agreement, flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="+")
    parser.add_argument("--candidate-root", default="hull_multi_ref_candidates_mv2_v12_am1")
    parser.add_argument("--out-root", default="photo_support_window_mv2_v12_am1")
    parser.add_argument("--radii", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--gt-samples", type=int, default=400000)
    parser.add_argument("--surface-tol-mm", type=float, default=12.0)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="smaller batches keep large NCC windows within memory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="device for projection, warp, and NCC scoring")
    args = parser.parse_args()
    out_dir = EVAL / args.out_root
    for scene in args.scenes:
        process(scene, args.candidate_root, out_dir, args.radii, args.gt_samples,
                args.surface_tol_mm, args.batch_size, args.device)


if __name__ == "__main__":
    main()
