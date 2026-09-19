#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""Iteratively remove only the current outer hull voxel after sparse MVS support.

This is an isolated experiment.  It deliberately differs from the rejected
``depth_consistency.py`` carve: an accepted ray removes *one* current outer
voxel, then the next iteration builds a new set of rays from the changed hull.
GT is never read.  Per-iteration candidate arrays and timings are retained for
ray-by-ray auditing.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
os.environ.setdefault("SAM_ROOT", str(REPO / "data" / "eval" / "mobilesamv2_fast"))
os.environ.setdefault("CAPTURES_ROOT", str(REPO / "data" / "captures_fast"))
sys.path.insert(0, str(REPO / "srp" / "stage1_hull"))
sys.path.insert(0, str(REPO / "srp" / "stage1_hull" / "experiments" / "hull_depth_interval_20260918"))

import depth_consistency as DC  # noqa: E402
import multi_ref_candidates as MC  # noqa: E402
from run_interval_ray_search import load_views  # noqa: E402


EVAL = REPO / "data" / "eval"


def auto_max_step(shape: tuple[int, ...]) -> int:
    """Largest possible grid traversal in voxels; never clips a hull interval."""
    return int(np.ceil(np.linalg.norm(np.asarray(shape, float))))


def write_input_hull(path: Path, occ: np.ndarray, source: np.lib.npyio.NpzFile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = {
        "occupancy": occ,
        "grid_min": source["grid_min"],
        "voxel_size": source["voxel_size"],
        "build_meta": source["build_meta"],
    }
    if "observed" in source:
        fields["observed"] = source["observed"]
    np.savez_compressed(path, **fields)


def select_and_support(candidates, occ, grid_min, voxel_size, views, patch_radius,
                       min_sources, ncc_reject, ncc_accept, ncc_margin, device):
    """Return one independently supported deeper point per eligible reference ray."""
    surface_indices = candidates["surface_indices"].astype(np.int32)
    source_rows = candidates["source_surface_rows"].astype(np.int32)
    refs = candidates["reference_views"].astype(np.int16)
    layers = candidates["layers"].astype(np.int32)
    normals = MC.IC.hull_normals(occ)[tuple(surface_indices[source_rows].T)]
    valid = (layers >= 0).all(2)
    rows, columns = np.nonzero(valid)
    candidate_indices = layers[rows, columns]
    points = grid_min + (candidate_indices + 0.5) * voxel_size

    started = time.perf_counter()
    flat_scores = DC.forced_reference_scores(
        points, normals[rows], refs[rows], views, patch_radius=patch_radius,
        device=device,
    )
    score_seconds = time.perf_counter() - started
    scores = np.full(valid.shape, np.nan, np.float32)
    scores[rows, columns] = flat_scores
    best_layer = np.argmax(np.where(np.isfinite(scores), scores, -np.inf), axis=1)
    best_scores = scores[np.arange(len(layers)), best_layer]
    outer_scores = scores[:, 0]
    best_indices = layers[np.arange(len(layers)), best_layer]
    selected_points = grid_min + (best_indices + 0.5) * voxel_size
    selected_points[~np.isfinite(best_scores)] = np.nan

    started = time.perf_counter()
    pixels = DC.original_pixels(surface_indices, source_rows, refs, grid_min, voxel_size, views)
    maps = DC.build_sparse_maps(refs, pixels, selected_points, best_scores, len(views))
    map_seconds = time.perf_counter() - started

    # These are the pre-existing sparse-MVS evidence checks, made explicit in
    # the output instead of being hidden inside a full-ray deletion operation.
    proposed = (
        (best_layer > 0) & np.isfinite(outer_scores) & np.isfinite(best_scores)
        & (outer_scores < ncc_reject) & (best_scores >= ncc_accept)
        & (best_scores >= outer_scores + ncc_margin)
    )
    started = time.perf_counter()
    supports = np.zeros(len(layers), np.int8)
    for row in np.flatnonzero(proposed):
        supports[row] = DC.source_support(
            selected_points[row], refs[row], maps, views, min_sources * voxel_size,
        )
    accepted = proposed & (supports >= min_sources)
    support_seconds = time.perf_counter() - started

    return {
        "surface_indices": surface_indices,
        "source_rows": source_rows,
        "refs": refs,
        "layers": layers,
        "pixels": pixels,
        "scores": scores,
        "outer_scores": outer_scores,
        "best_layer": best_layer,
        "best_indices": best_indices,
        "best_scores": best_scores,
        "supports": supports,
        "accepted": accepted,
        "candidate_points": int(len(points)),
        "scored_points": int(np.isfinite(flat_scores).sum()),
        "depth_map_entries": int(sum(len(m) for m in maps)),
        "score_seconds": score_seconds,
        "map_seconds": map_seconds,
        "support_seconds": support_seconds,
    }


def run(scene, input_root, interval_root, out_root, max_iterations, patch_radius,
        min_sources, ncc_reject, ncc_accept, ncc_margin, device):
    source_path = EVAL / input_root / scene / "hull.npz"
    source = np.load(source_path, allow_pickle=False)
    occ = source["occupancy"].astype(bool).copy()
    grid_min = source["grid_min"].astype(float)
    voxel_size = float(source["voxel_size"])
    max_step = auto_max_step(occ.shape)
    views, names = load_views(scene)

    target = EVAL / out_root / scene
    iteration_dir = target / "iterations"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    work_hull_root = f"{out_root}__iter_work_hull"
    work_candidate_root = f"{out_root}__iter_work_candidates"
    rows = []
    total_started = time.perf_counter()

    try:
        for iteration in range(1, max_iterations + 1):
            started = time.perf_counter()
            work_hull = EVAL / work_hull_root / scene / "hull.npz"
            write_input_hull(work_hull, occ, source)
            input_write_seconds = time.perf_counter() - started

            started = time.perf_counter()
            MC.build(scene, work_hull_root, interval_root, work_candidate_root,
                     stride=1, max_step=max_step)
            candidate_seconds = time.perf_counter() - started
            candidate_path = EVAL / work_candidate_root / scene / "multi_ref_candidates.npz"
            candidates = np.load(candidate_path, allow_pickle=False)

            result = select_and_support(
                candidates, occ, grid_min, voxel_size, views, patch_radius,
                min_sources, ncc_reject, ncc_accept, ncc_margin, device,
            )
            accepted_surface = result["surface_indices"][result["source_rows"][result["accepted"]]]
            remove_indices = np.unique(accepted_surface, axis=0) if len(accepted_surface) else np.empty((0, 3), np.int32)
            before = int(occ.sum())
            started = time.perf_counter()
            if len(remove_indices):
                occ[tuple(remove_indices.T)] = False
            delete_seconds = time.perf_counter() - started

            idir = iteration_dir / f"iteration_{iteration:02d}"
            idir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate_path, idir / "candidates.npz")
            np.savez_compressed(
                idir / "decisions.npz",
                reference_surface_indices=result["surface_indices"],
                source_surface_rows=result["source_rows"], reference_views=result["refs"],
                pixels=result["pixels"], candidate_indices=result["layers"],
                ncc_scores=result["scores"], outer_ncc=result["outer_scores"],
                selected_layer=result["best_layer"], selected_indices=result["best_indices"],
                selected_ncc=result["best_scores"], source_support=result["supports"],
                accepted=result["accepted"], removed_indices=remove_indices,
            )
            row = {
                "iteration": iteration,
                "occupied_before": before,
                "surface_voxels": int(len(result["surface_indices"])),
                "reference_rays": int(len(result["refs"])),
                "candidate_points": result["candidate_points"],
                "scored_points": result["scored_points"],
                "depth_map_entries": result["depth_map_entries"],
                "deeper_candidates": int((result["best_layer"] > 0).sum()),
                "accepted_rays": int(result["accepted"].sum()),
                "removed_outer_voxels": int(len(remove_indices)),
                "occupied_after": int(occ.sum()),
                "input_write_seconds": input_write_seconds,
                "candidate_seconds": candidate_seconds,
                "score_seconds": result["score_seconds"],
                "map_seconds": result["map_seconds"],
                "support_seconds": result["support_seconds"],
                "delete_seconds": delete_seconds,
                "iteration_seconds": time.perf_counter() - started + input_write_seconds + candidate_seconds
                                     + result["score_seconds"] + result["map_seconds"] + result["support_seconds"],
            }
            rows.append(row)
            (idir / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)
            if len(remove_indices) == 0:
                break
    finally:
        shutil.rmtree(EVAL / work_hull_root, ignore_errors=True)
        shutil.rmtree(EVAL / work_candidate_root, ignore_errors=True)

    meta = {
        "method": "iterative_outer_voxel_sparse_mvs",
        "source_hull": input_root,
        "views": names,
        "voxel_size_m": voxel_size,
        "candidate_stride_voxels": 1,
        "max_candidate_step_voxels": max_step,
        "patch_radius_pixels": patch_radius,
        "patch_width_pixels": 2 * patch_radius + 1,
        "minimum_other_view_support": min_sources,
        "source_match_tolerance_m": min_sources * voxel_size,
        "ncc_reject": ncc_reject,
        "ncc_accept": ncc_accept,
        "ncc_margin": ncc_margin,
        "delete_rule": "remove only unique current outer voxels of accepted rays",
        "uses_gt": False,
        "total_seconds": time.perf_counter() - total_started,
        "iterations": rows,
    }
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target / "hull.npz", occupancy=occ, grid_min=grid_min,
                        voxel_size=np.float64(voxel_size), observed=source["observed"],
                        build_meta=json.dumps(meta))
    np.savez_compressed(target / "instances.npz", occupancy=occ, labels=np.zeros(occ.shape, np.int16),
                        grid_min=grid_min, voxel_size=np.float64(voxel_size))
    with (target / "timings.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    with (target / "timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["iteration"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[{scene}] completed {len(rows)} iteration(s), {int(source['occupancy'].sum())}->{int(occ.sum())} voxels, "
          f"{meta['total_seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="+")
    parser.add_argument("--input-root", default="srp_hull_mv2_v12_am1")
    parser.add_argument("--interval-root", default="hull_depth_interval_mv2_v12_am1")
    parser.add_argument("--out-root", default="srp_hull_mv2_v12_am1_iterouter_depthcons")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--patch-radius", type=int, default=2, help="2 means a 5x5 RGB patch")
    parser.add_argument("--min-sources", type=int, default=2, help="other reference depth maps")
    parser.add_argument("--ncc-reject", type=float, default=0.10)
    parser.add_argument("--ncc-accept", type=float, default=0.25)
    parser.add_argument("--ncc-margin", type=float, default=0.10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.max_iterations < 1 or args.patch_radius < 0 or args.min_sources < 1:
        raise ValueError("max-iterations and min-sources must be positive; patch-radius must be nonnegative")
    for scene in args.scenes:
        run(scene, args.input_root, args.interval_root, args.out_root, args.max_iterations,
            args.patch_radius, args.min_sources, args.ncc_reject, args.ncc_accept,
            args.ncc_margin, args.device)


if __name__ == "__main__":
    main()
