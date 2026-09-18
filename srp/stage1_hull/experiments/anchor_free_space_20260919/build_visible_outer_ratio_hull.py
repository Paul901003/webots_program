#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""Build an isolated hull from a fixed visible-outer support-ratio rule.

Rule: a visible outer voxel V0 is removed only when at least min_views
reference views can score both NCC windows and every one selects a deeper,
identical 7x7/11x11 candidate. GT is never used in this decision.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[4]
os.environ.setdefault("SAM_ROOT", str(REPO / "data" / "eval" / "mobilesamv2_fast"))
os.environ.setdefault("CAPTURES_ROOT", str(REPO / "data" / "captures_fast"))
sys.path[:0] = [str(REPO / "srp" / "io"), str(REPO / "srp" / "stage1_hull"),
                str(REPO / "srp" / "stage1_hull" / "experiments" / "hull_depth_interval_20260918"),
                str(REPO / "srp" / "stage2_instances")]
import interval_candidates as IC  # noqa: E402
from depth_consistency import forced_reference_scores  # noqa: E402
from run_interval_ray_search import load_views  # noqa: E402
from anchor_free_space import combined_gt, proposal_metrics  # noqa: E402

EVAL = REPO / "data" / "eval"


def camera_counts(shape, indices, refs):
    counts = np.zeros(shape, np.uint8)
    for ref in np.unique(refs):
        cells = np.unique(indices[refs == ref], axis=0)
        counts[tuple(cells.T)] += 1
    return counts


def process(scene, candidate_root, source_root, out_root, min_views, ratio, batch_size, device, eval_gt):
    source_path = EVAL / source_root / scene / "hull.npz"
    candidate_path = EVAL / candidate_root / scene / "multi_ref_candidates.npz"
    source = np.load(source_path, allow_pickle=False)
    candidates = np.load(candidate_path, allow_pickle=False)
    occupancy = source["occupancy"].astype(bool)
    grid_min = source["grid_min"].astype(float)
    voxel_size = float(source["voxel_size"])
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
    normals = IC.hull_normals(occupancy)[tuple(surface_indices[source_rows[rows]].T)]
    scores = {}
    for radius in (3, 5):
        started = time.time()
        flat = forced_reference_scores(points, normals, refs[rows], views, patch_radius=radius,
                                      batch_size=batch_size, device=device)
        score = np.full(valid.shape, np.nan, np.float32); score[rows, columns] = flat
        scores[radius] = score
        print(f"{scene}: {2 * radius + 1}x{2 * radius + 1} NCC {time.time() - started:.2f}s", flush=True)

    score7, score11 = scores[3], scores[5]
    scoreable = np.isfinite(score7).any(1) & np.isfinite(score11).any(1)
    choice7 = np.argmax(np.where(np.isfinite(score7), score7, -np.inf), axis=1)
    choice11 = np.argmax(np.where(np.isfinite(score11), score11, -np.inf), axis=1)
    agreed = scoreable & np.all(layers[np.arange(len(layers)), choice7] == layers[np.arange(len(layers)), choice11], axis=1)
    deeper = agreed & (choice11 > 0)

    outer = surface_indices[source_rows]
    assessable = camera_counts(occupancy.shape, outer[scoreable], refs[scoreable])
    support = camera_counts(occupancy.shape, outer[deeper], refs[deeper])
    anchors = np.zeros_like(occupancy, bool)
    agreed_rows = np.flatnonzero(agreed)
    anchors[tuple(layers[agreed_rows, choice11[agreed_rows]].T)] = True
    support_ratio = np.divide(support, assessable, out=np.zeros_like(support, np.float32), where=assessable > 0)
    proposed = occupancy & (assessable >= min_views) & (support_ratio >= ratio) & ~anchors
    result_occupancy = occupancy & ~proposed

    out = EVAL / out_root / scene
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(str(source["build_meta"]))
    meta["visible_outer_ratio_rule"] = {"min_assessable_views": min_views, "support_ratio": ratio,
                                         "device_for_ncc": device, "source_hull": source_root}
    payload = {key: source[key] for key in source.files}
    payload["occupancy"] = result_occupancy
    payload["surface"] = IC.outer_surface(result_occupancy).astype(np.uint8)
    payload["build_meta"] = np.asarray(json.dumps(meta))
    np.savez_compressed(out / "hull.npz", **payload)
    np.savez_compressed(out / "proposal.npz", assessable_views=assessable, support_views=support,
                        support_ratio=support_ratio, removed=proposed, protected_anchors=anchors)
    summary = {"scene": scene, "source_hull": source_root, "device": device, "min_assessable_views": min_views,
               "support_ratio": ratio, "removed_voxels": int(proposed.sum()),
               "remaining_voxels": int(result_occupancy.sum()), "max_assessable_views": int(assessable.max()),
               "max_support_views": int(support.max())}
    if eval_gt:
        gt = combined_gt(scene, grid_min, voxel_size, occupancy.shape)
        baseline_real = int((occupancy & gt).sum())
        baseline_ghost = int((occupancy & ~gt).sum())
        summary["gt_evaluation"] = {
            "proposed_remove_voxels": int(proposed.sum()),
            "proposed_remove_ghost_voxels": int((proposed & ~gt).sum()),
            "proposed_remove_real_voxels": int((proposed & gt).sum()),
            "removed_is_ghost_precision": float((proposed & ~gt).sum() / proposed.sum()) if proposed.any() else None,
            "ghost_removed_recall": float((proposed & ~gt).sum() / baseline_ghost) if baseline_ghost else None,
            "real_voxels_removed_fraction": float((proposed & gt).sum() / baseline_real) if baseline_real else None,
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="+")
    parser.add_argument("--candidate-root", default="hull_multi_ref_candidates_mv2_v12_am1")
    parser.add_argument("--source-root", default="srp_hull_mv2_v12_am1")
    parser.add_argument("--out-root", default="srp_hull_mv2_v12_am1_visibleouter_ratio100_n3")
    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--eval-gt", action="store_true")
    args = parser.parse_args()
    if args.min_views < 1 or not 0 < args.ratio <= 1:
        raise ValueError("need min-views >= 1 and 0 < ratio <= 1")
    for scene in args.scenes:
        process(scene, args.candidate_root, args.source_root, args.out_root, args.min_views, args.ratio,
                args.batch_size, args.device, args.eval_gt)

if __name__ == "__main__":
    main()
