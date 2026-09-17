#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""Run the conservative RGB-only ray-search photo experiment on existing hulls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "srp" / "stage1_hull"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import photo_carve_warp as warp  # noqa: E402
from photo_carve_b import load_views  # noqa: E402
from ray_search import ray_search_carve, surface  # noqa: E402

EVAL = REPO / "data" / "eval"


def write_instances(path, occupancy, grid_min, voxel_size, source):
    np.savez_compressed(
        path / "instances.npz",
        labels=np.zeros_like(occupancy, dtype=np.int32),
        occupancy=occupancy,
        grid_min=grid_min,
        voxel_size=np.float64(voxel_size),
        build_meta=json.dumps({"src": source, "grayfill": True}),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="+")
    parser.add_argument("--in-root", default="srp_hull_mv2_v12_am1")
    parser.add_argument("--out-root", default="srp_hull_mv2_v12_am1_photo_raysearch")
    parser.add_argument("--ncc-reject", type=float, default=0.10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--ncc-accept", type=float, default=0.25)
    parser.add_argument("--ncc-margin", type=float, default=0.10)
    args = parser.parse_args()

    for number, scene in enumerate(args.scenes, 1):
        source = EVAL / args.in_root / scene / "hull.npz"
        target = EVAL / args.out_root / scene
        if not source.is_file():
            raise FileNotFoundError(source)
        if (target / "hull.npz").is_file() and not args.force:
            print(f"[skip] {scene}")
            continue
        data = np.load(source)
        occupancy = data["occupancy"].astype(bool)
        grid_min = data["grid_min"]
        voxel_size = float(data["voxel_size"])
        observed = data["observed"] if "observed" in data else occupancy
        views = warp.prep_views(load_views(scene, 12))
        carved, stats = ray_search_carve(
            occupancy, grid_min, voxel_size, views,
            max_steps=args.max_steps,
            ncc_accept=args.ncc_accept,
            ncc_reject=args.ncc_reject,
            ncc_margin=args.ncc_margin,
        )
        target.mkdir(parents=True, exist_ok=True)
        meta = {
            "src": args.in_root,
            "method": "ray_search_second_ncc_fgpatch",
            "num_views": 12,
            "allow_miss": 1,
            "max_steps": args.max_steps,
            "ncc_accept": args.ncc_accept,
            "ncc_margin": args.ncc_margin,
            "ncc_reject": args.ncc_reject,
            "stats": stats,
        }
        np.savez_compressed(
            target / "hull.npz",
            occupancy=carved,
            surface=surface(carved),
            observed=observed,
            grid_min=grid_min,
            voxel_size=np.float64(voxel_size),
            build_meta=json.dumps(meta),
        )
        write_instances(target, carved, grid_min, voxel_size, args.in_root)
        print(f"[{number}/{len(args.scenes)}] {scene}: {int(occupancy.sum())}->{int(carved.sum())} {stats}")


if __name__ == "__main__":
    main()
