#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""Run the foreground-patch guarded warped-NCC carving experiment."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "srp" / "stage1_hull"))
sys.path.insert(0, str(REPO / "srp" / "io"))

import photo_carve_warp as warp
from photo_carve_b import load_views


EVAL = REPO / "data" / "eval"
DEFAULT_INPUT = "srp_hull_mv2_v12_am1"
GROUPS = ("n1", "n3", "n4", "n5", "occ3", "occ4", "occ5", "stack3", "stack4", "stack5")


def surface(occupancy):
    return occupancy & ~ndimage.binary_erosion(
        occupancy, ndimage.generate_binary_structure(3, 1)
    )


def drop_small_components(occupancy, minimum=20):
    labels, count = ndimage.label(
        occupancy, ndimage.generate_binary_structure(3, 1)
    )
    if count == 0:
        return occupancy
    sizes = ndimage.sum(
        np.ones_like(labels), labels, index=np.arange(1, count + 1)
    )
    keep = np.nonzero(sizes >= minimum)[0] + 1
    return np.isin(labels, keep)


def default_out_root(input_root, ncc_reduce):
    return f"{input_root}_photo_fgpatch_{ncc_reduce}"


def available_scenes(source_root):
    return sorted(
        path.name
        for group in GROUPS
        for path in source_root.glob(f"{group}_scene*")
        if (path / "hull.npz").is_file()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="*", help="Exact scene names.")
    parser.add_argument("--in-root", default=DEFAULT_INPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out-root")
    parser.add_argument("--ncc-reduce", choices=("mean", "max", "second"), default="second")
    args = parser.parse_args()

    source_root = EVAL / args.in_root
    output_name = args.out_root or default_out_root(args.in_root, args.ncc_reduce)
    scenes = args.scenes or available_scenes(source_root)
    out_root = EVAL / output_name
    for number, scene in enumerate(scenes, start=1):
        source = source_root / scene / "hull.npz"
        if not source.is_file():
            raise FileNotFoundError(source)
        target = out_root / scene / "hull.npz"
        if target.is_file() and not args.force:
            print(f"[skip] {scene}: {target} exists")
            continue

        data = np.load(source)
        occupancy = data["occupancy"].astype(bool)
        grid_min = data["grid_min"]
        voxel_size = float(data["voxel_size"])
        observed = data["observed"] if "observed" in data else occupancy
        views = warp.prep_views(load_views(scene, 12))
        if len(views) < 2:
            print(f"[skip] {scene}: fewer than two views")
            continue

        carved = warp.carve_warp(
            occupancy, grid_min, voxel_size, views,
            0.1, 4, 2, 3, 2, 15, 20.0, fg_patch_guard=True, ncc_reduce=args.ncc_reduce,
        )
        carved = drop_small_components(carved)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "src": args.in_root,
            "sam": "mobilesamv2_fast",
            "num_views": 12,
            "allow_miss": 1,
            "min_comp": 20,
            "carve": f"warp_tilt20_ncc0.1_k4_min2_clu3_patch2_iter15_fgpatch_{args.ncc_reduce}",
        }
        np.savez_compressed(
            target,
            occupancy=carved,
            surface=surface(carved),
            observed=observed,
            grid_min=grid_min,
            voxel_size=np.float64(voxel_size),
            build_meta=json.dumps(metadata),
        )
        print(
            f"[{number}/{len(scenes)}] {scene}: "
            f"{int(occupancy.sum())}->{int(carved.sum())} vox"
        )


if __name__ == "__main__":
    main()
