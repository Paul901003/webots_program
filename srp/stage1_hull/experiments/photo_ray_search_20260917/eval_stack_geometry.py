#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""Compare geometry quality for the fixed legacy stack3/4/5 cohort."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))

import eval_mesh as mesh_eval  # noqa: E402

EVAL = REPO / "data" / "eval"
HERE = Path(__file__).resolve().parent
ROOTS = {
    "raw_am1": "srp_hull_mv2_v12_am1",
    "photo_second": "srp_hull_mv2_v12_am1_photo_fgpatch_second",
    "ray_search": "srp_hull_mv2_v12_am1_photo_raysearch",
}


def scenes():
    return sorted(
        path.name
        for group in ("stack3", "stack4", "stack5")
        for path in (EVAL / ROOTS["raw_am1"]).glob(f"{group}_scene*")
        if (path / "hull.npz").is_file()
    )


def quality(occupancy, gt):
    tp = int((occupancy & gt).sum())
    total_gt = int(gt.sum())
    total = int(occupancy.sum())
    return {
        "voxels": total,
        "coverage": tp / total_gt if total_gt else 0.0,
        "ghost": 1 - tp / total if total else 0.0,
        "bloat": total / total_gt if total_gt else 0.0,
    }


def main():
    rows = []
    aggregate = defaultdict(lambda: defaultdict(list))
    for number, scene in enumerate(scenes(), 1):
        raw = np.load(EVAL / ROOTS["raw_am1"] / scene / "hull.npz")
        shape = raw["occupancy"].shape
        gt_parts = mesh_eval.solid_mesh_occ(
            scene, raw["grid_min"], float(raw["voxel_size"]), shape
        )
        gt = np.zeros(shape, bool)
        for part in gt_parts.values():
            gt |= part
        for name, root in ROOTS.items():
            data = np.load(EVAL / root / scene / "hull.npz")
            result = quality(data["occupancy"].astype(bool), gt)
            rows.append({"scene": scene, "method": name, **result})
            for key, value in result.items():
                aggregate[name][key].append(value)
        print(f"[{number}/60] {scene}", flush=True)

    out = HERE / "stack_geometry.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene", "method", "voxels", "coverage", "ghost", "bloat"])
        writer.writeheader()
        writer.writerows(rows)
    print("\nmethod             coverage    ghost    bloat     voxels")
    for name in ROOTS:
        values = aggregate[name]
        print(
            f"{name:<17} {np.mean(values['coverage']) * 100:7.2f}%"
            f" {np.mean(values['ghost']) * 100:7.2f}%"
            f" {np.mean(values['bloat']):7.3f}"
            f" {np.mean(values['voxels']):10.1f}"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
