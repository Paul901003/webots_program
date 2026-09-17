#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""Trace mainline leak voxels into the center-photo div result on shared grids."""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
from stack_leak_nosep import EVAL, GEX, label_stats, load_gt, on_pairs

MAIN_INST = "srp_hull_divB_t50_reNNcSd_am1"
MAIN_GT = "srp_hull_gtlabel_am1"
PHOTO_INST = "srp_hull_divB_t50_reNNcSd_am1photo_fgpatch_second"
PHOTO_GT = "srp_hull_gtlabel_am1photo_fgpatch_second"


def load_scene(inst_root, gt_root, scene):
    gt, idx2name = load_gt(gt_root, scene)
    path = EVAL / inst_root / scene / "instances.npz"
    if gt is None or not path.is_file():
        return None
    pred = np.load(path)["labels"]
    if pred.shape != gt.shape:
        raise ValueError(f"{scene}: prediction and GT grids differ")
    return gt, pred, label_stats(gt, pred, idx2name)


def voxel_status(label, obj, dom):
    if label <= 0:
        return "unassigned"
    owner = dom.get(int(label))
    if owner == obj:
        return "correct"
    if owner is None:
        return "other"
    return "leak"


def merged(stats, upper, lower):
    iu = stats["name2idx"].get(upper)
    il = stats["name2idx"].get(lower)
    if iu is None or il is None:
        return None
    mu = stats["per"][iu]["main_i"]
    ml = stats["per"][il]["main_i"]
    return mu > 0 and mu == ml


def pair_flow(main, photo, upper, lower):
    main_gt, main_pred, main_stats = main
    photo_gt, photo_pred, photo_stats = photo
    flow = Counter()
    for name in (upper, lower):
        main_obj = main_stats["name2idx"].get(name)
        photo_obj = photo_stats["name2idx"].get(name)
        if main_obj is None or photo_obj is None:
            return None
        base_mask = main_gt == main_obj
        for base_label, photo_gt_label, photo_label in zip(
            main_pred[base_mask], photo_gt[base_mask], photo_pred[base_mask]
        ):
            if voxel_status(int(base_label), main_obj, main_stats["dom"]) != "leak":
                continue
            flow["base_leak"] += 1
            if photo_gt_label != photo_obj:
                flow["surface_changed"] += 1
                continue
            flow[voxel_status(int(photo_label), photo_obj, photo_stats["dom"])] += 1
    return {
        "flow": flow,
        "main_merged": merged(main_stats, upper, lower),
        "photo_merged": merged(photo_stats, upper, lower),
    }


def pct(count, total):
    return f"{100 * count / total:.1f}%" if total else "-"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-inst", default=MAIN_INST)
    parser.add_argument("--main-gt", default=MAIN_GT)
    parser.add_argument("--photo-inst", default=PHOTO_INST)
    parser.add_argument("--photo-gt", default=PHOTO_GT)
    parser.add_argument("--out", type=Path, default=HERE / "DIVB_PAIR_FLOW.md")
    args = parser.parse_args()

    scenes = sorted(
        path.parent.name
        for path in (EVAL / args.photo_inst).glob("stack*_scene*/instances.npz")
    )
    rows = []
    total = Counter()
    merge = Counter()
    for scene in scenes:
        main_data = load_scene(args.main_inst, args.main_gt, scene)
        photo_data = load_scene(args.photo_inst, args.photo_gt, scene)
        if main_data is None or photo_data is None:
            continue
        for upper, lower in on_pairs(scene):
            if upper in GEX or lower in GEX:
                continue
            result = pair_flow(main_data, photo_data, upper, lower)
            if result is None:
                continue
            flow = result["flow"]
            total.update(flow)
            merge["main"] += int(result["main_merged"])
            merge["photo"] += int(result["photo_merged"])
            merge["newly_merged"] += int(not result["main_merged"] and result["photo_merged"])
            merge["newly_separated"] += int(result["main_merged"] and not result["photo_merged"])
            rows.append((scene, upper, lower, flow, result))

    rows.sort(key=lambda row: (-row[3]["correct"], row[0], row[1], row[2]))
    total_leak = total["base_leak"]
    md = [
        "# Mainline Div vs Center-Photo Div: Leak Voxel Flow",
        "",
        "This report follows only voxels that are leakage in mainline div: their GT object",
        "is assigned to an instance dominated by a different GT object. The two grids share",
        "world coordinates. For each such voxel, the photo result is classified as correct,",
        "still leak, unassigned, other, or surface changed. Surface changed means the same",
        "world coordinate is no longer the same object's GT-labelled surface after photo carving;",
        "it is deliberately not counted as unassigned.",
        "",
        f"- On relations: {len(rows)}",
        f"- Mainline leak voxels traced: {total_leak}",
        f"- To correct: {pct(total['correct'], total_leak)}",
        f"- Still leak: {pct(total['leak'], total_leak)}",
        f"- To unassigned: {pct(total['unassigned'], total_leak)}",
        f"- Surface changed: {pct(total['surface_changed'], total_leak)}",
        f"- Other assigned state: {pct(total['other'], total_leak)}",
        f"- Main-pair merged: mainline {merge['main']}/{len(rows)}, photo {merge['photo']}/{len(rows)}",
        f"- Newly merged: {merge['newly_merged']}; newly separated: {merge['newly_separated']}",
        "",
        "| Scene | On pair | Main leak vox | To correct | Still leak | To unassigned | Surface changed | Other | Main merged -> photo merged |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for scene, upper, lower, flow, result in rows:
        n = flow["base_leak"]
        main_same = "yes" if result["main_merged"] else "no"
        photo_same = "yes" if result["photo_merged"] else "no"
        md.append(
            f"| {scene} | {upper} -> {lower} | {n} | "
            f"{pct(flow['correct'], n)} | {pct(flow['leak'], n)} | "
            f"{pct(flow['unassigned'], n)} | {pct(flow['surface_changed'], n)} | "
            f"{pct(flow['other'], n)} | {main_same} -> {photo_same} |"
        )
    args.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} for {len(rows)} on relations.")
    print(f"mainline leak voxels={total_leak}; correct={pct(total['correct'], total_leak)}; "
          f"unassigned={pct(total['unassigned'], total_leak)}; still_leak={pct(total['leak'], total_leak)}")


if __name__ == "__main__":
    main()
