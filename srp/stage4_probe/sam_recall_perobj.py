#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""sam_recall_perobj — 各物體層級 recall,比較 sam_only vs MobileSAMv2(哪些物體救回)。

每 (場景,視角,GT YCB 物體):該物 GT modal 遮罩 vs 該視角任一遮罩 best IoU ≥ 0.5 即「找到」。
三方法(sam_only / mobilesamv2_l2 / mobilesamv2_tiny_vit)共用 GT(只解一次),各載自己的遮罩。
依 Δrecall(mv2_l2 − sam_only)排序,列出 MobileSAMv2 救回最多的物體。
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

REPO = Path(__file__).resolve().parents[2]
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "srp" / "io")); from labels import LABELS  # data/labels 分層(類別/數量/場景)
THR = 0.5
GROUPS = ["n1", "n3", "n4", "n5", "occ3", "occ4", "occ5", "stack3", "stack4", "stack5"]
METHODS = {
    "sam_only": REPO / "data" / "eval" / "sam_only_fast",
    "mv2_l2":   REPO / "data" / "eval" / "mobilesamv2_fast",
    "mv2_tiny": REPO / "data" / "eval" / "mobilesamv2_tiny_vit_fast",
}


def load_sam(root, scene, view):
    d = root / scene / view / "masks"
    return [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) > 127 for f in sorted(d.glob("mask_*.png"))]


def iou(a, b):
    inter = int(np.logical_and(a, b).sum())
    return inter / int(np.logical_or(a, b).sum()) if inter else 0.0


def modal_by_view(scene):
    ann = LABELS / scene / "actual" / "annotations.json"
    if not ann.exists():
        return {}
    d = json.load(open(ann))
    cat = {c["id"]: c for c in d["categories"]}
    vname = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
    out = defaultdict(list)
    for a in d["annotations"]:
        c = cat[a["category_id"]]
        if c.get("supercategory") == "robot":
            continue
        out[vname[a["image_id"]]].append((c["name"], mask_utils.decode(a["segmentation"]).astype(bool)))
    return out


def main():
    perobj = {m: defaultdict(lambda: [0, 0]) for m in METHODS}   # method: obj: [n_obj_view, found]
    scenes = []
    for g in GROUPS:
        scenes += sorted(Path(p).name for p in glob.glob(str(METHODS["sam_only"] / f"{g}_scene*")))
    for si, scene in enumerate(scenes):
        modal = modal_by_view(scene)
        for view, objs in modal.items():
            sams = {m: load_sam(METHODS[m], scene, view) for m in METHODS}
            for name, gt in objs:
                for m in METHODS:
                    best = max((iou(s, gt) for s in sams[m]), default=0.0)
                    perobj[m][name][0] += 1
                    perobj[m][name][1] += int(best >= THR)
        if si % 50 == 0:
            print(f"  {si}/{len(scenes)}", flush=True)

    rows = []
    for name in perobj["sam_only"]:
        n = perobj["sam_only"][name][0]
        if n < 12:
            continue
        r_so = perobj["sam_only"][name][1] / n
        r_l2 = perobj["mv2_l2"][name][1] / max(1, perobj["mv2_l2"][name][0])
        r_tv = perobj["mv2_tiny"][name][1] / max(1, perobj["mv2_tiny"][name][0])
        rows.append((name, n, r_so, r_l2, r_tv, r_l2 - r_so))
    rows.sort(key=lambda x: -x[5])

    print(f"\n{'物體':<30}{'物視角':>7}{'sam_only':>9}{'mv2_l2':>8}{'mv2_tiny':>9}{'Δ(l2-so)':>10}")
    print("-" * 73)
    for name, n, so, l2, tv, d in rows:
        print(f"{name:<30}{n:>7}{so:>9.3f}{l2:>8.3f}{tv:>9.3f}{d:>+10.3f}")


if __name__ == "__main__":
    main()
