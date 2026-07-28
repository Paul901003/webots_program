#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""sam_recall_fast — 「SAM 找到多少 GT」(報告 §2.2 同法,吃 captures_fast 新資料)。

每 (場景, 視角, GT YCB 物體):該物 GT modal 遮罩 與該視角「任一」class-agnostic SAM 遮罩
比 best IoU,≥ THR(0.5) 即「找到」。指標:recall@.5 = 找到/(物,視角)數;meanIoU = best IoU 平均。

GT : data/labels/<scene>/actual/annotations.json —— view 名直接取 image file_name(view_el..),
     **排除 supercategory==robot(ur5e 手臂),只算 super=ycb 物體**。
SAM: data/eval/sam_only_fast/<scene>/<view>/masks/mask_*.png(class-agnostic 全遮罩)。
env: webots_visual_hull(pycocotools + cv2)。
"""
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

REPO = Path(__file__).resolve().parents[2]
SAM = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "sam_only_fast")))
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "srp" / "io")); from labels import LABELS  # data/labels 分層(類別/數量/場景)
THR = 0.5
GROUPS = ["n1", "n3", "n4", "n5", "occ3", "occ4", "occ5", "stack3", "stack4", "stack5"]


def load_sam_masks(scene, view):
    d = SAM / scene / view / "masks"
    return [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) > 127 for f in sorted(d.glob("mask_*.png"))]


def iou(a, b):
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    return inter / int(np.logical_or(a, b).sum())


def modal_by_view(scene):
    """{view_name: [(obj_name, modal_mask_bool), ...]};排除 super=robot。view 名取 image file_name。"""
    ann = LABELS / scene / "actual" / "annotations.json"
    if not ann.exists():
        return {}
    d = json.load(open(ann))
    cat = {c["id"]: c for c in d["categories"]}
    vname = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
    out = defaultdict(list)
    for a in d["annotations"]:
        c = cat[a["category_id"]]
        if c.get("supercategory") == "robot":   # ★排除 ur5e 手臂,只留 YCB
            continue
        m = mask_utils.decode(a["segmentation"]).astype(bool)
        out[vname[a["image_id"]]].append((c["name"], m))
    return out


def main():
    acc = defaultdict(lambda: [0, 0, 0.0])      # group:  [物視角數, 找到數, sum best IoU]
    perobj = defaultdict(lambda: [0, 0, 0.0])   # object: 同上
    for g in GROUPS:
        scenes = sorted(Path(p).name for p in glob.glob(str(SAM / f"{g}_scene*")))
        for scene in scenes:
            modal = modal_by_view(scene)
            for view, objs in modal.items():
                sams = load_sam_masks(scene, view)
                for name, m in objs:
                    best = max((iou(s, m) for s in sams), default=0.0)
                    hit = int(best >= THR)
                    acc[g][0] += 1; acc[g][1] += hit; acc[g][2] += best
                    perobj[name][0] += 1; perobj[name][1] += hit; perobj[name][2] += best
        n = acc[g][0]
        print(f"  [{g}] 完成 場景={len(scenes)} 物視角={n} "
              f"recall={acc[g][1]/n:.3f}" if n else f"  [{g}] 無場景", flush=True)

    # ── 各組 ──
    print(f"\n{'組':<10}{'物-視角數':>10}{'recall@.5':>11}{'meanIoU':>10}")
    tot = [0, 0, 0.0]
    for g in GROUPS:
        n, f, s = acc[g]
        if not n:
            continue
        tot[0] += n; tot[1] += f; tot[2] += s
        print(f"{g:<10}{n:>10}{f/n:>11.3f}{s/n:>10.3f}")
    if tot[0]:
        print(f"{'-'*41}")
        print(f"{'總計':<10}{tot[0]:>10}{tot[1]/tot[0]:>11.3f}{tot[2]/tot[0]:>10.3f}")

    # ── per-object 最常漏(物視角≥12) ──
    print(f"\n{'物體(最常漏, 物視角≥12)':<32}{'物-視角':>8}{'recall@.5':>11}{'meanIoU':>10}")
    rows = [(name, v[0], v[1] / v[0], v[2] / v[0]) for name, v in perobj.items() if v[0] >= 12]
    for name, n, r, mi in sorted(rows, key=lambda x: x[2])[:15]:
        print(f"{name:<32}{n:>8}{r:>11.3f}{mi:>10.3f}")


if __name__ == "__main__":
    main()
