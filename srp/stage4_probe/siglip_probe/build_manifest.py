#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""build_manifest.py — 判別力驗證共用清單:選抽樣場景、標定每個 MobileSAMv2 遮罩屬哪個 GT 物體。

流程(不用深度,只用 2D 遮罩):
  抽樣場景(n3/n4/n5/stack3/4/5 各 N_PER 場,sorted 取前 N_PER,可重現)。
  對每場 A-3 selected 12 視角、MobileSAMv2 全部遮罩:與「同視角的 GT modal(actual)遮罩」算 2D-IoU,
  取最大;>IOU_MIN 才算屬該 GT 物體(同 GT=同物、異 GT=異物),無主碎片與 ur5e(手臂)排除。
  面積 < MIN_AREA 的遮罩略過。
輸出 data/eval/_diag/siglip_probe/manifest.json:每筆 {scene,group,view,mask_rel,gt_name,area,iou}。
三組抽特徵腳本(CLIP-B32 / SigLIP2-B32 / SigLIP2-B16)都讀這同一份清單、依相同順序抽 → 基準一致。
用法: ./build_manifest.py [--n-per 8] [--iou-min 0.5] [--min-area 200]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "srp" / "io"))
import viewpoints as VP          # noqa: E402
from labels import LABELS        # noqa: E402

MV2 = REPO / "data" / "eval" / "mobilesamv2_fast"
OUT = REPO / "data" / "eval" / "_diag" / "siglip_probe"
GROUPS = ["n3", "n4", "n5", "stack3", "stack4", "stack5"]
EXCLUDE_CAT = {"ur5e"}


def load_gt_modal(scene):
    """回傳 {view: {gt_name: mask(bool)}}(排除手臂),取自 labels/<scene>/actual/annotations.json。"""
    d = json.loads((LABELS / scene / "actual" / "annotations.json").read_text())
    cat = {c["id"]: c["name"] for c in d["categories"]}
    vof = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
    out = {}
    for a in d["annotations"]:
        name = cat[a["category_id"]]
        if name in EXCLUDE_CAT:
            continue
        m = mask_utils.decode(a["segmentation"]).astype(bool)
        out.setdefault(vof[a["image_id"]], {})[name] = m
    return out


def iou2(a, b):
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else 0.0


def process(scene, sel, iou_min, min_area, rows):
    gt = load_gt_modal(scene)
    group = scene.split("_")[0]
    n_add = 0
    for view in sorted(sel):
        vd = MV2 / scene / view / "masks"
        if not vd.is_dir() or view not in gt:
            continue
        gts = gt[view]
        for mp in sorted(vd.glob("mask_*.png")):
            seg = cv2.imread(str(mp), 0) > 127
            area = int(seg.sum())
            if area < min_area:
                continue
            best_g, best_i = None, 0.0
            for g, gm in gts.items():
                i = iou2(seg, gm)
                if i > best_i:
                    best_i, best_g = i, g
            if best_g is None or best_i <= iou_min:
                continue
            rows.append({"scene": scene, "group": group, "view": view,
                         "mask_rel": str(mp.relative_to(REPO)), "gt_name": best_g,
                         "area": area, "iou": round(best_i, 4)})
            n_add += 1
    return n_add


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=8, dest="n_per")
    ap.add_argument("--iou-min", type=float, default=0.5, dest="iou_min")
    ap.add_argument("--min-area", type=int, default=200, dest="min_area")
    args = ap.parse_args()
    sel = set(VP.selected_view_names(12))
    rows = []
    scenes = []
    for g in GROUPS:
        gs = sorted(p.name for p in MV2.glob(f"{g}_scene*") if p.is_dir())[:args.n_per]
        scenes += gs
    for sc in scenes:
        n = process(sc, sel, args.iou_min, args.min_area, rows)
        print(f"[{sc}] +{n} 有主遮罩", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"n_per": args.n_per, "iou_min": args.iou_min, "min_area": args.min_area,
            "groups": GROUPS, "n_scenes": len(scenes), "n_masks": len(rows),
            "mask_source": "mobilesamv2_fast", "gt": "modal(actual)", "views": "A-3 selected 12"}
    (OUT / "manifest.json").write_text(
        json.dumps({"meta": meta, "items": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    # 統計每場有主遮罩、每物體遮罩數分佈
    from collections import Counter
    perscene = Counter(r["scene"] for r in rows)
    print(f"\n共 {len(scenes)} 場 / {len(rows)} 有主遮罩 → {OUT/'manifest.json'}")
    print(f"每場有主遮罩數: min={min(perscene.values())} max={max(perscene.values())} "
          f"mean={np.mean(list(perscene.values())):.1f}")


if __name__ == "__main__":
    main()
