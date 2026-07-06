#!/home/cho/.pyenv/versions/grounded_sam/bin/python3
"""sam_only.py — 只跑 SAM 全自動分割,輸出「所有」遮罩(不分類、不接 CLIP)。

可處理:單張影像、整個場景(所有 view)、或整組(n3 等所有場景)。

輸出(--output-dir;預設 data/eval/sam_only/<scene>/<view>/):
  overlay.png          所有遮罩以不同顏色疊在原圖上
  masks/mask_000.png … 每張遮罩二值圖(依面積大→小)
  meta.txt             每張遮罩 area / bbox / iou / stability

需在 grounded_sam 或 webots_visual_hull 環境執行(有 segment_anything)。

用法:
  ./sam_only/sam_only.py n3_scene0001                 # 整個場景(所有 view)
  ./sam_only/sam_only.py 3                            # 整組 n3 的所有場景
  ./sam_only/sam_only.py 1 3 4 5                      # 多組
  ./sam_only/sam_only.py --input-image <img>          # 單張(--output-dir 可自訂)
  FORCE=1 ./sam_only/sam_only.py 3                    # 重做(忽略已存在)
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

SO_DIR = Path(__file__).resolve().parent
REPO = SO_DIR.parent
GSA = REPO / "Grounded-Segment-Anything"
sys.path.insert(0, str(GSA / "segment_anything"))

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator  # noqa: E402

SAM_CHECKPOINT = str(GSA / "sam_vit_b_01ec64.pth")
SAM_ENCODER    = "vit_b"
CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures")))
OUT_ROOT = Path(os.environ.get("SAM_OUT_ROOT", str(REPO / "data" / "eval" / "sam_only")))
FORCE = os.environ.get("FORCE") == "1"


def resolve_views(targets):
    """targets: 場景名(n3_scene0001)或組號(3)。回傳 [view_XX.png 路徑, ...]"""
    views = []
    for a in targets:
        if "scene" in a:
            g = a.split("_")[0]
            d = CAPTURES / f"multi_{g}" / a
            if not d.is_dir():
                print(f"[warn] 找不到場景: {d}"); continue
            views += [v for v in sorted(d.glob("view_*.png")) if "_depth" not in v.name]
        else:
            g = f"n{a}" if a.isdigit() else a   # "3"→n3;"occ3"/"stack3"/"n3" 直接當組名
            for d in sorted((CAPTURES / f"multi_{g}").glob(f"{g}_scene*")):
                views += [v for v in sorted(d.glob("view_*.png")) if "_depth" not in v.name]
    return views


def default_out_dir(img_path: Path) -> Path:
    return OUT_ROOT / img_path.parent.name / img_path.stem


def process_image(amg, img_path: Path, out_dir: Path, alpha: float) -> int:
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        print(f"  [warn] 讀不到 {img_path}"); return 0
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    masks = amg.generate(image_rgb)
    masks.sort(key=lambda m: m["area"], reverse=True)
    rng = np.random.default_rng(0)
    overlay = image_bgr.astype(np.float32)
    meta = []
    for i, m in enumerate(masks):
        seg = m["segmentation"]
        color = rng.integers(0, 255, size=3).astype(np.float32)
        overlay[seg] = overlay[seg] * (1 - alpha) + color * alpha
        cv2.imwrite(str(mask_dir / f"mask_{i:03d}.png"), (seg.astype(np.uint8) * 255))
        x, y, w, h = [int(v) for v in m["bbox"]]
        meta.append(f"mask_{i:03d}  area={m['area']:8d}  bbox=({x},{y},{w},{h})  "
                    f"iou={m.get('predicted_iou', 0):.3f}  stab={m.get('stability_score', 0):.3f}")
    cv2.imwrite(str(out_dir / "overlay.png"), np.clip(overlay, 0, 255).astype(np.uint8))
    (out_dir / "meta.txt").write_text("\n".join(meta), encoding="utf-8")
    return len(masks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", help="場景名(n3_scene0001)或組號(3)")
    ap.add_argument("--input-image", default=None, help="單張影像(優先於 targets)")
    ap.add_argument("--output-dir", default=None, help="輸出目錄(單張時可指定;批次時忽略)")
    ap.add_argument("--points-per-side", type=int, default=32)
    ap.add_argument("--pred-iou-thresh", type=float, default=0.88)
    ap.add_argument("--stability-score-thresh", type=float, default=0.92)
    ap.add_argument("--min-mask-region-area", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # 決定要處理的影像清單與各自輸出
    if args.input_image:
        jobs = [(Path(args.input_image),
                 Path(args.output_dir) if args.output_dir else default_out_dir(Path(args.input_image)))]
    else:
        if not args.targets:
            sys.exit("請給場景名/組號(如 n3_scene0001 或 3),或用 --input-image")
        jobs = [(v, default_out_dir(v)) for v in resolve_views(args.targets)]
    if not jobs:
        sys.exit("沒有可處理的影像")

    print(f"載入 SAM ({SAM_ENCODER}, {args.device}) ... 影像數 {len(jobs)}")
    sam = sam_model_registry[SAM_ENCODER](checkpoint=SAM_CHECKPOINT).to(torch.device(args.device))
    amg = SamAutomaticMaskGenerator(
        sam, points_per_side=args.points_per_side, pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area)

    for i, (img_path, out_dir) in enumerate(jobs, 1):
        if out_dir.exists() and (out_dir / "overlay.png").exists() and not FORCE:
            print(f"  [{i}/{len(jobs)}] {img_path.parent.name}/{img_path.stem} 已存在,跳過")
            continue
        n = process_image(amg, img_path, out_dir, args.alpha)
        print(f"  [{i}/{len(jobs)}] {img_path.parent.name}/{img_path.stem}: {n} 張遮罩 → {out_dir}")

    print(f"\n完成。輸出根目錄: {OUT_ROOT}")


if __name__ == "__main__":
    main()
