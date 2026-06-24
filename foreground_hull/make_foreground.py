#!/home/cho/.pyenv/versions/grounded_sam/bin/python3
"""make_foreground.py — SAM 全自動切 → 扣掉背景(碰邊界/過大)→ 每 view 前景二值。

foreground_hull 流程第①步(完全不用 depth):
  ① make_foreground.py  每 view 出前景/背景二值遮罩
  ② split_hull.py       固定 cube 雕殼 + 連通元件分物體(不經 build_torchhull)

不分物體、不分類、不用 depth:前景 = 所有「不碰影像邊界且非過大」的 SAM 遮罩聯集。
輸出 view_XX_mask_foreground.png(給 split_hull.py 當單一類別 carve)+ 疊圖驗證。

輸出: data/eval/foreground/<scene>/
需在 grounded_sam 或 webots_visual_hull 環境(有 segment_anything)。

用法:
  ./foreground_hull/make_foreground.py n3_scene0001
  ./foreground_hull/make_foreground.py 3            # 整組
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

FH_DIR = Path(__file__).resolve().parent
REPO = FH_DIR.parent
GSA = REPO / "Grounded-Segment-Anything"
sys.path.insert(0, str(GSA / "segment_anything"))
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator  # noqa: E402

SAM_CHECKPOINT = str(GSA / "sam_vit_b_01ec64.pth")
CAPTURES = REPO / "data" / "captures"
OUT_ROOT = REPO / "data" / "eval" / "foreground"
FORCE = os.environ.get("FORCE") == "1"
MAX_AREA_FRAC = 0.5          # 面積 > 此比例 → 視為背景(整片桌面)
BORDER = 2                   # 距邊界幾 px 內算「碰邊界」


def touches_border(seg) -> bool:
    return bool(seg[:BORDER, :].any() or seg[-BORDER:, :].any()
               or seg[:, :BORDER].any() or seg[:, -BORDER:].any())


def foreground_of(amg, image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]
    fg = np.zeros((H, W), dtype=np.uint8)
    kept = 0
    for m in amg.generate(image_rgb):
        seg = m["segmentation"]
        if m["area"] > MAX_AREA_FRAC * H * W:   # 過大 → 背景
            continue
        if touches_border(seg):                 # 碰邊界 → 背景(桌面延伸到邊)
            continue
        fg |= seg.astype(np.uint8)
        kept += 1
    return fg, kept


def resolve_scenes(targets):
    scenes = []
    for a in targets:
        if "scene" in a:
            g = a.split("_")[0]; d = CAPTURES / f"multi_{g}" / a
            scenes.append(d) if d.is_dir() else print(f"[warn] 無 {d}")
        else:
            scenes += sorted((CAPTURES / f"multi_n{a}").glob(f"n{a}_scene*"))
    return scenes


def main():
    targets = sys.argv[1:] or ["3"]
    scenes = resolve_scenes(targets)
    if not scenes:
        sys.exit("沒有場景")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"載入 SAM ({device}) ... 場景 {len(scenes)}")
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT).to(device)
    amg = SamAutomaticMaskGenerator(sam, points_per_side=32, pred_iou_thresh=0.88,
                                    stability_score_thresh=0.92, min_mask_region_area=400)
    for scene_dir in scenes:
        scene = scene_dir.name
        out = OUT_ROOT / scene
        ov = out / "fg_overlay"
        out.mkdir(parents=True, exist_ok=True); ov.mkdir(exist_ok=True)
        views = [v for v in sorted(scene_dir.glob("view_*.png")) if "_depth" not in v.name]
        if any(out.glob("view_*_mask_foreground.png")) and not FORCE:
            print(f"  {scene} 已存在,跳過"); continue
        print(f"  {scene}: {len(views)} views")
        for v in views:
            img = cv2.imread(str(v))
            fg, kept = foreground_of(amg, img)
            cv2.imwrite(str(out / f"{v.stem}_mask_foreground.png"), fg * 255)
            vis = img.copy(); vis[fg > 0] = (0.4 * vis[fg > 0] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
            cv2.imwrite(str(ov / f"{v.stem}.png"), vis)
            print(f"    {v.stem}: 前景塊 {kept}, 前景像素 {int(fg.sum())}")
    print(f"\n完成 → {OUT_ROOT}")


if __name__ == "__main__":
    main()
