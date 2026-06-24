#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""masks.py — SAM 遮罩載入 + 地板/背景排除(Stage 1/2 共用,保持一致)。

地板排除規則(改善版):一塊遮罩視為地板/背景而排除,若
  面積 > max_frac×畫面          (整片大背景)
  或 (碰邊界 且 面積 > border_frac×畫面)   (沿底/邊洩漏的大塊,如 41% 地板)
其餘保留為物體前景。比舊版「只排最大且>50%」更能抓住碰邊界的中大型背景洩漏。
"""

from pathlib import Path

import cv2
import numpy as np

BORDER = 2


def touches_border(b):
    return bool(b[:BORDER].any() or b[-BORDER:].any()
                or b[:, :BORDER].any() or b[:, -BORDER:].any())


def kept_object_masks(view_dir, max_frac=0.5, border_frac=None):
    """回傳該視角保留(非地板)的 [(bool_mask, filename)]。
    地板判定:面積 > max_frac×畫面;border_frac 不為 None 時,額外排除「碰邊界且 > border_frac×畫面」。
    border_frac=None(預設)= 證實過的保守行為(只排 >max_frac 的大塊),避免誤刪真實物體。"""
    out = []
    for mp in sorted((Path(view_dir) / "masks").glob("mask_*.png")):
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        b = m > 127
        a = int(b.sum())
        if a == 0:
            continue
        H, W = b.shape
        if a > max_frac * H * W:
            continue
        if border_frac is not None and touches_border(b) and a > border_frac * H * W:
            continue
        out.append((b, mp.name))
    return out
