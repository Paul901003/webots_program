"""crop_util.py — 三組(CLIP/SigLIP2)共用的遮罩方形 crop 幾何。
沿用 mask_clip_cluster.sqcrop("mean") 的幾何:bbox→遮罩外填 fill→短邊補 fill 成正方形。
只回傳「未 resize」的方形 uint8 crop;resize 目標(224/256)與 normalize 由各模型腳本自理,
確保 crop 的幾何(定位/遮罩外填色策略)三組完全一致,只差各自標準前處理。
"""
import cv2
import numpy as np


def sqcrop_geom(rgb, seg, fill):
    """rgb(H,W,3 uint8), seg(H,W bool), fill=長度3的填色。回傳 (s,s,3) uint8 或 None。"""
    ys, xs = np.nonzero(seg)
    if xs.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    c = rgb[y0:y1, x0:x1].copy()
    c[~seg[y0:y1, x0:x1]] = fill
    h, w = c.shape[:2]
    s = max(h, w)
    out = np.empty((s, s, 3), np.uint8)
    out[:] = fill
    oy, ox = (s - h) // 2, (s - w) // 2
    out[oy:oy + h, ox:ox + w] = c
    return out
