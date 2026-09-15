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


def kept_object_masks(view_dir, max_frac=0.5, border_frac=0.2):
    """回傳該視角保留(非地板)的 [(bool_mask, filename)]。
    地板判定:面積 > max_frac×畫面;border_frac 不為 None 時,額外排除「碰邊界且 > border_frac×畫面」。
    border_frac=0.2(預設)= 額外排除碰邊界的中大型背景洩漏(如 24% 的紅色背景牆),避免污染分群;
    傳 border_frac=None 可退回舊保守行為(只排 >max_frac 的大塊)。"""
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


def mask_feats(view_dir, feat_file="clip_mean_feats.npy"):
    """回 {遮罩檔名: 語意特徵(np.ndarray) or None}。
    特徵檔(預設 clip_mean_feats.npy;siglip 版可傳 siglip_feats.npy)與 sorted(masks/*.png)「全部遮罩」對齊。
    用『檔名』查 → 與 border_frac 背景過濾徹底解耦。數量不符 raise(不靜默跳過,見 fail-loud 原則)。"""
    vd = Path(view_dir)
    p = vd / feat_file
    if not p.is_file():
        return {}
    arr = np.load(p)
    names = [q.name for q in sorted((vd / "masks").glob("mask_*.png"))]
    if len(arr) != len(names):        # 不該發生的異常 → 報錯,不靜默跳過(靜默會把 bug 藏很久)
        raise ValueError(
            f"{p}: 特徵數 {len(arr)} ≠ 遮罩數 {len(names)}，語意特徵與遮罩不同步；"
            f"請重產特徵（須對全部遮罩、與檔名對齊）")
    return {n: (None if np.isnan(a).any() else a) for n, a in zip(names, arr)}


def feats_list(view_dir, names, feat_file="clip_mean_feats.npy"):
    """依 names(遮罩檔名)順序回 [特徵 or None]。給 sem 三種取代 cached_feats,用檔名查、不重算。
    feat_file 可切換特徵來源(clip_mean_feats.npy / siglip_feats.npy)。"""
    fm = mask_feats(view_dir, feat_file)
    return [fm.get(n) for n in names]
