#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""dino_mask_feats.py — 用 DINOv2 取「遮罩語意特徵」,介面與 voxel_sem_cluster_donut.donut_feats 對齊。

★ 新檔,不動任何既有程式。供 voxel_sem_cluster_reassign_soliddrop.py 的 FEAT=dino 切換使用。

取法(與 CLIP 的摳圖填灰完全不同):
  整張原圖 resize 成 14 的倍數 -> DINOv2 vitb14 前向【一次】-> x_norm_patchtokens -> fmap(ph,pw,768)
  每個遮罩:cov_p = 該 patch 的 196 個像素落在遮罩內的比例(在 H14xW14 上算,與 patch 網格對齊)
            v = sum(cov_p * f_p) / sum(cov_p)   ← cov>0 納入 + 面積加權,再 L2
  依據:RESULT_gtmask_cluster_compare.md(60場 GT 遮罩)面積加權 v-measure 0.873 > 等權 0.831(+0.042);
        RESULT_dino_patch_coverage.md:原版 INTER_NEAREST 有 19.0% 特徵權重來自覆蓋<50% 的 patch。
  ⚠ 不做 F_BG 去偏:DINO 不摳圖、不填灰,沒有填充色偏置可扣 → 呼叫端請設 DEBIAS=0。

快取:存 <view_dir>/dino_donut_feats.npy (N x 768,對齊 sorted(masks/*.png) 全遮罩,非 kept = nan),
      與 clip_donut_feats.npy 同存法 → 可被 masks.mask_feats(vd, 檔名) 以遮罩檔名查。
      FORCE_FEAT=1 強制重算。

lazy-load:import 本模組不載 DINOv2;只有真的要算特徵才載(仿 mask_clip_cluster 的做法)。
"""
import os
from pathlib import Path

import numpy as np
import cv2
import torch

PATCH = 14                                  # dinov2_vitb14
DIM = 768
DONUT_FEAT = "dino_donut_feats.npy"         # 甜甜圈遮罩的 DINO 特徵(對應 clip_donut_feats.npy)
RAW_FEAT = "dino_feats.npy"                 # 原始遮罩的 DINO 特徵(對應 clip_mean_feats.npy)
DMEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
DSTD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_dev = "cuda" if torch.cuda.is_available() else "cpu"
_model = None


def _dino():
    """lazy-load:首次要算特徵時才載入模型。"""
    global _model
    if _model is None:
        _model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14',
                                verbose=False).to(_dev).eval()
    return _model


@torch.no_grad()
def dino_feats(rgb, ms):
    """整圖跑一次 DINOv2,對 ms 內每個遮罩回 768d 特徵(cov>0 納入 + 面積加權 + L2);無交集回 None。"""
    H, W = rgb.shape[:2]
    H14, W14 = (H // PATCH) * PATCH, (W // PATCH) * PATCH
    ph, pw = H14 // PATCH, W14 // PATCH
    im = cv2.resize(rgb, (W14, H14))
    x = (torch.from_numpy(im).permute(2, 0, 1).float() / 255 - DMEAN) / DSTD
    f = _dino().forward_features(x[None].to(_dev))['x_norm_patchtokens'][0]
    fmap = f.reshape(ph, pw, -1)
    out = []
    for m in ms:
        s14 = cv2.resize(m.astype(np.uint8), (W14, H14), interpolation=cv2.INTER_NEAREST)
        cov = s14.reshape(ph, PATCH, pw, PATCH).sum(axis=(1, 3)) / float(PATCH * PATCH)
        if cov.sum() <= 0:
            out.append(None); continue
        covt = torch.from_numpy(cov).float().to(_dev)
        v = (fmap * covt[:, :, None]).sum((0, 1)) / covt.sum()
        out.append((v / (v.norm() + 1e-9)).cpu().numpy().astype(np.float32))
    return out


def _cached(view_dir, rgb, ms, names, feat_file):
    """共用快取邏輯:與 donut_feats 完全相同的存法(對全遮罩存,非 kept 留 nan,依檔名查)。"""
    vd = Path(view_dir)
    fp = vd / feat_file
    allnames = [q.name for q in sorted((vd / "masks").glob("mask_*.png"))]
    if fp.is_file() and os.environ.get("FORCE_FEAT", "") != "1":
        arr = np.load(fp)
        if len(arr) == len(allnames):
            d = {allnames[i]: (None if np.isnan(arr[i]).any() else arr[i]) for i in range(len(allnames))}
            return {n: d.get(n) for n in names}
    name2m = {names[i]: ms[i] for i in range(len(names))}
    out_arr = np.full((len(allnames), DIM), np.nan, np.float32)
    idx = {n: i for i, n in enumerate(allnames)}
    kept_ms, kept_pos = [], []
    for n in names:
        if n in idx:
            kept_ms.append(name2m[n]); kept_pos.append(idx[n])
    if kept_ms:
        for pos, f in zip(kept_pos, dino_feats(rgb, kept_ms)):
            if f is not None:
                out_arr[pos] = np.asarray(f, np.float32)
    np.save(fp, out_arr)
    return {n: (None if (n not in idx or np.isnan(out_arr[idx[n]]).any()) else out_arr[idx[n]])
            for n in names}


def dino_donut_feats(view_dir, rgb, ms_donut, names):
    """甜甜圈遮罩的 DINO 特徵。介面/存法與 voxel_sem_cluster_donut.donut_feats 相同。"""
    return _cached(view_dir, rgb, ms_donut, names, DONUT_FEAT)


def dino_raw_feats(view_dir, rgb, ms, names):
    """原始(未挖洞)遮罩的 DINO 特徵。介面/存法同上,檔名 dino_feats.npy。"""
    return _cached(view_dir, rgb, ms, names, RAW_FEAT)
