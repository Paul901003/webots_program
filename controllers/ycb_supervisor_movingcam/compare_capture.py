#!/usr/bin/env python3
"""compare_capture.py — 比對「手臂移動拍攝」vs「相機瞬移拍攝」同視角的 RGB + depth 差異。

兩邊都用 view_el{el}_az{az} 命名 → 依檔名配對同一視角，逐張比 RGB 與 depth。

用法（webots_visual_hull 環境）:
  compare_capture.py <armmove_scene_dir> <movingcam_scene_dir> [out_dir]
例:
  /home/cho/.pyenv/versions/webots_visual_hull/bin/python3 \
    controllers/ycb_supervisor_movingcam/compare_capture.py \
    data/cmp_armmove/multi_n3/n3_scene0001 \
    data/cmp_movingcam/multi_n3/n3_scene0001
"""
import csv
import glob
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_FP = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
try:
    font_manager.fontManager.addfont(_FP)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_FP).get_name()
except Exception:
    pass

DEPTH_MIN, DEPTH_MAX = 0.05, 3.0   # 有效深度範圍(m);與 D455 maxRange 一致


def views_in(d):
    """回傳 {view_name: stem_path}，view_name 如 view_el30_az135。"""
    out = {}
    for p in glob.glob(os.path.join(d, "view_el*.png")):
        b = os.path.basename(p)
        if b.endswith("_depth.png"):
            continue
        out[b[:-4]] = os.path.join(d, b[:-4])
    return out


def load_rgb(stem):
    img = cv2.imread(stem + ".png")
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_depth(stem):
    f = stem + "_depth.npy"
    return np.load(f) if os.path.exists(f) else None


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    arm_dir, cam_dir = sys.argv[1], sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "data/eval/_diag/arm_vs_movingcam"
    os.makedirs(out_dir, exist_ok=True)
    scene = os.path.basename(arm_dir.rstrip("/"))

    va, vc = views_in(arm_dir), views_in(cam_dir)
    common = sorted(set(va) & set(vc))
    print(f"[cmp] {scene}: 手臂 {len(va)} 視角, 相機 {len(vc)} 視角, 共同 {len(common)} 個")
    if not common:
        sys.exit("無共同視角(檢查兩邊是否同場景/同視角集)")

    rows_summary = []
    n = len(common)
    fig, ax = plt.subplots(n, 6, figsize=(20, 3.0 * n))
    if n == 1:
        ax = ax.reshape(1, 6)
    for i, vn in enumerate(common):
        ra, rc = load_rgb(va[vn]), load_rgb(vc[vn])
        da, dc = load_depth(va[vn]), load_depth(vc[vn])

        # RGB 差異
        rgb_mad = rgb_pct = float("nan")
        if ra is not None and rc is not None and ra.shape == rc.shape:
            diff = np.abs(ra.astype(int) - rc.astype(int)).astype(np.uint8)
            rgb_mad = float(diff.mean())
            rgb_pct = float((diff.max(2) > 20).mean()) * 100
        else:
            diff = np.zeros_like(ra) if ra is not None else None

        # Depth 差異(只算兩邊都有效的像素)
        d_mean = d_p95 = d_max = float("nan"); valid_pct = 0.0; ddiff = None
        if da is not None and dc is not None and da.shape == dc.shape:
            mask = (np.isfinite(da) & np.isfinite(dc)
                    & (da > DEPTH_MIN) & (da < DEPTH_MAX)
                    & (dc > DEPTH_MIN) & (dc < DEPTH_MAX))
            ddiff = np.abs(da - dc)
            valid_pct = float(mask.mean()) * 100
            if mask.any():
                vals = ddiff[mask] * 1000.0   # mm
                d_mean = float(vals.mean()); d_p95 = float(np.percentile(vals, 95)); d_max = float(vals.max())

        rows_summary.append([scene, vn, round(rgb_mad, 2), round(rgb_pct, 2),
                             round(d_mean, 2), round(d_p95, 2), round(d_max, 2), round(valid_pct, 1)])

        # 畫:手臂RGB | 相機RGB | RGB差×3 | 手臂depth | 相機depth | depth差(mm)
        def show(a, img, title, **kw):
            a.imshow(img, **kw); a.set_title(title, fontsize=8); a.axis("off")
        if ra is not None: show(ax[i, 0], ra, f"手臂 {vn.replace('view_','')}")
        if rc is not None: show(ax[i, 1], rc, "相機瞬移")
        if diff is not None: show(ax[i, 2], (diff.astype(float) * 3).clip(0, 255).astype(np.uint8),
                                  f"RGB差×3 MAD={rgb_mad:.1f}")
        if da is not None: show(ax[i, 3], da, "手臂 depth", cmap="turbo", vmin=DEPTH_MIN, vmax=DEPTH_MAX)
        if dc is not None: show(ax[i, 4], dc, "相機 depth", cmap="turbo", vmin=DEPTH_MIN, vmax=DEPTH_MAX)
        if ddiff is not None: show(ax[i, 5], (ddiff * 1000).clip(0, 50), f"depth差mm 均={d_mean:.1f}",
                                  cmap="hot", vmin=0, vmax=50)

    fig.suptitle(f"{scene}  手臂移動 vs 相機瞬移（RGB + depth）", fontsize=13)
    fig.tight_layout()
    png = os.path.join(out_dir, f"{scene}.png")
    fig.savefig(png, dpi=85); plt.close(fig)

    csvp = os.path.join(out_dir, f"{scene}.csv")
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene", "view", "rgb_MAD", "rgb_diff%",
                    "depth_mean_mm", "depth_p95_mm", "depth_max_mm", "depth_valid%"])
        w.writerows(rows_summary)

    print(f"\n{'view':>20}{'rgbMAD':>8}{'rgb%':>7}{'dMean_mm':>9}{'dP95_mm':>8}{'dMax_mm':>8}")
    for r in rows_summary:
        print(f"{r[1]:>20}{r[2]:>8}{r[3]:>7}{r[4]:>9}{r[5]:>8}{r[6]:>8}")
    arr = np.array([[r[2], r[4]] for r in rows_summary if r[2] == r[2]], float)
    if len(arr):
        print(f"\n平均 RGB MAD={arr[:,0].mean():.2f}  平均 depth 差={arr[:,1].mean():.2f} mm")
    print(f"→ 圖: {png}\n→ 表: {csvp}")


if __name__ == "__main__":
    main()
