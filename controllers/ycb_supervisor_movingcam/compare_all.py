#!/usr/bin/env python3
"""compare_all.py — 批次比較 arm(手臂移動) vs cam(相機瞬移) 全場景的 RGB + depth 差異。

依 el/az 檔名配對兩邊共同視角(手臂 12 ⊂ 相機 39 → 取重疊 12)，逐場景算：
  RGB: MAD(平均絕對差)、%diff(>20 的像素比例)
  depth: 兩邊都有效像素的 |Δ|(mm) 之 mean/median/p95、有效率
輸出每場景一列的總表 CSV + 各組/整體統計(印出)。不畫圖(要看圖用 compare_capture.py 單場景)。

用法(webots_visual_hull 環境):
  compare_all.py [out_csv]
預設輸出 data/eval/_diag/arm_vs_cam/summary_all.csv
"""
import csv
import glob
import os
import sys

import numpy as np
import cv2

ARM_ROOT = "data/captures"
CAM_ROOT = "data/captures_multicam"
GROUPS = ["n3", "n4", "n5", "occ3", "occ4", "occ5", "stack3", "stack4", "stack5"]
DMIN, DMAX = 0.05, 3.0


def views(d):
    out = {}
    for p in glob.glob(os.path.join(d, "view_el*.png")):
        b = os.path.basename(p)
        if b.endswith("_depth.png"):
            continue
        out[b[:-4]] = os.path.join(d, b[:-4])
    return out


def scene_stats(arm_dir, cam_dir):
    va, vc = views(arm_dir), views(cam_dir)
    common = sorted(set(va) & set(vc))
    if not common:
        return None
    rgb_mad, rgb_pct = [], []
    dmean, dmed, dp95, dvalid = [], [], [], []
    for vn in common:
        ra = cv2.imread(va[vn] + ".png")
        rc = cv2.imread(vc[vn] + ".png")
        if ra is not None and rc is not None and ra.shape == rc.shape:
            diff = np.abs(ra.astype(np.int16) - rc.astype(np.int16))
            rgb_mad.append(float(diff.mean()))
            rgb_pct.append(float((diff.max(2) > 20).mean()) * 100)
        fa, fc = va[vn] + "_depth.npy", vc[vn] + "_depth.npy"
        if os.path.exists(fa) and os.path.exists(fc):
            da, dc = np.load(fa), np.load(fc)
            if da.shape == dc.shape:
                m = (np.isfinite(da) & np.isfinite(dc)
                     & (da > DMIN) & (da < DMAX) & (dc > DMIN) & (dc < DMAX))
                if m.any():
                    v = np.abs(da[m] - dc[m]) * 1000.0
                    dmean.append(float(v.mean()))
                    dmed.append(float(np.median(v)))
                    dp95.append(float(np.percentile(v, 95)))
                    dvalid.append(float(m.mean()) * 100)
    return {
        "n": len(common),
        "rgb_mad": np.mean(rgb_mad) if rgb_mad else float("nan"),
        "rgb_pct": np.mean(rgb_pct) if rgb_pct else float("nan"),
        "d_mean": np.mean(dmean) if dmean else float("nan"),
        "d_med": np.mean(dmed) if dmed else float("nan"),
        "d_p95": np.mean(dp95) if dp95 else float("nan"),
        "d_valid": np.mean(dvalid) if dvalid else float("nan"),
    }


def main():
    out_csv = sys.argv[1] if len(sys.argv) > 1 else "data/eval/_diag/arm_vs_cam/summary_all.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows = []
    for g in GROUPS:
        for arm_dir in sorted(glob.glob(f"{ARM_ROOT}/multi_{g}/{g}_scene*")):
            sc = os.path.basename(arm_dir)
            cam_dir = f"{CAM_ROOT}/multi_{g}/{sc}"
            if not os.path.isdir(cam_dir):
                continue
            s = scene_stats(arm_dir, cam_dir)
            if s is None:
                continue
            rows.append((g, sc, s))
        done = sum(1 for r in rows if r[0] == g)
        print(f"[{g}] 已比 {done} 場景", flush=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "scene", "n_common", "rgb_MAD", "rgb_diff%",
                    "depth_mean_mm", "depth_median_mm", "depth_p95_mm", "depth_valid%"])
        for g, sc, s in rows:
            w.writerow([g, sc, s["n"], f"{s['rgb_mad']:.3f}", f"{s['rgb_pct']:.3f}",
                        f"{s['d_mean']:.3f}", f"{s['d_med']:.3f}", f"{s['d_p95']:.3f}",
                        f"{s['d_valid']:.2f}"])

    # 統計
    def col(rows, key):
        return np.array([r[2][key] for r in rows if r[2][key] == r[2][key]], float)
    print("\n===== 各組平均 (RGB MAD / depth mean / depth median / depth p95, mm) =====")
    print(f"{'組':8s}{'場景':>5}{'RGB_MAD':>9}{'d_mean':>9}{'d_med':>8}{'d_p95':>8}")
    for g in GROUPS:
        gr = [r for r in rows if r[0] == g]
        if not gr:
            continue
        print(f"{g:8s}{len(gr):>5}{col(gr,'rgb_mad').mean():>9.2f}"
              f"{col(gr,'d_mean').mean():>9.2f}{col(gr,'d_med').mean():>8.2f}{col(gr,'d_p95').mean():>8.2f}")
    print(f"{'總計':8s}{len(rows):>5}{col(rows,'rgb_mad').mean():>9.2f}"
          f"{col(rows,'d_mean').mean():>9.2f}{col(rows,'d_med').mean():>8.2f}{col(rows,'d_p95').mean():>8.2f}")
    print(f"\n→ 總表: {out_csv}  ({len(rows)} 場景)")


if __name__ == "__main__":
    main()
