#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""viz_removed_masks.py — 把每視角「被去除的 SAM 遮罩」疊在 RGB 上,12 視角 montage,供目視核對過濾。

顏色:綠=進分群的物體遮罩(kept 且非手臂dropped)、紅=背景/桌面去除(不在 kept_object_masks)、
      藍=手臂/夾爪 drop(kept 但 ≥ARM_DROP_THR 落在 FK 手臂剪影 srp_arm_masks 內)。
背景去除 = masks.py kept_object_masks(面積/碰邊/桌背景參考);手臂 = srp_arm_masks(FK 剪影)。
用法: SAM_ROOT=$PWD/data/eval/mobilesamv2_fast CAPTURES_ROOT=$PWD/data/captures_fast \
      ./viz_removed_masks.py <scene...> [--arm-thr 0.5] [--out <dir>]
輸出: <out>/<scene>_removed.png(預設 out=srp/stage2_instances/removed_viz)
"""
import os
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "io"))
import masks as MK        # noqa: E402
import viewpoints as VP   # noqa: E402

REPO = HERE.parent.parent
SAM = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "mobilesamv2_fast")))
CAP = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures_fast")))
ARM = Path(os.environ.get("ARM_MASK_ROOT", str(REPO / "data" / "eval" / "srp_arm_masks")))
GREEN = np.array([0, 200, 0]); RED = np.array([220, 30, 30]); BLUE = np.array([40, 90, 230])


def view_overlay(sc, vn, arm_thr):
    vd = SAM / sc / vn
    group = sc.split("_")[0]
    rgbp = CAP / f"multi_{group}" / sc / f"{vn}.png"           # RGB 在 captures,不在 SAM 目錄
    if not rgbp.is_file() or not (vd / "masks").is_dir():
        return None
    rgb = cv2.cvtColor(cv2.imread(str(rgbp)), cv2.COLOR_BGR2RGB).astype(np.float32)
    H, W = rgb.shape[:2]
    kept = {fn for _, fn in MK.kept_object_masks(vd)}
    ap = ARM / sc / f"{vn}_arm.png"
    arm = (cv2.imread(str(ap), 0) > 127) if ap.is_file() else None
    ov = rgb.copy()
    n_obj = n_bg = n_arm = 0
    for mp in sorted((vd / "masks").glob("mask_*.png")):
        m = cv2.imread(str(mp), 0) > 127
        if not m.any():
            continue
        if mp.name not in kept:
            col = RED; n_bg += 1                                  # 背景/桌面去除
        elif arm is not None and (m & arm).sum() / max(int(m.sum()), 1) >= arm_thr:
            col = BLUE; n_arm += 1                                # 手臂/夾爪 drop
        else:
            col = GREEN; n_obj += 1                               # 進分群
        ov[m] = 0.5 * ov[m] + 0.5 * col
    if arm is not None:                                          # 手臂剪影輪廓(白)
        cnts, _ = cv2.findContours(arm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (255, 255, 255), 2)
    ov = cv2.resize(ov.astype(np.uint8), (W // 2, H // 2))
    cv2.putText(ov, f"{vn}  obj{n_obj} bg{n_bg} arm{n_arm}", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return ov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--arm-thr", type=float, default=0.5)
    ap.add_argument("--out", default=str(HERE / "removed_viz"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for sc in a.scenes:
        tiles = []
        for vn in sorted(VP.selected_view_names(12)):
            t = view_overlay(sc, vn, a.arm_thr)
            if t is not None:
                tiles.append(t)
        if not tiles:
            print(f"[skip] {sc} 無視角"); continue
        h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]
        cols = 4; rows = (len(tiles) + cols - 1) // cols
        while len(tiles) < rows * cols:
            tiles.append(np.zeros((h, w, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
        p = out / f"{sc}_removed.png"
        cv2.imwrite(str(p), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"[存] {p}  (綠=物體 紅=背景 藍=手臂夾爪;白框=FK手臂剪影)")


if __name__ == "__main__":
    main()
