#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""make_vlm_inputs_batch.py — 批次產出 VLM 測試用的裁切影像(擴大測試版)。

★ 新檔。承 make_vlm_inputs.py(3 案例手動版)的發現:crop 讓 LLaVA 的堆疊偵測 0/3 → 3/3,
   maskbg 無效(驗證 DISC Sec.II-C 的 domain shift 說法)。本版只產 crop,擴大到 90 場。

視角選擇(★可部署,不用 GT):在 el30(最低仰角)的 A-3 視角中,取 kept_object_masks
   面積總和最大者。低仰角較能看出上下堆疊(3 案例觀察,待本次驗證)。
裁切:kept_object_masks(扣手臂剪影)聯集的外接框 + margin 40px,不做任何塗色。

輸出: vlm_inputs_batch/<scene>.png 與 index.json(含場景、視角、是否有 on 對、on 對內容)
用法: ./make_vlm_inputs_batch.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "srp" / "io")); sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import masks as MK          # noqa: E402
import viewpoints as VP     # noqa: E402
from stack_leak_nosep import on_pairs   # noqa: E402

SAM = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "mobilesamv2_fast")))
CAP = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures_fast")))
ARM = Path(os.environ.get("ARM_MASK_ROOT", str(REPO / "data" / "eval" / "srp_arm_masks")))
HULL = REPO / "data" / "eval" / "srp_hull_mv2_v12_am1"
ARM_THR = 0.5
MARGIN = 40
GEX = {"skillet_lid", "windex_bottle", "colored_wood_blocks", "dice"}
OUT = HERE / "vlm_inputs_batch"


def kept_union(sc, vn, shape):
    vd = SAM / sc / vn
    if not (vd / "masks").is_dir():
        return None
    ms = [m for m, _ in MK.kept_object_masks(vd)]
    ap = ARM / sc / f"{vn}_arm.png"
    arm = (cv2.imread(str(ap), 0) > 127) if ap.is_file() else None
    if arm is not None:
        ms = [m for m in ms if (m & arm).sum() / max(int(m.sum()), 1) < ARM_THR]
    if not ms:
        return None
    u = np.zeros(shape, bool)
    for m in ms:
        u |= m
    return u


def main():
    import glob
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = []
    for g in ["stack3", "stack4", "stack5"]:
        scenes += [Path(p).parent.name for p in glob.glob(str(HULL / f"{g}_scene*/hull.npz"))]
    occ = sorted(Path(p).parent.name for p in glob.glob(str(HULL / "occ*_scene*/hull.npz")))[:30]
    scenes = sorted(set(scenes)) + occ
    el30 = [v for v in sorted(VP.selected_view_names(12)) if "el30" in v]
    idx = []
    for sc in scenes:
        best = None
        for vn in el30:                        # ★ 可部署:低仰角中取保留遮罩面積最大者
            p = CAP / f"multi_{sc.split('_')[0]}" / sc / f"{vn}.png"
            if not p.is_file():
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            u = kept_union(sc, vn, img.shape[:2])
            if u is None:
                continue
            a = int(u.sum())
            if best is None or a > best[0]:
                best = (a, vn, img, u)
        if best is None:
            print(f"[skip] {sc}"); continue
        _, vn, img, u = best
        H, W = img.shape[:2]
        ys, xs = np.nonzero(u)
        y0 = max(int(ys.min()) - MARGIN, 0); y1 = min(int(ys.max()) + MARGIN, H)
        x0 = max(int(xs.min()) - MARGIN, 0); x1 = min(int(xs.max()) + MARGIN, W)
        cv2.imwrite(str(OUT / f"{sc}.png"), img[y0:y1, x0:x1])
        ps = [(a_, b_) for (a_, b_) in on_pairs(sc) if a_ not in GEX and b_ not in GEX]
        idx.append({"scene": sc, "view": vn, "group": sc.split("_")[0],
                    "has_on": bool(ps), "on_pairs": ps,
                    "crop_wh": [x1 - x0, y1 - y0]})
    (OUT / "index.json").write_text(json.dumps(idx, ensure_ascii=False, indent=1))
    n_on = sum(1 for r in idx if r["has_on"])
    print(f"[輸出] {OUT}  共 {len(idx)} 場;有 on 對 {n_on}、無 {len(idx)-n_on}")


if __name__ == "__main__":
    main()
