#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""make_vlm_inputs.py — 為 VLM 測試產出三種輸入影像(單一變因:輸入形式)。

★ 新檔。動機:LLaVA 在原圖上把垂直堆疊讀成水平並排(RESULT_llava_scene_desc/missing_probe),
   懷疑畫面中的夾爪與背景牆干擾。但 DISC 論文 Sec.II-C 明確指出【遮罩去背會造成 domain shift、
   降低 zero-shot 能力】,所以不能只做去背,要三種並列比較。

三種輸入(都【不用 GT】,用管線現成的 kept_object_masks + 手臂剪影過濾,故可部署):
  orig    : 原圖不動(對照基準)
  crop    : 裁切到「保留遮罩聯集」的外接框 + margin(去掉下方夾爪、上方背景牆;不做任何遮罩塗色)
  maskbg  : 保留遮罩聯集內的像素,其餘塗中性灰(驗證 DISC 說的 domain shift)

輸出: srp/stage4_probe/vlm_inputs/<scene>__<view>__{orig,crop,maskbg}.png
用法: ./make_vlm_inputs.py   (案例寫死在 CASES,與 llava_* 腳本一致)
env : SAM_ROOT CAPTURES_ROOT ARM_MASK_ROOT
"""
import os
import sys
from pathlib import Path

import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "srp" / "io"))
import masks as MK   # noqa: E402

SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "mobilesamv2_fast")))
CAP = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures_fast")))
ARM = Path(os.environ.get("ARM_MASK_ROOT", str(REPO / "data" / "eval" / "srp_arm_masks")))
ARM_DROP_THR = float(os.environ.get("ARM_DROP_THR", "0.5"))
OUT = HERE / "vlm_inputs"
MARGIN = 40          # crop 外擴像素,避免切到物體邊緣
GRAY = (124, 116, 104)   # 中性灰(接近 ImageNet/CLIP 均值)

CASES = [("stack3_scene0005", "view_el30_az135"),
         ("stack4_scene0007", "view_el45_az225"),
         ("stack4_scene0010", "view_el30_az195")]


def keep_union(sc, vn, shape):
    """管線口徑的「物體像素」聯集:kept_object_masks 再扣掉落在手臂剪影內的遮罩。"""
    vd = SAM_ROOT / sc / vn
    if not (vd / "masks").is_dir():
        return None
    km = MK.kept_object_masks(vd)
    ms = [m for m, _ in km]
    ap = ARM / sc / f"{vn}_arm.png"
    arm = (cv2.imread(str(ap), 0) > 127) if ap.is_file() else None
    if arm is not None:
        ms = [m for m in ms if (m & arm).sum() / max(int(m.sum()), 1) < ARM_DROP_THR]
    if not ms:
        return None
    u = np.zeros(shape, bool)
    for m in ms:
        u |= m
    return u


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for sc, vn in CASES:
        p = CAP / f"multi_{sc.split('_')[0]}" / sc / f"{vn}.png"
        img = cv2.imread(str(p))
        if img is None:
            print(f"[skip] 缺影像 {p}"); continue
        H, W = img.shape[:2]
        u = keep_union(sc, vn, (H, W))
        if u is None:
            print(f"[skip] {sc}/{vn} 無保留遮罩"); continue
        ys, xs = np.nonzero(u)
        y0 = max(int(ys.min()) - MARGIN, 0); y1 = min(int(ys.max()) + MARGIN, H)
        x0 = max(int(xs.min()) - MARGIN, 0); x1 = min(int(xs.max()) + MARGIN, W)
        base = f"{sc}__{vn}"
        cv2.imwrite(str(OUT / f"{base}__orig.png"), img)
        cv2.imwrite(str(OUT / f"{base}__crop.png"), img[y0:y1, x0:x1])
        mb = img.copy(); mb[~u] = GRAY
        cv2.imwrite(str(OUT / f"{base}__maskbg.png"), mb)
        print(f"[{sc}/{vn}] 保留像素 {int(u.sum())}  crop={x1-x0}x{y1-y0}(原 {W}x{H})", flush=True)
    print(f"[輸出] {OUT}")


if __name__ == "__main__":
    main()
