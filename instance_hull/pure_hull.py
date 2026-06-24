#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""pure_hull.py — class-agnostic 純角錐 visual hull 關聯(不用 GT、不用標籤、不用深度)。

流程:
  ① 每視角前景 = 所有 SAM 遮罩聯集(預設沿用 面積>30%/碰邊界 濾除來定義 silhouette;
     --no-bg-filter 可關掉,但地板會被雕進來)。
  ② 角錐雕刻:體素落在前景的視角數 >= keep_frac×N → 保留(多視角輪廓交集 = visual hull)。
  ③ 3D 連通元件(6-鄰接,純幾何,不看遮罩號)→ 每元件 = 一個物體。
  ④ 元件 >= min_vox → instance;各視角遮罩 = 重投影與該元件重疊夠的 SAM 遮罩(供貼標籤/評估)。
輸出: data/eval/pure_hull/<scene>/instances.json(schema 同其他關聯方法)。
重用 eval_clip_match 的 load_views/project + associate_voxel 的 WS/常數。需 webots_visual_hull。

用法: ./instance_hull/pure_hull.py 1 3 4 5 [--voxel 0.015] [--keep-frac 0.6] [--min-vox 8]
       [--assign-iou 0.1] [--no-bg-filter]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_clip_match as E
import associate_voxel as av

OUT_ROOT = E.EVAL_ROOT / "pure_hull"
MAXFRAC = 0.30
BORDER = 2


def touches_border(b):
    return bool(b[:BORDER].any() or b[-BORDER:].any() or b[:, :BORDER].any() or b[:, -BORDER:].any())


def load_view_masks(scene, vn, no_bg):
    """回傳 [(bool_mask, filename)],已(可選)濾背景。"""
    out = []
    for mp in sorted((E.SAM_ROOT / scene / vn / "masks").glob("mask_*.png")):
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        b = m > 127
        H, W = b.shape
        a = int(b.sum())
        if a == 0:
            continue
        if not no_bg and (a > MAXFRAC * H * W or touches_border(b)):
            continue
        out.append((b, mp.name))
    return out


def process_scene(scene, args):
    views = E.load_views(scene)
    if len(views) < 2:
        print(f"[skip] {scene}: views<2"); return None
    # 體素格
    xs = np.arange(*av.WS_X, args.voxel); ys = np.arange(*av.WS_Y, args.voxel); zs = np.arange(*av.WS_Z, args.voxel)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    shape = gx.shape
    P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    proj = {vn: E.project(P, v) for vn, v in views.items()}

    # 各視角:遮罩清單 + 前景聯集
    vmasks = {vn: load_view_masks(scene, vn, args.no_bg) for vn in views}
    cnt = np.zeros(len(P), np.int32); nv = 0
    for vn in views:
        fg = None
        for b, _ in vmasks[vn]:
            fg = b if fg is None else (fg | b)
        if fg is None:
            continue
        ui, wi, inb = proj[vn]
        ins = np.zeros(len(P), bool); ins[inb] = fg[wi[inb], ui[inb]]
        cnt += ins; nv += 1
    if nv < 2:
        print(f"[skip] {scene}: 有效視角<2"); return None
    keep = cnt >= max(2, int(np.ceil(args.keep_frac * nv)))
    if not keep.any():
        print(f"[skip] {scene}: 雕殼為空"); return None

    # 3D 連通元件(6-鄰接)
    occ = keep.reshape(shape)
    lab, n = ndimage.label(occ, structure=ndimage.generate_binary_structure(3, 1))
    labflat = lab.ravel()

    instances = []
    for c in range(1, n + 1):
        vidx = np.where(labflat == c)[0]
        if len(vidx) < args.min_vox:
            continue
        center = P[vidx].mean(axis=0)
        occ_c = np.zeros(len(P), bool); occ_c[vidx] = True
        # 各視角:重投影元件(填補)→ 找重疊夠的 SAM 遮罩
        per_view = {}
        Pc = P[vidx]
        for vn, v in views.items():
            rm = E.reproject_occ(Pc, v, args.voxel)
            if not rm.any():
                continue
            files = []
            for b, fn in vmasks[vn]:
                inter = int((b & rm).sum())
                if inter == 0:
                    continue
                # 該遮罩大部分落在元件投影內,或元件投影大部分被該遮罩覆蓋
                if inter / int(b.sum()) >= args.assign_iou or inter / int(rm.sum()) >= args.assign_iou:
                    files.append(fn)
            if files:
                per_view[vn] = sorted(files)
        if len(per_view) < 2:
            continue
        instances.append({"center": [round(float(x), 4) for x in center],
                          "n_vox": int(len(vidx)), "support": len(per_view),
                          "masks": per_view})

    instances.sort(key=lambda a: -a["n_vox"])
    out_dir = OUT_ROOT / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "instances.json").write_text(json.dumps(
        {"scene": scene, "voxel": args.voxel,
         "centers": [i["center"] for i in instances], "instances": instances},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{scene}] 視角{nv} 元件{n} → instances {len(instances)}")
    return len(instances)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["n3_scene0001"])
    ap.add_argument("--voxel", type=float, default=0.015)
    ap.add_argument("--keep-frac", type=float, default=0.6, dest="keep_frac")
    ap.add_argument("--min-vox", type=int, default=8, dest="min_vox")
    ap.add_argument("--assign-iou", type=float, default=0.1, dest="assign_iou",
                    help="SAM 遮罩有此比例落在元件投影內就算屬於它")
    ap.add_argument("--no-bg-filter", action="store_true", dest="no_bg",
                    help="關閉 面積/邊界 背景濾除(地板會被雕入)")
    args = ap.parse_args()
    scenes = E.resolve_scenes(args.scenes or ["n3_scene0001"])
    tot = 0
    for i, sc in enumerate(scenes, 1):
        print(f"[{i}/{len(scenes)}]", end=" ")
        try:
            r = process_scene(sc, args)
            if r:
                tot += r
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")
    print(f"\n== pure_hull: {len(scenes)} 場景, 共 {tot} instances → data/eval/pure_hull/ ==")


if __name__ == "__main__":
    main()
