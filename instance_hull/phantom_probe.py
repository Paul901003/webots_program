#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""phantom_probe.py — 實證:三個分數能否把「真幻影 hull」跟「真實 hull」分開。

真實/幻影標記:hull 對全部 GT 的最佳 3D IoU,best_iou>=0.5=真實,<0.5=真幻影。
三個候選分數(每 hull):
  ① see_through  : 體素投到「全部視角」,落在背景(所有前景遮罩之外)的視角比例平均。高=幻影。
  ② self_consist : hull 重投影到它的來源視角 vs 該視角來源遮罩聯集 的 IoU 平均。低=幻影。
  ③ exclusive    : hull 重投影遮罩中「不被其他 hull 重投影覆蓋」的比例平均(explaining-away)。低=冗餘/幻影。
輸出每方法:真實 vs 幻影 的分數分佈 + AUC(分得開的程度,0.5=無區別、→1 完全分開)。
重用 eval_clip_match 的雕殼/投影/重投影。需 webots_visual_hull。
用法: ./instance_hull/phantom_probe.py 3 4 5 --roots instance_hull instance_hull_voxel_ml epipolar
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_clip_match as E
import associate_voxel as av


def fg_union(scene, views):
    """每視角:所有 SAM 前景遮罩聯集(bool)。"""
    out = {}
    for vn in views:
        seg = None
        for mp in sorted((E.SAM_ROOT / scene / vn / "masks").glob("mask_*.png")):
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            b = m > 127
            seg = b if seg is None else (seg | b)
        if seg is not None:
            out[vn] = seg
    return out


def src_union(inst, scene, vn):
    """instance 在某視角用到的來源遮罩聯集。"""
    seg = None
    for f in inst["masks"].get(vn, []):
        m = cv2.imread(str(E.SAM_ROOT / scene / vn / "masks" / f), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        b = m > 127
        seg = b if seg is None else (seg | b)
    return seg


def process(root, scene, pitch=0.01, keep_frac=0.6):
    views = E.load_views(scene)
    if len(views) < 2:
        return []
    ij = E.EVAL_ROOT / root / scene / "instances.json"
    if not ij.is_file():
        return []
    instances = json.loads(ij.read_text())["instances"]
    gt = E.load_gt_masks(scene)
    gt_names = [n for n in gt if n in E.TXT_IDX]
    if not gt_names:
        return []
    xs = np.arange(*av.WS_X, pitch); ys = np.arange(*av.WS_Y, pitch); zs = np.arange(*av.WS_Z, pitch)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    proj = {vn: E.project(P, v) for vn, v in views.items()}
    gt_occ = {nm: E.carve_from_masks(gt[nm], views, proj, P, keep_frac) for nm in gt_names}
    gt_occ = {k: v for k, v in gt_occ.items() if v is not None}
    if not gt_occ:
        return []
    fg = fg_union(scene, views)

    recon = [E.recon_occupancy(inst, scene, views, proj, P, keep_frac) for inst in instances]
    valid = [i for i in range(len(instances)) if recon[i] is not None]
    # 預算各 hull 各視角重投影遮罩(exclusive 用)
    reproj = {}  # i -> {vn: bool mask}
    for i in valid:
        Pocc = P[recon[i]]
        reproj[i] = {vn: E.reproject_occ(Pocc, views[vn], pitch) for vn in views}

    rows = []
    for i in valid:
        occ = recon[i]; Pocc = P[occ]
        best_iou = max(E.iou3(occ, gt_occ[nm]) for nm in gt_occ)
        is_phantom = int(best_iou < 0.5)

        # ① see_through:每體素投到所有視角,背景視角比例
        in_img = np.zeros(len(Pocc), np.int32); bg = np.zeros(len(Pocc), np.int32)
        for vn, v in views.items():
            X = (Pocc - v["C"]) @ v["R"] @ av.BODY_TO_OPENCV.T
            z = X[:, 2]; ok = z > 1e-6
            u = np.round(v["fx"] * X[:, 0] / np.where(ok, z, 1) + v["cx"]).astype(np.int64)
            w = np.round(v["fx"] * X[:, 1] / np.where(ok, z, 1) + v["cy"]).astype(np.int64)
            inb = ok & (u >= 0) & (u < v["W"]) & (w >= 0) & (w < v["H"])
            in_img += inb
            if vn in fg:
                hit = np.zeros(len(Pocc), bool)
                hit[inb] = fg[vn][w[inb], u[inb]]
                bg += (inb & ~hit)
        m = in_img > 0
        see_through = float(np.mean(bg[m] / in_img[m])) if m.any() else 0.0

        # ② self_consist:重投影 vs 來源遮罩聯集
        sc_ious = []
        for vn in inst_views(instances[i]):
            if vn not in views:
                continue
            su = src_union(instances[i], scene, vn)
            if su is None:
                continue
            sc_ious.append(E.iou3(reproj[i][vn], su))
        self_consist = float(np.mean(sc_ious)) if sc_ious else 0.0

        # ③ exclusive:重投影中不被其他 hull 覆蓋的比例
        ex = []
        for vn in views:
            ri = reproj[i][vn]
            a = int(ri.sum())
            if a == 0:
                continue
            others = np.zeros_like(ri)
            for j in valid:
                if j != i:
                    others |= reproj[j][vn]
            ex.append(int((ri & ~others).sum()) / a)
        exclusive = float(np.mean(ex)) if ex else 0.0

        rows.append({"method": root, "scene": scene, "hull": i, "best_iou": round(best_iou, 4),
                     "phantom": is_phantom, "see_through": round(see_through, 4),
                     "self_consist": round(self_consist, 4), "exclusive": round(exclusive, 4)})
    return rows


def inst_views(inst):
    return list(inst["masks"].keys())


def auc(real, phan, higher_is_phantom):
    """P(隨機幻影分數比隨機真實更像幻影)。"""
    real = np.asarray(real); phan = np.asarray(phan)
    if len(real) == 0 or len(phan) == 0:
        return float("nan")
    cnt = 0; tot = len(real) * len(phan)
    for p in phan:
        if higher_is_phantom:
            cnt += np.sum(p > real) + 0.5 * np.sum(p == real)
        else:
            cnt += np.sum(p < real) + 0.5 * np.sum(p == real)
    return cnt / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["3", "4", "5"])
    ap.add_argument("--roots", nargs="*", default=["instance_hull", "instance_hull_voxel_ml", "epipolar"])
    ap.add_argument("--csv", default=str(E.EVAL_ROOT / "phantom_probe.csv"))
    args = ap.parse_args()
    scenes = E.resolve_scenes(args.scenes)
    allrows = []
    for r in args.roots:
        n = 0
        for sc in scenes:
            try:
                rs = process(r, sc); allrows += rs; n += len(rs)
            except Exception as e:
                import traceback; traceback.print_exc(); print(f"[err] {r}/{sc}: {e}")
        print(f"{r}: {n} hull")
    import csv
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    print(f"→ {len(allrows)} 列 → {args.csv}\n")

    for r in args.roots:
        rs = [x for x in allrows if x["method"] == r]
        real = [x for x in rs if x["phantom"] == 0]
        phan = [x for x in rs if x["phantom"] == 1]
        print(f"===== {r}  (真實 {len(real)} / 幻影 {len(phan)}) =====")
        for score, hip in (("see_through", True), ("self_consist", False), ("exclusive", False)):
            rv = [x[score] for x in real]; pv = [x[score] for x in phan]
            a = auc(rv, pv, hip)
            print(f"  {score:<13} 真實均={np.mean(rv):.3f} 幻影均={np.mean(pv):.3f}  "
                  f"AUC={a:.3f}  ({'高=幻影' if hip else '低=幻影'})")
        print()


if __name__ == "__main__":
    main()
