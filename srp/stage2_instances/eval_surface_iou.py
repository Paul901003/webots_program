#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""eval_surface_iou — modal 表面 voxel 3D-IoU 找到率(鎖定評估,看結果前定義)。

共用 hull 表面 voxel(所有方法同一套):
  GT:每個表面 voxel 對「它可見(zbuffer 最前)」的視角投影,看落進哪個物體 modal 遮罩;
      多數決 → 該 voxel 的 GT 物體。GTsurf(o)=屬 o 的表面 voxel。無任何投票=背景。
  預測:該 root 的 instances.npz labels 在表面 voxel 上(避開 instances.json 遮罩檔問題)。
  配對:預測 instance ↔ GT 物體 以表面 voxel 3D-IoU 做 Hungarian;IoU≥門檻算找到。
排除:GT 物體排 GLOBAL_EXCLUDE(兩不相干:預測 instance 主屬排除物者不計幻影)。
記錄:per-scene found@{0.3,0.5,0.7}、mIoU、幻影;per-object 最佳 IoU 存 CSV。
用法: SAM_ROOT/CAPTURES 預設 fast。 ./eval_surface_iou.py --roots r1,r2 [scenes...]
"""
import sys
import os
import json
import argparse
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
from pycocotools import mask as cocomask
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import camera as cam
import viewpoints as VP
import labels as L
import cg_associate as CG

EVAL = REPO / "data" / "eval"
HULL = os.environ.get("HULL_ROOT_NAME", "srp_hull_mv2_v12_am1_photo")   # env 覆蓋:am1 vs am1_photo(無光雕)
CAP = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures_fast")))
GEX = {"skillet_lid", "windex_bottle", "colored_wood_blocks", "dice"}
TAUS = [0.3, 0.5, 0.7]
GROUPS = ["n3", "n4", "n5", "occ3", "occ4", "occ5", "stack3", "stack4", "stack5"]


def surface(o):
    s = np.zeros_like(o)
    s[1:-1, 1:-1, 1:-1] = o[1:-1, 1:-1, 1:-1] & ~(o[:-2, 1:-1, 1:-1] & o[2:, 1:-1, 1:-1] & o[1:-1, :-2, 1:-1] &
                                                  o[1:-1, 2:, 1:-1] & o[1:-1, 1:-1, :-2] & o[1:-1, 1:-1, 2:])
    return s


def modal_masks(sc, vn):
    ann = json.loads((L.label_dir(sc) / "actual" / "annotations.json").read_text())
    cat = {c["id"]: c["name"] for c in ann["categories"]}
    id2v = {im["id"]: Path(im["file_name"]).stem for im in ann["images"]}
    out = {}
    for a in ann["annotations"]:
        if a["category_id"] == 1 or id2v[a["image_id"]] != vn:
            continue
        s = a["segmentation"]
        c = s["counts"].encode() if isinstance(s["counts"], str) else s["counts"]
        out[cat[a["category_id"]].split("_", 1)[-1]] = cocomask.decode({"size": s["size"], "counts": c}).astype(bool)
    return out


def gt_surface_labels(sc, Pw, vs, views):
    """回 (gt_lab[N] 物體 short 名或 None, onames)。每表面 voxel 多數決其可見視角落進的 modal 物體。"""
    sdir = CAP / f"multi_{sc.split('_')[0]}" / sc
    votes = defaultdict(lambda: np.zeros(len(Pw), int))   # obj -> per-voxel 票數
    for vn in views:
        pf = sdir / f"{vn}_pose.json"
        if not pf.is_file():
            continue
        modal = modal_masks(sc, vn)
        if not modal:
            continue
        H, W = next(iter(modal.values())).shape
        C, Rb = cam.load_pose(pf); Rwc, t = cam.pose_to_w2c(C, Rb); K = cam.intrinsics(W, H)
        X = Pw @ Rwc.T + t; zc = X[:, 2]; ok = zc > 1e-6; zz = np.where(ok, zc, 1.0)
        px = np.round(K[0, 0] * X[:, 0] / zz + K[0, 2]).astype(int)
        py = np.round(K[1, 1] * X[:, 1] / zz + K[1, 2]).astype(int)
        inb = ok & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        va = CG.zbuffer_visible(Pw, C, Rb, W, H, vs).reshape(H, W)
        ii = np.where(inb)[0]
        vis_i = ii[va[py[ii], px[ii]] == ii]              # 該視角可見的 voxel index
        for o, m in modal.items():
            hit = vis_i[m[py[vis_i], px[vis_i]]]
            votes[o][hit] += 1
    if not votes:
        return np.full(len(Pw), None, object), []
    onames = list(votes)
    V = np.stack([votes[o] for o in onames], 1)           # (N, n_obj)
    best = V.argmax(1); tot = V.sum(1)
    gt_lab = np.array([onames[best[i]] if tot[i] > 0 else None for i in range(len(Pw))], object)
    return gt_lab, onames


def iou(a, b):
    inter = int((a & b).sum())
    return inter / int((a | b).sum()) if inter else 0.0


def scene_gt(sc, views):
    """每場算一次:回 (surf, gt_objs, gt_mask{o:bool over surf})。GT 各 root 共用。"""
    hp = EVAL / HULL / sc / "hull.npz"
    if not hp.is_file():
        return None
    zz = np.load(hp); occ = zz["occupancy"]; gm = zz["grid_min"]; vs = float(zz["voxel_size"])
    surf = surface(occ); sidx = np.argwhere(surf); Pw = gm + (sidx + 0.5) * vs
    if len(sidx) == 0:
        return None
    gt_lab, _ = gt_surface_labels(sc, Pw, vs, views)
    gt_objs = [o for o in set(x for x in gt_lab if x is not None) if o not in GEX]
    gt_mask = {o: (gt_lab == o) for o in gt_objs}
    return surf, gt_objs, gt_mask


def eval_scene(sc, root, gtinfo):
    ip = EVAL / root / sc / "instances.npz"
    if not ip.is_file():
        return None
    surf, gt_objs, gt_mask = gtinfo
    pred = np.load(ip)["labels"][surf]                    # 每表面 voxel 的預測 instance
    pred_ids = [int(k) for k in np.unique(pred) if k > 0]
    pred_mask = {k: (pred == k) for k in pred_ids}
    # IoU 矩陣 → Hungarian
    if gt_objs and pred_ids:
        M = np.zeros((len(gt_objs), len(pred_ids)))
        for gi, o in enumerate(gt_objs):
            for pj, k in enumerate(pred_ids):
                M[gi, pj] = iou(gt_mask[o], pred_mask[k])
        ri, ci = linear_sum_assignment(-M)
        matched = {gt_objs[r]: (pred_ids[c], M[r, c]) for r, c in zip(ri, ci)}
    else:
        matched = {}
    rows = []
    for o in gt_objs:
        k, io = matched.get(o, (None, 0.0))
        rows.append((sc, o, io))
    # 幻影:未配到任何 GT 的預測 instance(主屬非排除 GT 或背景)
    matched_pred = {v[0] for v in matched.values()}
    phantom = sum(1 for k in pred_ids if k not in matched_pred)
    return rows, phantom, len(pred_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", required=True)
    ap.add_argument("scenes", nargs="*")
    args = ap.parse_args()
    scenes = args.scenes
    if not scenes:
        for g in GROUPS:
            scenes += sorted(Path(p).name for p in glob.glob(str(EVAL / HULL / f"{g}_scene*")))
    views = sorted(VP.selected_view_names(12))
    roots = args.roots.split(",")
    print(f"場景={len(scenes)},視角={len(views)},roots={roots}\n", flush=True)

    import csv
    print("[GT] 算每場表面 GT(各 root 共用)...", flush=True)
    gtcache = {}
    for i, sc in enumerate(scenes):
        g = scene_gt(sc, views)
        if g is not None:
            gtcache[sc] = g
        if (i + 1) % 60 == 0:
            print(f"    {i+1}/{len(scenes)}", flush=True)
    print("[GT] 完成\n", flush=True)

    for root in roots:
        allrows = []; phantom = 0; n_pred = 0; nsc = 0
        for sc in scenes:
            if sc not in gtcache:
                continue
            r = eval_scene(sc, root, gtcache[sc])
            if r is None:
                continue
            rows, ph, npd = r
            allrows += rows; phantom += ph; n_pred += npd; nsc += 1
        with open(Path(__file__).parent / f"eval_surface_iou_{root}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["scene", "obj", "best_iou"]); w.writerows(allrows)
        ngt = len(allrows)
        ious = np.array([r[2] for r in allrows])
        print(f"── {root}  (場{nsc} GT物體{ngt} 預測inst{n_pred} 幻影{phantom})")
        print(f"   {'門檻':<6}{'找到':>6}{'找到率':>8}{'配對mIoU':>10}")
        for t in TAUS:
            fnd = int((ious >= t).sum())
            print(f"   {t:<6}{fnd:>6}{fnd/max(ngt,1):>8.3f}{ious[ious>=t].mean() if (ious>=t).any() else 0:>10.3f}")
        print(f"   全部配對 mIoU(含未達門檻) = {ious.mean():.3f}\n")


if __name__ == "__main__":
    main()
