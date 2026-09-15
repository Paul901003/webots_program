#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""build_hull_gt.py — 讀共用 GT(gt_reproj,由 build_gt.py 產) + 算該方法每個 hull 的重投影對照。

hull 重投影 = occupancy voxel「立方體直接投影」(每 voxel 8 角投影取 2D bbox 填、聯集,不取表面/不三角化)。
對每個 hull、每個 GT 物體:在該物體無遮擋視角把 hull 重投影跟 modal mask 算 2D IoU,取平均;
平均 > HIT_IOU = 命中。一物體被多 hull 命中 → 認 avg 最高、其餘 redundant。
存 <root>/<scene>/hull_gt.{json,npz,csv}(只存 hull 重投影+對照,GT 在 gt_reproj 共用、不重存)。
用法: ./build_hull_gt.py --root <method> [scene|group|(空=全部)]  (需先跑 build_gt.py)
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import camera as cam            # noqa: E402
import viewpoints as VP         # noqa: E402
from labels import LABELS       # noqa: E402

EVAL = REPO / "data" / "eval"
CAPTURES = REPO / "data" / "captures_fast"
GT_OUT = EVAL / "gt_reproj"
HIT_IOU = 0.8


def reproject(vox_world, C, Rb, W, H, vs):
    """hull occupancy voxel (n,3) → 2D silhouette bool mask。
    每 voxel 當 vs³ 立方體,投影 8 角取 2D bounding box 填、聯集(直接投影、不做其他處理)。"""
    Rwc, t = cam.pose_to_w2c(C, Rb)
    K = cam.intrinsics(W, H); fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
    h = vs * 0.5
    corners = np.array([[dx, dy, dz] for dx in (-h, h) for dy in (-h, h) for dz in (-h, h)])  # (8,3)
    X = (vox_world[:, None, :] + corners[None, :, :]) @ Rwc.T + t     # (n,8,3)
    zc = X[..., 2]; ok = zc > 1e-9
    zz = np.where(ok, zc, 1.0)
    u = np.where(ok, fx * X[..., 0] / zz + cx, np.nan)
    v = np.where(ok, fy * X[..., 1] / zz + cy, np.nan)
    valid = ok.any(1)
    umin = np.clip(np.floor(np.nanmin(u, 1)), 0, W); umax = np.clip(np.ceil(np.nanmax(u, 1)), 0, W)
    vmin = np.clip(np.floor(np.nanmin(v, 1)), 0, H); vmax = np.clip(np.ceil(np.nanmax(v, 1)), 0, H)
    ui0 = umin.astype(int); ui1 = umax.astype(int); vi0 = vmin.astype(int); vi1 = vmax.astype(int)
    keep = valid & (ui1 > ui0) & (vi1 > vi0)
    ui0, ui1, vi0, vi1 = ui0[keep], ui1[keep], vi0[keep], vi1[keep]
    # 2D 差分陣列:每個 bbox 四角 ±1,兩次前綴和還原覆蓋計數 → 一次填完所有 voxel(取代逐 voxel 迴圈)
    diff = np.zeros((H + 1, W + 1), np.int32)
    np.add.at(diff, (vi0, ui0), 1); np.add.at(diff, (vi0, ui1), -1)
    np.add.at(diff, (vi1, ui0), -1); np.add.at(diff, (vi1, ui1), 1)
    return diff.cumsum(0).cumsum(1)[:H, :W] > 0


def iou2(a, b):
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else 0.0


def load_gt(scene):
    gd = GT_OUT / scene
    if not (gd / "gt.json").is_file():
        return None
    gj = json.loads((gd / "gt.json").read_text())
    gz = np.load(gd / "gt.npz")
    modal = {}
    for key in gz.files:
        if key.startswith("modal_"):
            g, vn = key[len("modal_"):].rsplit("__", 1)
            modal[(g, vn)] = gz[key]
    return gj, modal


def load_amodal(scene):
    """amodal 2D 完整輪廓(labels/amodal RLE, 全視角);不受遮擋、不做 OCC 過濾。→ {(g,vn): mask}。
    解嚴重遮擋物(如堆疊底層)在 modal 版被 OCC_THRESH=0.9 砍光 modal → 假漏 的問題。"""
    from pycocotools import mask as mask_utils
    ann = LABELS / scene / "amodal" / "annotations.json"
    if not ann.is_file():
        return None
    d = json.loads(ann.read_text())
    catn = {c["id"]: c["name"] for c in d["categories"]}
    vof = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
    out = {}
    for a in d["annotations"]:
        out[(catn[a["category_id"]], vof[a["image_id"]])] = mask_utils.decode(a["segmentation"]).astype(bool)
    return out


def process(scene, root, out_root, match="amodal"):
    suf = ""   # amodal 為唯一正確評估(modal 有系統假漏),統一寫 hull_gt.*,不再並存
    ip = EVAL / root / scene / "instances.npz"
    if not ip.is_file():
        print(f"[skip] {scene}: 無 {ip}"); return None
    if os.environ.get("FORCE", "") != "1" and (out_root / scene / f"hull_gt{suf}.json").is_file():
        return None   # 續跑保護:已算過就跳過
    gt = load_gt(scene)
    if gt is None:
        print(f"[skip] {scene}: 無共用 GT(先跑 build_gt.py)"); return None
    gj, modal = gt
    gt_names = gj["gt_objects"]; unocc = gj["unoccluded_views"]
    if match == "amodal":
        gtmask = load_amodal(scene)
        if not gtmask:
            print(f"[skip] {scene}: 無 amodal GT"); return None
    else:
        gtmask = modal

    z = np.load(ip); labels = z["labels"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
    group = scene.split("_")[0]; sdir = CAPTURES / f"multi_{group}" / scene
    hull_ids = [k for k in range(1, int(labels.max()) + 1) if (labels == k).any()]
    # 只用 A-3 12 視角:per_gt 對 unocc[g]∩A-3-12 平均(12視角 metric),報告也顯示這 12 視角
    need_views = sorted(set(VP.selected_view_names(12)))
    poses = {vn: cam.load_pose(sdir / f"{vn}_pose.json")
             for vn in need_views if (sdir / f"{vn}_pose.json").is_file()}
    # amodal 完整→用全 12 A-3 視角;modal→只用該物體無遮擋視角(unocc)
    gview = {g: need_views for g in gt_names} if match == "amodal" else {g: unocc[g] for g in gt_names}
    H, W = next(iter(gtmask.values())).shape if gtmask else (720, 1280)

    reproj = {}; avg_iou = {k: {} for k in hull_ids}; detail = {k: {} for k in hull_ids}
    for k in hull_ids:
        Pw = gm + (np.argwhere(labels == k) + 0.5) * vs
        for vn in need_views:
            if vn in poses:
                C, Rb = poses[vn]
                reproj[(k, vn)] = reproject(Pw, C, Rb, W, H, vs)
        for g in gt_names:
            ious = [iou2(reproj[(k, vn)], gtmask[(g, vn)])
                    for vn in gview[g] if (k, vn) in reproj and (g, vn) in gtmask]
            a = float(np.mean(ious)) if ious else 0.0
            avg_iou[k][g] = a
            detail[k][g] = {"match_views": gview[g], "ious": [round(x, 4) for x in ious], "avg": round(a, 4)}

    hit = {k: (max(avg_iou[k], key=avg_iou[k].get)
               if avg_iou[k] and max(avg_iou[k].values()) > HIT_IOU else None) for k in hull_ids}
    by_g = defaultdict(list)
    for k in hull_ids:
        if hit[k]:
            by_g[hit[k]].append(k)
    redundant = set()
    for g, ks in by_g.items():
        for k in sorted(ks, key=lambda k: -avg_iou[k][g])[1:]:
            redundant.add(k)

    out = out_root / scene; out.mkdir(parents=True, exist_ok=True)
    (out / f"hull_gt{suf}.json").write_text(json.dumps({
        "scene": scene, "root": root, "n_hull": len(hull_ids), "gt_objects": gt_names,
        "params": {"HIT_IOU": HIT_IOU, "match": match, "reproj": "voxel-cube-bbox", "gt": "gt_reproj(shared)"},
        "hulls": {str(k): {"hit": hit[k], "redundant": k in redundant, "per_gt": detail[k]} for k in hull_ids},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    arrs = {"grid_min": gm, "voxel_size": vs, "shape": np.array(labels.shape), "hull_ids": np.array(hull_ids)}
    for k in hull_ids:
        arrs[f"hull_{k}"] = (labels == k)
    for (k, vn), m in reproj.items():
        arrs[f"reproj_{k}__{vn}"] = m
    np.savez_compressed(out / f"hull_gt{suf}.npz", **arrs)
    with open(out / f"hull_gt{suf}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hull_id", "gt_name", "match_views", "avg_reproj_iou", "hit", "redundant"])
        for k in hull_ids:
            for g in gt_names:
                w.writerow([k, g, len(gview[g]), round(avg_iou[k][g], 4),
                            "Y" if hit[k] == g else "", "Y" if (hit[k] == g and k in redundant) else ""])
    nfound = len({hit[k] for k in hull_ids if hit[k]})
    print(f"[{scene}] {len(hull_ids)}hull×{len(gt_names)}GT → 找到 {nfound}/{len(gt_names)} "
          f"(命中 {sum(1 for k in hull_ids if hit[k] and k not in redundant)}, 多餘 {len(redundant)})", flush=True)
    return nfound


def resolve(root, targets):
    base = EVAL / root
    if not targets:
        return sorted(p.parent.name for p in base.glob("*_scene*/instances.npz"))
    out = []
    for a in targets:
        out.append(a) if "scene" in a else out.extend(
            p.parent.name for p in base.glob(f"{a}_scene*/instances.npz"))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--root", required=True)
    ap.add_argument("--match", default="amodal", choices=["modal", "amodal"])
    args = ap.parse_args()
    out_root = EVAL / args.root
    for sc in resolve(args.root, args.targets):
        try:
            process(sc, args.root, out_root, args.match)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")


if __name__ == "__main__":
    main()
