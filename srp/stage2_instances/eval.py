#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""eval.py — Stage 2 instance 指派評估(D1 找到率/過檢/3D IoU、D2 vs 3D 連通 baseline)。

GT(per-object,公平 hull-vs-hull):用 amodal 遮罩(data/labels/<scene>/amodal/,每物體每視角
完整輪廓,以拍攝 pose 渲染、對齊我們的相機)carve 出**每物體 GT 視覺 hull**,落在預測的同一個網格。
凹腔/非水密問題兩邊一致(都填),不像 mesh route 會低估 hull。

預測:讀 instances.npz 的 labels 網格(label k = 第 k 個 instance)。
配對:預測 instance ↔ GT 物體 以 3D IoU 做 Hungarian 最佳配對。

輸出每場景:D1 找到率@0.25/門檻(recall)、過檢/幻影(precision)、配對 mIoU、實例數 vs GT;
            D2 我們 instance 數 vs 純 3D 連通元件數 vs GT 數。需 webots_visual_hull。

用法: ./srp/stage2_instances/eval.py n3_scene0001 n3_scene0030 [--iou 0.25]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage1_hull"))
import camera as cam            # noqa: E402
from carve import carve_visual_hull   # noqa: E402

# 路徑可用 env 覆寫(新資料:captures_fast + srp_hull_fast + 獨立 GT 快取避免撞舊)
CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures")))
LABELS = REPO / "data" / "labels"
HULL_ROOT = Path(os.environ.get("HULL_ROOT", str(REPO / "data" / "eval" / "srp_hull")))
GT_CACHE = Path(os.environ.get("GT_CACHE", str(REPO / "data" / "eval" / "gt_hull_cache")))


def gt_object_hulls(scene, grid_min, grid_max, vs, shape):
    """用 amodal 遮罩 carve 每物體 GT 視覺 hull。回傳 {name: occ(bool, 同網格)}。
    GT hull 只依 場景+網格(box/voxel),與 am/cover/agree 無關 → 快取一次,跨參數組合重用。"""
    cp = GT_CACHE / scene / f"gt_v{int(round(vs * 10000))}.npz"
    if cp.is_file():
        z = np.load(cp)
        if np.allclose(z["grid_min"], grid_min) and tuple(z["shape"]) == tuple(shape):
            return {str(n): z["occ"][i] for i, n in enumerate(z["names"])}

    ann = LABELS / scene / "amodal" / "annotations.json"
    if not ann.is_file():
        return None
    d = json.loads(ann.read_text())
    cat = {c["id"]: c["name"] for c in d["categories"]}
    view_of = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}   # id → view_XX
    group = scene.split("_")[0]
    sdir = CAPTURES / f"multi_{group}" / scene

    # 每物體:各視角 (mask, K, extr)
    per_obj = {name: {"masks": [], "Ks": [], "extr": []} for name in cat.values()}
    for a in d["annotations"]:
        name = cat[a["category_id"]]
        view = view_of[a["image_id"]]
        pose = sdir / f"{view}_pose.json"
        if not pose.is_file():
            continue
        m = mask_utils.decode(a["segmentation"]).astype(bool)
        if m.sum() == 0:
            continue
        H, W = m.shape
        C, R_body = cam.load_pose(pose)
        per_obj[name]["masks"].append(m)
        per_obj[name]["Ks"].append(cam.intrinsics(W, H))
        per_obj[name]["extr"].append(cam.pose_to_w2c(C, R_body))

    out = {}
    for name, v in per_obj.items():
        if len(v["masks"]) < 2:
            continue
        hull = carve_visual_hull(v["masks"], v["Ks"], v["extr"], grid_min, grid_max, vs,
                                 table_z=0.0)
        if hull.occupancy.any():
            out[name] = hull.occupancy
    if out:                                  # 寫快取(跨參數組合重用)
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cp, names=np.array(list(out)),
                            occ=np.stack([out[n] for n in out]),
                            grid_min=np.asarray(grid_min, float), shape=np.asarray(shape))
    return out


def iou3(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def _suf(tag):
    return f"_{tag}" if tag else ""


def process(scene, iou_thresh, hull_root=HULL_ROOT, tag=""):
    ip = hull_root / scene / f"instances{_suf(tag)}.npz"
    if not ip.is_file():
        print(f"[skip] {scene}: 找不到 {ip}"); return None
    z = np.load(ip)
    labels = z["labels"]; grid_min = z["grid_min"]; vs = float(z["voxel_size"])
    shape = labels.shape
    grid_max = grid_min + np.array(shape) * vs
    pred_occ = [labels == k for k in range(1, int(labels.max()) + 1) if (labels == k).any()]

    gt = gt_object_hulls(scene, grid_min, grid_max, vs, shape)
    if not gt:
        print(f"[skip] {scene}: 無 GT hull"); return None
    gt_names = list(gt); gt_occ = list(gt.values())

    nP, nG = len(pred_occ), len(gt_occ)
    M = np.zeros((nP, nG))
    for i in range(nP):
        for j in range(nG):
            M[i, j] = iou3(pred_occ[i], gt_occ[j])
    matched = {}
    if nP and nG:
        ri, cj = linear_sum_assignment(-M)
        for i, j in zip(ri, cj):
            matched[i] = (j, M[i, j])

    found025 = sum(1 for _, (j, v) in matched.items() if v >= 0.25)
    found_t = sum(1 for _, (j, v) in matched.items() if v >= iou_thresh)
    mious = [v for _, (j, v) in matched.items() if v >= iou_thresh]
    phantom = nP - found_t
    n_3dcc = ndimage.label((labels > 0), ndimage.generate_binary_structure(3, 1))[1]
    recall = found_t / nG
    precision = found_t / nP if nP else 0.0
    miou = float(np.mean(mious)) if mious else 0.0
    print(f"[{scene}] GT{nG} pred{nP} 3Dcc{n_3dcc} | found@.25={found025}/{nG} "
          f"found@{iou_thresh}={found_t}/{nG} 幻影={phantom} | "
          f"recall={recall:.2f} prec={precision:.2f} mIoU={miou:.2f}")
    return {"scene": scene, "n_gt": nG, "n_pred": nP, "n_3dcc": n_3dcc,
            "found_025": found025, "found_t": found_t, "phantom": phantom,
            "recall": round(recall, 3), "precision": round(precision, 3),
            "mean_iou": round(miou, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--iou", type=float, default=0.25)
    ap.add_argument("--root", default="srp_hull", help="讀取根目錄名 data/eval/<root>/")
    ap.add_argument("--tag", default="", help="讀 instances 的檔名後綴(如 am1_cvsmall)")
    ap.add_argument("--csv", default=None, help="CSV 路徑(預設 data/eval/<root>/d1d2_<tag>.csv)")
    args = ap.parse_args()
    hull_root = REPO / "data" / "eval" / args.root
    csv_path = args.csv or str(hull_root / f"d1d2{_suf(args.tag)}.csv")
    rows = []
    for sc in args.scenes:
        try:
            r = process(sc, args.iou, hull_root, args.tag)
            if r:
                rows.append(r)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")
    if not rows:
        return
    agg = {k: round(float(np.mean([r[k] for r in rows])), 3)
           for k in ("recall", "precision", "mean_iou")}
    tg = sum(r["n_gt"] for r in rows); tf = sum(r["found_t"] for r in rows)
    tph = sum(r["phantom"] for r in rows)
    print(f"\n== {len(rows)} 場景 | 總 found {tf}/{tg} 幻影 {tph} | "
          f"平均 recall={agg['recall']} prec={agg['precision']} mIoU={agg['mean_iou']} "
          f"(IoU門檻 {args.iou}) ==")
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"→ {csv_path}")


if __name__ == "__main__":
    main()
