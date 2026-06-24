#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""eval_clip_match.py — 評估(不收斂):CLIP 特徵↔名詞配對 + 漏檢/過檢/3D IoU。

對某方法的全部 hull(不剪枝):
  ① hull 特徵 = 各視角遮罩查預存 CLIP 影像特徵 → 面積加權聚合 → L2 norm。
  ② 配對 = cosine(hull 特徵, 場景物體名詞文字特徵) 貪婪一對一 → hull↔GT。
  ③ 3D IoU = 重建 hull 體素 vs GT visual hull(GT 遮罩 carve,同格 pitch)。
  ④ 指標:漏檢率、過檢率(a)=多餘hull/總hull、過檢率(c)=有多餘hull的場景比例、3D IoU。
只量測,不改 hull 生成。需 webots_visual_hull;特徵須先由 precompute_clip.py 算好。

用法: ./instance_hull/eval_clip_match.py 1 3 4 5 --root=epipolar
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import associate_voxel as av

REPO = av.REPO
CAPTURES = av.CAPTURES
SAM_ROOT = av.SAM_ROOT
EVAL_ROOT = REPO / "data" / "eval"
LABELS = REPO / "data" / "labels"
TEXT_FEATS = EVAL_ROOT / "clip_text_feats.npz"


# ── 共用工具(自包含;原 eval_3diou.py 的乾淨部分,GT 改由 carve_from_masks 雕殼)──
def resolve_scenes(targets):
    out = []
    for a in targets:
        if "scene" in a:
            out.append(a)
        else:
            out += [d.name for d in sorted((CAPTURES / f"multi_n{a}").glob(f"n{a}_scene*"))]
    return out


def load_views(scene):
    g = scene.split("_")[0]
    sdir = CAPTURES / f"multi_{g}" / scene
    sam = SAM_ROOT / scene
    views = {}
    for vdir in sorted(sam.glob("view_*")):
        pp = sdir / f"{vdir.name}_pose.json"
        any_m = next((vdir / "masks").glob("mask_*.png"), None)
        if not pp.is_file() or any_m is None:
            continue
        C, R = av.load_pose(pp)
        H, W = cv2.imread(str(any_m), cv2.IMREAD_GRAYSCALE).shape
        fx, cx, cy = av.intrinsics(W, H)
        views[vdir.name] = {"C": C, "R": R, "fx": fx, "cx": cx, "cy": cy, "W": W, "H": H}
    return views


def project(P, v):
    X = (P - v["C"]) @ v["R"] @ av.BODY_TO_OPENCV.T
    z = X[:, 2]; valid = z > 1e-6
    u = np.full(len(P), -1.0); w = np.full(len(P), -1.0)
    u[valid] = v["fx"] * X[valid, 0] / z[valid] + v["cx"]
    w[valid] = v["fx"] * X[valid, 1] / z[valid] + v["cy"]
    ui = np.round(u).astype(np.int64); wi = np.round(w).astype(np.int64)
    inb = valid & (ui >= 0) & (ui < v["W"]) & (wi >= 0) & (wi < v["H"])
    return ui, wi, inb


def recon_occupancy(inst, scene, views, proj, P, keep_frac):
    cnt = np.zeros(len(P), dtype=np.int32); nv = 0
    for vn, files in inst["masks"].items():
        if vn not in views:
            continue
        seg = None
        for f in files:
            m = cv2.imread(str(SAM_ROOT / scene / vn / "masks" / f), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            b = m > 127
            seg = b if seg is None else (seg | b)
        if seg is None:
            continue
        ui, wi, inb = proj[vn]
        ins = np.zeros(len(P), dtype=bool); ins[inb] = seg[wi[inb], ui[inb]]
        cnt += ins; nv += 1
    if nv < 2:
        return None
    keep_min = max(2, int(math.ceil(keep_frac * nv)))
    occ = cnt >= keep_min
    return occ if occ.any() else None

_TXT = np.load(TEXT_FEATS, allow_pickle=True)
TXT_NAMES = list(_TXT["names"])
TXT_FEATS_ARR = _TXT["feats"]
TXT_IDX = {n: i for i, n in enumerate(TXT_NAMES)}


def load_feat_cache(scene):
    """{view_name: (files_list, feats(M×512))}"""
    cache = {}
    for vdir in sorted((SAM_ROOT / scene).glob("view_*")):
        npy = vdir / "clip_feats.npy"; fj = vdir / "clip_feats_files.json"
        if npy.is_file() and fj.is_file():
            files = json.loads(fj.read_text())
            cache[vdir.name] = (files, np.load(npy))
    return cache


def hull_per_view_feats(inst, scene, fcache):
    """{view: (feat(512) L2norm, area_sum)};每視角把該物體用到的遮罩特徵面積加權聚合。"""
    out = {}
    for vn, files in inst["masks"].items():
        fc = fcache.get(vn)
        if fc is None:
            continue
        flist, feats = fc
        fidx = {f: i for i, f in enumerate(flist)}
        acc = np.zeros(512, np.float64); aw = 0.0
        for f in files:
            if f not in fidx:
                continue
            m = cv2.imread(str(SAM_ROOT / scene / vn / "masks" / f), cv2.IMREAD_GRAYSCALE)
            area = float((m > 127).sum()) if m is not None else 1.0
            acc += area * feats[fidx[f]]; aw += area
        if aw > 0:
            v = acc / aw; n = np.linalg.norm(v)
            if n > 0:
                out[vn] = ((v / n).astype(np.float32), aw)
    return out


def combine_feats(pv):
    """各視角特徵面積加權 → 物體聚合特徵。"""
    if not pv:
        return None
    acc = np.zeros(512, np.float64); aw = 0.0
    for f, a in pv.values():
        acc += a * f; aw += a
    if aw == 0:
        return None
    v = acc / aw; n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else None


def load_gt_masks(scene):
    p = LABELS / scene / "actual" / "annotations.json"
    if not p.is_file():
        return {}
    coco = json.loads(p.read_text()); cats = {c["id"]: c["name"] for c in coco["categories"]}
    out = {}
    for a in coco["annotations"]:
        nm = cats[a["category_id"]]
        if nm == "ur5e":
            continue
        rle = a["segmentation"]
        if isinstance(rle["counts"], str):
            rle = {"counts": rle["counts"].encode(), "size": rle["size"]}
        out.setdefault(nm, {})[f"view_{int(a['image_id']):02d}"] = mu.decode(rle).astype(bool)
    return out


def carve_from_masks(view_masks, views, proj, P, keep_frac):
    """view_masks: {view_name: bool mask};回傳體素佔據 bool。"""
    cnt = np.zeros(len(P), np.int32); nv = 0
    for vn, seg in view_masks.items():
        if vn not in proj:
            continue
        ui, wi, inb = proj[vn]
        ins = np.zeros(len(P), bool); ins[inb] = seg[wi[inb], ui[inb]]
        cnt += ins; nv += 1
    if nv < 2:
        return None
    occ = cnt >= max(2, int(np.ceil(keep_frac * nv)))
    return occ if occ.any() else None


def iou3(a, b):
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else 0.0


def reproject_occ(Pocc, v, pitch):
    """instance 體素 → 投影回某視角 → 2D 二值遮罩(投影點+閉合填縫)。"""
    X = (Pocc - v["C"]) @ v["R"] @ av.BODY_TO_OPENCV.T
    z = X[:, 2]; valid = z > 1e-6
    u = v["fx"] * X[:, 0] / np.where(valid, z, 1) + v["cx"]
    w = v["fx"] * X[:, 1] / np.where(valid, z, 1) + v["cy"]
    ui = np.round(u).astype(np.int64); wi = np.round(w).astype(np.int64)
    inb = valid & (ui >= 0) & (ui < v["W"]) & (wi >= 0) & (wi < v["H"])
    img = np.zeros((v["H"], v["W"]), np.uint8)
    if not inb.any():
        return img.astype(bool)
    img[wi[inb], ui[inb]] = 1
    dmed = float(np.median(z[inb]))
    k = int(max(3, round(pitch * v["fx"] / max(dmed, 1e-3))))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return img.astype(bool)


def reproj2d_per_view(occ, P, nm, gt, views, pitch):
    """matched hull 重投影遮罩 vs GT 遮罩,回傳 {view: IoU}。"""
    if occ is None:
        return {}
    Pocc = P[occ]; out = {}
    for vn, gmask in gt.get(nm, {}).items():
        if vn not in views:
            continue
        rm = reproject_occ(Pocc, views[vn], pitch)
        if rm.shape == gmask.shape:
            out[vn] = iou3(rm, gmask)
    return out


def process_scene(scene, root, args):
    views = load_views(scene)
    if len(views) < 2:
        return None
    inst_path = EVAL_ROOT / root / scene / "instances.json"
    if not inst_path.is_file():
        return None
    instances = json.loads(inst_path.read_text())["instances"]
    fcache = load_feat_cache(scene)
    gt = load_gt_masks(scene)
    gt_names = [n for n in gt if n in TXT_IDX]
    if not gt_names:
        return None

    # 體素格 + 投影
    xs = np.arange(*av.WS_X, args.pitch); ys = np.arange(*av.WS_Y, args.pitch); zs = np.arange(*av.WS_Z, args.pitch)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    proj = {vn: project(P, v) for vn, v in views.items()}

    # hull 逐視角特徵 + 聚合特徵 + 重建佔據
    pv_list, feats, recon = [], [], []
    for inst in instances:
        pv = hull_per_view_feats(inst, scene, fcache)
        pv_list.append(pv)
        feats.append(combine_feats(pv))
        recon.append(recon_occupancy(inst, scene, views, proj, P, args.keep_frac))
    valid = [i for i in range(len(instances)) if feats[i] is not None]
    n_hull = len(valid)

    # GT visual hull(每物體用 GT 遮罩 carve)
    gt_occ = {}
    for nm in gt_names:
        occ = carve_from_masks(gt[nm], views, proj, P, args.keep_frac)
        if occ is not None:
            gt_occ[nm] = occ
    gt_names = list(gt_occ.keys())
    if not gt_names:
        return None

    # 每個 hull:CLIP 標籤(argmax cosine)+ 對「該標籤 GT」的 3D IoU。合格 = 標籤對 AND IoU>=tau。
    tau = args.iou_th
    hull_info = []           # (i, lab, cos, v3, v2, rv, qual)
    n_qual = 0
    hull_rows = []           # 逐 hull 明細(幻影拆解用):best_iou=對全部 GT 的最佳 3D IoU
    for i in valid:
        sims = [(float(feats[i] @ TXT_FEATS_ARR[TXT_IDX[nm]]), nm) for nm in gt_names]
        cos_i, lab = max(sims)
        # 對全部 GT 算 3D IoU:取對標籤 GT 的(v3)與全域最佳(best_gt/best_iou)
        ious = {nm: (iou3(recon[i], gt_occ[nm]) if recon[i] is not None else 0.0) for nm in gt_names}
        v3 = ious.get(lab, 0.0)
        best_gt = max(ious, key=ious.get) if ious else ""
        best_iou = ious.get(best_gt, 0.0)
        rv = reproj2d_per_view(recon[i], P, lab, gt, views, args.pitch)
        v2 = float(np.mean(list(rv.values()))) if rv else 0.0
        qual = v3 >= tau
        n_qual += int(qual)
        hull_info.append((i, lab, cos_i, v3, v2, rv, qual))
        hull_rows.append({"scene": scene, "hull": i, "pred_label": lab, "cos": round(cos_i, 4),
                          "iou_label": round(v3, 4), "best_gt": best_gt, "best_iou": round(best_iou, 4),
                          "qualify": int(qual), "label_ok": int(lab == best_gt),
                          "n_hull": n_hull, "n_gt": len(gt_names)})

    # 逐物體對應:每個真實物體取「標籤=它 且 IoU>=tau」中 IoU 最高的 hull 當代表(有就算找到)
    matched = {}            # nm -> (i, cos, v3, v2, rv)
    for i, lab, cos_i, v3, v2, rv, qual in hull_info:
        if qual and (lab not in matched or v3 > matched[lab][2]):
            matched[lab] = (i, cos_i, v3, v2, rv)

    # 逐物體列(含漏掉的);對應上才有代表 hull
    gt_rows, view_rows = [], []
    for nm in gt_names:
        if nm in matched:
            i, cos_i, v3, v2, rv = matched[nm]
            gt_rows.append({"scene": scene, "gt_name": nm, "found": 1, "hull": i,
                            "cos": round(cos_i, 4), "iou3d": round(v3, 4), "reproj2d": round(v2, 4),
                            "n_hull": n_hull, "n_gt": len(gt_names)})
            nt = TXT_FEATS_ARR[TXT_IDX[nm]]
            for vn in sorted(set(pv_list[i]) | set(rv)):
                cv = round(float(pv_list[i][vn][0] @ nt), 4) if vn in pv_list[i] else ""
                view_rows.append({"scene": scene, "gt_name": nm, "hull": i, "view": vn,
                                  "cos": cv, "reproj2d": round(rv[vn], 4) if vn in rv else "",
                                  "iou3d_obj": round(v3, 4)})
        else:
            gt_rows.append({"scene": scene, "gt_name": nm, "found": 0, "hull": -1,
                            "cos": "", "iou3d": "", "reproj2d": "", "n_hull": n_hull, "n_gt": len(gt_names)})

    found = len(matched)                     # 對應到代表 hull 的真實物體數
    miss = len(gt_names) - found
    extra = n_hull - found                   # 多餘 hull(過檢):重複 + 幻影/誤配
    dup = n_qual - found                     # 其中:合格但多餘(同物體多個合格 hull)
    phantom = n_hull - n_qual                # 其中:不合格(IoU<tau,誤配/幻影)
    mio = [r["iou3d"] for r in gt_rows if r["found"]]
    summary = {
        "scene": scene, "root": root, "tau": tau, "n_hull": n_hull, "n_gt": len(gt_names),
        "found": found, "n_qualify": n_qual,
        "miss_rate": round(miss / len(gt_names), 4),                      # 漏檢:沒對應到 hull 的物體
        "over_rate": round(extra / n_hull, 4) if n_hull else 0.0,        # 過檢:多餘 hull / 總 hull
        "dup_rate": round(dup / n_hull, 4) if n_hull else 0.0,           # 過檢中:重複合格 / 總 hull
        "phantom_rate": round(phantom / n_hull, 4) if n_hull else 0.0,   # 過檢中:幻影/誤配 / 總 hull
        "mean_iou3d": round(float(np.mean(mio)), 4) if mio else 0.0}     # 對應上物體的 3D IoU
    out_dir = EVAL_ROOT / root / scene / "eval_clip"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{scene}] hull={n_hull} GT={len(gt_names)} found={found} "
          f"漏檢={summary['miss_rate']:.2f} 過檢={summary['over_rate']:.2f}"
          f"(重複{summary['dup_rate']:.2f}/幻影{summary['phantom_rate']:.2f}) 3DIoU={summary['mean_iou3d']:.3f}")
    return summary, gt_rows, view_rows, hull_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["n3_scene0001"])
    ap.add_argument("--root", default="epipolar")
    ap.add_argument("--pitch", type=float, default=0.01)
    ap.add_argument("--keep-frac", type=float, default=0.6, dest="keep_frac")
    ap.add_argument("--iou-th", type=float, default=0.5, dest="iou_th", help="位置算對的 3D IoU 門檻")
    ap.add_argument("--csv", default=None, help="per-object CSV(每場景每物體一列;append)")
    ap.add_argument("--csv-view", default=None, dest="csv_view", help="per-view CSV(每物體每視角一列;append)")
    ap.add_argument("--csv-hull", default=None, dest="csv_hull", help="per-hull CSV(每 hull 一列,含 best_iou/幻影拆解;append)")
    args = ap.parse_args()
    scenes = resolve_scenes(args.scenes or ["n3_scene0001"])
    sums, allrows, vrows, hrows = [], [], [], []
    for i, sc in enumerate(scenes, 1):
        print(f"[{i}/{len(scenes)}]", end=" ")
        try:
            res = process_scene(sc, args.root, args)
            if res:
                s, grows, vws, hls = res
                sums.append(s)
                for r in grows:
                    allrows.append({"method": args.root, **r})
                for r in vws:
                    vrows.append({"method": args.root, **r})
                for r in hls:
                    hrows.append({"method": args.root, **r})
            else:
                print(f"{sc}: skip")
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[error] {sc}: {e}")

    def dump(path, rows, cols):
        if not (path and rows):
            return
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        new = not p.is_file()
        with open(p, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"→ {len(rows)} 列 → {p}")
    dump(args.csv_hull, hrows, ["method", "scene", "hull", "pred_label", "cos", "iou_label", "best_gt", "best_iou", "qualify", "label_ok", "n_hull", "n_gt"])
    dump(args.csv, allrows, ["method", "scene", "gt_name", "found", "hull", "cos", "iou3d", "reproj2d", "n_hull", "n_gt"])
    dump(args.csv_view, vrows, ["method", "scene", "gt_name", "hull", "view", "cos", "reproj2d", "iou3d_obj"])
    if sums:
        # 各指標皆「每場景先逐物體對應(特徵定標籤+位置算 3D IoU)算出該場景比率,再跨場景等權平均」
        print(f"\n==== {args.root}: {len(sums)} 場景  (tau={args.iou_th}) ====")
        print(f"漏檢率(沒對應到 hull 的真實物體) = {np.mean([s['miss_rate'] for s in sums]):.3f}")
        print(f"過檢率(多餘 hull / 總 hull)     = {np.mean([s['over_rate'] for s in sums]):.3f}")
        print(f"  其中 重複合格 / 總 hull       = {np.mean([s['dup_rate'] for s in sums]):.3f}")
        print(f"  其中 幻影誤配 / 總 hull       = {np.mean([s['phantom_rate'] for s in sums]):.3f}")
        print(f"對應上物體的 3D IoU 平均        = {np.mean([s['mean_iou3d'] for s in sums]):.3f}")


if __name__ == "__main__":
    main()
