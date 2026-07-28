#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""eval_v3.py — 收斂前兩軸評估(規格 v3)。

逐 GT 物體,掃過該方法**所有** hull(都在 256³ 同格雕殼):
  3D 軸  : best_iou = max_h IoU(hull殼, GT殼);3D 找到 ⟺ best_iou >= 0.5。
  名稱軸 : 名稱找到 ⟺ ∃ hull 其 CLIP 標籤=GT名 且 IoU(hull,GT) >= 0.1(錨定)。
  過檢   : 收斂後才比,本程式不計。
IoU = 兩立體殼的體素交集/聯集。殼 = 各視角指定遮罩在 256³ 投票雕出(門檻同 hull_common)。
CLIP 標籤重用 eval_clip_match 的特徵聚合 + 文字特徵。需 webots_visual_hull。

用法: ./instance_hull/eval_v3.py <scenes> --root instance_hull [--csv out.csv]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hull_common as HC
import eval_clip_match as E   # 重用:load_feat_cache / hull_per_view_feats / combine_feats / TXT

import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "srp" / "io")); from labels import LABELS  # data/labels 分層(類別/數量/場景)


def load_gt_amodal(scene):
    """讀 amodal(完整、無遮擋)GT 遮罩:data/labels/<scene>/amodal/annotations.json。"""
    p = LABELS / scene / "amodal" / "annotations.json"
    if not p.is_file():
        return None
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

COV = 0.5        # 幾何找到:覆蓋率 |hull∩GT|/|GT| 門檻(且 hull 只覆蓋一個 GT)
NAME_COS = 0.25  # 名稱對:CLIP 影像↔名詞 cosine 門檻(在「找到」的 hull 上,argmax 標籤須 = 該 GT)


def carve_obj(view_masks, proj, P, allow_miss=0):
    """view_masks: {view: bool};在 256³ 投票雕殼。allow_miss=容許漏幾個視角(GT 用 0=嚴格)。"""
    votes = np.zeros(len(P), np.int16); nv = 0
    for vn, seg in view_masks.items():
        if vn not in proj:
            continue
        ui, wi, inb = proj[vn]
        hit = np.zeros(len(P), bool); hit[inb] = seg[wi[inb], ui[inb]]
        votes += hit; nv += 1
    if nv < 2:
        return None
    occ = votes >= (nv - allow_miss)
    return occ if occ.any() else None


def iou(a, b):
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else 0.0


def load_occ_idx(root, scene):
    """載入 association 存的每 instance 體素索引(對齊 instances.json 順序);無則 None。"""
    p = HC.EVAL_ROOT / root / scene / "occ.npz"
    if not p.is_file():
        return None
    d = np.load(p)
    counts = d["counts"]; idx = d["idx"]
    out = []; off = 0
    for c in counts:
        out.append(idx[off:off + int(c)]); off += int(c)
    return out


def inst_view_masks(inst, scene):
    out = {}
    for vn, files in inst["masks"].items():
        seg = None
        for f in files:
            m = cv2.imread(str(HC.SAM_ROOT / scene / vn / "masks" / f), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            b = m > 127
            seg = b if seg is None else (seg | b)
        if seg is not None:
            out[vn] = seg
    return out


def process_scene(scene, root, P, shape):
    views = HC.load_views(scene)
    if len(views) < 2:
        return None
    ij = HC.EVAL_ROOT / root / scene / "instances.json"
    if not ij.is_file():
        return None
    instances = json.loads(ij.read_text()).get("instances", [])
    gt = load_gt_amodal(scene)                # 完整(amodal)GT 遮罩
    if gt is None:
        print(f"[{scene}] 無 amodal GT(先跑 generate_amodal_masks.py)"); return None
    gt_names = [n for n in gt if n in E.TXT_IDX]
    if not gt_names:
        return None
    proj = HC.project_all(P, views)

    # GT 殼(256³,嚴格雕 K=0:所有視角都要包含 → 無遮擋理想殼)
    gt_occ = {}
    for nm in gt_names:
        o = carve_obj(gt[nm], proj, P)
        if o is not None:
            gt_occ[nm] = o
    gt_names = list(gt_occ.keys())
    if not gt_names:
        return None

    # instance 體素:優先載入 association 存的 occ.npz(免重雕);否則從遮罩重雕
    occ_idx_list = load_occ_idx(root, scene)

    # 每 hull:CLIP 標籤/cosine + 對各 GT 的覆蓋率(用體素索引相交,快)
    fcache = E.load_feat_cache(scene)
    inst_lab, inst_cos, inst_cov, inst_ncov, n_valid = [], [], [], [], 0
    gt_size = {nm: int(gt_occ[nm].sum()) for nm in gt_names}
    for i, inst in enumerate(instances):
        # 取該 instance 的體素索引
        if occ_idx_list is not None and i < len(occ_idx_list):
            vidx = occ_idx_list[i]
        else:
            o = carve_obj(inst_view_masks(inst, scene), proj, P)
            vidx = np.where(o)[0] if o is not None else None
        # CLIP 標籤
        pv = E.hull_per_view_feats(inst, scene, fcache)
        feat = E.combine_feats(pv)
        if feat is None:
            inst_lab.append(None); inst_cos.append(0.0)
        else:
            sims = [(float(feat @ E.TXT_FEATS_ARR[E.TXT_IDX[nm]]), nm) for nm in gt_names]
            c, l = max(sims); inst_lab.append(l); inst_cos.append(c)
        if vidx is None or len(vidx) == 0:
            inst_cov.append(None); inst_ncov.append(0); continue
        n_valid += 1
        cv = {nm: (int(gt_occ[nm][vidx].sum()) / gt_size[nm] if gt_size[nm] else 0.0) for nm in gt_names}
        inst_cov.append(cv)
        inst_ncov.append(sum(1 for nm in gt_names if cv[nm] > COV))

    rows = []
    for nm in gt_names:
        found3d = 0; name_found = 0; best_cov = 0.0
        for i in range(len(instances)):
            cvd = inst_cov[i]
            if cvd is None:
                continue
            cv = cvd[nm]
            if cv > best_cov:
                best_cov = cv
            single = (cv > COV and inst_ncov[i] == 1)   # 找到的幾何條件:覆蓋率>0.5 且只罩一個 GT
            if single:
                found3d = 1
            # 名稱對:在「找到」的 hull 上,標籤=此 GT 且 cosine>=門檻
            if single and inst_lab[i] == nm and inst_cos[i] >= NAME_COS:
                name_found = 1
        rows.append({"scene": scene, "gt_name": nm,
                     "best_cov": round(best_cov, 4),
                     "found3d": found3d, "name_found": name_found,
                     "n_hull": n_valid, "n_gt": len(gt_names)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["n3_scene0001"])
    ap.add_argument("--root", default="instance_hull")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    scenes = HC.resolve_scenes(args.scenes)
    P, shape = HC.build_grid()
    allrows = []
    for i, sc in enumerate(scenes, 1):
        try:
            r = process_scene(sc, args.root, P, shape)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}"); r = None
        if r:
            allrows += r
            n_gt = len(r)
            print(f"[{i}/{len(scenes)}] {sc}: GT{n_gt} "
                  f"3D漏檢{1-np.mean([x['found3d'] for x in r]):.2f} "
                  f"名稱漏檢{1-np.mean([x['name_found'] for x in r]):.2f} "
                  f"best_cov均{np.mean([x['best_cov'] for x in r]):.3f}")
        else:
            print(f"[{i}/{len(scenes)}] {sc}: skip")
    if args.csv and allrows:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
        print(f"→ {len(allrows)} 列 → {args.csv}")
    if allrows:
        # 逐場景比率 → 場景平均
        import collections
        by = collections.defaultdict(list)
        for x in allrows:
            by[x["scene"]].append(x)
        miss3d = [1 - np.mean([x["found3d"] for x in v]) for v in by.values()]
        missnm = [1 - np.mean([x["name_found"] for x in v]) for v in by.values()]
        bc = [np.mean([x["best_cov"] for x in v]) for v in by.values()]
        print(f"\n==== {args.root}: {len(by)} 場景 ====")
        print(f"3D 漏檢率(無 hull 覆蓋率>0.5 且只罩一個 GT)= {np.mean(miss3d):.3f}")
        print(f"名稱漏檢率(找到的 hull 上 argmax=該GT 且 cos>={NAME_COS})= {np.mean(missnm):.3f}")
        print(f"best_cov 平均(全 GT)= {np.mean(bc):.3f}")


if __name__ == "__main__":
    main()
