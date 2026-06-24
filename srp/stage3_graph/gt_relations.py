#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""gt_relations.py — 從 GT 生成關係標籤(MRG):on(支撐)+ blocks_access(視覺遮擋)。

對齊 plan_check_schema:on(X,Y)=X 在上 Y 在下;blocks_access(X,Y)=X 擋住接近 Y(此版用視覺遮擋為證據)。
兩關係用不同 GT 來源(皆模擬器真值,不用預測、不用深度):
  on            ← GT 物體位姿 + YCB mesh 頂點(世界座標)→ xy-AABB footprint + top/bot/質心 + 幾何判準。
  blocks_access ← GT amodal(單物完整)− modal(整場景含遮擋)遮罩 = 被遮區域,找蓋住它的物體 = 遮擋者。
輸出 data/labels/<scene>/relations.json。此為 plan「REGRAD 式」的幾何+遮罩近似(物理 drop-test 留 v2)。
需 webots_visual_hull(trimesh/pycocotools)。
用法: ./srp/stage3_graph/gt_relations.py <scenes>  (n3_scene0001 / 組號略,直接列場景)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import eval_mesh as EM          # noqa: E402  (ycb_center, aa_to_mat, load_mesh, gt_objects)
import eval_reproj2d as RP      # noqa: E402  (load_modal_by_view)

LABELS = REPO / "data" / "labels"

# on 幾何門檻
PEN, GAP, ON_XY = 0.015, 0.03, 0.30
# blocks_access(視覺遮擋)門檻
OCC_MIN = 0.10        # 物體 i 在某視角被遮 ≥ 此比例才算被遮
OCCLUDER_MIN = 0.30   # 遮擋者 j 需蓋住 i 被遮區域 ≥ 此比例
MIN_VIEWS = 2         # 需在 ≥ 此視角數出現才記 blocks_access


def obj_geom(scene):
    """每物體世界座標幾何 {id: dict(xmin,xmax,ymin,ymax,top,bot,cenz,area)};id 同名以 #k 區分。"""
    out, seen = {}, {}
    for o in EM.gt_objects(scene):
        name = o["name"]; m = EM.load_mesh(name)
        if m is None:
            continue
        R = EM.aa_to_mat(o.get("rotation_axis_angle", [0, 1, 0, 0])[:3],
                         o.get("rotation_axis_angle", [0, 1, 0, 0])[3])
        V = (m.vertices - EM.ycb_center(name)) @ R.T + np.asarray(o["position_m"], float)
        k = seen.get(name, 0); seen[name] = k + 1
        oid = name if k == 0 and EM_count(scene, name) == 1 else f"{name}#{k}"
        out[oid] = {"xmin": V[:, 0].min(), "xmax": V[:, 0].max(),
                    "ymin": V[:, 1].min(), "ymax": V[:, 1].max(),
                    "top": V[:, 2].max(), "bot": V[:, 2].min(),
                    "cenz": float(V[:, 2].mean()),
                    "area": (V[:, 0].max() - V[:, 0].min()) * (V[:, 1].max() - V[:, 1].min())}
    return out


def EM_count(scene, name):
    return sum(1 for o in EM.gt_objects(scene) if o["name"] == name)


def xy_overlap(a, b):
    dx = max(0.0, min(a["xmax"], b["xmax"]) - max(a["xmin"], b["xmin"]))
    dy = max(0.0, min(a["ymax"], b["ymax"]) - max(a["ymin"], b["ymin"]))
    return dx * dy


def compute_on(geom):
    rels = []
    for X in geom:
        for Y in geom:
            if X == Y:
                continue
            gx, gy = geom[X], geom[Y]
            contact = -PEN <= (gx["bot"] - gy["top"]) <= GAP
            overlap = xy_overlap(gx, gy) / gx["area"] if gx["area"] > 0 else 0
            above = gx["cenz"] > gy["cenz"]
            if contact and overlap >= ON_XY and above:
                rels.append({"type": "on", "x": X, "y": Y,
                             "gap": round(float(gx["bot"] - gy["top"]), 4),
                             "xy_overlap": round(float(overlap), 3)})
    return rels


def load_amodal_by_view(scene):
    ann = LABELS / scene / "amodal" / "annotations.json"
    if not ann.is_file():
        return None
    d = json.loads(ann.read_text())
    cat = {c["id"]: c["name"] for c in d["categories"]}
    view_of = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
    out = {}
    for a in d["annotations"]:
        m = mask_utils.decode(a["segmentation"]).astype(bool)
        if m.sum() == 0:
            continue
        out.setdefault(view_of[a["image_id"]], {})[cat[a["category_id"]]] = m
    return out


def compute_blocks(scene):
    amodal = load_amodal_by_view(scene)
    modal, _ = RP.load_modal_by_view(scene)
    if not amodal or not modal:
        return []
    pair = {}   # (occluder j, occluded i) -> [n_views, max_frac]
    for v in amodal:
        if v not in modal:
            continue
        am, mo = amodal[v], modal[v]
        for i, am_i in am.items():
            mo_i = mo.get(i, np.zeros_like(am_i))
            hidden = am_i & ~mo_i
            hf = int(hidden.sum()) / int(am_i.sum()) if am_i.sum() else 0
            if hf < OCC_MIN:
                continue
            # 找蓋住被遮區域最多的物體 j
            best_j, best_cov = None, 0.0
            for j, mo_j in mo.items():
                if j == i:
                    continue
                cov = int((hidden & mo_j).sum()) / int(hidden.sum()) if hidden.sum() else 0
                if cov > best_cov:
                    best_cov, best_j = cov, j
            if best_j is not None and best_cov >= OCCLUDER_MIN:
                k = (best_j, i)
                p = pair.setdefault(k, [0, 0.0])
                p[0] += 1; p[1] = max(p[1], hf)
    rels = []
    for (j, i), (nv, mf) in pair.items():
        if nv >= MIN_VIEWS:
            rels.append({"type": "blocks_access", "x": j, "y": i,
                         "n_views": nv, "max_occ_frac": round(mf, 3)})
    return rels


def process(scene):
    geom = obj_geom(scene)
    if not geom:
        print(f"[skip] {scene}: 無 GT 物體"); return None
    on = compute_on(geom)
    blocks = compute_blocks(scene)
    out = {"scene": scene, "objects": list(geom),
           "relations": on + blocks,
           "params": {"PEN": PEN, "GAP": GAP, "ON_XY": ON_XY,
                      "OCC_MIN": OCC_MIN, "OCCLUDER_MIN": OCCLUDER_MIN, "MIN_VIEWS": MIN_VIEWS}}
    (LABELS / scene).mkdir(parents=True, exist_ok=True)
    (LABELS / scene / "relations.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{scene}] 物{len(geom)} → on {len(on)} blocks {len(blocks)}  "
          f"{[(r['x'],'on',r['y']) for r in on] + [(r['x'],'blk',r['y']) for r in blocks]}")
    return len(on), len(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    args = ap.parse_args()
    to, tb, n = 0, 0, 0
    for sc in args.scenes:
        try:
            r = process(sc)
            if r:
                to += r[0]; tb += r[1]; n += 1
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")
    print(f"\n== {n} 場景 | on {to} blocks_access {tb} → data/labels/<scene>/relations.json ==")


if __name__ == "__main__":
    main()
