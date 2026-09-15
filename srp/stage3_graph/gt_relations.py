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
import cv2
from pycocotools import mask as mask_utils

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import eval_mesh as EM          # noqa: E402  (ycb_center, aa_to_mat, load_mesh, gt_objects)

import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "srp" / "io")); from labels import LABELS  # data/labels 分層(類別/數量/場景)

# on 幾何門檻
PEN, GAP, ON_XY = 0.015, 0.03, 0.30
# blocks_access(視覺遮擋)門檻 —— 每視角各自輸出,不跨視角累計
OCC_MIN = 0.0         # 保留(記錄用);實際不用比例門檻,改以腐蝕濾邊界噪點
OCCLUDER_MIN = 0.0    # 0=蓋住被遮區最多的物體即遮擋者,不設覆蓋門檻
# 被遮區腐蝕 1px:濾掉 amodal(單物渲染)vs modal(整場景渲染)在物體邊緣的抗鋸齒噪點
# (那種假 hidden 是 1~2px 薄層,腐蝕後歸零);真實遮擋是實質區塊,腐蝕後仍在。
_ERODE_K = np.ones((3, 3), np.uint8)
_ERODE_ITERS = 1


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


def load_modal_by_view(scene):
    """actual 遮罩(含遮擋的可見輪廓);view 名用 file_name stem(與 amodal 對稱),排除 ur5e。
    不用 eval_reproj2d 版:那以 camera_pos 對最近拍攝視角,view 命名與 amodal 的 file_name 不一致
    (amodal 34 view / 位置映射版僅 12 unique → 交集空 → blocks 恆 0)。"""
    ann = LABELS / scene / "actual" / "annotations.json"
    if not ann.is_file():
        return None
    d = json.loads(ann.read_text())
    cat = {c["id"]: c["name"] for c in d["categories"]}
    view_of = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
    out = {}
    for a in d["annotations"]:
        name = cat[a["category_id"]]
        if name == "ur5e":
            continue
        m = mask_utils.decode(a["segmentation"]).astype(bool)
        if m.sum() == 0:
            continue
        out.setdefault(view_of[a["image_id"]], {})[name] = m
    return out


def compute_blocks(scene):
    """每視角各自輸出遮擋(不跨視角累計):每視角每 (遮擋者 j → 被遮者 i) 一條,帶 view。"""
    amodal = load_amodal_by_view(scene)
    modal = load_modal_by_view(scene)
    if not amodal or not modal:
        return []
    rels = []
    for v in sorted(amodal):
        if v not in modal:
            continue
        am, mo = amodal[v], modal[v]
        view_rels = {}                          # (遮擋者 j, 被遮者 i) -> {hidden, occ_frac, occluder_cov}
        for i, am_i in am.items():
            mo_i = mo.get(i, np.zeros_like(am_i))
            hidden = am_i & ~mo_i
            hidden = cv2.erode(np.ascontiguousarray(hidden, np.uint8), _ERODE_K, iterations=_ERODE_ITERS)  # 腐蝕1px濾邊界噪點
            H = int(hidden.sum())
            if H == 0:                 # 腐蝕後空=薄邊界抗鋸齒噪點,非實質遮擋
                continue
            hf = H / int(am_i.sum()) if am_i.sum() else 0
            # 找蓋住被遮區域最多的物體 j
            best_j, best_cov = None, 0.0
            for j, mo_j in mo.items():
                if j == i:
                    continue
                cov = int((hidden & mo_j).sum()) / H
                if cov > best_cov:
                    best_cov, best_j = cov, j
            if best_j is not None and best_cov >= OCCLUDER_MIN:
                view_rels[(best_j, i)] = {"hidden": H, "occ_frac": round(hf, 3), "occluder_cov": round(best_cov, 3)}
        # 互遮取主向:同視角同一對物體若雙向都判到,只留被遮較多(hidden 大)的那向,消除物理上不可能的互遮
        for (j, i), d in view_rels.items():
            rev = view_rels.get((i, j))
            if rev is not None and (rev["hidden"] > d["hidden"] or (rev["hidden"] == d["hidden"] and (i, j) < (j, i))):
                continue                        # 反向被遮更多(或相等時保留字典序小者)→ 此向為次向噪點,丟
            d2 = {k: d[k] for k in ("occ_frac", "occluder_cov")}
            rels.append({"type": "blocks_access", "x": j, "y": i, "view": v, **d2})
    return rels


def process(scene):
    geom = obj_geom(scene)
    if not geom:
        print(f"[skip] {scene}: 無 GT 物體"); return None
    on = compute_on(geom)
    blocks = compute_blocks(scene)      # 每視角逐條(blocks 為 view 級,on 為全局)
    out = {"scene": scene, "objects": list(geom),
           "relations": on + blocks,
           "params": {"PEN": PEN, "GAP": GAP, "ON_XY": ON_XY,
                      "OCC_MIN": OCC_MIN, "OCCLUDER_MIN": OCCLUDER_MIN}}
    (LABELS / scene).mkdir(parents=True, exist_ok=True)
    (LABELS / scene / "relations.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    nbv = len({(r["x"], r["y"]) for r in blocks})   # 不重複的 (遮擋者,被遮者) 對數
    print(f"[{scene}] 物{len(geom)} → on {len(on)} blocks {len(blocks)}條/{nbv}對  "
          f"{[(r['x'],'on',r['y']) for r in on]}")
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
