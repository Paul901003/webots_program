#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""a1_rule.py — A1 probing(規則基線):幾何規則能否復現 GT 關係(on / blocks_access)。

評估以**三元組 (type, x, y) 精確配對**為準(主體+受體+類型全對才算 TP),非數量。
規則(與 gt_relations 同義,但作用在「重建幾何」上):
  on(X,Y)          : X 底 ≈ Y 頂(接觸) + footprint 重疊 + X 在上。
  blocks_access(X,Y): 每視角「重投影遮罩 − IoU最高 SAM 遮罩」= 被遮區,誰蓋最多=遮擋者(與 GT 同套遮罩差)。
兩種幾何來源:
  mesh : GT 實心 mesh 佔據(eval_mesh.solid_mesh_occ)→ on 近恆等(上界);blocks 測幾何能否復現視覺遮擋。
  hull : srp 重建 instance(labels;對到 GT 名)→ 真實免深度可行性(含 hull 噪声)= plan E3。
用法: ./srp/stage4_probe/a1_rule.py [scenes...] [--geom mesh|hull|both]
  不給 scenes → 全 stack/occ 120 場。
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import camera as cam            # noqa: E402
import eval_mesh as EM          # noqa: E402
import masks as MK              # noqa: E402  (kept_object_masks:SAM 遮罩)

# 路徑可用 env 覆寫(fast 資料:HULL_ROOT=srp_hull_fast、CAPTURES_ROOT=captures_fast、SAM_ROOT=sam_only_fast)
HULL = Path(os.environ.get("HULL_ROOT", str(REPO / "data" / "eval" / "srp_hull")))
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "srp" / "io")); from labels import LABELS  # data/labels 分層(類別/數量/場景)
CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures")))
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "sam_only")))

PEN, GAP, ON_XY = 0.015, 0.03, 0.30
OCC_MIN, OCCLUDER_MIN = 0.10, 0.30
CLOSE_K = 9   # 重投影稀疏點閉運算填實心的 kernel(體素 5mm 在 ~0.65m 投影約 5–6px)
DS = 4        # 舊 z-buffer raster 降採樣(已不被 rule_blocks 使用,保留供參考)


def entity_geom(occ, gm, vs):
    idx = np.array(np.nonzero(occ)).T
    c = gm + (idx + 0.5) * vs
    return {"c": c, "xmin": c[:, 0].min(), "xmax": c[:, 0].max(),
            "ymin": c[:, 1].min(), "ymax": c[:, 1].max(),
            "top": c[:, 2].max(), "bot": c[:, 2].min(), "cenz": float(c[:, 2].mean()),
            "area": (c[:, 0].max() - c[:, 0].min()) * (c[:, 1].max() - c[:, 1].min())}


def rule_on(G):
    rels = []
    for X in G:
        for Y in G:
            if X == Y:
                continue
            gx, gy = G[X], G[Y]
            dx = max(0.0, min(gx["xmax"], gy["xmax"]) - max(gx["xmin"], gy["xmin"]))
            dy = max(0.0, min(gx["ymax"], gy["ymax"]) - max(gx["ymin"], gy["ymin"]))
            ov = dx * dy / gx["area"] if gx["area"] > 0 else 0
            if -PEN <= gx["bot"] - gy["top"] <= GAP and ov >= ON_XY and gx["cenz"] > gy["cenz"]:
                rels.append(("on", X, Y))
    return rels


def raster(c, K, Rwc, t, H, W):
    """回傳 (mask bool, depth float[min z per pixel, inf 空])。"""
    X = c @ Rwc.T + t
    z = X[:, 2]; ok = z > 1e-9
    u = np.round(K[0, 0] * X[:, 0] / np.where(ok, z, 1) / DS + K[0, 2] / DS).astype(int)
    v = np.round(K[1, 1] * X[:, 1] / np.where(ok, z, 1) / DS + K[1, 2] / DS).astype(int)
    inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    depth = np.full((H, W), np.inf)
    np.minimum.at(depth, (v[inb], u[inb]), z[inb])
    return depth < np.inf, depth


def reproject_mask(c, K, Rwc, t, H, W):
    """體素中心 full-res 投影 → 閉運算填實心 bool mask(完整輪廓,對應 GT amodal)。"""
    X = c @ Rwc.T + t
    z = X[:, 2]; ok = z > 1e-9
    zz = np.where(ok, z, 1.0)
    u = np.round(K[0, 0] * X[:, 0] / zz + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * X[:, 1] / zz + K[1, 2]).astype(int)
    inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    m = np.zeros((H, W), np.uint8)
    m[v[inb], u[inb]] = 1
    if m.any():
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((CLOSE_K, CLOSE_K), np.uint8))
    return m.astype(bool)


def _iou2d(a, b):
    inter = int(np.logical_and(a, b).sum())
    return inter / int(np.logical_or(a, b).sum()) if inter else 0.0


def rule_blocks(G, scene):
    """每視角「重投影遮罩 − IoU最高 SAM 遮罩」= 被遮區 → 遮擋者;四元組 (blocks_access, X, Y, view)。
    與 gt_relations.compute_blocks(amodal−modal)完全同套遮罩差:reproj=amodal、vis(IoU最高SAM)=modal。"""
    grp = scene.split("_")[0]
    sdir = CAPTURES / f"multi_{grp}" / scene
    poses = sorted(sdir.glob("view_*_pose.json"))
    if not poses:
        return []
    K = cam.intrinsics(1280, 720)
    names = list(G)
    rels = []
    for pf in poses:
        vname = pf.stem.replace("_pose", "")           # view_elXX_azYY(與 SAM/amodal 命名一致)
        sam = MK.kept_object_masks(SAM_ROOT / scene / vname)   # [(bool_mask, filename)]
        if not sam:
            continue
        C, Rb = cam.load_pose(pf); Rwc, t = cam.pose_to_w2c(C, Rb)
        # 每實體:重投影完整輪廓 reproj + 與其 IoU 最高的 SAM 遮罩當可見輪廓 vis
        reproj, vis = {}, {}
        for n in names:
            r = reproject_mask(G[n]["c"], K, Rwc, t, 720, 1280)
            if not r.any():
                continue
            best_s, best_iou = None, 0.0
            for s, _ in sam:
                iou = _iou2d(r, s)
                if iou > best_iou:
                    best_iou, best_s = iou, s
            reproj[n] = r
            vis[n] = best_s if best_s is not None else np.zeros_like(r)
        for Y in reproj:
            hidden = reproj[Y] & ~vis[Y]               # 完整輪廓 − 可見 = 被遮區
            hf = int(hidden.sum()) / int(reproj[Y].sum()) if reproj[Y].sum() else 0
            if hf < OCC_MIN:
                continue
            best_x, best_cov = None, 0.0               # 遮擋者 = 其可見 SAM 蓋住被遮區最多者
            for X in reproj:
                if X == Y:
                    continue
                cov = int((hidden & vis[X]).sum()) / int(hidden.sum()) if hidden.sum() else 0
                if cov > best_cov:
                    best_cov, best_x = cov, X
            if best_x is not None and best_cov >= OCCLUDER_MIN:
                rels.append(("blocks_access", best_x, Y, vname))
    return rels


def mesh_entities(scene, gm, vs, shape):
    gt = EM.solid_mesh_occ(scene, gm, vs, shape)
    return {n: entity_geom(occ, gm, vs) for n, occ in gt.items()} if gt else {}


def hull_entities(scene, gm, vs, shape):
    ip = HULL / scene / "instances.npz"
    if not ip.is_file():
        return {}
    labels = np.load(ip)["labels"]
    gt = EM.solid_mesh_occ(scene, gm, vs, shape)
    if not gt:
        return {}
    names = list(gt); meshes = [gt[n] for n in names]
    insts = [k for k in range(1, int(labels.max()) + 1) if (labels == k).any()]
    occs = [labels == k for k in insts]
    M = np.array([[EM.iou3(o, m) for m in meshes] for o in occs]) if occs else np.zeros((0, 0))
    name_of = {}
    if len(occs) and len(names):
        ri, cj = linear_sum_assignment(-M)
        for i, j in zip(ri, cj):
            name_of[i] = names[j] if M[i, j] > 0 else f"inst{insts[i]}"
    return {name_of.get(i, f"inst{insts[i]}"): entity_geom(occs[i], gm, vs) for i in range(len(occs))}


def gt_triples(scene):
    """on = 全局三元組 (type,x,y);blocks_access = 每視角四元組 (type,x,y,view)。"""
    f = LABELS / scene / "relations.json"
    if not f.is_file():
        return None
    out = set()
    for r in json.loads(f.read_text())["relations"]:
        if r["type"] == "blocks_access":
            out.add((r["type"], r["x"], r["y"], r["view"]))
        else:
            out.add((r["type"], r["x"], r["y"]))
    return out


def score(pred, gt):
    """回傳 per-type {tp,fp,fn}。"""
    res = {}
    for typ in ("on", "blocks_access"):
        P = {t for t in pred if t[0] == typ}; G = {t for t in gt if t[0] == typ}
        res[typ] = [len(P & G), len(P - G), len(G - P)]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--geom", choices=["mesh", "hull", "both"], default="both")
    args = ap.parse_args()
    scenes = args.scenes or sorted(
        Path(p).name for p in glob.glob(str(HULL / "stack*")) + glob.glob(str(HULL / "occ*")))
    geoms = ["mesh", "hull"] if args.geom == "both" else [args.geom]

    tot = {g: {"on": [0, 0, 0], "blocks_access": [0, 0, 0]} for g in geoms}
    for sc in scenes:
        gt = gt_triples(sc)
        if gt is None:
            continue
        hp = HULL / sc / "hull.npz"
        if not hp.is_file():
            continue
        z = np.load(hp); gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = z["occupancy"].shape
        for g in geoms:
            G = mesh_entities(sc, gm, vs, shape) if g == "mesh" else hull_entities(sc, gm, vs, shape)
            if not G:
                continue
            pred = set(rule_on(G)) | set(rule_blocks(G, sc))
            s = score(pred, gt)
            for typ in tot[g]:
                for k in range(3):
                    tot[g][typ][k] += s[typ][k]

    print(f"{'geom':<6}{'type':<15}{'TP':>5}{'FP':>5}{'FN':>5}{'P':>7}{'R':>7}{'F1':>7}")
    for g in geoms:
        for typ in ("on", "blocks_access"):
            tp, fp, fn = tot[g][typ]
            P = tp / (tp + fp) if tp + fp else 0
            R = tp / (tp + fn) if tp + fn else 0
            F = 2 * P * R / (P + R) if P + R else 0
            print(f"{g:<6}{typ:<15}{tp:>5}{fp:>5}{fn:>5}{P:>7.3f}{R:>7.3f}{F:>7.3f}")


if __name__ == "__main__":
    main()
