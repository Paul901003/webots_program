#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""eval_mesh.py — 殼 vs 真實 mesh(強制實心)評估:找到率 + 重疊度 + 冗餘程度。

與 eval.py(vs GT 視覺 hull)不同:這裡比的是**真實實心 mesh**,才能量出殼的「冗餘」
(visual hull 的 shadow/ghost 過估計)。凹腔被填實(force solid),不把固有凹面限制算進冗餘。

GT 實心佔據(robust,不靠 mesh 水密):
  mesh.sample(N) 取密集表面點 → 標到網格體素 → scipy.binary_fill_holes 填內部 → 實心。
  (開口物如 bowl 的空腔若連通到外界可能未完全填滿,屬已知近似。)快取於 data/eval/gt_mesh_cache/。

配對 + 指標(每個配對到的 instance 殼 H vs GT 實心 mesh M):
  - 配對:殼×mesh 的 3D IoU 做 Hungarian 一對一。
  - found:配對 IoU ≥ 門檻(逐一一殼一mesh) → recall = found/GT 數。
  - 重疊度:IoU = |H∩M|/|H∪M|;覆蓋 cover = |H∩M|/|M|(殼是否包住真物)。
  - 冗餘:純度 purity = |H∩M|/|H|(冗餘比 = 1−purity);膨脹比 bloat = |H|/|M|。
只跑多物體場景(n3/n4/n5)較有意義,但任意場景皆可。需 webots_visual_hull(trimesh/scipy)。

用法: ./srp/stage2_instances/eval_mesh.py <scenes> --root srp_sweep --tag am0_cvlarge_ag50 [--iou 0.25]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "srp" / "io")); from labels import LABELS  # data/labels 分層(類別/數量/場景)
ASSETS = REPO / "urdfs" / "ycb_assets"
EVAL = REPO / "data" / "eval"
MESH_CACHE = EVAL / "gt_mesh_cache"
GEO = json.loads((REPO / "controllers" / "ycb_supervisor" / "ycb_geometries.json").read_text())
N_SAMPLE = 200000


def ycb_center(name):
    c = GEO.get(name, {}).get("center", {"x": 0, "y": 0, "z": 0})
    return np.array([c["x"], c["y"], c["z"]], float)


def aa_to_mat(axis, ang):
    axis = np.asarray(axis, float); n = np.linalg.norm(axis)
    if n < 1e-9 or abs(ang) < 1e-12:
        return np.eye(3)
    x, y, z = axis / n; c, s = np.cos(ang), np.sin(ang); C = 1 - c
    return np.array([[c + x*x*C, x*y*C - z*s, x*z*C + y*s],
                     [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
                     [z*x*C - y*s, z*y*C + x*s, c + z*z*C]])


def load_mesh(name):
    for fn in ("textured.obj", "nontextured.ply", "nontextured.stl"):
        p = ASSETS / name / "google_16k" / fn
        if p.is_file():
            return trimesh.load(str(p), force="mesh")
    return None


def gt_objects(scene):
    ann = LABELS / scene / "actual" / "annotations.json"
    if not ann.is_file():
        return []
    d = json.loads(ann.read_text())
    return d["images"][0].get("objects", []) if d.get("images") else []


def solid_mesh_occ(scene, grid_min, vs, shape):
    """每物體真實實心佔據(同網格)。回傳 {name: occ(bool)}。快取。"""
    cp = MESH_CACHE / scene / f"solid_v{int(round(vs*10000))}.npz"
    if cp.is_file():
        z = np.load(cp)
        if np.allclose(z["grid_min"], grid_min) and tuple(z["shape"]) == tuple(shape):
            return {str(n): z["occ"][i] for i, n in enumerate(z["names"])}
    out = {}
    for o in gt_objects(scene):
        name = o["name"]; m = load_mesh(name)
        if m is None:
            continue
        R = aa_to_mat(o.get("rotation_axis_angle", [0, 1, 0, 0])[:3],
                      o.get("rotation_axis_angle", [0, 1, 0, 0])[3])
        Vw = (m.vertices - ycb_center(name)) @ R.T + np.asarray(o["position_m"], float)
        mw = trimesh.Trimesh(vertices=Vw, faces=m.faces, process=False)
        pts, _ = trimesh.sample.sample_surface(mw, N_SAMPLE)
        idx = np.floor((pts - grid_min) / vs).astype(int)
        ok = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
        surf = np.zeros(shape, bool)
        idx = idx[ok]; surf[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        solid = ndimage.binary_fill_holes(surf)         # 強制實心(填封閉內部)
        if solid.any():
            out[name] = solid
    if out:
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cp, names=np.array(list(out)),
                            occ=np.stack([out[n] for n in out]),
                            grid_min=np.asarray(grid_min, float), shape=np.asarray(shape))
    return out


def iou3(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / float(u) if u else 0.0


def process(scene, tag, root, cover_thr, rows):
    ip = EVAL / root / scene / f"instances{('_'+tag) if tag else ''}.npz"
    if not ip.is_file():
        return None
    z = np.load(ip); labels = z["labels"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
    shape = labels.shape
    hulls = [labels == k for k in range(1, int(labels.max()) + 1) if (labels == k).any()]
    gt = solid_mesh_occ(scene, gm, vs, shape)
    if not gt:
        return None
    names = list(gt); meshes = [gt[n] for n in names]
    nH, nG = len(hulls), len(meshes)
    M = np.zeros((nH, nG))                         # 配對仍用 IoU(穩健的一對一對應)
    for i in range(nH):
        for j in range(nG):
            M[i, j] = iou3(hulls[i], meshes[j])
    matched = {}
    if nH and nG:
        ri, cj = linear_sum_assignment(-M)
        for i, j in zip(ri, cj):
            matched[j] = i
    found = 0
    for j, name in enumerate(names):
        if j in matched:
            i = matched[j]; H = hulls[i]; Mm = meshes[j]
            inter = int(np.logical_and(H, Mm).sum()); hv = int(H.sum()); mv = int(Mm.sum())
            cover = inter / mv                     # 覆蓋率:殼包住真物的比例
            is_found = cover >= cover_thr           # 「找到」= 覆蓋率達門檻(不懲罰膨脹)
            rows.append({"scene": scene, "object": name, "iou": round(M[i, j], 3),
                         "cover": round(cover, 3), "purity": round(inter/hv, 3),
                         "redundancy": round(1 - inter/hv, 3), "bloat": round(hv/mv, 3),
                         "found": int(is_found)})
            found += int(is_found)
        else:
            rows.append({"scene": scene, "object": name, "iou": 0, "cover": 0,
                         "purity": 0, "redundancy": 0, "bloat": 0, "found": 0})
    return found, nG, nH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--root", default="srp_hull")
    ap.add_argument("--tag", default="")
    ap.add_argument("--cover-thr", type=float, default=0.5, dest="cover_thr",
                    help="找到判準:殼覆蓋真物比例 ≥ 此值(預設 0.5;不懲罰膨脹)")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    rows = []; tot_found = tot_gt = tot_hull = 0
    for sc in args.scenes:
        try:
            r = process(sc, args.tag, args.root, args.cover_thr, rows)
            if r:
                tot_found += r[0]; tot_gt += r[1]; tot_hull += r[2]
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")
    fr = [r for r in rows if r["found"]]
    if not fr:
        print("無配對"); return
    mean = lambda k: round(float(np.mean([r[k] for r in fr])), 3)
    print(f"== tag={args.tag or '(baseline)'} | 物體 {tot_gt} 殼 {tot_hull} | "
          f"found {tot_found}/{tot_gt}(recall {tot_found/tot_gt:.3f}) | "
          f"配對均: IoU {mean('iou')} 覆蓋 {mean('cover')} 純度 {mean('purity')} "
          f"冗餘 {mean('redundancy')} 膨脹 {mean('bloat')} ==")
    csv_path = args.csv or str(EVAL / args.root / f"mesh_eval{('_'+args.tag) if args.tag else ''}.csv")
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"→ {csv_path}")


if __name__ == "__main__":
    main()
