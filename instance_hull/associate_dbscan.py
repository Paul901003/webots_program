#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""associate_dbscan.py — associate 的 DBSCAN 版(實驗,結果獨立存放)。

與 associate_hdbscan.py 同樣做「多視角幾何關聯」,但分群換成 DBSCAN:
  HDBSCAN+回併在「細長物(沿長軸切成多群)」和「相鄰物體(回併會誤併)」之間
  threading 不過去(沒有單一 merge_margin 能同時對)。DBSCAN 用「點與點局部連通」
  分群 → 細長物的候選點沿軸連成一條(不看整體形狀),物體間的空檔又斷得開,
  窗口比「群中心距回併」寬得多,且不需要回併步驟。

設計:
  ① eps 自適應:用 k-近鄰距離(k=min_samples)的中位數 × eps_factor,
     讓尺度由候選點的局部間距決定,而非手設固定長度。
  ② min_samples 綁 MIN_SUPPORT(小絕對值,與候選點總數無關)→ 稀疏單物體場景不歸零。
  ③ per-mask 物理半徑認領遮罩(同 hdbscan 版)。無回併。

輸出: data/eval/instance_hull_dbscan/<scene>/instances.json(+ assoc_report.txt)
       —— 與 instance_hull/、instance_hull_hdbscan/ 完全分開。
需在 webots_visual_hull 環境(sklearn;numpy/cv2)。

用法: ./instance_hull/associate_dbscan.py n3_scene0001   (或組號 3 / 多組 1 3 4 5)
       [--min-samples 4] [--eps-factor 1.5] [--eps 固定值override]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parents[1]
CAPTURES = REPO / "data" / "captures"
SAM_ROOT = REPO / "data" / "eval" / "sam_only"
OUT_ROOT = REPO / "data" / "eval" / "instance_hull_dbscan"   # ← 獨立目錄

HFOV_RAD = 1.4746

# ── 背景遮罩過濾 ──────────────────────────────────────────────────────────────
MAX_AREA_FRAC = 0.30
BORDER = 2

# ── 工作空間範圍 ──────────────────────────────────────────────────────────────
WS_X = (-0.05, 0.75)
WS_Y = (-0.45, 0.45)
WS_Z = (-0.05, 0.45)

# ── 候選交會點門檻 ────────────────────────────────────────────────────────────
RAY_TAU = 0.05
MIN_SUPPORT = 3

# ── per-mask 認領半徑下限/上限 ────────────────────────────────────────────────
RADIUS_FLOOR = 0.02
RADIUS_CAP = 0.12

# ── eps 自適應夾限(避免極端場景估出離譜 eps)──────────────────────────────────
EPS_FLOOR = 0.005
EPS_CAP = 0.06


def rpy_to_R(roll, pitch, yaw):
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


OPENCV_TO_BODY = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
BODY_TO_OPENCV = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)


def load_pose(pose_path):
    meta = json.loads(pose_path.read_text(encoding="utf-8"))
    if "position_m" not in meta and isinstance(meta.get("camera"), dict):
        meta = meta["camera"]
    p = meta["position_m"]
    C = np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
    r = meta["rotation_rpy_rad"]
    return C, rpy_to_R(r["roll"], r["pitch"], r["yaw"])


def intrinsics(W, H):
    fx = W / (2.0 * math.tan(HFOV_RAD / 2.0))
    return fx, fx, W / 2.0, H / 2.0


def pixel_to_ray(u, v, C, R, K):
    fx, fy, cx, cy = K
    d_cv = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    d_world = R @ (OPENCV_TO_BODY @ d_cv)
    return d_world / np.linalg.norm(d_world)


def closest_point(o1, d1, o2, d2):
    b = float(d1 @ d2)
    denom = 1.0 - b * b
    if abs(denom) < 1e-9:
        return None, 1e9, -1, -1
    w0 = o1 - o2
    d = float(d1 @ w0)
    e = float(d2 @ w0)
    s = (b * e - d) / denom
    t = (e - b * d) / denom
    p1, p2 = o1 + s * d1, o2 + t * d2
    return (p1 + p2) * 0.5, float(np.linalg.norm(p1 - p2)), s, t


def in_workspace(p):
    return (WS_X[0] <= p[0] <= WS_X[1] and WS_Y[0] <= p[1] <= WS_Y[1]
            and WS_Z[0] <= p[2] <= WS_Z[1])


def point_ray_dist(c, o, d):
    v = c - o
    proj = float(v @ d)
    if proj <= 0:
        return 1e9
    return float(np.linalg.norm(v - proj * d))


def adaptive_eps(cand, k, factor):
    """eps = 第 k 近鄰距離的中位數 × factor(尺度由候選點局部間距決定)。"""
    k = min(k, len(cand) - 1)
    if k < 1:
        return EPS_FLOOR
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(cand)
    dists, _ = nbrs.kneighbors(cand)
    kth = dists[:, -1]                 # 每點到第 k 近鄰的距離
    eps = float(np.median(kth)) * factor
    return min(max(eps, EPS_FLOOR), EPS_CAP)


def touches_border(seg):
    return bool(seg[:BORDER].any() or seg[-BORDER:].any()
                or seg[:, :BORDER].any() or seg[:, -BORDER:].any())


def load_view_masks(view_dir):
    mask_dir = view_dir / "masks"
    out = []
    for mp in sorted(mask_dir.glob("mask_*.png")):
        seg = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if seg is None:
            continue
        seg = seg > 127
        H, W = seg.shape
        area = int(seg.sum())
        if area == 0 or area > MAX_AREA_FRAC * H * W or touches_border(seg):
            continue
        ys, xs = np.nonzero(seg)
        out.append({"uv": (float(xs.mean()), float(ys.mean())),
                    "area": area, "file": mp.name})
    return out


def resolve_scenes(targets):
    scenes = []
    for a in targets:
        if "scene" in a:
            scenes.append(a)
        else:
            scenes += [d.name for d in sorted((CAPTURES / f"multi_n{a}").glob(f"n{a}_scene*"))]
    return scenes


def process_scene(scene, args):
    group = scene.split("_")[0]
    scene_dir = CAPTURES / f"multi_{group}" / scene
    sam_dir = SAM_ROOT / scene
    if not sam_dir.is_dir():
        print(f"[skip] {scene}: 找不到 SAM 遮罩 {sam_dir}(先跑 sam_only.py)")
        return

    out_dir = OUT_ROOT / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 每 view:姿態 + 遮罩 + 質心射線 + fx
    views = []
    for vdir in sorted(sam_dir.glob("view_*")):
        name = vdir.name
        pose_path = scene_dir / f"{name}_pose.json"
        if not pose_path.is_file():
            continue
        C, R = load_pose(pose_path)
        masks = load_view_masks(vdir)
        if not masks:
            continue
        any_mp = next((sam_dir / name / "masks").glob("mask_*.png"))
        H, W = cv2.imread(str(any_mp), cv2.IMREAD_GRAYSCALE).shape
        K = intrinsics(W, H)
        fx = K[0]
        for m in masks:
            m["ray"] = pixel_to_ray(m["uv"][0], m["uv"][1], C, R, K)
        views.append({"name": name, "C": C, "fx": fx, "masks": masks})
    if len(views) < 2:
        print(f"[skip] {scene}: 有效 view < 2,無法三角化")
        return
    print(f"{scene}: {len(views)} views,遮罩數 {[len(v['masks']) for v in views]}")

    # 2) 跨 view 兩兩射線交會 → 候選點
    cand = []
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            for mi in views[i]["masks"]:
                for mj in views[j]["masks"]:
                    p, dist, s, t = closest_point(views[i]["C"], mi["ray"],
                                                  views[j]["C"], mj["ray"])
                    if p is None or s <= 0 or t <= 0:
                        continue
                    if dist <= RAY_TAU and in_workspace(p):
                        cand.append(p)
    if len(cand) < MIN_SUPPORT:
        print(f"[skip] {scene}: 候選交會點 {len(cand)} 太少")
        return
    cand = np.asarray(cand)

    # 3) DBSCAN:min_samples 綁 MIN_SUPPORT、eps 由 k-近鄰自適應(或 --eps override)
    msamp = args.min_samples
    eps = args.eps if args.eps is not None else adaptive_eps(cand, msamp, args.eps_factor)
    labels = DBSCAN(eps=eps, min_samples=msamp).fit_predict(cand)
    uniq = sorted(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"候選交會點: {len(cand)}  eps={eps:.4f} min_samples={msamp} "
          f"→ DBSCAN {len(uniq)} 群 (+{n_noise} noise)")
    centers = [np.median(cand[labels == lab], axis=0) for lab in uniq]

    # 4) per-mask 物理半徑認領
    flat = [(vi, mi) for vi, v in enumerate(views) for mi in range(len(v["masks"]))]
    assign = {}
    cluster_pts = {k: [] for k in range(len(centers))}
    for (vi, mi) in flat:
        v = views[vi]
        m = v["masks"][mi]
        o, d, fx = v["C"], m["ray"], v["fx"]
        best_k, best_dist = None, 1e9
        for k, c in enumerate(centers):
            pr = point_ray_dist(c, o, d)
            if pr >= best_dist:
                continue
            r = float(np.linalg.norm(c - o))
            rho = math.sqrt(m["area"] / math.pi)
            phys_r = r * rho / fx * args.radius_margin
            phys_r = min(max(phys_r, RADIUS_FLOOR), RADIUS_CAP)
            if pr <= phys_r:
                best_k, best_dist = k, pr
        if best_k is not None:
            assign.setdefault(best_k, {}).setdefault(v["name"], []).append(m["file"])
            cluster_pts[best_k].append(o + max(0.0, float((centers[best_k] - o) @ d)) * d)

    # 5) 組 instance(support >= MIN_SUPPORT),中心用認領點精修。無回併。
    instances = []
    for k, c in enumerate(centers):
        per_view = assign.get(k, {})
        if len(per_view) < MIN_SUPPORT:
            continue
        center = np.mean(cluster_pts[k], axis=0) if cluster_pts[k] else np.asarray(c)
        instances.append({"center": [round(float(x), 4) for x in center],
                          "support": len(per_view), "masks": per_view})
    instances.sort(key=lambda a: -a["support"])
    print(f"保留 instance 數: {len(instances)}")

    # 6) GT 驗證
    gt = []
    mani = scene_dir / "scene_manifest.json"
    if mani.is_file():
        m = json.loads(mani.read_text(encoding="utf-8"))
        for o in m["actual"]["viewpoints"][0]["objects"]:
            gt.append((o["name"], np.array(o["position_m"], dtype=np.float64)))
    report = [f"scene: {scene}  (DBSCAN 版)", f"views: {len(views)}",
              f"eps={eps:.4f} min_samples={msamp} eps_factor={args.eps_factor} "
              f"radius_margin={args.radius_margin}",
              f"候選點={len(cand)}  DBSCAN群={len(uniq)} noise={n_noise}",
              f"instances: {len(instances)}  (GT 物體數: {len(gt)})", ""]
    for k, inst in enumerate(instances):
        c = np.array(inst["center"])
        line = (f"inst_{k:02d}: center=({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f}) "
                f"support={inst['support']}/{len(views)}")
        if gt:
            name, dmin = min(((n, float(np.linalg.norm(c - p))) for n, p in gt),
                             key=lambda a: a[1])
            line += f"  最近GT={name} ({dmin*100:.1f}cm)"
        report.append(line)
    txt = "\n".join(report)
    print("\n" + txt)

    (out_dir / "instances.json").write_text(
        json.dumps({"scene": scene, "method": "dbscan",
                    "centers": [i["center"] for i in instances],
                    "instances": instances}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "assoc_report.txt").write_text(txt + "\n", encoding="utf-8")
    print(f"\n→ {out_dir}/instances.json、assoc_report.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["n3_scene0001"], help="場景名或組號")
    ap.add_argument("--min-samples", type=int, default=4, dest="min_samples",
                    help="DBSCAN min_samples(綁 MIN_SUPPORT 的小絕對值)")
    ap.add_argument("--eps-factor", type=float, default=1.5, dest="eps_factor",
                    help="自適應 eps = 第k近鄰距離中位數 × 此係數")
    ap.add_argument("--eps", type=float, default=None,
                    help="固定 eps(override 自適應;預設 None=自適應)")
    ap.add_argument("--radius-margin", type=float, default=1.3, dest="radius_margin",
                    help="per-mask 物理半徑 gate 的放大係數")
    args = ap.parse_args()
    scenes = resolve_scenes(args.scenes or ["n3_scene0001"])
    if not scenes:
        sys.exit("沒有場景")
    for i, scene in enumerate(scenes, 1):
        print(f"\n===== [{i}/{len(scenes)}] {scene} =====")
        try:
            process_scene(scene, args)
        except Exception as e:
            print(f"[error] {scene}: {e}")


if __name__ == "__main__":
    main()
