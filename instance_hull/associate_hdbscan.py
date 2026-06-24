#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""associate_hdbscan.py — associate.py 的 HDBSCAN 版(實驗,結果獨立存放)。

與 associate.py 同樣做「多視角幾何關聯」,但把對應機制換成自適應:
  舊版:貪婪 set-cover + 固定 CLAIM_R(6cm)+ 固定 MERGE_D(9cm)
  本版:① HDBSCAN 對候選交會點分群(變密度、自動決定群數、丟 noise)
            → 取代 CLAIM_R/MERGE_D 的固定半徑,尺度由資料密度決定
        ② per-mask「物理半徑」認領遮罩:用遮罩自身像素大小 + 該中心到相機
            的距離換算物理半徑(仍不用深度)→ 物體大小自適應

主參數 min_cluster_size 是「整數計數」(一物體至少幾個交會點支持),有物理
意義、與物體尺度/距離脫鉤,對未知物體比固定長度門檻穩健。

輸出: data/eval/instance_hull_hdbscan/<scene>/instances.json(+ assoc_report.txt)
       —— 與 associate.py 的 data/eval/instance_hull/ 完全分開,不覆蓋舊結果。
需在 webots_visual_hull 環境(sklearn>=1.3 的 HDBSCAN;numpy/cv2)。

用法: ./instance_hull/associate_hdbscan.py n3_scene0001   (或組號 3 / 多組 1 3 4 5)
       [--min-cluster-size 5] [--radius-margin 1.3]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import HDBSCAN

REPO = Path(__file__).resolve().parents[1]
CAPTURES = REPO / "data" / "captures"
SAM_ROOT = REPO / "data" / "eval" / "sam_only"
OUT_ROOT = REPO / "data" / "eval" / "instance_hull_hdbscan"   # ← 獨立目錄

HFOV_RAD = 1.4746

# ── 背景遮罩過濾(同 associate.py)────────────────────────────────────────────
MAX_AREA_FRAC = 0.30
BORDER = 2

# ── 工作空間範圍(同 associate.py)────────────────────────────────────────────
WS_X = (-0.05, 0.75)
WS_Y = (-0.45, 0.45)
WS_Z = (-0.05, 0.45)

# ── 候選交會點門檻(僅用來產候選點,真正分群交給 HDBSCAN)──────────────────
RAY_TAU = 0.05             # 兩射線最近距離 < 此(m)→ 收為候選點(放寬一點,分群會濾)
MIN_SUPPORT = 3            # 一個 instance 至少幾個 view 支持

# ── per-mask 認領半徑的下限/上限與 margin(取代固定 CLAIM_R)──────────────────
RADIUS_FLOOR = 0.02        # 物理半徑下限(m),避免極小遮罩 gate 太緊
RADIUS_CAP = 0.12          # 物理半徑上限(m),避免巨大遮罩 gate 把鄰物吃掉


# ── 姿態 / 內參 / 投影(與 associate.py 一致)──────────────────────────────────
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


def merge_instances(insts, margin):
    """細長物回併:中心距 < (兩群物理半徑之和)×margin → 合併。每次併最近的一對,迭代到無可併。
    insts 元素:{center(np), masks(dict), pts(list[np]), radii(list[float]), radius(float)}。"""
    while len(insts) > 1:
        best = None  # (dist, i, j)
        for i in range(len(insts)):
            for j in range(i + 1, len(insts)):
                dist = float(np.linalg.norm(insts[i]["center"] - insts[j]["center"]))
                thr = (insts[i]["radius"] + insts[j]["radius"]) * margin
                if dist < thr and (best is None or dist < best[0]):
                    best = (dist, i, j)
        if best is None:
            break
        _, i, j = best
        a, b = insts[i], insts[j]
        for vn, files in b["masks"].items():
            a["masks"].setdefault(vn, []).extend(files)
        a["pts"].extend(b["pts"])
        a["radii"].extend(b["radii"])
        a["center"] = np.mean(a["pts"], axis=0)
        a["radius"] = float(np.median(a["radii"]))
        insts.pop(j)
    return insts


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
        # 影像尺寸:讀一張遮罩拿 shape
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
    if len(cand) < 20:
        print(f"[skip] {scene}: 候選交會點 {len(cand)} 太少")
        return
    cand = np.asarray(cand)
    # min_cluster_size 綁 MIN_SUPPORT,不綁候選點比例:
    # 一個真物體至少被 MIN_SUPPORT 個 view 看到 → 至少 C(MIN_SUPPORT,2) 個交會點。
    # 這是「小的絕對數、與候選點總數無關」,稀疏單物體場景(n1)才不會被高門檻濾成 0 群。
    # 門檻刻意設低 → 寧可過度分割(尤其細長物),再靠步驟 5 的物理半徑回併收回來。
    if args.min_cluster_size is None:
        mcs = max(3, math.comb(MIN_SUPPORT, 2))
    else:
        mcs = args.min_cluster_size
    msamp = args.min_samples if args.min_samples is not None else mcs
    print(f"候選交會點: {len(cand)}  → min_cluster_size={mcs}")

    # 3) HDBSCAN 分群(取代固定半徑 set-cover + merge)
    clusterer = HDBSCAN(min_cluster_size=mcs, min_samples=msamp,
                        cluster_selection_epsilon=args.cluster_eps,
                        cluster_selection_method="eom")
    labels = clusterer.fit_predict(cand)
    uniq = sorted(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"HDBSCAN: {len(uniq)} 群 (+{n_noise} noise / {len(cand)})")
    # 群中心:用群內點的中位數(穩健)
    centers = [np.median(cand[labels == lab], axis=0) for lab in uniq]

    # 4) per-mask 物理半徑認領:每塊遮罩指派給「點線距 < 自身物理半徑」且最近的群
    flat = [(vi, mi) for vi, v in enumerate(views) for mi in range(len(v["masks"]))]
    assign = {}   # cluster_idx -> {view_name: [files]}
    cluster_pts = {k: [] for k in range(len(centers))}    # 認領遮罩的最近點(精修中心用)
    cluster_radii = {k: [] for k in range(len(centers))}  # 認領遮罩的物理半徑(回併用)
    for (vi, mi) in flat:
        v = views[vi]
        m = v["masks"][mi]
        o, d, fx = v["C"], m["ray"], v["fx"]
        best_k, best_dist, best_pr = None, 1e9, None
        for k, c in enumerate(centers):
            pr = point_ray_dist(c, o, d)
            if pr >= best_dist:
                continue
            r = float(np.linalg.norm(c - o))                 # 中心到相機距離(幾何,非深度)
            rho = math.sqrt(m["area"] / math.pi)             # 等效像素半徑
            phys_r = r * rho / fx * args.radius_margin       # 物理半徑 gate
            phys_r = min(max(phys_r, RADIUS_FLOOR), RADIUS_CAP)
            if pr <= phys_r:
                best_k, best_dist, best_pr = k, pr, phys_r
        if best_k is not None:
            assign.setdefault(best_k, {}).setdefault(v["name"], []).append(m["file"])
            cluster_pts[best_k].append(o + max(0.0, float((centers[best_k] - o) @ d)) * d)
            cluster_radii[best_k].append(best_pr)

    # 5) 組 raw instance(先不套 support 門檻),中心用認領點精修、半徑取認領遮罩中位數
    raw = []
    for k, c in enumerate(centers):
        per_view = assign.get(k, {})
        if not per_view:
            continue
        center = np.mean(cluster_pts[k], axis=0) if cluster_pts[k] else np.asarray(c)
        radius = float(np.median(cluster_radii[k])) if cluster_radii[k] else RADIUS_FLOOR
        raw.append({"center": center, "masks": per_view,
                    "pts": list(cluster_pts[k]), "radii": list(cluster_radii[k]),
                    "radius": radius})

    # 5b) 細長物回併(物理半徑自適應):質心射線法把細長物(刀/叉)沿長軸切成多群,
    #     兩群中心距 < (各自物理半徑之和)×merge_margin → 視為同物體,合併。
    #     用物理半徑(遮罩大小×距離換算)當門檻 → 尺度自適應、不用固定長度、不用深度。
    n_before = len(raw)
    raw = merge_instances(raw, args.merge_margin)
    print(f"raw 群={n_before} → 回併後={len(raw)}")

    # 5c) 才套 MIN_SUPPORT:回併後不足支持 view 的(雜訊小群)丟掉
    instances = []
    for inst in raw:
        if len(inst["masks"]) < MIN_SUPPORT:
            continue
        instances.append({"center": [round(float(x), 4) for x in inst["center"]],
                          "support": len(inst["masks"]), "masks": inst["masks"]})
    instances.sort(key=lambda a: -a["support"])
    print(f"保留 instance 數: {len(instances)}")

    # 6) GT 驗證
    gt = []
    mani = scene_dir / "scene_manifest.json"
    if mani.is_file():
        m = json.loads(mani.read_text(encoding="utf-8"))
        for o in m["actual"]["viewpoints"][0]["objects"]:
            gt.append((o["name"], np.array(o["position_m"], dtype=np.float64)))
    report = [f"scene: {scene}  (HDBSCAN 版)", f"views: {len(views)}",
              f"min_cluster_size={mcs} min_samples={msamp} cluster_eps={args.cluster_eps} "
              f"radius_margin={args.radius_margin} merge_margin={args.merge_margin}",
              f"候選點={len(cand)}  HDBSCAN群={len(uniq)} noise={n_noise}",
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
        json.dumps({"scene": scene, "method": "hdbscan",
                    "centers": [i["center"] for i in instances],
                    "instances": instances}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "assoc_report.txt").write_text(txt + "\n", encoding="utf-8")
    print(f"\n→ {out_dir}/instances.json、assoc_report.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["n3_scene0001"], help="場景名或組號")
    ap.add_argument("--min-cluster-size", type=int, default=None, dest="min_cluster_size",
                    help="HDBSCAN 一物體至少幾個交會點(預設 None=綁 MIN_SUPPORT,小絕對值)")
    ap.add_argument("--min-samples", type=int, default=None, dest="min_samples",
                    help="HDBSCAN min_samples(預設=min_cluster_size)")
    ap.add_argument("--cluster-eps", type=float, default=0.0, dest="cluster_eps",
                    help="HDBSCAN cluster_selection_epsilon(>0 會合併相近子群;預設 0)")
    ap.add_argument("--radius-margin", type=float, default=1.3, dest="radius_margin",
                    help="per-mask 物理半徑 gate 的放大係數")
    ap.add_argument("--merge-margin", type=float, default=1.5, dest="merge_margin",
                    help="細長物回併門檻:中心距 < (半徑和)×此係數 → 併")
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
