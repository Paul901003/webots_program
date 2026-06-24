#!/home/cho/.pyenv/versions/grounded_sam/bin/python3
"""associate.py — B 方法第一步:多視角「幾何」關聯(不靠標籤、不靠深度)。

把 sam_only 在各 view 切出的 class-agnostic 遮罩,純用相機姿態做幾何關聯,
分成各物體 instance:同一物體在不同 view 的遮罩,其「質心射線」會在 3D 交會。

流程:
  ① 讀各 view 的 SAM 遮罩,濾掉背景(過大/碰邊界)
  ② 每個遮罩 → 質心像素 → 反投影成世界射線(origin=相機位置, dir=世界方向)
  ③ 跨 view 兩兩射線求最近交會點 → 落在工作空間內且夠近 → instance 中心假設
  ④ 3D 鄰近度合併假設 → instance 中心;對每個中心反投影回各 view 數支持度
  ⑤ 每個 instance 收集各 view 支持它的遮罩(含過度分割的零件)→ instances.json
  ⑥ 用 GT(scene_manifest actual)驗證:instance 數、中心與 GT 物體距離

輸出: data/eval/instance_hull/<scene>/instances.json (+ assoc_overlay/ 反投影驗證圖)
需在 grounded_sam 環境(numpy/cv2;不需 torch)。

用法:
  ./instance_hull/associate.py n3_scene0001
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
CAPTURES = REPO / "data" / "captures"
SAM_ROOT = REPO / "data" / "eval" / "sam_only"
OUT_ROOT = REPO / "data" / "eval" / "instance_hull"

# ── 相機 / 影像參數(與 build_torchhull 一致)──────────────────────────────────
HFOV_RAD = 1.4746

# ── 背景遮罩過濾 ──────────────────────────────────────────────────────────────
MAX_AREA_FRAC = 0.30        # 面積 > 此比例 → 背景(桌面/整片)
BORDER = 2                  # 碰邊界(桌面延伸到邊)→ 背景

# ── 工作空間範圍(已知,用來剔除離譜的交會點)─────────────────────────────────
WS_X = (-0.05, 0.75)
WS_Y = (-0.45, 0.45)
WS_Z = (-0.05, 0.45)

# ── 關聯門檻 ──────────────────────────────────────────────────────────────────
RAY_TAU = 0.04             # 兩射線最近距離 < 此(m)→ 產生候選中心
CLAIM_R = 0.06             # 點到射線距離 < 此(m,≈物體半徑)→ 該中心「認領」此遮罩
MIN_SUPPORT = 3            # 至少幾個 view 支持才算一個 instance
MERGE_D = 0.09             # 大物體殘塊回併:中心距已接受 instance < 此(m)→ 併入
                           # (須 < 物體間最小中心距,否則會把相鄰物體誤併)


# ── 姿態 / 內參 / 投影 ────────────────────────────────────────────────────────
def rpy_to_R(roll, pitch, yaw):
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


# opencv optical(x右,y下,z前) ↔ webots body(x前,y左,z上)
OPENCV_TO_BODY = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
BODY_TO_OPENCV = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)


def load_pose(pose_path):
    meta = json.loads(pose_path.read_text(encoding="utf-8"))
    if "position_m" not in meta and isinstance(meta.get("camera"), dict):
        meta = meta["camera"]
    p = meta["position_m"]
    C = np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
    r = meta["rotation_rpy_rad"]
    R = rpy_to_R(r["roll"], r["pitch"], r["yaw"])     # camera_to_world (body)
    return C, R


def intrinsics(W, H):
    fx = W / (2.0 * math.tan(HFOV_RAD / 2.0))
    return fx, fx, W / 2.0, H / 2.0


def pixel_to_ray(u, v, C, R, K):
    fx, fy, cx, cy = K
    d_cv = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    d_world = R @ (OPENCV_TO_BODY @ d_cv)
    return d_world / np.linalg.norm(d_world)


def project(p_world, C, R, K, W, H):
    fx, fy, cx, cy = K
    p_cv = BODY_TO_OPENCV @ (R.T @ (p_world - C))
    if p_cv[2] <= 1e-6:
        return None
    u = fx * p_cv[0] / p_cv[2] + cx
    v = fy * p_cv[1] / p_cv[2] + cy
    if 0 <= u < W and 0 <= v < H:
        return int(round(u)), int(round(v))
    return None


def closest_point(o1, d1, o2, d2):
    """兩條射線(單位方向)最近交會;回傳 (中點, 距離, s, t)。s,t<0 表在相機後方。"""
    b = float(d1 @ d2)
    denom = 1.0 - b * b
    if abs(denom) < 1e-9:                 # 近平行
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


# ── 讀遮罩 ────────────────────────────────────────────────────────────────────
def touches_border(seg):
    return bool(seg[:BORDER].any() or seg[-BORDER:].any()
                or seg[:, :BORDER].any() or seg[:, -BORDER:].any())


def load_view_masks(view_dir):
    """回傳 [(seg(bool), centroid(u,v), area), ...] 已濾背景。"""
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
        out.append({"seg": seg, "uv": (float(xs.mean()), float(ys.mean())),
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


def process_scene(scene):
    group = scene.split("_")[0]
    scene_dir = CAPTURES / f"multi_{group}" / scene
    sam_dir = SAM_ROOT / scene
    if not sam_dir.is_dir():
        print(f"[skip] {scene}: 找不到 SAM 遮罩 {sam_dir}(先跑 sam_only.py)")
        return

    out_dir = OUT_ROOT / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 每 view:讀姿態 + 遮罩 + 各遮罩射線
    views = []   # {name, C, R, K, W, H, masks:[{seg,uv,area,file, ray}]}
    for vdir in sorted(sam_dir.glob("view_*")):
        name = vdir.name
        pose_path = scene_dir / f"{name}_pose.json"
        if not pose_path.is_file():
            continue
        C, R = load_pose(pose_path)
        masks = load_view_masks(vdir)
        if not masks:
            continue
        H, W = masks[0]["seg"].shape
        K = intrinsics(W, H)
        for m in masks:
            m["ray"] = pixel_to_ray(m["uv"][0], m["uv"][1], C, R, K)
        views.append({"name": name, "C": C, "R": R, "K": K, "W": W, "H": H, "masks": masks})
    if len(views) < 2:
        print(f"[skip] {scene}: 有效 view < 2,無法三角化")
        return
    print(f"{scene}: {len(views)} views,遮罩數 {[len(v['masks']) for v in views]}")

    # 2) 跨 view 兩兩射線交會 → 候選中心
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
    if not cand:
        print(f"[skip] {scene}: 沒有任何跨視角射線交會落在工作空間內")
        return
    cand = np.array(cand)
    print(f"候選交會點: {len(cand)}")

    # 3) 貪婪 set-cover:候選中心用「點到射線距離 < CLAIM_R」認領遮罩,
    #    取支持 view 數最多者當一個 instance,認領後把遮罩移出池子,重複。
    #    → 物體遮罩被吃完後該處不再有高支持候選,instance 數自然收斂到物體數。
    def point_ray_dist(c, o, d):
        v = c - o
        proj = float(v @ d)
        if proj <= 0:
            return 1e9
        return float(np.linalg.norm(v - proj * d))

    # 攤平所有遮罩,標記是否已被認領
    flat = []   # (vi, mi)
    for vi, v in enumerate(views):
        for mi in range(len(v["masks"])):
            flat.append((vi, mi))
    claimed = set()

    instances = []
    while True:
        best = None   # (n_views, center, claim_list, views_set)
        for c in cand:
            claim, vset = [], set()
            for (vi, mi) in flat:
                if (vi, mi) in claimed:
                    continue
                if point_ray_dist(c, views[vi]["C"], views[vi]["masks"][mi]["ray"]) <= CLAIM_R:
                    claim.append((vi, mi)); vset.add(vi)
            if best is None or len(vset) > best[0]:
                best = (len(vset), c, claim, vset)
        if best is None or best[0] < MIN_SUPPORT:
            break
        n_views, c, claim, vset = best
        # 用認領遮罩的最近點精修中心
        pts = []
        for (vi, mi) in claim:
            o, d = views[vi]["C"], views[vi]["masks"][mi]["ray"]
            pts.append(o + max(0.0, float((c - o) @ d)) * d)
        center = np.mean(pts, axis=0) if pts else c
        per_view = {}
        for (vi, mi) in claim:
            per_view.setdefault(views[vi]["name"], []).append(views[vi]["masks"][mi]["file"])
        instances.append({"center": [round(float(x), 4) for x in center],
                          "support": n_views, "masks": per_view})
        claimed |= set(claim)
    print(f"set-cover instance 數: {len(instances)}")

    # 3b) 大物體殘塊回併:弱 instance 中心離某強 instance < MERGE_D → 併入(遮罩合併)
    instances.sort(key=lambda a: -a["support"])
    merged = []
    for inst in instances:
        c = np.array(inst["center"])
        host = next((h for h in merged
                     if np.linalg.norm(c - np.array(h["center"])) <= MERGE_D), None)
        if host is None:
            merged.append(inst)
        else:
            for vn, files in inst["masks"].items():
                host["masks"].setdefault(vn, []).extend(files)
            host["support"] = len(host["masks"])
    instances = merged
    print(f"回併後 instance 數: {len(instances)}")

    # 5) GT 驗證
    gt = []
    mani = scene_dir / "scene_manifest.json"
    if mani.is_file():
        m = json.loads(mani.read_text(encoding="utf-8"))
        for o in m["actual"]["viewpoints"][0]["objects"]:
            gt.append((o["name"], np.array(o["position_m"], dtype=np.float64)))
    report = [f"scene: {scene}", f"views: {len(views)}",
              f"instances: {len(instances)}  (GT 物體數: {len(gt)})", ""]
    for k, inst in enumerate(instances):
        c = np.array(inst["center"])
        line = f"inst_{k:02d}: center=({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f}) support={inst['support']}/{len(views)}"
        if gt:
            dists = [(name, float(np.linalg.norm(c - pos))) for name, pos in gt]
            name, dmin = min(dists, key=lambda a: a[1])
            line += f"  最近GT={name} ({dmin*100:.1f}cm)"
        report.append(line)
    txt = "\n".join(report)
    print("\n" + txt)

    (out_dir / "instances.json").write_text(
        json.dumps({"scene": scene, "centers": [i["center"] for i in instances],
                    "instances": instances}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "assoc_report.txt").write_text(txt + "\n", encoding="utf-8")
    print(f"\n→ {out_dir}/instances.json、assoc_report.txt")


def main():
    targets = sys.argv[1:] or ["n3_scene0001"]
    scenes = resolve_scenes(targets)
    if not scenes:
        sys.exit("沒有場景")
    for i, scene in enumerate(scenes, 1):
        print(f"\n===== [{i}/{len(scenes)}] {scene} =====")
        try:
            process_scene(scene)
        except Exception as e:
            print(f"[error] {scene}: {e}")


if __name__ == "__main__":
    main()
