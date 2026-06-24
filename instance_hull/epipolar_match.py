#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""epipolar_match.py — 概念驗證:Descriptor-free 多視角 region matching(Doi et al. ACCV2020 風格)。

純幾何跨視角 instance 對應(不用特徵、不用深度):
  ① 每個 mask 內取樣點 → 用 fundamental matrix 投到別視角畫 epipolar 線 → 疊成 band。
  ② 邊權重 = 別視角 mask 與該 band 的交集程度(覆蓋比例),對稱化 → 相似度矩陣 A。
  ③ SymNMF(A≈HHᵀ)圖聚類 → 每群=一物體跨視角的 masks。
先固定 k=GT 物體數驗證對應對不對(自動 k 之後再加)。

需 webots_visual_hull 環境(numpy/cv2)。
用法: ./instance_hull/epipolar_match.py n3_scene0001 [n5_scene0031] [--samples 200]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

REPO = Path(__file__).resolve().parents[1]
CAPTURES = REPO / "data" / "captures"
SAM_ROOT = REPO / "data" / "eval" / "sam_only"
HFOV_RAD = 1.4746
MAX_AREA_FRAC = 0.30
BORDER = 2
BODY_TO_OPENCV = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)


def rpy_to_R(roll, pitch, yaw):
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]], dtype=np.float64)


def load_pose(pose_path):
    meta = json.loads(pose_path.read_text(encoding="utf-8"))
    if "position_m" not in meta and isinstance(meta.get("camera"), dict):
        meta = meta["camera"]
    p = meta["position_m"]; r = meta["rotation_rpy_rad"]
    C = np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
    return C, rpy_to_R(r["roll"], r["pitch"], r["yaw"])


def Kmat(W, H):
    fx = W / (2.0 * math.tan(HFOV_RAD / 2.0))
    return np.array([[fx, 0, W / 2.0], [0, fx, H / 2.0], [0, 0, 1.0]])


def proj_matrix(C, R, K):
    Rt = BODY_TO_OPENCV @ R.T              # world → optical rotation
    t = (-Rt @ C).reshape(3, 1)
    return K @ np.hstack([Rt, t])          # 3x4


def skew(e):
    return np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])


def fundamental(Pi, Pj, Ci):
    e_j = Pj @ np.append(Ci, 1.0)          # epipole:相機 i 中心在 view j 的像
    return skew(e_j) @ Pj @ np.linalg.pinv(Pi)


def touches_border(b):
    return bool(b[:BORDER].any() or b[-BORDER:].any() or b[:, :BORDER].any() or b[:, -BORDER:].any())


def load_views(scene):
    group = scene.split("_")[0]
    scene_dir = CAPTURES / f"multi_{group}" / scene
    sam_dir = SAM_ROOT / scene
    views = []
    for vdir in sorted(sam_dir.glob("view_*")):
        pose_path = scene_dir / f"{vdir.name}_pose.json"
        if not pose_path.is_file():
            continue
        masks = []
        for mp in sorted((vdir / "masks").glob("mask_*.png")):
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            b = m > 127; H, W = b.shape
            if b.sum() == 0 or b.sum() > MAX_AREA_FRAC * H * W or touches_border(b):
                continue
            masks.append((b, mp.name))
        if not masks:
            continue
        C, R = load_pose(pose_path)
        H, W = masks[0][0].shape
        K = Kmat(W, H)
        views.append({"name": vdir.name, "C": C, "R": R, "K": K, "W": W, "H": H,
                      "P": proj_matrix(C, R, K), "masks": masks})
    return views


def draw_band(F, src_pts, Wj, Hj):
    """src_pts: (n,2) view i 的取樣點;畫出它們在 view j 的 epipolar 線 → band 二值圖。"""
    band = np.zeros((Hj, Wj), np.uint8)
    homo = np.hstack([src_pts, np.ones((len(src_pts), 1))])   # (n,3)
    lines = (F @ homo.T).T                                     # (n,3) 每條線 a,b,c
    for a, b, c in lines:
        if abs(a) < 1e-9 and abs(b) < 1e-9:
            continue
        if abs(b) >= abs(a):                                  # 用 u=0、u=W 取兩端
            p1 = (0, int(round(-c / b))); p2 = (Wj - 1, int(round(-(c + a * (Wj - 1)) / b)))
        else:
            p1 = (int(round(-c / a)), 0); p2 = (int(round(-(c + b * (Hj - 1)) / a)), Hj - 1)
        ok, q1, q2 = cv2.clipLine((0, 0, Wj, Hj), p1, p2)
        if ok:
            cv2.line(band, q1, q2, 1, 1)
    return band.astype(bool)


def build_affinity(views, n_samples, rng, sigma):
    # 攤平所有 mask
    flat = [(vi, mi) for vi, v in enumerate(views) for mi in range(len(v["masks"]))]
    idx = {(vi, mi): k for k, (vi, mi) in enumerate(flat)}
    n = len(flat)
    A = np.zeros((n, n))
    # 預存每 mask 取樣點
    samp = {}
    for vi, mi in flat:
        b = views[vi]["masks"][mi][0]
        ys, xs = np.nonzero(b)
        sel = rng.choice(len(xs), size=min(n_samples, len(xs)), replace=False)
        samp[(vi, mi)] = np.stack([xs[sel], ys[sel]], axis=1).astype(np.float64)
    # 預存 fundamental matrices
    F = {}
    for i in range(len(views)):
        for j in range(len(views)):
            if i == j:
                continue
            F[(i, j)] = fundamental(views[i]["P"], views[j]["P"], views[i]["C"])
    # 對每個 (mask_a in i, view j):畫 band,測 view j 各 mask 覆蓋比例 → 有向權重
    W_dir = np.zeros((n, n))
    for (vi, mi) in flat:
        pts = samp[(vi, mi)]
        for j, vj in enumerate(views):
            if j == vi:
                continue
            band = draw_band(F[(vi, j)], pts, vj["W"], vj["H"])
            ba = band.sum()
            if ba == 0:
                continue
            for mj in range(len(vj["masks"])):
                d = vj["masks"][mj][0]
                inter = int((band & d).sum())
                if inter > 0:
                    # Step3(論文):非對稱權重 = 覆蓋比例(band 覆蓋 R_j 的比例),無因次[0,1]
                    #   (論文無高斯項、無 self-tuning σ;此處用覆蓋比例,IoU 變體)
                    W_dir[idx[(vi, mi)], idx[(j, mj)]] = inter / float(d.sum())
    # Step4(論文):對稱化用幾何平均 √(w_ij·w_ji);同視角已為 0
    A = np.sqrt(np.maximum(W_dir, 0.0) * np.maximum(W_dir.T, 0.0))
    np.fill_diagonal(A, 0.0)
    return A, flat, idx


def symnmf(A, k, iters=2000, seed=0):
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    H = rng.random((n, k)) * math.sqrt(max(A.mean(), 1e-6) / k + 1e-6)
    for _ in range(iters):
        AH = A @ H
        HHtH = H @ (H.T @ H)
        H *= 0.5 + 0.5 * AH / (HHtH + 1e-9)
    return H


def gt_assign(scene, views):
    """每個 SAM mask 指派給最大 IoU 的 GT 物體(該視角),回傳 {(vi,mi):gtname}。"""
    coco_p = REPO / "data" / "labels" / scene / "actual" / "annotations.json"
    if not coco_p.is_file():
        return {}, []
    coco = json.loads(coco_p.read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    gtm = {}
    for a in coco["annotations"]:
        nm = cats[a["category_id"]]
        if nm == "ur5e":
            continue
        rle = a["segmentation"]
        if isinstance(rle["counts"], str):
            rle = {"counts": rle["counts"].encode(), "size": rle["size"]}
        gtm.setdefault(int(a["image_id"]), {})[nm] = mu.decode(rle).astype(bool)
    assign = {}
    for vi, v in enumerate(views):
        vid = int(v["name"].split("_")[1])
        gts = gtm.get(vid, {})
        for mi, (b, _) in enumerate(v["masks"]):
            best, bi = None, 0.2
            for nm, g in gts.items():
                if g.shape != b.shape:
                    continue
                iou = (b & g).sum() / max((b | g).sum(), 1)
                if iou > bi:
                    bi, best = iou, nm
            assign[(vi, mi)] = best
    names = sorted({n for n in assign.values() if n})
    return assign, names


# ── 工作空間(Step7 3D 體素投票用;論文無紀錄,標準補)──────────────────────────
WS_X = (-0.05, 0.75); WS_Y = (-0.45, 0.45); WS_Z = (-0.02, 0.40)
OUT_ROOT = REPO / "data" / "eval" / "epipolar"


def eigengap_k(A, kmin, kmax):
    """[補,非論文] 正規化親和力 D^-1/2 A D^-1/2 的特徵值,取 [kmin,kmax] 內最大 eigengap 處為 K。"""
    d = A.sum(1)
    di = 1.0 / np.sqrt(np.maximum(d, 1e-9))
    L = (A * di[:, None]) * di[None, :]
    ev = np.sort(np.linalg.eigvalsh(L))[::-1]          # 由大到小
    kmin = max(2, kmin); kmax = min(kmax, len(ev) - 1)
    best_k, best_gap = kmin, -1.0
    for K in range(kmin, kmax + 1):
        g = ev[K - 1] - ev[K]                          # 第 K 與第 K+1 特徵值間隙
        if g > best_gap:
            best_gap, best_k = g, K
    return best_k


def precompute_proj(views, voxel):
    """體素格 + 每視角投影像素(只算一次)。"""
    xs = np.arange(*WS_X, voxel); ys = np.arange(*WS_Y, voxel); zs = np.arange(*WS_Z, voxel)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    proj = []
    for v in views:
        X = (P - v["C"]) @ v["R"] @ BODY_TO_OPENCV.T
        z = X[:, 2]; valid = z > 1e-6
        u = np.full(len(P), -1.0); w = np.full(len(P), -1.0)
        fx, cx, cy = v["K"][0, 0], v["K"][0, 2], v["K"][1, 2]
        u[valid] = fx * X[valid, 0] / z[valid] + cx
        w[valid] = fx * X[valid, 1] / z[valid] + cy
        ui = np.round(u).astype(np.int64); wi = np.round(w).astype(np.int64)
        inb = valid & (ui >= 0) & (ui < v["W"]) & (wi >= 0) & (wi < v["H"])
        proj.append((ui, wi, inb))
    return P, proj


def vote_cluster(mem, views, P, proj, keep_frac, min_vox):
    """[補,非論文] 對一群 mask 做 3D 體素投票:落在該群各視角遮罩內的視角數 >= keep → 保留。"""
    vmask, vfiles = {}, {}
    for vi, mi in mem:
        seg, fn = views[vi]["masks"][mi]
        vmask.setdefault(vi, np.zeros_like(seg)); vmask[vi] |= seg
        vfiles.setdefault(vi, []).append(fn)
    gv = sorted(vmask)
    if len(gv) < 2:
        return None
    M = len(P)
    cnt = np.zeros(M, dtype=np.int32)
    for vi in gv:
        ui, wi, inb = proj[vi]
        ins = np.zeros(M, dtype=bool)
        ins[inb] = vmask[vi][wi[inb], ui[inb]]
        cnt += ins
    keep_min = max(2, int(math.ceil(keep_frac * len(gv))))
    keep = cnt >= keep_min
    if int(keep.sum()) < min_vox:
        return None
    center = P[keep].mean(0)
    per_view = {views[vi]["name"]: vfiles[vi] for vi in gv}
    return {"center": [round(float(x), 4) for x in center],
            "support": len(per_view), "n_vox": int(keep.sum()), "masks": per_view}


def dedup(objs, dist):
    """[補,非論文] 中心距 < dist 的物體合併(同物體被同視角懲罰拆出的副本群)。"""
    objs = sorted(objs, key=lambda o: -o["n_vox"])
    out = []
    for o in objs:
        c = np.array(o["center"]); host = None
        for h in out:
            if np.linalg.norm(c - np.array(h["center"])) < dist:
                host = h; break
        if host is None:
            out.append(o)
        else:
            for vn, fs in o["masks"].items():
                host["masks"].setdefault(vn, []).extend(fs)
            host["support"] = len(host["masks"])
    return out


def process(scene, args):
    from collections import Counter
    views = load_views(scene)
    if len(views) < 2:
        print(f"[skip] {scene}"); return
    assign, gt_names = gt_assign(scene, views)
    rng = np.random.default_rng(0)
    A, flat, idx = build_affinity(views, args.samples, rng, 0.0)
    # 自動 k:範圍 [單視角最大遮罩數, N](同視角懲罰 → K 下界);eigengap 選 K(補,非論文)
    kfloor = max(len(v["masks"]) for v in views)
    if args.k:
        K = args.k
    else:
        K = eigengap_k(A, kfloor, len(flat))
    H = symnmf(A, K)
    lab = H.argmax(1)
    # Step7:每群 3D 體素投票 → 濾無效 → 去重(補,非論文)
    P, proj = precompute_proj(views, args.voxel)
    objs = []
    for c in range(K):
        mem = [flat[i] for i in range(len(flat)) if lab[i] == c]
        if not mem:
            continue
        o = vote_cluster(mem, views, P, proj, args.keep_frac, args.min_vox)
        if o is not None:
            objs.append(o)
    objs = dedup(objs, args.merge_dist)
    objs = [o for o in objs if o["support"] >= args.min_obj_views]   # [補]濾弱支持物體
    objs.sort(key=lambda o: -o["support"])

    print(f"\n===== {scene} =====")
    print(f"views={len(views)} masks={len(flat)} GT物體={len(gt_names)} | "
          f"K(eigengap)={K} → 體素濾+去重後 instances={len(objs)}")
    gt = []
    mani = CAPTURES / f"multi_{scene.split('_')[0]}" / scene / "scene_manifest.json"
    if mani.is_file():
        for o in json.loads(mani.read_text())["actual"]["viewpoints"][0]["objects"]:
            gt.append((o["name"], np.array(o["position_m"])))
    for k2, o in enumerate(objs):
        c = np.array(o["center"]); line = f"  inst_{k2:02d}: support={o['support']} vox={o['n_vox']}"
        if gt:
            nm, dm = min(((n, float(np.linalg.norm(c - p))) for n, p in gt), key=lambda a: a[1])
            line += f"  最近GT={nm} ({dm*100:.1f}cm)"
        print(line)
    out_dir = OUT_ROOT / scene; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "instances.json").write_text(json.dumps(
        {"scene": scene, "method": "epipolar", "centers": [o["center"] for o in objs],
         "instances": objs}, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["n3_scene0001"])
    ap.add_argument("--samples", type=int, default=200, help="每遮罩取樣點數(畫對極線)")
    ap.add_argument("--k", type=int, default=0, help="0=自動k(eigengap);>0 強制")
    ap.add_argument("--voxel", type=float, default=0.015, help="[補]3D 體素邊長")
    ap.add_argument("--keep-frac", type=float, default=0.6, dest="keep_frac", help="[補]體素投票保留門檻")
    ap.add_argument("--min-vox", type=int, default=8, dest="min_vox", help="[補]群最小體素數(濾無效)")
    ap.add_argument("--merge-dist", type=float, default=0.06, dest="merge_dist", help="[補]副本群去重中心距")
    ap.add_argument("--min-obj-views", type=int, default=3, dest="min_obj_views", help="[補]物體最少支持視角數")
    args = ap.parse_args()
    scenes = []
    for a in args.scenes:
        if "scene" in a:
            scenes.append(a)
        else:
            scenes += [d.name for d in sorted((CAPTURES / f"multi_n{a}").glob(f"n{a}_scene*"))]
    for i, sc in enumerate(scenes, 1):
        print(f"[{i}/{len(scenes)}]", end=" ")
        try:
            process(sc, args)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[error] {sc}: {e}")


if __name__ == "__main__":
    main()
