#!/home/cho/.pyenv/versions/grounded_sam/bin/python3
"""clip_hull.py — 外觀(CLIP)配對 + visual hull 幾何驗證的關聯。

雙線交叉驗證(各補對方弱點):
  外觀單用→不同物體偶爾長得像會誤配;幾何單用→相鄰物體幻影橋會誤併。
  → 先用外觀把「長得像」的遮罩配成群,再對每群「各自」carve visual hull:
    不同物體不在同一外觀群 → 各自雕殼 → 不會幻影橋焊在一起;
    同款不同位置的物體(同外觀群)→ 其 hull 在 3D 分成兩坨 → 連通元件切開。

流程:
  ① 每塊 SAM 遮罩 → CLIP 特徵(摳遮罩→encode→L2 normalize)
  ② 跨/同視角任兩遮罩 cos ≥ --sim → 連邊;連通元件 = 外觀群
  ③ 每外觀群:用該群遮罩 carve 體素(落在群遮罩內視角數 >= keep)→ 3D 連通元件 = 物體
  ④ 物體需 ≥ --min-views 視角、≥ --min-vox 體素 → instances.json

需在 grounded_sam 環境(clip/torch/cv2);carve 用 numpy(重用 associate_voxel 常數/UF)。
用法: ./instance_hull/clip_hull.py n5_scene0031 [--sim 0.85] [--min-views 2]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import clip

sys.path.insert(0, str(Path(__file__).resolve().parent))
import associate_voxel as av   # noqa: E402

REPO = av.REPO
CAPTURES = av.CAPTURES
SAM_ROOT = av.SAM_ROOT
OUT_ROOT = REPO / "data" / "eval" / "clip_hull"


def load_views(scene):
    group = scene.split("_")[0]
    scene_dir = CAPTURES / f"multi_{group}" / scene
    sam_dir = SAM_ROOT / scene
    views = []
    for vdir in sorted(sam_dir.glob("view_*")):
        name = vdir.name
        pose_path = scene_dir / f"{name}_pose.json"
        img = cv2.imread(str(scene_dir / f"{name}.png"))
        if not pose_path.is_file() or img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        masks = []
        for mp in sorted((vdir / "masks").glob("mask_*.png")):
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            b = m > 127
            if b.sum() == 0 or b.sum() > av.MAX_AREA_FRAC * H * W or av.touches_border(b):
                continue
            masks.append((b, mp.name))
        if not masks:
            continue
        C, R = av.load_pose(pose_path)
        fx, cx, cy = av.intrinsics(W, H)
        views.append({"name": name, "C": C, "R": R, "fx": fx, "cx": cx, "cy": cy,
                      "W": W, "H": H, "rgb": rgb, "masks": masks})
    return views


CLIP_MEAN = np.array([123, 116, 103], dtype=np.uint8)   # ImageNet 均值(0-255),比黑底接近 CLIP 分佈


def square_mean_crop(rgb, seg):
    """遮罩外填均值色 + 以長邊做正方形 crop(物體置中),避免 CLIP center-crop 砍掉細長零件。"""
    ys, xs = np.nonzero(seg)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = rgb[y0:y1, x0:x1].copy()
    crop[~seg[y0:y1, x0:x1]] = CLIP_MEAN          # 背景(box內遮罩外)→ 均值
    h, w = crop.shape[:2]
    side = max(h, w)
    canvas = np.empty((side, side, 3), dtype=np.uint8); canvas[:] = CLIP_MEAN
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop           # 置中,四周補均值 → 正方形
    return Image.fromarray(canvas)


def clip_features(views, model, prep, device):
    items, crops = [], []
    for vi, v in enumerate(views):
        for mi, (seg, _) in enumerate(v["masks"]):
            crops.append(prep(square_mean_crop(v["rgb"], seg)))
            items.append((vi, mi))
    if not crops:
        return items, np.zeros((0, 512))
    feats = []
    with torch.no_grad():
        for i in range(0, len(crops), 256):
            batch = torch.stack(crops[i:i + 256]).to(device)
            f = model.encode_image(batch).float()
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
    return items, np.concatenate(feats, 0)


def appearance_groups(items, feats, tau):
    """完全連結(complete-linkage)貪婪分群:一群內『所有配對』都 >= tau(非單連結傳遞)。
    擋掉鏈式過併 —— 不會因為有一條路徑就把不相似的頭尾併在一起。"""
    n = len(items)
    if n == 0:
        return []
    sim = feats @ feats.T
    unassigned = set(range(n))
    groups = []
    while unassigned:
        ua = list(unassigned)
        # 種子:在未指派中鄰居(>=tau)最多者
        seed = max(ua, key=lambda i: int(sum(sim[i, j] >= tau for j in ua)))
        group = [seed]
        # 依與種子相似度由高到低嘗試加入,需與『群內所有成員』都 >= tau
        cands = sorted((j for j in ua if j != seed and sim[seed, j] >= tau),
                       key=lambda j: -sim[seed, j])
        for j in cands:
            if all(sim[j, g] >= tau for g in group):
                group.append(j)
        groups.append(group)
        unassigned -= set(group)
    return groups


def carve_group(group, items, views, args):
    """對一個外觀群:用該群遮罩 carve + 3D 連通元件 → 物體 list。"""
    # 該群每視角的 union seg + 檔名
    gv = {}   # vi -> {"seg":bool, "files":[...]}
    for k in group:
        vi, mi = items[k]
        seg, fname = views[vi]["masks"][mi]
        d = gv.setdefault(vi, {"seg": np.zeros_like(seg), "files": []})
        d["seg"] |= seg
        d["files"].append(fname)
    gviews = sorted(gv)
    if len(gviews) < args.min_views:
        return []

    vx = args.voxel
    xs = np.arange(*av.WS_X, vx); ys = np.arange(*av.WS_Y, vx); zs = np.arange(*av.WS_Z, vx)
    nx, ny, nz = len(xs), len(ys), len(zs)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    M = P.shape[0]
    INS = np.zeros((M, len(gviews)), dtype=bool)
    for col, vi in enumerate(gviews):
        v = views[vi]; seg = gv[vi]["seg"]
        X = (P - v["C"]) @ v["R"] @ av.BODY_TO_OPENCV.T
        z = X[:, 2]; valid = z > 1e-6
        u = np.full(M, -1.0); w = np.full(M, -1.0)
        u[valid] = v["fx"] * X[valid, 0] / z[valid] + v["cx"]
        w[valid] = v["fx"] * X[valid, 1] / z[valid] + v["cy"]
        ui = np.round(u).astype(np.int64); wi = np.round(w).astype(np.int64)
        inb = valid & (ui >= 0) & (ui < v["W"]) & (wi >= 0) & (wi < v["H"])
        ins = np.zeros(M, dtype=bool)
        ins[inb] = seg[wi[inb], ui[inb]]
        INS[:, col] = ins
    keep_min = max(args.min_views, int(math.ceil(args.keep_frac * len(gviews))))
    keep = INS.sum(1) >= keep_min
    if keep.sum() == 0:
        return []

    # 3D 空間連通(6-鄰接)
    grid = keep.reshape(nx, ny, nz)
    idx = -np.ones((nx, ny, nz), dtype=np.int64)
    nk = int(keep.sum()); idx[grid] = np.arange(nk)
    uf = av.UF(nk)
    for axis in range(3):
        sa = [slice(None)] * 3; sb = [slice(None)] * 3
        sa[axis] = slice(0, -1); sb[axis] = slice(1, None)
        ia = idx[tuple(sa)].ravel(); ib = idx[tuple(sb)].ravel()
        m = (ia >= 0) & (ib >= 0)
        for a, b in zip(ia[m], ib[m]):
            uf.union(int(a), int(b))
    roots = np.array([uf.find(i) for i in range(nk)])
    uniq, inv, counts = np.unique(roots, return_inverse=True, return_counts=True)
    keptP = P[keep]; keptINS = INS[keep]

    objs = []
    for ci in np.argsort(-counts):
        if counts[ci] < args.min_vox:
            continue
        sel = inv == ci
        sub_ins = keptINS[sel]
        thresh = max(1, int(0.05 * sel.sum()))
        per_view = {}
        for col, vi in enumerate(gviews):
            if int(sub_ins[:, col].sum()) >= thresh:
                per_view[views[vi]["name"]] = gv[vi]["files"]
        if len(per_view) < args.min_views:
            continue
        center = keptP[sel].mean(0)
        objs.append({"center": [round(float(x), 4) for x in center],
                     "support": len(per_view), "n_vox": int(counts[ci]), "masks": per_view})
    return objs


def merge_close(instances, dist):
    """把中心距 < dist 的 instance 合併(同物體被外觀過度分割的碎片;迭代併最近一對)。"""
    insts = [{"center": np.array(i["center"], float), "n_vox": i["n_vox"],
              "masks": {v: set(f) for v, f in i["masks"].items()}} for i in instances]
    while len(insts) > 1:
        best = None
        for a in range(len(insts)):
            for b in range(a + 1, len(insts)):
                d = float(np.linalg.norm(insts[a]["center"] - insts[b]["center"]))
                if d < dist and (best is None or d < best[0]):
                    best = (d, a, b)
        if best is None:
            break
        _, a, b = best
        A, B = insts[a], insts[b]
        for v, fs in B["masks"].items():
            A["masks"].setdefault(v, set()).update(fs)
        tot = A["n_vox"] + B["n_vox"]
        A["center"] = (A["center"] * A["n_vox"] + B["center"] * B["n_vox"]) / tot
        A["n_vox"] = tot
        insts.pop(b)
    return [{"center": [round(float(x), 4) for x in i["center"]],
             "support": len(i["masks"]), "n_vox": i["n_vox"],
             "masks": {v: sorted(fs) for v, fs in i["masks"].items()}} for i in insts]


def process_scene(scene, model, prep, device, args):
    views = load_views(scene)
    if len(views) < 2:
        print(f"[skip] {scene}: 有效 view < 2"); return
    items, feats = clip_features(views, model, prep, device)
    groups = appearance_groups(items, feats, args.sim)
    instances = []
    for g in groups:
        instances += carve_group(g, items, views, args)
    # 空間合併:完全連結會把同物體拆成多群→多 hull,但它們位置重疊(同物體);
    # 不同物體相距遠(且外觀群已隔開)→ 小距離合併只併回同物體碎片,不誤併。
    instances = merge_close(instances, args.merge_dist)
    instances.sort(key=lambda a: -a["support"])

    gt = []
    mani = CAPTURES / f"multi_{scene.split('_')[0]}" / scene / "scene_manifest.json"
    if mani.is_file():
        for o in json.loads(mani.read_text())["actual"]["viewpoints"][0]["objects"]:
            gt.append((o["name"], np.array(o["position_m"])))
    report = [f"scene: {scene}  (CLIP+hull, sim={args.sim})",
              f"views={len(views)} 遮罩總數={len(items)} 外觀群={len(groups)} "
              f"→ instances={len(instances)}  (GT 物體數: {len(gt)})", ""]
    for k, inst in enumerate(instances):
        c = np.array(inst["center"]); line = f"inst_{k:02d}: support={inst['support']} vox={inst['n_vox']}"
        if gt:
            nm, dmin = min(((n, float(np.linalg.norm(c - p))) for n, p in gt), key=lambda a: a[1])
            line += f"  最近GT={nm} ({dmin*100:.1f}cm)"
        report.append(line)
    txt = "\n".join(report); print(txt)
    out_dir = OUT_ROOT.parent / f"clip_hull_s{round(args.sim * 100)}" / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "instances.json").write_text(json.dumps(
        {"scene": scene, "method": "clip_hull",
         "centers": [i["center"] for i in instances], "instances": instances},
        indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "assoc_report.txt").write_text(txt + "\n", encoding="utf-8")
    print(f"→ {out_dir}/instances.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["n5_scene0031"])
    ap.add_argument("--sim", type=float, default=0.85, help="CLIP 餘弦相似度配對門檻")
    ap.add_argument("--voxel", type=float, default=0.015)
    ap.add_argument("--keep-frac", type=float, default=0.6, dest="keep_frac")
    ap.add_argument("--min-views", type=int, default=2, dest="min_views")
    ap.add_argument("--min-vox", type=int, default=8, dest="min_vox")
    ap.add_argument("--merge-dist", type=float, default=0.05, dest="merge_dist",
                    help="空間合併:中心距 < 此(m)的碎片併回同物體")
    args = ap.parse_args()
    scenes = av.resolve_scenes(args.scenes or ["n5_scene0031"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, prep = clip.load("ViT-B/32", device=device); model.eval()
    if not scenes:
        sys.exit("沒有場景")
    for i, scene in enumerate(scenes, 1):
        print(f"\n===== [{i}/{len(scenes)}] {scene} =====")
        try:
            process_scene(scene, model, prep, device, args)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[error] {scene}: {e}")


if __name__ == "__main__":
    main()
