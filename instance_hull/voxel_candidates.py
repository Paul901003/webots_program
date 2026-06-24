#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""voxel_candidates.py — 程式①:幾何找候選 hull(不做語意判定)。

流程:
  carve 體素(落在遮罩內的視角數 >= keep_frac×N)→ 多標籤連通(相鄰體素遮罩集合有交集)
  → 每個連通元件 = 一個候選 hull,記錄它各視角對應到的 mask 檔名。
  **不限制支持視角數**(min_vox 小、不要求 >=2 view),把所有候選都吐出來;
  「這些 mask 是否真同一物體」交給程式② 用 CLIP/DINO 特徵相似度判定。

輸出: data/eval/voxel_candidates/<scene>/candidates.json
  { scene, voxel, candidates:[ {id, center[3], n_vox, n_views, masks:{view:[檔名]}} ] }

需在 webots_visual_hull 環境。重用 associate_voxel 的幾何函式。
用法: ./instance_hull/voxel_candidates.py n3_scene0001  (或組號 3 / 多組 1 3 4 5)
       [--voxel 0.015] [--keep-frac 0.6] [--agree-frac 0.5] [--min-vox 2]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import associate_voxel as av   # noqa: E402

REPO = av.REPO
CAPTURES = av.CAPTURES
SAM_ROOT = av.SAM_ROOT
OUT_ROOT = REPO / "data" / "eval" / "voxel_candidates"


def process_scene(scene, args):
    group = scene.split("_")[0]
    scene_dir = CAPTURES / f"multi_{group}" / scene
    sam_dir = SAM_ROOT / scene
    if not sam_dir.is_dir():
        print(f"[skip] {scene}: 找不到 SAM 遮罩 {sam_dir}")
        return
    out_dir = OUT_ROOT / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 各視角:姿態 + label/bitmask + 檔名
    views = []
    for vdir in sorted(sam_dir.glob("view_*")):
        name = vdir.name
        pose_path = scene_dir / f"{name}_pose.json"
        if not pose_path.is_file():
            continue
        label, bitmask, files = av.load_label_image(vdir)
        if label is None:
            continue
        C, R = av.load_pose(pose_path)
        H, W = label.shape
        fx, cx, cy = av.intrinsics(W, H)
        views.append({"name": name, "C": C, "R": R, "fx": fx, "cx": cx, "cy": cy,
                      "W": W, "H": H, "label": label, "bitmask": bitmask, "files": files})
    n = len(views)
    if n < 2:
        print(f"[skip] {scene}: 有效 view < 2")
        return

    # 2) 體素格
    vx = args.voxel
    xs = np.arange(*av.WS_X, vx); ys = np.arange(*av.WS_Y, vx); zs = np.arange(*av.WS_Z, vx)
    nx, ny, nz = len(xs), len(ys), len(zs)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    M = P.shape[0]

    # 3) 投影 → 單標籤 L + 多標籤 bitmask B
    L = np.zeros((M, n), dtype=np.int32)
    B = np.zeros((M, n), dtype=np.int64)
    for vi, v in enumerate(views):
        X = (P - v["C"]) @ v["R"] @ av.BODY_TO_OPENCV.T
        z = X[:, 2]; valid = z > 1e-6
        u = np.full(M, -1.0); vv = np.full(M, -1.0)
        u[valid] = v["fx"] * X[valid, 0] / z[valid] + v["cx"]
        vv[valid] = v["fx"] * X[valid, 1] / z[valid] + v["cy"]
        ui = np.round(u).astype(np.int64); vj = np.round(vv).astype(np.int64)
        inb = valid & (ui >= 0) & (ui < v["W"]) & (vj >= 0) & (vj < v["H"])
        L[inb, vi] = v["label"][vj[inb], ui[inb]]
        B[inb, vi] = v["bitmask"][vj[inb], ui[inb]]

    # 4) carving
    keep_min = max(2, int(math.ceil(args.keep_frac * n)))
    keep = (L > 0).sum(axis=1) >= keep_min
    nkeep = int(keep.sum())
    if nkeep == 0:
        print(f"[skip] {scene}: carving 後無體素")
        return

    # 5) 多標籤連通(相鄰體素 bitmask 有交集 >= agree_frac)
    grid_keep = keep.reshape(nx, ny, nz)
    idx_grid = -np.ones((nx, ny, nz), dtype=np.int64)
    idx_grid[grid_keep] = np.arange(nkeep)
    Bk = B[keep]
    uf = av.UF(nkeep)

    def agree(a_ids, b_ids):
        Ba, Bb = Bk[a_ids], Bk[b_ids]
        both = (Ba != 0) & (Bb != 0)
        den = both.sum(1)
        eq = (((Ba & Bb) != 0) & both).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(den > 0, eq / np.maximum(den, 1), 0.0)
        return (den > 0) & (frac >= args.agree_frac)

    for axis in range(3):
        sl_a = [slice(None)] * 3; sl_b = [slice(None)] * 3
        sl_a[axis] = slice(0, -1); sl_b[axis] = slice(1, None)
        ia = idx_grid[tuple(sl_a)].ravel(); ib = idx_grid[tuple(sl_b)].ravel()
        m = (ia >= 0) & (ib >= 0); ia, ib = ia[m], ib[m]
        if ia.size == 0:
            continue
        ok = agree(ia, ib)
        for x, y in zip(ia[ok], ib[ok]):
            uf.union(int(x), int(y))

    roots = np.array([uf.find(i) for i in range(nkeep)])
    uniq, inv, counts = np.unique(roots, return_inverse=True, return_counts=True)
    kept_P = P[keep]

    # 6) 每個元件 = 候選 hull(不限視角數),記錄各視角 mask 檔名
    cands = []
    for ci in np.argsort(-counts):
        if counts[ci] < args.min_vox:
            continue
        sel = inv == ci
        comp_P = kept_P[sel]; comp_B = Bk[sel]
        center = comp_P.mean(axis=0)
        thresh = max(1, int(0.05 * sel.sum()))
        per_view = {}
        for vi, v in enumerate(views):
            bm = comp_B[:, vi]
            chosen = [k + 1 for k in range(min(len(v["files"]), 63))
                      if int((((bm >> np.int64(k)) & 1) == 1).sum()) >= thresh]
            files = [v["files"][l - 1] for l in chosen if 0 < l <= len(v["files"])]
            if files:
                per_view[v["name"]] = files
        if not per_view:                       # 連 1 個視角都沒對到才丟
            continue
        cands.append({"id": len(cands),
                      "center": [round(float(x), 4) for x in center],
                      "n_vox": int(counts[ci]),
                      "n_views": len(per_view),
                      "masks": per_view})

    # GT 數參考(僅報告用)
    gt_n = 0
    mani = scene_dir / "scene_manifest.json"
    if mani.is_file():
        gt_n = len(json.loads(mani.read_text())["actual"]["viewpoints"][0]["objects"])
    print(f"{scene}: {n} views, carved={nkeep}, 候選 hull={len(cands)}  (GT 物體數參考: {gt_n})")
    for c in cands:
        print(f"  hull_{c['id']:02d}: vox={c['n_vox']:4d} views={c['n_views']:2d} "
              f"center=({c['center'][0]:+.3f},{c['center'][1]:+.3f},{c['center'][2]:+.3f})")

    (out_dir / "candidates.json").write_text(
        json.dumps({"scene": scene, "voxel": vx, "keep_frac": args.keep_frac,
                    "candidates": cands}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"→ {out_dir}/candidates.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["n3_scene0001"])
    ap.add_argument("--voxel", type=float, default=0.015)
    ap.add_argument("--keep-frac", type=float, default=0.6, dest="keep_frac")
    ap.add_argument("--agree-frac", type=float, default=0.5, dest="agree_frac")
    ap.add_argument("--min-vox", type=int, default=2, dest="min_vox",
                    help="候選 hull 最小體素數(小,不限視角數)")
    args = ap.parse_args()
    scenes = av.resolve_scenes(args.scenes or ["n3_scene0001"])
    if not scenes:
        sys.exit("沒有場景")
    for i, scene in enumerate(scenes, 1):
        print(f"\n===== [{i}/{len(scenes)}] {scene} =====")
        try:
            process_scene(scene, args)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[error] {scene}: {e}")


if __name__ == "__main__":
    main()
