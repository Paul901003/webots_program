#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""associate.py — Stage 2 跨視角 instance 指派(規格實作)。

對齊 plan/pipeline_and_experiments.md 的 Stage 2:用精確外參 + voxel 當「橋」,把各視角的
local SAM 遮罩 ID 焊成 global instance(幾何一致性,非外觀)。3D 不切割,只在佔據網格上標記。

流程:
  ① 讀 Stage 1 的 hull.npz(occupancy + grid)。
  ② 每視角建 label 圖(每塊 SAM 遮罩一個 id,排除地板;小遮罩覆蓋大遮罩)。
  ③ 每個佔據 voxel 投影回各視角 → 取得「跨視角 label 向量」。
  ④ 6-鄰接 + 「相鄰兩 voxel 在共同可見視角上 label 一致比例 ≥ agree_frac」才 union
     → 連通塊 = 一個 global instance(自動切開相接物體、合併過度分割碎片)。
  ⑤ 輸出帶 instance 標籤的佔據網格 + instances.json(各 instance 的 voxel 數/質心/各視角遮罩檔)。

輸出: data/eval/srp_hull/<scene>/instances.npz(labels)+ instances.json
需 webots_visual_hull(numpy/cv2/scipy)。沿用 srp.io.camera。
用法: ./srp/stage2_instances/associate.py n3_scene0001 [--agree-frac 0.5] [--min-vox 8] [--min-views 2]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
import camera as cam   # noqa: E402
import masks as MK     # noqa: E402


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

CAPTURES = REPO / "data" / "captures"
SAM_ROOT = REPO / "data" / "eval" / "sam_only"
HULL_ROOT = REPO / "data" / "eval" / "srp_hull"
MAX_AREA_FRAC = 0.50
BORDER = 2


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def view_label_image(view_dir, cover="small"):
    """回傳 (label int16 圖, filenames)。label k(>=1)=第 k 塊保留遮罩(id 依面積遞減固定)。
    cover='small':重疊處小遮罩勝(小者後畫,預設);'large':大遮罩勝(大者後畫)。
    地板/背景排除沿用 srp.io.masks.kept_object_masks(與 Stage 1 一致)。"""
    km = MK.kept_object_masks(view_dir)          # [(bool_mask, filename)]
    if not km:
        return None, []
    km = sorted(km, key=lambda x: -int(x[0].sum()))   # id1=最大 ... idn=最小(files 順序固定)
    H, W = km[0][0].shape
    img = np.zeros((H, W), np.int16)
    files = [name for _, name in km]
    # 畫圖順序決定重疊勝者(後畫者勝):small→大先畫小後畫;large→小先畫大後畫
    draw = range(len(km)) if cover == "small" else range(len(km) - 1, -1, -1)
    for idx in draw:
        img[km[idx][0]] = idx + 1
    return img, files


def _suf(tag):
    return f"_{tag}" if tag else ""


def process(scene, agree_frac, min_vox, min_views, min_frac, hull_root=HULL_ROOT,
            cover="large", out_root=None, hull_tag="", tag=""):
    out_root = out_root or hull_root
    hp = hull_root / scene / f"hull{_suf(hull_tag)}.npz"
    if not hp.is_file():
        print(f"[skip] {scene}: 找不到 {hp}(先跑 run_scene.py)"); return None
    z = np.load(hp)
    occ = z["occupancy"]; grid_min = z["grid_min"]; vs = float(z["voxel_size"])
    shape = occ.shape
    occ_idx = np.flatnonzero(occ.ravel())
    nk = len(occ_idx)
    if nk == 0:
        print(f"[skip] {scene}: 空 hull"); return None

    # 佔據 voxel 世界座標(GPU)
    dev = _device()
    gi, gj, gk = np.unravel_index(occ_idx, shape)
    P = grid_min + (np.stack([gi, gj, gk], 1) + 0.5) * vs
    P_t = torch.tensor(P, dtype=torch.float32, device=dev)        # (nk,3)

    # 各視角 label 向量(投影在 GPU 上算)
    group = scene.split("_")[0]
    sdir = CAPTURES / f"multi_{group}" / scene
    views = sorted((SAM_ROOT / scene).glob("view_*"))
    vnames, files_per_view, L_cols = [], {}, []
    for vdir in views:
        pose = sdir / f"{vdir.name}_pose.json"
        if not pose.is_file():
            continue
        img, files = view_label_image(vdir, cover)
        if img is None:
            continue
        H, W = img.shape
        C, R_body = cam.load_pose(pose)
        R_w2c, t = cam.pose_to_w2c(C, R_body)
        K = cam.intrinsics(W, H)
        Rt = torch.tensor(R_w2c, dtype=torch.float32, device=dev)
        tt = torch.tensor(t, dtype=torch.float32, device=dev)
        img_t = torch.tensor(img.astype(np.int32), device=dev).reshape(-1)
        X = P_t @ Rt.T + tt
        z = X[:, 2]; ok = z > 1e-9
        zz = torch.where(ok, z, torch.ones_like(z))
        u = torch.round(K[0, 0] * X[:, 0] / zz + K[0, 2]).long()
        v = torch.round(K[1, 1] * X[:, 1] / zz + K[1, 2]).long()
        inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        lab = torch.zeros(nk, dtype=torch.int32, device=dev)
        flat = (v[inb] * W + u[inb])
        lab[inb] = img_t[flat]
        L_cols.append(lab)
        files_per_view[vdir.name] = files
        vnames.append(vdir.name)
    if len(L_cols) < min_views:
        print(f"[skip] {scene}: 有效視角 < {min_views}"); return None
    L_t = torch.stack(L_cols, dim=1)              # (nk, V) on GPU
    L = L_t.cpu().numpy().astype(np.int16)         # 供 per-instance 統計

    # 6-鄰接 邊(CPU 建索引)
    idx3 = -np.ones(shape, np.int64)
    idx3[gi, gj, gk] = np.arange(nk)
    ea, eb = [], []
    for axis in range(3):
        sa = [slice(None)] * 3; sb = [slice(None)] * 3
        sa[axis] = slice(0, -1); sb[axis] = slice(1, None)
        ia = idx3[tuple(sa)].ravel(); ib = idx3[tuple(sb)].ravel()
        m = (ia >= 0) & (ib >= 0)
        ea.append(ia[m]); eb.append(ib[m])
    a = np.concatenate(ea); b = np.concatenate(eb)

    # agree 一致性(GPU)→ 過關的邊
    uf = UF(nk)
    if len(a):
        a_t = torch.tensor(a, dtype=torch.long, device=dev)
        b_t = torch.tensor(b, dtype=torch.long, device=dev)
        La = L_t[a_t]; Lb = L_t[b_t]
        both = (La > 0) & (Lb > 0)
        den = both.sum(1)
        same = ((La == Lb) & both).sum(1)
        ok = (den > 0) & (same >= agree_frac * den.to(torch.float32))
        ea_ok = a_t[ok].cpu().numpy(); eb_ok = b_t[ok].cpu().numpy()
        for x, y in zip(ea_ok, eb_ok):
            uf.union(int(x), int(y))

    roots = np.array([uf.find(i) for i in range(nk)])
    uniq, inv, counts = np.unique(roots, return_inverse=True, return_counts=True)

    # 幻影修剪:絕對 min_vox 與「相對最大塊比例 min_frac」取大者為門檻
    big = int(counts.max())
    vox_thresh = max(min_vox, int(np.ceil(min_frac * big)))

    # 依大小給 instance 標籤(過濾 < vox_thresh)
    order = np.argsort(-counts)
    labels_flat = np.zeros(occ.size, np.int32)
    instances = []
    inst_id = 0
    for ci in order:
        if counts[ci] < vox_thresh:
            continue
        sel = inv == ci
        comp_occ = occ_idx[sel]
        sub_L = L[sel]
        inst_id += 1
        labels_flat[comp_occ] = inst_id
        center = P[sel].mean(0)
        thresh = max(1, int(0.05 * int(sel.sum())))
        per_view = {}
        for col, vn in enumerate(vnames):
            vals, cnts = np.unique(sub_L[:, col][sub_L[:, col] > 0], return_counts=True)
            fs = [files_per_view[vn][int(val) - 1] for val, c in zip(vals, cnts) if c >= thresh]
            if fs:
                per_view[vn] = sorted(fs)
        instances.append({"instance": inst_id, "n_vox": int(sel.sum()),
                          "center": [round(float(x), 4) for x in center],
                          "support_views": len(per_view), "masks": per_view})

    out_dir = out_root / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"instances{_suf(tag)}.npz",
                        labels=labels_flat.reshape(shape), grid_min=grid_min, voxel_size=vs)
    (out_dir / f"instances{_suf(tag)}.json").write_text(json.dumps(
        {"scene": scene, "voxel": vs, "n_instances": len(instances),
         "instances": instances}, indent=2, ensure_ascii=False), encoding="utf-8")
    n3d = ndimage.label(occ, ndimage.generate_binary_structure(3, 1))[1]
    print(f"[{scene}] 佔據{nk} 純3D連通{n3d} → instance {len(instances)} "
          f"(voxel數: {[i['n_vox'] for i in instances]})")
    return len(instances)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--agree-frac", type=float, default=0.5, dest="agree_frac")
    ap.add_argument("--min-vox", type=int, default=8, dest="min_vox")
    ap.add_argument("--min-views", type=int, default=2, dest="min_views")
    ap.add_argument("--min-frac", type=float, default=0.05, dest="min_frac",
                    help="幻影修剪:instance voxel 數 < 最大塊×此比例 即刪(預設 0.05)")
    ap.add_argument("--root", default="srp_hull", help="輸出 instances 根目錄 data/eval/<root>/")
    ap.add_argument("--hull-root", default=None, dest="hull_root",
                    help="讀 hull.npz 的根目錄(預設=--root;可指向共用 hull,寫到別的 --root 不蓋別組)")
    ap.add_argument("--cover", choices=("small", "large"), default="large",
                    help="重疊遮罩歸屬:large=大者勝(預設,掃描證實較穩)、small=小者勝")
    ap.add_argument("--hull-tag", default="", dest="hull_tag",
                    help="讀 hull 的檔名後綴(如 am1 → hull_am1.npz)")
    ap.add_argument("--tag", default="",
                    help="寫 instances 的檔名後綴(如 am1_cvsmall → instances_am1_cvsmall.*)")
    args = ap.parse_args()
    out_root = REPO / "data" / "eval" / args.root
    hull_root = REPO / "data" / "eval" / (args.hull_root or args.root)
    for sc in args.scenes:
        try:
            process(sc, args.agree_frac, args.min_vox, args.min_views, args.min_frac,
                    hull_root, args.cover, out_root, args.hull_tag, args.tag)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")


if __name__ == "__main__":
    main()
