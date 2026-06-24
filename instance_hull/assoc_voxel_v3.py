#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""assoc_voxel_v3.py — #3 voxel(規格 v3):全場投票雕刻 + 單標籤 agree 連通。

  ① 每視角:排除最大塊(>50%)後,其餘遮罩 → 標籤圖(每像素取覆蓋它的「最小遮罩號」)。
  ② 體素投票:落在前景(label>0)的視角數 >= 門檻(hull_common,~95%)→ 保留。
  ③ 連通:保留體素 6-鄰接,且「相鄰兩體素跨視角遮罩號一致比例 >= agree_frac」才連(單標籤)。
  ④ 連通塊 → instance;各視角對應遮罩 = 塊內體素該視角出現夠多的遮罩號;支持視角 >= min_views。
共用 hull_common(盒 0.7×0.7×0.35 / 256³ / 投票門檻 / 半球座標)。輸出 data/eval/v3/instance_hull_voxel/。
用法: ./instance_hull/assoc_voxel_v3.py 3 4 5  [--agree-frac 0.5] [--min-views 3]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hull_common as HC
import associate_voxel as av   # 用其 UF

OUT_ROOT = HC.EVAL_ROOT / "v3" / "instance_hull_voxel"


def view_label_image(scene, vn, H, W):
    """回傳 (label_img int16, files);label k(>=1)= kept_masks[k-1],小號覆蓋大號。"""
    km = HC.kept_masks(HC.load_masks(scene, vn), H, W)
    img = np.zeros((H, W), np.int16)
    files = []
    for k in range(len(km) - 1, -1, -1):     # 反向:小號最後寫 → 小號勝
        img[km[k][0]] = k + 1
    files = [fn for _, fn in km]
    return img, files


def process_scene(scene, args):
    views = HC.load_views(scene)
    if len(views) < 2:
        print(f"[skip] {scene}: views<2"); return None
    P, shape = HC.build_grid()
    proj = HC.project_all(P, views)
    M = len(P)
    vlist = list(views.keys())

    # 每視角:label 圖 → 逐體素標籤 L[:,col];投票
    L = np.zeros((M, len(vlist)), np.int16)
    files_per_view = {}
    votes = np.zeros(M, np.int16); nv = 0
    for col, vn in enumerate(vlist):
        v = views[vn]
        img, files = view_label_image(scene, vn, v["H"], v["W"])
        files_per_view[vn] = files
        ui, wi, inb = proj[vn]
        lab = np.zeros(M, np.int16); lab[inb] = img[wi[inb], ui[inb]]
        L[:, col] = lab
        votes += (lab > 0); nv += 1
    if nv < 2:
        return None
    occ = votes >= HC.vote_threshold(nv)
    if not occ.any():
        print(f"[skip] {scene}: 雕殼空"); return None

    # 保留體素索引 + kept 內部編號
    occ_idx = np.where(occ)[0]
    nk = len(occ_idx)
    idx_map = -np.ones(M, np.int64); idx_map[occ_idx] = np.arange(nk)
    Lk = L[occ_idx]                       # (nk, V)

    # 6-鄰接 + agree 連通
    uf = av.UF(nk)
    grid_occ = occ.reshape(shape)
    idx3 = -np.ones(shape, np.int64); idx3[grid_occ] = np.arange(nk)
    for axis in range(3):
        sa = [slice(None)] * 3; sb = [slice(None)] * 3
        sa[axis] = slice(0, -1); sb[axis] = slice(1, None)
        ia = idx3[tuple(sa)].ravel(); ib = idx3[tuple(sb)].ravel()
        m = (ia >= 0) & (ib >= 0)
        a = ia[m]; b = ib[m]
        if len(a) == 0:
            continue
        La = Lk[a]; Lb = Lk[b]
        both = (La > 0) & (Lb > 0)
        den = both.sum(1)
        same = ((La == Lb) & both).sum(1)
        ok = (den > 0) & (same >= args.agree_frac * den)
        for x, y in zip(a[ok], b[ok]):
            uf.union(int(x), int(y))

    roots = np.array([uf.find(i) for i in range(nk)])
    uniq, inv, counts = np.unique(roots, return_inverse=True, return_counts=True)

    recs = []   # (inst_dict, comp_voxel_flat_idx)
    for ci in np.argsort(-counts):
        sel = inv == ci
        comp = occ_idx[sel]
        sub_L = Lk[sel]
        nvox = int(sel.sum())
        thresh = max(1, int(0.05 * nvox))
        per_view = {}
        for col, vn in enumerate(vlist):
            labs = sub_L[:, col]
            vals, cnts = np.unique(labs[labs > 0], return_counts=True)
            files = [files_per_view[vn][int(val) - 1] for val, c in zip(vals, cnts) if c >= thresh]
            if files:
                per_view[vn] = sorted(files)
        if len(per_view) < args.min_views:
            continue
        center = P[comp].mean(0)
        recs.append(({"center": [round(float(x), 4) for x in center],
                      "n_vox": nvox, "support": len(per_view), "masks": per_view},
                     comp.astype(np.int32)))

    recs.sort(key=lambda a: -a[0]["n_vox"])
    instances = [r[0] for r in recs]
    comps = [r[1] for r in recs]
    out_dir = OUT_ROOT / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "instances.json").write_text(json.dumps(
        {"scene": scene, "centers": [i["center"] for i in instances], "instances": instances},
        indent=2, ensure_ascii=False), encoding="utf-8")
    # 存每 instance 的體素索引(供 eval 直接算指標,免重雕)
    counts_arr = np.array([len(c) for c in comps], np.int64)
    idx_arr = np.concatenate(comps) if comps else np.zeros(0, np.int32)
    np.savez_compressed(out_dir / "occ.npz", counts=counts_arr, idx=idx_arr, res=HC.RES)
    print(f"[{scene}] nv{nv} kept{nk} 元件{len(uniq)} → instances {len(instances)}")
    return len(instances)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["n3_scene0001"])
    ap.add_argument("--agree-frac", type=float, default=0.5, dest="agree_frac")
    ap.add_argument("--min-views", type=int, default=3, dest="min_views")
    args = ap.parse_args()
    scenes = HC.resolve_scenes(args.scenes)
    for i, sc in enumerate(scenes, 1):
        print(f"[{i}/{len(scenes)}]", end=" ")
        try:
            process_scene(sc, args)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")


if __name__ == "__main__":
    main()
