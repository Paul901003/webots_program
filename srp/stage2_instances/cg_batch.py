#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""cg_batch.py — 全域/批次版 cg 關聯(取代逐視角增量,治「稀疏離散視角」的種子/橋接問題)。

不逐幀 argmax:一次收集全 12 視角所有 detection(footprint z-buffer + largest_cc + CLIP 特徵),
對每一對 detection 算 agg=(1+PHYS_BIAS)*spatial+(1-PHYS_BIAS)*visual(spatial=voxel 重疊係數
|vi∩vj|/min、visual=CLIP 餘弦,同 ConceptGraphs 分數),agg≥SIM_THRESHOLD 連邊 → 連通元件 = 物體。
前/後側 detection 雖不直接重疊,但會經「相鄰視角逐段重疊的鏈」被連通元件接起來。

輸出 data/eval/<out_root>/<scene>/instances.{npz,json}(labels 只含表面 voxel;json 含每物體來源遮罩)。
env: HULL_ROOT SAM_ROOT CAPTURES_ROOT ARM_MASK_ROOT OUT_ROOT PHYS_BIAS SIM_THRESHOLD
用法: ./cg_batch.py [scene|group|(空=全部)] [--n-views 12]
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import camera as cam          # noqa: E402
import masks as MK            # noqa: E402
import viewpoints as VP       # noqa: E402
import cg_associate as CG     # noqa: E402  重用 zbuffer_visible / mask_subtract_contained / largest_cc / 參數

HULL_ROOT = CG.HULL_ROOT; SAM_ROOT = CG.SAM_ROOT; CAPTURES = CG.CAPTURES
SIM_THRESHOLD = CG.SIM_THRESHOLD
MIN_VOX_DET = CG.MIN_VOX_DET
MIN_OBJ_VOX = CG.MIN_OBJ_VOX
# 邊分數 agg = SPATIAL_W*spatial + VIS_W*visual(spatial 權重固定 1,調 visual 權重 VIS_W)
SPATIAL_W = float(os.environ.get("SPATIAL_W", "1.0"))
VIS_W = float(os.environ.get("VIS_W", "1.0"))
# DROP_MERGED=1:無條件丟掉「包住 ≥2 個其他遮罩」的合體遮罩(常是把堆疊物體切成一張的橋接源頭)
DROP_MERGED = os.environ.get("DROP_MERGED") == "1"
MERGED_CONTAIN = 0.8    # 子遮罩 >此比例落在大遮罩內 = 被包含


def merged_mask_keep(masks, th=MERGED_CONTAIN, min_contained=2):
    """回 keep list:遮罩若『面積較大且包住 ≥min_contained 個其他遮罩』→ 判為合體遮罩,丟(keep=False)。"""
    n = len(masks); areas = [int(m.sum()) for m in masks]; keep = [True] * n
    for i in range(n):
        if areas[i] == 0:
            keep[i] = False; continue
        cnt = 0
        for j in range(n):
            if i == j or areas[j] == 0:
                continue
            if int((masks[i] & masks[j]).sum()) / areas[j] > th and areas[i] > areas[j]:
                cnt += 1
        if cnt >= min_contained:
            keep[i] = False
    return keep


def collect_detections(scene, n_views, P, gi, gj, gk, shape, vs):
    """回 dets: list of (vox_idx ndarray, feat_norm, view, mask_name)。全視角一次收完,不分順序。"""
    group = scene.split("_")[0]
    sdir = CAPTURES / f"multi_{group}" / scene
    dets = []
    for vn in sorted(VP.selected_view_names(n_views)):
        vd = SAM_ROOT / scene / vn
        pose = sdir / f"{vn}_pose.json"
        cf = vd / "clip_mean_feats.npy"
        if not (vd.is_dir() and pose.is_file() and cf.is_file()):
            continue
        km = MK.kept_object_masks(vd)
        if not km:
            continue
        featmap = MK.mask_feats(vd)           # 用檔名查(與過濾解耦);數量不符會在 mask_feats raise
        if DROP_MERGED:                       # 丟合體遮罩(包住≥2個其他遮罩)
            keep = merged_mask_keep([m for m, _ in km])
            km = [km[k] for k in range(len(km)) if keep[k]]
            if not km:
                continue
        H, W = km[0][0].shape
        C, Rb = cam.load_pose(pose)
        vox_at = CG.zbuffer_visible(P, C, Rb, W, H, vs)
        arm = CG.load_arm(scene, vn, (H, W))
        masks_sub = CG.mask_subtract_contained([m for m, _ in km])
        names = [nm for _, nm in km]
        for mi, m in enumerate(masks_sub):
            ft = featmap.get(names[mi])       # 以檔名查特徵
            if ft is None:
                continue
            mm = m.ravel()
            if arm is not None:
                mm = mm & (~arm.ravel())
            v = vox_at[mm]; v = v[v >= 0]
            if len(v) == 0:
                continue
            vset = CG.largest_cc(set(np.unique(v).tolist()), gi, gj, gk, shape)
            if len(vset) < MIN_VOX_DET:
                continue
            fn = ft / (np.linalg.norm(ft) + 1e-9)
            dets.append((np.fromiter(vset, np.int64), fn.astype(np.float32), vn, names[mi]))
    return dets


def process(scene, n_views, out_root):
    hp = HULL_ROOT / scene / "hull.npz"
    if not hp.is_file():
        print(f"[skip] {scene}: 無 hull"); return None
    z = np.load(hp)
    shape = z["occupancy"].shape; gm = z["grid_min"]; vs = float(z["voxel_size"])
    if "surface" not in z.files:
        print(f"[skip] {scene}: 無 surface"); return None
    surf = z["surface"].astype(bool)
    sflat = np.flatnonzero(surf.ravel()); ns = len(sflat)
    if ns == 0:
        print(f"[skip] {scene}: 空 surface"); return None
    gi, gj, gk = np.unravel_index(sflat, shape)
    P = gm + (np.stack([gi, gj, gk], 1) + 0.5) * vs

    dets = collect_detections(scene, n_views, P, gi, gj, gk, shape, vs)
    N = len(dets)
    if N == 0:
        print(f"[skip] {scene}: 無 detection"); return None

    # detection×voxel 稀疏矩陣 → 兩兩共享 voxel 數
    rows, cols = [], []
    for i, (vx, *_ ) in enumerate(dets):
        rows.extend([i] * len(vx)); cols.extend(vx.tolist())
    D = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(N, ns))
    inter = np.asarray((D @ D.T).todense())               # (N,N) 共享 voxel 數
    sizes = np.array([len(d[0]) for d in dets], np.float32)
    minsz = np.minimum.outer(sizes, sizes)
    spatial = inter / np.maximum(minsz, 1.0)              # voxel 重疊係數
    F = np.stack([d[1] for d in dets])                    # (N,512) 已 L2
    visual = F @ F.T                                       # (N,N) 餘弦
    agg = SPATIAL_W * spatial + VIS_W * visual
    np.fill_diagonal(agg, -np.inf)

    adj = (agg >= SIM_THRESHOLD)                           # 建圖:agg 過門檻連邊
    ncomp, comp = connected_components(sp.csr_matrix(adj), directed=False)

    # 每連通元件 = 一個物體:voxel 聯集 + 來源遮罩
    obj_vox = [set() for _ in range(ncomp)]
    obj_masks = [dict() for _ in range(ncomp)]
    for i, (vx, _, vn, nm) in enumerate(dets):
        c = comp[i]
        obj_vox[c].update(vx.tolist())
        obj_masks[c].setdefault(vn, set()).add(nm)
    order = sorted(range(ncomp), key=lambda c: -len(obj_vox[c]))
    order = [c for c in order if len(obj_vox[c]) >= MIN_OBJ_VOX]

    labels = np.zeros(shape, np.int32)
    inst_list = []
    for newid, c in enumerate(order, 1):
        li = np.fromiter(obj_vox[c], np.int64)
        labels[gi[li], gj[li], gk[li]] = newid
        inst_list.append({"instance": newid, "n_vox": len(obj_vox[c]),
                          "masks": {v: sorted(fs) for v, fs in sorted(obj_masks[c].items())}})

    out = REPO / "data" / "eval" / out_root / scene
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "instances.npz", labels=labels, grid_min=gm, voxel_size=vs)
    (out / "instances.json").write_text(json.dumps({
        "scene": scene, "method": "cg_batch", "n_detections": N, "n_objects": len(order),
        "voxels_per_obj": sorted((len(obj_vox[c]) for c in order), reverse=True),
        "params": {"SIM_THRESHOLD": SIM_THRESHOLD, "SPATIAL_W": SPATIAL_W, "VIS_W": VIS_W,
                   "MIN_VOX_DET": MIN_VOX_DET, "MIN_OBJ_VOX": MIN_OBJ_VOX},
        "instances": inst_list,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{scene}] {N} detection → {ncomp} 連通元件 → {len(order)} 物體 "
          f"voxel/物體={[len(obj_vox[c]) for c in order]}", flush=True)
    return len(order)


def resolve(t):
    if not t:
        return sorted(Path(p).parent.name for p in glob.glob(str(HULL_ROOT / "*_scene*/hull.npz")))
    out = []
    for a in t:
        if "scene" in a:
            out.append(a)
        else:
            out += [Path(p).parent.name for p in glob.glob(str(HULL_ROOT / f"{a}_scene*/hull.npz"))]
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--n-views", type=int, default=12)
    args = ap.parse_args()
    out_root = os.environ.get("OUT_ROOT", "srp_hull_cgbatch")
    scenes = resolve(args.targets)
    for i, sc in enumerate(scenes):
        try:
            process(sc, args.n_views, out_root)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")
        if (i + 1) % 30 == 0:
            print(f"...{i+1}/{len(scenes)}", flush=True)


if __name__ == "__main__":
    main()
