#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""semantic_split.py — 用 SAM 遮罩 CLIP 語意矛盾切開黏連/重疊的 hull。

想法(與 clip_hull 2D 分群不同):對已雕好的 hull,每 occupied voxel 投影各視角 → 落在的遮罩
(排手臂)→ 取該遮罩預存的 CLIP 特徵。一個 voxel 跨視角落的遮罩特徵若**分成兩群語意不相似**
(2-means 群間 cos 距離 ≥ SEM_THR,兩群各 ≥ MIN_SUPPORT 視角)→ 代表該處是兩個不同物體重疊
投影的黏連區。高矛盾 voxel 當**交界**移除 → 連通元件=物體核心(種子)→ 其餘 voxel 指派回最近種子。

輸入:hull.npz(HULL_ROOT)、SAM 遮罩+clip_feats(SAM_ROOT)、arm 剪影(ARM_MASK_ROOT)、pose(CAPTURES_ROOT)。
輸出:data/eval/<root>/<scene>/instances.npz(labels)+ instances.json(同 associate 格式,可餵 eval.py)。
需 webots_visual_hull(numpy/scipy/cv2;只讀 clip_feats.npy,不載 CLIP 模型)。
用法: ./srp/stage2_instances/semantic_split.py n5_scene0031 [--sem-thr 0.2] [--min-support 2] [--root srp_hull_sem]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.cluster.vq import kmeans2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
import camera as cam   # noqa: E402
import masks as MK     # noqa: E402

CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures")))
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "sam_only")))
HULL_ROOT = Path(os.environ.get("HULL_ROOT", str(REPO / "data" / "eval" / "srp_hull")))
ARM_MASK_ROOT = Path(os.environ.get("ARM_MASK_ROOT", str(REPO / "data" / "eval" / "srp_arm_masks")))
MAX_AREA_FRAC = 0.50


def load_arm_mask(scene, view):
    p = ARM_MASK_ROOT / scene / f"{view}_arm.png"
    if not p.is_file():
        return None
    a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return (a > 127) if a is not None else None


def view_label_image(view_dir, cover="large"):
    """label int16 圖(label k = 第 k 塊保留遮罩,面積遞減)+ files(對應 mask 檔名)。與 associate 一致。"""
    km = MK.kept_object_masks(view_dir)
    if not km:
        return None, []
    km = sorted(km, key=lambda x: -int(x[0].sum()))
    H, W = km[0][0].shape
    img = np.zeros((H, W), np.int16)
    files = [name for _, name in km]
    draw = range(len(km)) if cover == "small" else range(len(km) - 1, -1, -1)
    for idx in draw:
        img[km[idx][0]] = idx + 1
    return img, files


def conflict_score(F, min_support):
    """voxel 的跨視角遮罩特徵集 F(n×512,L2 已正規化)→ 語意矛盾分數 = 2 群中心 cos 距離。
    兩群各需 ≥ min_support 支持才算真矛盾,否則 0。"""
    n = len(F)
    if n < 2 * min_support:
        return 0.0
    try:
        cent, lab = kmeans2(F.astype(np.float64), 2, minit="++", seed=0)
    except Exception:
        return 0.0
    sizes = np.bincount(lab, minlength=2)
    if sizes.min() < min_support:
        return 0.0
    c0 = cent[0] / (np.linalg.norm(cent[0]) + 1e-9)
    c1 = cent[1] / (np.linalg.norm(cent[1]) + 1e-9)
    return float(1.0 - c0 @ c1)


def _suf(tag):
    return f"_{tag}" if tag else ""


def process(scene, sem_thr, min_support, min_vox, out_root, hull_root=HULL_ROOT, tag=""):
    hp = hull_root / scene / "hull.npz"
    if not hp.is_file():
        print(f"[skip] {scene}: 找不到 {hp}"); return None
    z = np.load(hp)
    occ = z["occupancy"]; grid_min = z["grid_min"]; vs = float(z["voxel_size"]); shape = occ.shape
    occ_idx = np.flatnonzero(occ.ravel()); nk = len(occ_idx)
    if nk == 0:
        print(f"[skip] {scene}: 空 hull"); return None
    gi, gj, gk = np.unravel_index(occ_idx, shape)
    P = grid_min + (np.stack([gi, gj, gk], 1) + 0.5) * vs

    # 每 voxel 蒐集跨視角落的遮罩 CLIP 特徵
    group = scene.split("_")[0]
    sdir = CAPTURES / f"multi_{group}" / scene
    feats_per_vox = [[] for _ in range(nk)]
    for vdir in sorted((SAM_ROOT / scene).glob("view_*")):
        pose = sdir / f"{vdir.name}_pose.json"
        cf = vdir / "clip_feats.npy"; cff = vdir / "clip_feats_files.json"
        if not (pose.is_file() and cf.is_file() and cff.is_file()):
            continue
        img, files = view_label_image(vdir)
        if img is None:
            continue
        arm = load_arm_mask(scene, vdir.name)
        if arm is not None and arm.shape == img.shape:
            img[arm] = 0
        feats = np.load(cf); cfiles = json.loads(cff.read_text())
        fname2feat = {fn: feats[i] for i, fn in enumerate(cfiles)}
        H, W = img.shape
        C, Rb = cam.load_pose(pose); Rwc, t = cam.pose_to_w2c(C, Rb); K = cam.intrinsics(W, H)
        X = P @ Rwc.T + t; zc = X[:, 2]; ok = zc > 1e-9; zz = np.where(ok, zc, 1.0)
        u = np.round(K[0, 0] * X[:, 0] / zz + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * X[:, 1] / zz + K[1, 2]).astype(int)
        inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.where(inb)[0]
        labv = img[v[idx], u[idx]]
        for vi, l in zip(idx, labv):
            if l > 0:
                f = fname2feat.get(files[l - 1])
                if f is not None:
                    feats_per_vox[vi].append(f)

    # 每 voxel 語意矛盾分數
    s = np.zeros(nk)
    for i in range(nk):
        if len(feats_per_vox[i]) >= 2 * min_support:
            s[i] = conflict_score(np.array(feats_per_vox[i]), min_support)

    # 切開:高矛盾 voxel 當交界移除 → 連通元件=種子 → 其餘指派回最近種子
    conf = s >= sem_thr
    core = np.zeros(shape, bool)
    core[gi[~conf], gj[~conf], gk[~conf]] = True
    seeds, nseed = ndimage.label(core, ndimage.generate_binary_structure(3, 1))
    # 過濾小種子(併回背景,稍後由最近種子吸收)
    if nseed > 0:
        sizes = np.bincount(seeds.ravel())
        small = {k for k in range(1, nseed + 1) if sizes[k] < min_vox}
        if small:
            mask_small = np.isin(seeds, list(small))
            seeds[mask_small] = 0
    # 重編號種子
    uniq = [k for k in np.unique(seeds) if k > 0]
    remap = {k: i + 1 for i, k in enumerate(uniq)}
    seed_lbl = np.zeros(shape, np.int32)
    for k, i in remap.items():
        seed_lbl[seeds == k] = i
    n_inst = len(uniq)
    if n_inst == 0:
        print(f"[skip] {scene}: 無有效種子"); return None

    # 所有 occ voxel 指派到最近種子(限 occ 內:對非種子 occ voxel 用 3D 最近種子 EDT)
    labels_flat = np.zeros(occ.size, np.int32)
    if n_inst == 1:
        labels_flat[occ_idx] = 1
    else:
        # EDT:對「非種子」格算到最近種子的索引,取該種子 label
        _, (ii, jj, kk) = ndimage.distance_transform_edt(seed_lbl == 0, return_indices=True)
        nearest = seed_lbl[ii, jj, kk]
        assigned = seed_lbl.copy()
        assigned[seed_lbl == 0] = nearest[seed_lbl == 0]
        lab3d = assigned * occ   # 只保留 occ
        labels_flat = lab3d.ravel().astype(np.int32)

    # 輸出
    out_dir = out_root / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = labels_flat.reshape(shape)
    np.savez_compressed(out_dir / f"instances{_suf(tag)}.npz",
                        labels=labels, grid_min=grid_min, voxel_size=vs)
    instances = []
    for i in range(1, int(labels.max()) + 1):
        m = labels == i
        nv = int(m.sum())
        if nv == 0:
            continue
        ci, cj, ck = np.where(m)
        center = grid_min + (np.array([ci.mean(), cj.mean(), ck.mean()]) + 0.5) * vs
        instances.append({"instance": i, "n_vox": nv,
                          "center": [round(float(x), 4) for x in center]})
    (out_dir / f"instances{_suf(tag)}.json").write_text(json.dumps(
        {"scene": scene, "voxel": vs, "n_instances": len(instances),
         "instances": instances}, indent=2, ensure_ascii=False), encoding="utf-8")
    n_conf = int(conf.sum())
    n3d = ndimage.label(occ, ndimage.generate_binary_structure(3, 1))[1]
    print(f"[{scene}] 佔據{nk} 矛盾voxel{n_conf} 純3D連通{n3d} → instance {len(instances)} "
          f"(voxel數: {[x['n_vox'] for x in instances]})")
    return len(instances)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--sem-thr", type=float, default=0.20, dest="sem_thr",
                    help="語意矛盾閾值(2群 cos 距離);越小越敏感切越多")
    ap.add_argument("--min-support", type=int, default=2, dest="min_support",
                    help="兩語意群各需的最少支持視角數")
    ap.add_argument("--min-vox", type=int, default=20, dest="min_vox",
                    help="種子最小體素數(小於此併回,由最近種子吸收)")
    ap.add_argument("--root", default="srp_hull_sem", help="輸出根 data/eval/<root>/")
    ap.add_argument("--hull-root", default=None, dest="hull_root", help="讀 hull 的根(預設 HULL_ROOT env)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out_root = REPO / "data" / "eval" / args.root
    hull_root = Path(REPO / "data" / "eval" / args.hull_root) if args.hull_root else HULL_ROOT
    for sc in args.scenes:
        try:
            process(sc, args.sem_thr, args.min_support, args.min_vox, out_root, hull_root, args.tag)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")


if __name__ == "__main__":
    main()
