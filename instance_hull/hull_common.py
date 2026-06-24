#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""hull_common.py — 六方法共用基底(規格 v3),保證各方法雕刻/背景/工作空間逐位元一致。

提供:
  常數      : BOX(0.7×0.7×0.35 半球 AABB)、RES=256、半球(WS_CENTER/WS_RADIUS)。
  背景      : foreground_excl_largest —— 每視角排除「最大塊且 >50% 畫面」(=floor),其餘聯集。
  工作空間  : in_halfsphere(p)。
  幾何      : load_views(poses+內參,重用 av)、build_grid()、project()。
  雕刻      : vote_threshold(nv) = nv - round(0.05*nv)(參考 python-visual-hull,~允許 5% 視角漏)。
  連通      : components6(occ, shape, predicate=None) —— 6-鄰接 union-find,可加條件。
不用深度、不用 GT。座標/內參與 associate_voxel(av)完全相同。
"""

import json
from pathlib import Path

import cv2
import numpy as np

import associate_voxel as av  # 內參/姿態/常數來源(已核對與其他程式一致)

REPO = av.REPO
CAPTURES = av.CAPTURES
SAM_ROOT = av.SAM_ROOT
EVAL_ROOT = REPO / "data" / "eval"

# ── 工作空間(半球)與體素盒(其 AABB)──
WS_CENTER = np.array([0.35, 0.0, 0.0], dtype=np.float64)
WS_RADIUS = 0.35
BOX_X = (0.0, 0.70)
BOX_Y = (-0.35, 0.35)
BOX_Z = (0.0, 0.35)
RES = 256
BG_FRAC = 0.50   # 最大塊面積 > 此比例才當背景排除


def in_halfsphere(p):
    return ((p[0] - 0.35) ** 2 + p[1] ** 2 + p[2] ** 2 <= WS_RADIUS ** 2) and (p[2] >= 0.0)


def resolve_scenes(targets):
    out = []
    for a in targets:
        if "scene" in str(a):
            out.append(a)
        else:
            out += [d.name for d in sorted((CAPTURES / f"multi_n{a}").glob(f"n{a}_scene*"))]
    return out


def load_views(scene):
    """{view: {C,R,fx,cx,cy,W,H}};內參/姿態用 av(與六方法一致)。"""
    g = scene.split("_")[0]
    sdir = CAPTURES / f"multi_{g}" / scene
    sam = SAM_ROOT / scene
    views = {}
    for vdir in sorted(sam.glob("view_*")):
        pp = sdir / f"{vdir.name}_pose.json"
        any_m = next((vdir / "masks").glob("mask_*.png"), None)
        if not pp.is_file() or any_m is None:
            continue
        C, R = av.load_pose(pp)
        H, W = cv2.imread(str(any_m), cv2.IMREAD_GRAYSCALE).shape
        fx, cx, cy = av.intrinsics(W, H)
        views[vdir.name] = {"C": C, "R": R, "fx": fx, "cx": cx, "cy": cy, "W": W, "H": H}
    return views


def load_masks(scene, vn):
    """回傳 [(bool_mask, filename)](非空)。"""
    out = []
    for mp in sorted((SAM_ROOT / scene / vn / "masks").glob("mask_*.png")):
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        b = m > 127
        if b.sum() == 0:
            continue
        out.append((b, mp.name))
    return out


def kept_masks(masks, H, W):
    """套背景規則:若最大塊面積 > BG_FRAC*畫面 → 排除它;回傳保留的 [(mask,filename)]。"""
    if not masks:
        return []
    areas = [int(b.sum()) for b, _ in masks]
    j = int(np.argmax(areas))
    drop = areas[j] > BG_FRAC * H * W
    return [m for i, m in enumerate(masks) if not (drop and i == j)]


def foreground(masks, H, W):
    """保留遮罩的聯集(silhouette)。"""
    km = kept_masks(masks, H, W)
    fg = None
    for b, _ in km:
        fg = b if fg is None else (fg | b)
    return fg


def build_grid():
    xs = np.linspace(*BOX_X, RES); ys = np.linspace(*BOX_Y, RES); zs = np.linspace(*BOX_Z, RES)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)
    return P, (RES, RES, RES)


_B2O = av.BODY_TO_OPENCV.T.astype(np.float32)


def project(P, v):
    X = (P - v["C"].astype(np.float32)) @ v["R"].astype(np.float32) @ _B2O
    z = X[:, 2]; ok = z > 1e-6
    u = np.empty(len(P), np.float32); w = np.empty(len(P), np.float32)
    u[ok] = v["fx"] * X[ok, 0] / z[ok] + v["cx"]
    w[ok] = v["fx"] * X[ok, 1] / z[ok] + v["cy"]
    ui = np.round(u).astype(np.int32); wi = np.round(w).astype(np.int32)
    inb = ok & (ui >= 0) & (ui < v["W"]) & (wi >= 0) & (wi < v["H"])
    return ui, wi, inb


def vote_threshold(nv):
    """參考 python-visual-hull:容許 ~5% 視角漏。"""
    return nv - round(0.05 * nv)


def project_all(P, views):
    return {vn: project(P, v) for vn, v in views.items()}


def carve_union(scene, views, P, proj):
    """全場雕刻:每視角 silhouette=保留遮罩聯集,投票 >= 門檻 → 佔據 bool。回傳 (occ, nv)。"""
    votes = np.zeros(len(P), np.int16); nv = 0
    for vn, v in views.items():
        fg = foreground(load_masks(scene, vn), v["H"], v["W"])
        if fg is None:
            continue
        ui, wi, inb = proj[vn]
        hit = np.zeros(len(P), bool); hit[inb] = fg[wi[inb], ui[inb]]
        votes += hit; nv += 1
    if nv < 2:
        return None, nv
    occ = votes >= vote_threshold(nv)
    return (occ if occ.any() else None), nv


def components6(occ_flat, shape, min_vox=0):
    """6-鄰接連通元件(scipy)。回傳 list[ndarray(flat voxel indices)],依大小遞減,過濾 < min_vox。"""
    from scipy import ndimage
    lab, n = ndimage.label(occ_flat.reshape(shape),
                           structure=ndimage.generate_binary_structure(3, 1))
    flat = lab.ravel()
    comps = []
    for c in range(1, n + 1):
        idx = np.where(flat == c)[0]
        if len(idx) >= min_vox:
            comps.append(idx)
    comps.sort(key=len, reverse=True)
    return comps
