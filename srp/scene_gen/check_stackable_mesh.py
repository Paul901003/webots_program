#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""check_stackable_mesh.py — 用視覺 mesh 幾何判定 YCB 物體是否可堆疊。

作法(不看碰撞 primitive、不看 bounding box,只看真實 mesh):
  對每個 textured.obj,在 x-y footprint 鋪 GRID×GRID 網格,垂直往下打射線,
  每格取最高命中 z=頂面高度、最低命中 z=底面高度,得兩張高度圖,再算:
    bottom_flat = 底面貼齊最低點(≤z_min+δ)的格比例   → 能穩站
    top_flat    = 頂面貼齊最高點(≥z_max−δ)的格比例   → 頂能放東西
    top_span    = 頂面平坦格的 x-y 涵蓋寬 / footprint 寬 → 平頂夠不夠大
  判定 stackable = bottom_flat≥Tb 且 top_flat≥Tt 且 top_span≥Ts。

前提:mesh 以 z 為重力軸(自然直立姿)。錨點物(罐/盒 vs 球/香蕉)驗證此假設。
用法: ./check_stackable_mesh.py            # 全 64 物
      ./check_stackable_mesh.py --anchors  # 只跑 4 個錨點驗方法
需 webots_visual_hull(trimesh)。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "urdfs" / "ycb_assets"
OUTDIR = REPO / "data" / "eval" / "_diag" / "stackable"

GRID = 64
BAND = 0.010           # 1cm:近頂/近底的搜尋帶(容許淺內凹的罐蓋/罐底)
SLAB = 0.003           # 3mm 薄層厚度:一塊「單一水平高度」的平面才算平
TB, TT, TS = 0.50, 0.50, 0.50   # bottom_flat / top_flat / top_span 門檻


def densest_level(heights, ref, band, slab):
    """在 [ref, ref+band](含負向,由呼叫端調正負)的高度帶內,用 slab 厚薄層掃描,
    回傳 (涵蓋最多格的那層 mask, 該層高度)。band>0 往上、<0 往下。"""
    lo, hi = (ref, ref + band) if band >= 0 else (ref + band, ref)
    near = (heights >= lo) & (heights <= hi)
    if not near.any():
        return np.zeros_like(heights, bool), None
    hs = heights[near]
    # slab/2 為半寬,以每個近帶高度為中心試,取涵蓋最多者
    best_mask, best_cnt = None, -1
    for c in np.unique(np.round(hs / (slab / 2)) * (slab / 2)):
        m = near & (np.abs(heights - c) <= slab / 2)
        if m.sum() > best_cnt:
            best_cnt, best_mask, best_c = int(m.sum()), m, c
    return best_mask, best_c

ANCHORS = ["005_tomato_soup_can", "003_cracker_box", "056_tennis_ball", "011_banana"]


def mesh_path(name):
    p = ASSETS / name / "google_16k" / "textured.obj"
    return p if p.is_file() else None


def analyze(name):
    mp = mesh_path(name)
    if mp is None:
        return None
    m = trimesh.load(str(mp), process=False, force="mesh")
    if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
        return None
    lo, hi = m.bounds
    size = hi - lo
    zmin, zmax = float(lo[2]), float(hi[2])

    # x-y footprint 網格中心
    xs = np.linspace(lo[0], hi[0], GRID + 1); xs = (xs[:-1] + xs[1:]) / 2
    ys = np.linspace(lo[1], hi[1], GRID + 1); ys = (ys[:-1] + ys[1:]) / 2
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    n = len(pts)

    # 從上方往 -z 打射線,收集每條射線全部命中
    origins = np.column_stack([pts, np.full(n, zmax + 0.05)])
    dirs = np.tile([0.0, 0.0, -1.0], (n, 1))
    locs, idx_ray, _ = m.ray.intersects_location(origins, dirs, multiple_hits=True)

    z_top = np.full(n, np.nan)
    z_bot = np.full(n, np.nan)
    if len(idx_ray):
        # 逐命中更新每條 ray 的 max=頂 / min=底
        for r, zval in zip(idx_ray, locs[:, 2]):
            if np.isnan(z_top[r]) or zval > z_top[r]:
                z_top[r] = zval
            if np.isnan(z_bot[r]) or zval < z_bot[r]:
                z_bot[r] = zval

    valid = ~np.isnan(z_top)
    nvalid = int(valid.sum())
    if nvalid == 0:
        return {"name": name, "size": size.tolist(), "note": "no ray hit"}

    # 近頂/近底帶內找「單一水平高度」的最大平面(彎頂/開口杯在此都只剩薄條/薄環→小)
    tz = np.where(valid, z_top, -np.inf)
    bz = np.where(valid, z_bot, np.inf)
    top_flat_mask, _ = densest_level(tz, zmax, -BAND, SLAB)
    bot_flat_mask, _ = densest_level(bz, zmin, BAND, SLAB)
    top_plateau = float(top_flat_mask.sum()) / nvalid
    bottom_plateau = float(bot_flat_mask.sum()) / nvalid

    # top_span:平頂高原格在 x-y 的涵蓋寬 / footprint 寬(取 x,y 兩軸較小者)
    if top_flat_mask.any():
        tp = pts[top_flat_mask]
        spanx = (tp[:, 0].max() - tp[:, 0].min()) / max(size[0], 1e-9)
        spany = (tp[:, 1].max() - tp[:, 1].min()) / max(size[1], 1e-9)
        top_span = float(min(spanx, spany))
    else:
        top_span = 0.0

    # top_cavity:頂面中央 50% footprint 的凹陷深度(z_max − 中央中位頂高);開口杯會很大
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    central = valid & (np.abs(pts[:, 0] - cx) <= size[0] / 4) & (np.abs(pts[:, 1] - cy) <= size[1] / 4)
    top_cavity = float(zmax - np.nanmedian(z_top[central])) if central.any() else 0.0

    # 兩個角色分開判:
    #   top_ok  = 可當「上物」:只要底面平穩(放得上去)
    #   base_ok = 可當「底物/中間物」:底面平穩 且 頂面是夠大的平面(能被疊)
    top_ok = bottom_plateau >= TB
    base_ok = top_ok and top_plateau >= TT and top_span >= TS
    return {
        "name": name,
        "size_cm": [round(float(s) * 100, 1) for s in size],
        "bottom_plateau": round(bottom_plateau, 3),
        "top_plateau": round(top_plateau, 3),
        "top_span": round(top_span, 3),
        "top_cavity_cm": round(top_cavity * 100, 1),
        "top_ok": bool(top_ok),
        "base_ok": bool(base_ok),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", action="store_true", help="只跑 4 個錨點驗方法")
    args = ap.parse_args()

    names = ANCHORS if args.anchors else sorted(p.name for p in ASSETS.iterdir()
                                                if p.is_dir() and mesh_path(p.name))
    rows = []
    print(f"{'物體':<28}{'尺寸cm(x,y,z)':<20}{'bot':>6}{'top':>6}{'span':>6}{'凹cm':>6}  角色")
    print("-" * 84)
    for nm in names:
        r = analyze(nm)
        if r is None:
            print(f"{nm:<28} (無 mesh/空)")
            continue
        if "note" in r:
            print(f"{nm:<28} {r['note']}  size={r['size']}")
            continue
        rows.append(r)
        sz = ",".join(f"{v:g}" for v in r["size_cm"])
        role = "底+頂" if r["base_ok"] else ("僅頂" if r["top_ok"] else "  ✗")
        print(f"{r['name']:<28}{sz:<20}{r['bottom_plateau']:>6}{r['top_plateau']:>6}"
              f"{r['top_span']:>6}{r['top_cavity_cm']:>6}  {role}")

    if not args.anchors and rows:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        keys = ["name", "size_cm", "bottom_plateau", "top_plateau", "top_span",
                "top_cavity_cm", "top_ok", "base_ok"]
        with open(OUTDIR / "stackable_mesh.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows:
                rr = dict(r); rr["size_cm"] = ",".join(map(str, r["size_cm"]))
                w.writerow(rr)
        (OUTDIR / "stackable_mesh.json").write_text(
            json.dumps({"params": {"grid": GRID, "band": BAND, "slab": SLAB,
                                   "Tb": TB, "Tt": TT, "Ts": TS}, "rows": rows},
                       ensure_ascii=False, indent=2))
        base = [r["name"] for r in rows if r["base_ok"]]
        toponly = [r["name"] for r in rows if r["top_ok"] and not r["base_ok"]]
        print(f"\n底物池 base_ok(頂+底都平){len(base)} 個:")
        for nm in base:
            print(f"  {nm}")
        print(f"\n上物池 = base_ok {len(base)} + 僅頂 {len(toponly)} = {len(base)+len(toponly)} 個")
        print("  僅頂(底平但頂不平,只能當上物):")
        for nm in toponly:
            print(f"  {nm}")
        print(f"\n→ {OUTDIR}/stackable_mesh.csv + .json")


if __name__ == "__main__":
    main()
