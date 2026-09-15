#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""舊 367 場:讀現成 am1 hull → warped-NCC 光度雕(與品質分析同參數)→ 存 srp_hull_mv2_v12_am1_photo。
不重建 hull、不動 am1;occupancy=雕後、surface重算、observed沿用am1、build_meta記provenance。"""
import sys, json, time
import numpy as np
from scipy import ndimage
from pathlib import Path
sys.path.insert(0, "srp/stage1_hull"); sys.path.insert(0, "srp/io")
import photo_carve_warp as W
from photo_carve_b import load_views

def surface(o):
    return o & ~ndimage.binary_erosion(o, ndimage.generate_binary_structure(3, 1))

SRC = Path("data/eval/srp_hull_mv2_v12_am1")
OUT = "srp_hull_mv2_v12_am1_photo"
GROUPS = ("n1", "n3", "n4", "n5", "occ3", "occ4", "occ5", "stack3", "stack4", "stack5")
scenes = sorted(p.name for g in GROUPS for p in SRC.glob(f"{g}_scene*")
                if (p / "hull.npz").is_file())
print(f"舊場景共 {len(scenes)} 場 → {OUT}", flush=True)
done = []; t0 = time.time()
for i, sc in enumerate(scenes):
    try:
        z = np.load(SRC / sc / "hull.npz")
        occ0 = z["occupancy"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
        observed = z["observed"] if "observed" in z else occ0
        views = W.prep_views(load_views(sc, 12))
        if len(views) < 2:
            print(f"[skip] {sc}: 視角<2", flush=True); continue
        carved = W.carve_warp(occ0, gm, vs, views, 0.1, 4, 2, 3, 2, 15, 20.0)
        # 濾掉光度雕產生的小碎塊(<MIN_COMP voxel);同時取連通元件當檢視用 labels
        MIN_COMP = 20
        lab, n = ndimage.label(carved, ndimage.generate_binary_structure(3, 1))
        if n > 0:
            sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
            keep = np.nonzero(sizes >= MIN_COMP)[0] + 1
            carved = np.isin(lab, keep)
            lab2 = np.zeros_like(lab)                       # 重新編號 1..k(供檢視上色)
            for newk, oldk in enumerate(keep, 1):
                lab2[lab == oldk] = newk
            lab2 = lab2.astype(np.int32)
        else:
            lab2 = lab.astype(np.int32)
        meta = {"src": "srp_hull_mv2_v12_am1", "sam": "mobilesamv2_fast",
                "num_views": 12, "allow_miss": 1, "min_comp": MIN_COMP,
                "carve": "warp_tilt20_ncc0.1_k4_min2_clu3_patch2_iter15"}
        d = Path(f"data/eval/{OUT}/{sc}"); d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "hull.npz", occupancy=carved, surface=surface(carved),
                            observed=observed, grid_min=gm, voxel_size=np.float64(vs),
                            build_meta=json.dumps(meta))
        # 檢視用 instances.npz:純 hull 無語意 → labels 全 0,grayfill 整顆顯灰(不可用連通元件假裝語意)
        np.savez_compressed(d / "instances.npz", labels=np.zeros_like(carved, np.int32),
                            occupancy=carved, grid_min=gm, voxel_size=np.float64(vs),
                            build_meta=json.dumps({"src": "am1_photo 純hull無語意,整顆灰(grayfill)"}))
        done.append(sc)
        if (i + 1) % 20 == 0 or i < 3:
            dt = time.time() - t0
            print(f"[{i+1}/{len(scenes)}] {sc}: {int(occ0.sum())}→{int(carved.sum())} vox "
                  f"({dt:.0f}s, {dt/(i+1):.1f}s/場)", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}", flush=True)
print(f"\n完成 {len(done)}/{len(scenes)} 場, {time.time()-t0:.0f}s → data/eval/{OUT}", flush=True)
