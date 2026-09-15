#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""am1 的『切掉(FN)』來自哪:逐 GT 物體算 hull 覆蓋率,分『整個不見/部分/幾乎全蓋』,
並算總 FN 有多少來自『整個不見的物體』vs『被蓋物體的表面薄皮』。"""
import numpy as np, sys, json
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "srp/io"); sys.path.insert(0, "srp/stage2_instances")
import eval_mesh as EM
scenes = [p.name for g in ("stack3", "stack4", "stack5")
          for p in sorted(Path("data/eval/srp_hull_mv2_v12_am1").glob(f"{g}_scene*")) if (p / "hull.npz").is_file()]

buckets = {"整個不見(<10%)": 0, "大缺(10-50%)": 0, "部分(50-90%)": 0, "幾乎全蓋(>90%)": 0}
fn_from_missing = 0; fn_from_covered = 0; tot_obj = 0
per_scene_missing = []
for i, sc in enumerate(scenes):
    try:
        z = np.load(f"data/eval/srp_hull_mv2_v12_am1/{sc}/hull.npz")
        occ = z["occupancy"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
        gt = EM.solid_mesh_occ(sc, gm, vs, occ.shape)
        if not gt: continue
        nmiss = 0
        for name, g in gt.items():
            G = int(g.sum())
            if G == 0: continue
            tp = int((occ & g).sum()); cov = tp / G; fn = G - tp
            tot_obj += 1
            if cov < 0.1: buckets["整個不見(<10%)"] += 1; fn_from_missing += fn; nmiss += 1
            elif cov < 0.5: buckets["大缺(10-50%)"] += 1; fn_from_covered += fn
            elif cov < 0.9: buckets["部分(50-90%)"] += 1; fn_from_covered += fn
            else: buckets["幾乎全蓋(>90%)"] += 1; fn_from_covered += fn
        per_scene_missing.append(nmiss)
        if (i + 1) % 15 == 0: print(f"  {i+1}/{len(scenes)}", flush=True)
    except Exception as e:
        print(f"  [err] {sc}: {e}", flush=True)

print(f"\n===== am1『切掉』來源拆解, {len(scenes)}場, 共 {tot_obj} 個 GT 物體 =====")
print("每個 GT 物體被 am1 hull 蓋到多少:")
for k, v in buckets.items():
    print(f"  {k:<16}: {v:4d} 個物體 ({v/tot_obj*100:4.1f}%)")
tot_fn = fn_from_missing + fn_from_covered
print(f"\n總『切掉(FN)』voxel 來源:")
print(f"  來自『整個不見的物體』: {fn_from_missing/tot_fn*100:4.1f}%")
print(f"  來自『有蓋到的物體之表面薄皮/缺角』: {fn_from_covered/tot_fn*100:4.1f}%")
print(f"\n平均每場『整個不見』物體數: {np.mean(per_scene_missing):.2f}  (有缺物的場: {sum(1 for m in per_scene_missing if m>0)}/{len(per_scene_missing)})")
