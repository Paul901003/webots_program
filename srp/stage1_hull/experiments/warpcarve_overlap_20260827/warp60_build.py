#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""60 場:建 12 視角 hull + warped-NCC 雕刻(tilt20 ncc0.2)+ 打包 orig/carved hull root。"""
import sys, json, time
import numpy as np
from scipy import ndimage
from pathlib import Path
sys.path.insert(0, "srp/stage1_hull"); sys.path.insert(0, "srp/io")
import photo_carve_warp as W
from photo_carve_b import load_views, build_hull

def surface(o):
    return o & ~ndimage.binary_erosion(o, ndimage.generate_binary_structure(3, 1))

SAM = Path("data/eval/sam_only_fast")
scenes = sorted(p.name for g in ("stack3", "stack4", "stack5")
                for p in SAM.glob(f"{g}_scene*") if p.is_dir())
print(f"共 {len(scenes)} 場", flush=True)
done = []
t0 = time.time()
for i, sc in enumerate(scenes):
    try:
        views = W.prep_views(load_views(sc, 12))
        if len(views) < 2:
            print(f"[skip] {sc}: 視角<2", flush=True); continue
        occ, gm, vs = build_hull(views, 0.005)
        carved = W.carve_warp(occ, gm, vs, views, 0.2, 4, 2, 3, 2, 15, 20.0)
        lab, n = ndimage.label(carved, ndimage.generate_binary_structure(3, 1))
        if n > 0:
            sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
            carved = np.isin(lab, np.nonzero(sizes >= max(5, sizes.max() * 0.01))[0] + 1)
        for root, o in [("srp_hull_warp_orig", occ), ("srp_hull_warp_carved", carved)]:
            d = Path(f"data/eval/{root}/{sc}"); d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / "hull.npz", occupancy=o, surface=surface(o), observed=o,
                                grid_min=gm, voxel_size=np.float64(vs),
                                build_meta=json.dumps({"src": root, "carve": "warp_tilt20_ncc0.2"}))
        done.append(sc)
        print(f"[{i+1}/{len(scenes)}] {sc}: {int(occ.sum())}→{int(carved.sum())} vox", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}", flush=True)
Path("/tmp/claude-1000/-home-cho-webots-program/338e7fee-7000-4d61-a82a-92a2c5d8b46c/scratchpad/warp60_scenes.txt").write_text("\n".join(done))
print(f"\n建置完成 {len(done)}/{len(scenes)} 場, {time.time()-t0:.0f}s", flush=True)
