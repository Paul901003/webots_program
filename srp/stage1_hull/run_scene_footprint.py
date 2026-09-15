#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""run_scene_footprint.py — run_scene.py 的變體:雕刻改用 carve_footprint(整個 voxel footprint OR)。

不動 run_scene.py / carve.py。作法:import run_scene 復用其 load_scene/BOX/main 的所有邏輯,
只覆寫 process()——把 carve 換成 carve_footprint.carve_visual_hull,並在 build_meta 標明 carve="footprint_or"。
輸出 hull.npz 與 run_scene 同格式(occupancy/observed/grid_min/voxel_size/build_meta),下游可直接吃。

用法(與 run_scene 相同旗標): SAM_ROOT=.. CAPTURES_ROOT=.. ARM_MASK_ROOT=.. \
  ./run_scene_footprint.py <scenes...> --num-views 12 --allow-miss 1 --root srp_hull_mv2_v12_am1_fp
"""
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_scene as RS               # noqa: E402  復用 load_scene/BOX/main
import carve_footprint as CF         # noqa: E402  footprint OR 雕刻


def process(scene, voxel, use_table, allow_miss, out_root=RS.OUT_ROOT, tag="", num_views=None,
            miss_frac=None):
    masks, Ks, extr, names = RS.load_scene(scene, num_views)
    if len(masks) < 2:
        print(f"[skip] {scene}: 有效視角 < 2")
        return None
    eff_miss = round(miss_frac * len(masks)) if miss_frac is not None else allow_miss
    hull = CF.carve_visual_hull(masks, Ks, extr, RS.BOX_MIN, RS.BOX_MAX, voxel,
                                table_z=(RS.TABLE_Z if use_table else None),
                                allow_miss=eff_miss)
    n_comp = ndimage.label(hull.occupancy, ndimage.generate_binary_structure(3, 1))[1]
    out_dir = out_root / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    build_meta = {
        "script": "run_scene_footprint.py", "carve": "footprint_or", "footprint_rmax": CF.RMAX,
        "built": _dt.datetime.now().isoformat(timespec="seconds"),
        "sam_root": RS.SAM_ROOT.name, "captures_root": RS.CAPTURES.name, "arm_root": RS.ARM_MASK_ROOT.name,
        "n_views": len(masks), "views": names, "num_views_arg": num_views,
        "voxel_size": voxel, "allow_miss": eff_miss, "miss_frac": miss_frac,
        "use_table": use_table, "table_z": (RS.TABLE_Z if use_table else None),
        "outside_is_background": True, "box_min": RS.BOX_MIN.tolist(), "box_max": RS.BOX_MAX.tolist(),
    }
    np.savez_compressed(out_dir / f"hull{RS._suf(tag)}.npz",
                        occupancy=hull.occupancy, observed=hull.observed,
                        grid_min=hull.grid_min, voxel_size=hull.voxel_size,
                        build_meta=json.dumps(build_meta, ensure_ascii=False))
    ftag = f" (frac {miss_frac})" if miss_frac is not None else ""
    print(f"[{scene}] (footprint OR) 視角{len(masks)} voxel{voxel} allow_miss={eff_miss}{ftag} → "
          f"佔據 {int(hull.occupancy.sum())} vox ({hull.volume()*1e3:.2f} L) 連通元件 {n_comp} | "
          f"observed {int(hull.observed.sum())}/{hull.observed.size}")
    return hull


if __name__ == "__main__":
    RS.process = process            # 讓 RS.main() 呼叫我們的 process
    RS.main()
