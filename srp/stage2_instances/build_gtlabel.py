#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""build_gtlabel.py — 建「GT 完美標籤」root(同前景 hull 表面 + GT 遮罩 footprint 投票)。落地版,取代 job-tmp build_gtlabel_all.py。

用途:給某前景 hull,對每個表面 voxel 用 GT modal 遮罩投票決定它「真正屬哪個物體」,
      當作 leak/沒分開 指標(stack_leak_nosep.py)的完美分割基準。
作法(與原 job-tmp 一致):
  表面 voxel → 對每視角,zbuffer_visible 在 occupancy 上找每像素最前 voxel(→ 對應表面 voxel);
  每張 GT modal 遮罩內的像素,讓其最前表面 voxel 投該物體一票;多數決 → 該表面 voxel 的 GT 物體。
  labels: 1..N=各物體(onames 排序),0=無票(過估區);labelmap 存 build_meta。
一致:12 視角(A-3 selected);GT 來源 data/labels/<...>/actual/annotations.json;遮擋=hull occupancy。
surface 缺失時即時算(occ & ~6鄰侵蝕)——photo hull(am1_fp_photo)無 surface 欄。

用法:
  ./build_gtlabel.py --hull srp_hull_mv2_v12_am1_fp_photo --out srp_hull_gtlabel_am1fpphoto [scenes...]
  (scenes 空=全 303 多物場;已存在則 skip,FORCE=1 重做)
"""
import os
import sys
import json
import glob
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "io"))
import camera as cam            # noqa: E402
import viewpoints as VP         # noqa: E402
import cg_associate as CG       # noqa: E402
from labels import label_dir    # noqa: E402
from pycocotools import mask as RLE  # noqa: E402

REPO = HERE.parent.parent
E = REPO / "data" / "eval"
CAP = REPO / "data" / "captures_fast"


def surface_of(occ):
    st = ndimage.generate_binary_structure(3, 1)
    return occ & ~ndimage.binary_erosion(occ, st)


def build(sc, hull, rootout):
    out = E / rootout / sc
    if os.environ.get("FORCE", "") != "1" and (out / "instances.npz").is_file():
        return "skip"
    hp = E / hull / sc / "hull.npz"
    if not hp.is_file():
        return "no_hull"
    z = np.load(hp, allow_pickle=True)
    occ = z["occupancy"].astype(bool)
    surf = z["surface"].astype(bool) if "surface" in z.files else surface_of(occ)
    gm = z["grid_min"]
    vs = float(z["voxel_size"])
    shape = surf.shape
    vox = np.argwhere(surf)
    M = len(vox)
    ovox = np.argwhere(occ)
    oP = gm + (ovox + 0.5) * vs
    sg = np.full(shape, -1, np.int64)
    sg[tuple(vox.T)] = np.arange(M)
    s_of_o = sg[tuple(ovox.T)]
    d = label_dir(sc)
    af = d / "actual" / "annotations.json"
    if not af.is_file():
        return "no_ann"
    ann = json.load(open(af))
    id2n = {c["id"]: c["name"].split("_", 1)[-1] for c in ann["categories"] if c["name"] != "ur5e"}
    onames = sorted(set(id2n.values()))
    oidx = {o: i for i, o in enumerate(onames)}
    fn2img = {Path(im["file_name"]).stem: im["id"] for im in ann["images"]}
    seg = {(a["image_id"], a["category_id"]): a["segmentation"] for a in ann["annotations"]}
    g = sc.split("_")[0]
    votes = np.zeros((M, len(onames)))
    for vn in sorted(VP.selected_view_names(12)):
        img = fn2img.get(vn)
        pf = CAP / f"multi_{g}" / sc / f"{vn}_pose.json"
        if img is None or not pf.is_file():
            continue
        C, Rb = cam.load_pose(pf)
        va = CG.zbuffer_visible(oP, C, Rb, 1280, 720, vs)
        o2 = np.full(len(va), -1, np.int64)
        mm = va >= 0
        o2[mm] = s_of_o[va[mm]]
        vox_at = o2.reshape(720, 1280)
        ys, xs = np.where(vox_at >= 0)
        own = vox_at[ys, xs]
        for cid, oname in id2n.items():
            if (img, cid) not in seg:
                continue
            m = RLE.decode(seg[(img, cid)]).astype(bool)
            hit = m[ys, xs]
            if hit.any():
                np.add.at(votes[:, oidx[oname]], own[hit], 1.0)
    lab = np.zeros(M, int)
    has = votes.max(1) > 0
    lab[has] = votes.argmax(1)[has] + 1
    grid = np.zeros(shape, np.int32)
    grid[tuple(vox.T)] = lab
    out.mkdir(parents=True, exist_ok=True)
    meta = {"script": "build_gtlabel.py(前景hull+GT遮罩footprint投票)", "hull": hull,
            "labelmap": {i + 1: o for i, o in enumerate(onames)},
            "note": "1..N=各物體;0=過估區(無GT遮罩票)"}
    np.savez_compressed(out / "instances.npz", labels=grid, grid_min=gm, voxel_size=vs,
                        occupancy=occ, build_meta=json.dumps(meta, ensure_ascii=False))
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--hull", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scenes = a.scenes
    if not scenes:
        scenes = sorted(Path(p).parent.name for p in glob.glob(str(E / a.hull / "*_scene*" / "hull.npz")))
        scenes = [s for s in scenes if not s.startswith("n1_")]
    ok = skip = 0
    stat = {}
    for i, sc in enumerate(scenes):
        r = build(sc, a.hull, a.out)
        ok += (r == "ok")
        skip += (r == "skip")
        stat[r] = stat.get(r, 0) + 1
        if (i + 1) % 50 == 0:
            print(f"  {a.out} ..{i+1}/{len(scenes)} (ok={ok} skip={skip})", flush=True)
    print(f"{a.out}: 完成 {stat}")


if __name__ == "__main__":
    main()
