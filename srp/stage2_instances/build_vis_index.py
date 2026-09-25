#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""build_vis_index.py — 建「表面 voxel ↔ GT modal 遮罩」的可見性索引(hull 層級,與分群方法無關)。

★ 新檔,不動任何既有腳本。

為什麼存這個:
  評估要算「每群 voxel 投影回各視角的可見遮罩」與「GT modal 遮罩」的 2D IoU。
  逐次重算 z-buffer 很貴(實測 12 視角 1.4~2.7s/場),但 z-buffer 只取決於【hull occupancy + 相機位姿】,
  與 instance 分群方法【完全無關】→ 同一 hull 下所有方法共用一份索引即可。
  且不必存 per-pixel z-buffer(實測 42 MB/場、303 場 13 GB);只要存「每個表面 voxel 擁有幾個最前像素、
  其中幾個落在各 GT 物體遮罩內」就能【精確】還原 IoU(實測 0.34 MB/場、303 場 0.10 GB,小 124 倍)。

為什麼精確而非近似:z-buffer 定義下每個像素只屬於【一個】最前表面 voxel,故逐 voxel 加總 == 逐像素計數。
  |pm[k,v]|            = sum_{p in k} tot[v][p]
  |pm[k,v] & gt[o,v]|  = sum_{p in k} gtc[v][p][o]
  |pm | gt|            = |pm| + gtarea[v][o] - |pm & gt|

存什麼(每場 <HULL_ROOT>/<scene>/vis_index_12v.npz):
  views      (V,)        視角名(A-3 selected 12)
  objs       (O,)        GT 物體名(已去 YCB 前綴;含 GEX,評估端自行排除)
  gtarea     (V,O) int32 各視角各物體的 GT modal 遮罩面積(像素)
  tot_v/tot_p/tot_c      稀疏三元組:視角index / 表面voxel index / 該 voxel 擁有的最前像素數
  gtc_v/gtc_p/gtc_o/gtc_c 稀疏四元組:視角 / voxel / 物體index / 落在該物體 GT 遮罩內的像素數
  n_surf     ()          表面 voxel 總數(索引對齊 np.nonzero(hull['surface']) 的順序)
  build_meta ()          provenance:hull root、視角、GT 來源、voxel_size、影像尺寸

一致性:讀取端(match_eval.py)必須核對 build_meta 的 hull_root 與 n_surf,不符 raise(不靜默用錯索引)。
排除項:不在此排除 GEX(GT 物體全存),由評估端決定,避免索引綁死某次的排除設定。
用法: ./build_vis_index.py [scenes|groups|(空=303多物場)]  [FORCE=1 重建]
env : HULL_ROOT(完整路徑,預設 srp_hull_mv2_v12_am1) CAPTURES_ROOT
"""
import os
import re
import sys
import glob
import json
import argparse
import datetime as _dt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "srp" / "io")); sys.path.insert(0, str(HERE))
import camera as cam                 # noqa: E402
import viewpoints as VP              # noqa: E402
import cg_associate as CG            # noqa: E402
from labels import label_dir         # noqa: E402
from pycocotools import mask as RLE  # noqa: E402

HULL_ROOT = Path(os.environ.get("HULL_ROOT", str(REPO / "data" / "eval" / "srp_hull_mv2_v12_am1")))
CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures_fast")))
FORCE = os.environ.get("FORCE", "") == "1"
W, H = 1280, 720
OUT_NAME = "vis_index_12v.npz"
GRP_RE = r"^(n3|n4|n5|occ3|occ4|occ5|stack3|stack4|stack5)_scene"


def gt_modal(sc):
    """回 (objs, {view: {obj: mask}});物體名去 YCB 前綴,排除 ur5e 與全空遮罩。"""
    f = label_dir(sc) / "actual" / "annotations.json"
    if not f.is_file():
        return None, None
    d = json.loads(f.read_text())
    id2n = {c["id"]: c["name"].split("_", 1)[-1] for c in d["categories"] if c["name"] != "ur5e"}
    v_of = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
    per = {}
    for a in d["annotations"]:
        nm = id2n.get(a["category_id"])
        if nm is None:
            continue
        m = RLE.decode(a["segmentation"]).astype(bool)
        if m.sum() == 0:
            continue
        per.setdefault(v_of[a["image_id"]], {})[nm] = m
    objs = sorted({o for d_ in per.values() for o in d_})
    return objs, per


def build(sc):
    hp = HULL_ROOT / sc / "hull.npz"
    out = HULL_ROOT / sc / OUT_NAME
    if not hp.is_file():
        return "no-hull"
    if out.is_file() and not FORCE:
        return "skip"
    objs, per = gt_modal(sc)
    if not objs:
        return "no-gt"
    z = np.load(hp)
    occ = z["occupancy"]; surf = z["surface"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
    vox = np.array(np.nonzero(surf)).T; M = len(vox)          # 索引順序 = np.nonzero(surface)
    ovox = np.array(np.nonzero(occ)).T; oP = gm + (ovox + 0.5) * vs
    sg = np.full(surf.shape, -1, np.int64); sg[tuple(vox.T)] = np.arange(M)
    surf_of_occ = sg[tuple(ovox.T)]                            # 每 occ voxel → surf index(-1=內部)
    oi = {o: i for i, o in enumerate(objs)}
    views = sorted(VP.selected_view_names(12))
    gtarea = np.zeros((len(views), len(objs)), np.int32)
    tv, tp, tc = [], [], []
    gv, gp, go, gc = [], [], [], []
    sdir = CAPTURES / f"multi_{sc.split('_')[0]}" / sc
    for vi, vn in enumerate(views):
        pf = sdir / f"{vn}_pose.json"
        if not pf.is_file():
            continue
        C, Rb = cam.load_pose(pf)
        va = CG.zbuffer_visible(oP, C, Rb, W, H, vs)           # 實心遮擋
        flat = np.full(len(va), -1, np.int64); m = va >= 0; flat[m] = surf_of_occ[va[m]]
        vox_at = flat.reshape(H, W)
        vis = vox_at >= 0
        p, c = np.unique(vox_at[vis], return_counts=True)      # 每 voxel 擁有的最前像素數
        tv.append(np.full(len(p), vi, np.int32)); tp.append(p.astype(np.int32)); tc.append(c.astype(np.int32))
        for o, mask in (per.get(vn) or {}).items():
            gtarea[vi, oi[o]] = int(mask.sum())
            sel = vis & mask
            if not sel.any():
                continue
            p2, c2 = np.unique(vox_at[sel], return_counts=True)
            gv.append(np.full(len(p2), vi, np.int32)); gp.append(p2.astype(np.int32))
            go.append(np.full(len(p2), oi[o], np.int16)); gc.append(c2.astype(np.int32))
    cat = lambda L, d: (np.concatenate(L) if L else np.array([], d))
    meta = {"script": "build_vis_index.py", "hull_root": HULL_ROOT.name,
            "captures_root": CAPTURES.name, "gt": "labels/<scene>/actual/annotations.json (modal)",
            "views": views, "n_views": len(views), "image_wh": [W, H], "voxel_size": vs,
            "n_surf": int(M), "occluder": "solid occupancy (CG.zbuffer_visible)",
            "note": "索引順序 = np.nonzero(hull['surface']);未排除 GEX,由評估端處理",
            "built": _dt.datetime.now().isoformat(timespec="seconds")}
    np.savez_compressed(out, views=np.array(views), objs=np.array(objs), gtarea=gtarea,
                        tot_v=cat(tv, np.int32), tot_p=cat(tp, np.int32), tot_c=cat(tc, np.int32),
                        gtc_v=cat(gv, np.int32), gtc_p=cat(gp, np.int32),
                        gtc_o=cat(go, np.int16), gtc_c=cat(gc, np.int32),
                        n_surf=np.int64(M), build_meta=json.dumps(meta, ensure_ascii=False))
    return "ok"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("targets", nargs="*")
    a = ap.parse_args()
    scenes = []
    for x in a.targets:
        if "scene" in x:
            scenes.append(x)
        else:
            scenes += [Path(p).parent.name for p in glob.glob(str(HULL_ROOT / f"{x}_scene*/hull.npz"))]
    if not scenes:
        scenes = [Path(p).parent.name for p in glob.glob(str(HULL_ROOT / "*_scene*/hull.npz"))]
        scenes = [s for s in scenes if re.match(GRP_RE, s)]     # 303 多物場;排 n1 與 b 組
    scenes = sorted(set(scenes))
    print(f"場景={len(scenes)}  hull={HULL_ROOT.name}", flush=True)
    from collections import Counter
    st = Counter()
    for i, sc in enumerate(scenes, 1):
        st[build(sc)] += 1
        if i % 50 == 0 or i == len(scenes):
            print(f"  {i}/{len(scenes)}  {dict(st)}", flush=True)
    print(f"完成:{dict(st)}")


if __name__ == "__main__":
    main()
