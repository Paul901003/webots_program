#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""build_gt.py — 算每場景「共用 GT」(no-cache、源頭 captures_fast),存 data/eval/gt_reproj/<scene>/。
各方法的 build_hull_gt 都讀這同一份 GT → 保證各方法評估基準完全一致。
GT 內容(每物體):
  modal mask(各無遮擋視角)、YCB mesh 實心 occ(3D)、amodal-hull occ(3D)、無遮擋視角清單。
網格(grid_min/vs/shape)取自 base hull(--hull-root, 預設 srp_hull_mobilesamv2_bf)。
存: gt.npz(mesh_occ_<obj>/amodalhull_<obj>/modal_<obj>__<view>, grid), gt.json(物體/無遮擋視角/參數)。
用法: ./build_gt.py [scene|group|(空=全部)] [--hull-root srp_hull_mobilesamv2_bf]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage1_hull"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import camera as cam            # noqa: E402
from labels import LABELS       # noqa: E402
import eval_mesh as EM          # noqa: E402
import eval as EV               # noqa: E402

EVAL = REPO / "data" / "eval"
CAPTURES = REPO / "data" / "captures_fast"
EV.CAPTURES = CAPTURES                          # amodal carve pose 走 captures_fast
GT_OUT = EVAL / "gt_reproj"
OCC_THRESH = 0.9


def load_object_masks(scene):
    def load(kind):
        d = json.loads((LABELS / scene / kind / "annotations.json").read_text())
        catn = {c["id"]: c["name"] for c in d["categories"]}
        vof = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
        out = {}
        for a in d["annotations"]:
            m = mask_utils.decode(a["segmentation"]).astype(bool)
            out.setdefault(catn[a["category_id"]], {})[vof[a["image_id"]]] = m
        return out
    return load("actual"), load("amodal")


def unoccluded_views(modal, amodal):
    out = {}
    for name, av in amodal.items():
        vs = []
        for v, am in av.items():
            aa = int(am.sum())
            mm = modal.get(name, {}).get(v)
            if mm is not None and aa > 0 and mm.sum() / aa >= OCC_THRESH:
                vs.append(v)
        out[name] = sorted(vs)
    return out


def process(scene, hull_root):
    hp = EVAL / hull_root / scene / "hull.npz"
    if not hp.is_file():
        print(f"[skip] {scene}: 無 base hull {hp}"); return None
    z = np.load(hp); gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = z["occupancy"].shape
    gmax = gm + np.array(shape) * vs
    modal, amodal = load_object_masks(scene)
    if not amodal:
        print(f"[skip] {scene}: 無 amodal GT"); return None
    unocc = unoccluded_views(modal, amodal)
    gt_names = sorted(amodal.keys())

    mesh_occ = EM.solid_mesh_occ(scene, gm, vs, shape, use_cache=False)
    amodal_hull = EV.gt_object_hulls(scene, gm, gmax, vs, shape, use_cache=False)

    out = GT_OUT / scene; out.mkdir(parents=True, exist_ok=True)
    arrs = {"grid_min": gm, "voxel_size": vs, "shape": np.array(shape)}
    for g in mesh_occ:
        arrs[f"meshocc_{g}"] = mesh_occ[g]
    for g in amodal_hull:
        arrs[f"amodalhull_{g}"] = amodal_hull[g]
    for g in gt_names:                                  # 只存無遮擋視角的 modal(評估只用這些)
        for vn in unocc[g]:
            mm = modal.get(g, {}).get(vn)
            if mm is not None:
                arrs[f"modal_{g}__{vn}"] = mm
    np.savez_compressed(out / "gt.npz", **arrs)
    (out / "gt.json").write_text(json.dumps({
        "scene": scene, "hull_root": hull_root, "gt_objects": gt_names,
        "params": {"OCC_THRESH": OCC_THRESH, "captures": "captures_fast", "cache": False},
        "unoccluded_views": {g: unocc[g] for g in gt_names},
        "has_mesh": sorted(mesh_occ.keys()), "has_amodalhull": sorted(amodal_hull.keys()),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{scene}] GT {len(gt_names)}物體 | mesh {len(mesh_occ)} amodalhull {len(amodal_hull)} | "
          f"無遮擋視角數={[len(unocc[g]) for g in gt_names]}", flush=True)
    return len(gt_names)


def resolve(hull_root, targets):
    base = EVAL / hull_root
    if not targets:
        return sorted(p.parent.name for p in base.glob("*_scene*/hull.npz"))
    out = []
    for a in targets:
        out.append(a) if "scene" in a else out.extend(
            p.parent.name for p in base.glob(f"{a}_scene*/hull.npz"))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--hull-root", default="srp_hull_mobilesamv2_bf")
    args = ap.parse_args()
    for sc in resolve(args.hull_root, args.targets):
        try:
            process(sc, args.hull_root)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")


if __name__ == "__main__":
    main()
