#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""stack_separation — 統計「堆疊(on 關係)有沒有被分成兩個實例」(表面空間)。

對每個 stack 場的 on 關係(上物 x、下物 y):
  取 hull 表面 voxel;各物體 GT 表面 = surf ∩ GT 實心 occ;pred 表面 = surf ∩ labels。
  某物體「主實例」= 覆蓋其 GT 表面最多的 pred label(需覆蓋 > COVER_THR 才算找到)。
  分開 = 上物主實例 ≠ 下物主實例,且兩者都找到。
排除:上或下物屬 GLOBAL_EXCLUDE(colored_wood_blocks 等)→ 不計(記數)。
用法: ./stack_separation.py --roots srp_hull_span3d,srp_hull_semcluster_surf_am1photo
"""
import sys
import json
import argparse
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
sys.path.insert(0, str(REPO / "srp" / "io"))
import eval_mesh as EM
from labels import label_dir

EVAL = REPO / "data" / "eval"
GEX = {"skillet_lid", "windex_bottle", "colored_wood_blocks", "dice"}
COVER_THR = 0.3


def surface(o):
    s = np.zeros_like(o)
    s[1:-1, 1:-1, 1:-1] = o[1:-1, 1:-1, 1:-1] & ~(o[:-2, 1:-1, 1:-1] & o[2:, 1:-1, 1:-1] & o[1:-1, :-2, 1:-1] &
                                                  o[1:-1, 2:, 1:-1] & o[1:-1, 1:-1, :-2] & o[1:-1, 1:-1, 2:])
    return s


def on_pairs(sc):
    try:
        rel = json.loads((label_dir(sc) / "relations.json").read_text())
    except Exception:
        return []
    out = []
    for r in rel.get("relations", []):
        if r.get("type") == "on":
            x = r.get("x", "").split("_", 1)[-1]; y = r.get("y", "").split("_", 1)[-1]
            if x and y:
                out.append((x, y))
    return out


def main_inst(surf_flat_labels, gt_obj_surf_mask):
    """回 (主實例label, 覆蓋率)。gt_obj_surf_mask=該物體表面 voxel 布林(在 surf 順序上)。"""
    tot = int(gt_obj_surf_mask.sum())
    if tot == 0:
        return 0, 0.0
    labs = surf_flat_labels[gt_obj_surf_mask]
    labs = labs[labs > 0]
    if len(labs) == 0:
        return 0, 0.0
    vals, cnts = np.unique(labs, return_counts=True)
    k = int(vals[cnts.argmax()])
    return k, int(cnts.max()) / tot


def eval_root(root, scenes):
    stat = defaultdict(lambda: {"evaluable": 0, "both_found": 0, "separated": 0, "gex": 0})
    for sc in scenes:
        pairs = on_pairs(sc)
        if not pairs:
            continue
        grp = sc.split("_")[0]
        ip = EVAL / root / sc / "instances.npz"
        hp = EVAL / "srp_hull_mv2_v12_am1_photo" / sc / "hull.npz"
        if not (ip.is_file() and hp.is_file()):
            continue
        labels = np.load(ip)["labels"]
        zz = np.load(hp); occ = zz["occupancy"]; gm = zz["grid_min"]; vs = float(zz["voxel_size"])
        surf = surface(occ)
        sidx = np.argwhere(surf)
        surf_labels = labels[surf]                    # 每個表面 voxel 的 pred label
        gtocc = EM.solid_mesh_occ(sc, gm, vs, labels.shape)
        # 各物體表面布林(在 sidx 順序)
        obj_surf = {}
        for name, oc in gtocc.items():
            short = name.split("_", 1)[-1]
            obj_surf[short] = oc[surf]                 # 布林 over surf voxels
        for (up, lo) in pairs:
            s = stat[grp]
            if up in GEX or lo in GEX:
                s["gex"] += 1; continue
            if up not in obj_surf or lo not in obj_surf:
                continue
            s["evaluable"] += 1
            ku, cu = main_inst(surf_labels, obj_surf[up])
            kl, cl = main_inst(surf_labels, obj_surf[lo])
            found_u = ku > 0 and cu > COVER_THR
            found_l = kl > 0 and cl > COVER_THR
            if found_u and found_l:
                s["both_found"] += 1
                if ku != kl:
                    s["separated"] += 1
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", required=True)
    args = ap.parse_args()
    scenes = []
    for g in ["stack3", "stack4", "stack5"]:
        scenes += sorted(Path(p).name for p in glob.glob(str(EVAL / "srp_hull_span3d" / f"{g}_scene*")))
    print(f"堆疊場 on 關係分離統計(表面空間,COVER>{COVER_THR});排 GLOBAL_EXCLUDE\n")
    for root in args.roots.split(","):
        stat = eval_root(root, scenes)
        tot = {"evaluable": 0, "both_found": 0, "separated": 0, "gex": 0}
        print(f"── {root}")
        print(f"   {'組':<8}{'可評on':>7}{'兩物都找到':>11}{'分開':>6}{'分開率(/可評)':>14}")
        for g in ["stack3", "stack4", "stack5"]:
            s = stat[g]
            for k in tot: tot[k] += s[k]
            r = s["separated"] / s["evaluable"] if s["evaluable"] else 0
            print(f"   {g:<8}{s['evaluable']:>7}{s['both_found']:>11}{s['separated']:>6}{r:>13.2f}")
        r = tot["separated"] / tot["evaluable"] if tot["evaluable"] else 0
        print(f"   {'全部':<8}{tot['evaluable']:>7}{tot['both_found']:>11}{tot['separated']:>6}{r:>13.2f}"
              f"   (GLOBAL_EXCLUDE 排除 {tot['gex']} 個 on)\n")


if __name__ == "__main__":
    main()
