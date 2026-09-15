#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""d2_stack_clean — D2 堆疊分離「乾淨分開」正確判準(表面空間;同時罰過切與誤併)。

每個 stack 場 on 關係(上物 u、下物 l,兩物皆非 GLOBAL_EXCLUDE):
  表面 voxel = hull surface;每物 GT 表面 = surf ∩ GT 實心 occ;pred 表面 = surf ∩ labels。
  主群 C*(o) = 覆蓋 o GT 表面最多的 pred label(>0)。
  recall(o)    = |C*(o) ∩ GTsurf(o)| / |GTsurf(o)|                 (低 = 過切)
  precision(o) = |C*(o) ∩ GTsurf(o)| / |C*(o) ∩ (任一物體 GTsurf)|  (低 = 誤併/混物)
  clean(o) = recall≥τ 且 precision≥τ。
  ★ 乾淨分開(成功) = C*(u)≠C*(l)  且 clean(u) 且 clean(l)。
拆解也報:separated(只分開)、clean_both(只各自乾淨),看失敗是併掉還是過切。
用法: ./d2_stack_clean.py --roots srp_hull_span3d,srp_hull_semcluster_surf_am1photo
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
import os
import eval_mesh as EM
import camera as cam
import viewpoints as VP
import cg_associate as CG
from labels import label_dir

EVAL = REPO / "data" / "eval"
HULL = os.environ.get("HULL_ROOT_NAME", "srp_hull_mv2_v12_am1_photo")   # env 覆蓋:am1 vs am1_photo(無光雕)
CAP = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures_fast")))
GEX = {"skillet_lid", "windex_bottle", "colored_wood_blocks", "dice"}
TAUS = [0.3, 0.5, 0.7]
OBS_ONLY = os.environ.get("OBS_ONLY", "1") == "1"   # 1=recall/prec 分母只算「≥1視角zbuffer看得到」的表面(排不可見背面)


def observable_mask(sc, surf_idx, Pw, vs):
    """回 bool over surf voxels:該表面 voxel 是否在 12 視角中至少一個 zbuffer 最前(可觀測)。"""
    obs = np.zeros(len(surf_idx), bool)
    sdir = CAP / f"multi_{sc.split('_')[0]}" / sc
    import cv2, glob as _g
    for vn in sorted(VP.selected_view_names(12)):
        pf = sdir / f"{vn}_pose.json"
        if not pf.is_file():
            continue
        mp = _g.glob(str(EVAL / "mobilesamv2_fast" / sc / vn / "masks" / "mask_*.png"))
        if not mp:
            continue
        H, W = cv2.imread(mp[0], 0).shape
        C, Rb = cam.load_pose(pf)
        va = CG.zbuffer_visible(Pw, C, Rb, W, H, vs).reshape(H, W)
        Rwc, t = cam.pose_to_w2c(C, Rb); K = cam.intrinsics(W, H)
        X = Pw @ Rwc.T + t; zc = np.clip(X[:, 2], 1e-6, None); ok = X[:, 2] > 1e-6
        u = np.round(K[0, 0] * X[:, 0] / zc + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * X[:, 1] / zc + K[1, 2]).astype(int)
        inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ii = np.where(inb)[0]
        obs[ii[va[v[ii], u[ii]] == ii]] = True
    return obs


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


def obj_stats(surf_labels, obj_bool, any_obj_bool):
    """回 (C*label, recall, precision)。obj_bool/any_obj_bool 為 surf 順序布林。"""
    tot = int(obj_bool.sum())
    if tot == 0:
        return 0, 0.0, 0.0
    labs = surf_labels[obj_bool]; labs = labs[labs > 0]
    if len(labs) == 0:
        return 0, 0.0, 0.0
    vals, cnts = np.unique(labs, return_counts=True)
    k = int(vals[cnts.argmax()]); tp = int(cnts.max())
    recall = tp / tot
    in_c = surf_labels == k
    denom = int((in_c & any_obj_bool).sum())          # C* 內屬「任一物體」的表面 voxel
    precision = tp / denom if denom > 0 else 0.0
    return k, recall, precision


def eval_root(root, scenes):
    stat = {t: defaultdict(lambda: {"eval": 0, "sep": 0, "clean_both": 0, "success": 0}) for t in TAUS}
    for sc in scenes:
        pairs = on_pairs(sc)
        if not pairs:
            continue
        grp = sc.split("_")[0]
        ip = EVAL / root / sc / "instances.npz"
        hp = EVAL / HULL / sc / "hull.npz"
        if not (ip.is_file() and hp.is_file()):
            continue
        labels = np.load(ip)["labels"]
        zz = np.load(hp); occ = zz["occupancy"]; gm = zz["grid_min"]; vs = float(zz["voxel_size"])
        surf = surface(occ)
        surf_labels = labels[surf]
        gtocc = EM.solid_mesh_occ(sc, gm, vs, labels.shape)
        obj_surf = {name.split("_", 1)[-1]: oc[surf] for name, oc in gtocc.items()}
        if OBS_ONLY:   # 分母只算可觀測表面(排永遠看不到的背面)
            sidx = np.argwhere(surf); Pw = gm + (sidx + 0.5) * vs
            obs = observable_mask(sc, sidx, Pw, vs)
            obj_surf = {o: (m & obs) for o, m in obj_surf.items()}
        any_obj = np.zeros(surf_labels.shape, bool)
        for v in obj_surf.values():
            any_obj |= v
        for (u, l) in pairs:
            if u in GEX or l in GEX or u not in obj_surf or l not in obj_surf:
                continue
            ku, ru, pu = obj_stats(surf_labels, obj_surf[u], any_obj)
            kl, rl, pl = obj_stats(surf_labels, obj_surf[l], any_obj)
            sep = (ku > 0 and kl > 0 and ku != kl)
            for t in TAUS:
                s = stat[t][grp]
                s["eval"] += 1
                cu = ru >= t and pu >= t
                cl = rl >= t and pl >= t
                if sep:
                    s["sep"] += 1
                if cu and cl:
                    s["clean_both"] += 1
                if sep and cu and cl:
                    s["success"] += 1
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", required=True)
    ap.add_argument("scenes", nargs="*", help="顯式場景(空=預設 stack3/4/5)")
    args = ap.parse_args()
    scenes = list(args.scenes)
    if not scenes:
        for g in ["stack3", "stack4", "stack5"]:
            scenes += sorted(Path(p).name for p in glob.glob(str(EVAL / "srp_hull_span3d" / f"{g}_scene*")))
    print("D2 堆疊乾淨分離(表面空間);成功=分開 且 上下物各自 recall≥τ 且 precision≥τ;排 GLOBAL_EXCLUDE\n")
    for root in args.roots.split(","):
        stat = eval_root(root, scenes)
        print(f"── {root}")
        print(f"   {'τ':<5}{'可評on':>7}{'分開':>6}{'各自乾淨':>9}{'★乾淨分開':>11}{'成功率':>8}")
        for t in TAUS:
            tot = {"eval": 0, "sep": 0, "clean_both": 0, "success": 0}
            for g in stat[t]:                       # 全部出現的組(含 stkb),不再寫死 stack3/4/5
                for k in tot:
                    tot[k] += stat[t][g][k]
            r = tot["success"] / tot["eval"] if tot["eval"] else 0
            print(f"   {t:<5}{tot['eval']:>7}{tot['sep']:>6}{tot['clean_both']:>9}{tot['success']:>11}{r:>7.2f}")
        print()


if __name__ == "__main__":
    main()
