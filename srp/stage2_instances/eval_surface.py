#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""eval_surface.py — 在「表面 voxel 空間」統一評估實例分離(公平比 cg 表面 vs 四方法實心)。

對每場景:取 hull surface voxel;GT 物體表面 = surf ∩ GT 實心 occ;pred instance 表面 = surf ∩ labels。
- recall: 每個 GT 物體是否被某 pred 覆蓋(pred 覆蓋該 GT 表面 > COVER_THR,且該 pred 主 GT = 它)
- mixed : 一個 pred 跨 ≥2 GT(純度 < PURITY)的數量
- inst  : pred instance 數
四方法 labels 是實心,取 ∩surf 得表面,與 cg 同基準。
用法: ./eval_surface.py --root srp_hull_cg [scene|group|(空=全部)]
"""
import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import eval_mesh as EM        # noqa: E402

HULL = REPO / "data" / "eval" / "srp_hull_v12"
COVER_THR = 0.3               # pred 覆蓋 GT 表面 > 此值 → 該 GT 被找到
PURITY = 0.7                  # pred 純度 < 此值 → mixed(跨物體)


def prep_gt(scene):
    """回 (surf, gsurf) 或 None。GT 每場景只算一次,供多方法共用。"""
    hp = HULL / scene / "hull.npz"
    if not hp.is_file():
        return None
    zh = np.load(hp)
    if "surface" not in zh.files:
        return None
    surf = zh["surface"].astype(bool); gm = zh["grid_min"]; vs = float(zh["voxel_size"]); shape = surf.shape
    gt = EM.solid_mesh_occ(scene, gm, vs, shape)
    gsurf = {n: (surf & (o > 0)) for n, o in gt.items() if int((surf & (o > 0)).sum()) >= 20}
    return (surf, gsurf) if gsurf else None


def eval_labels(scene, root, prep):
    surf, gsurf = prep
    ip = REPO / "data" / "eval" / root / scene / "instances.npz"
    if not ip.is_file():
        return None
    labels = np.load(ip)["labels"]
    if labels.shape != surf.shape:
        return None
    lab_s = np.where(surf, labels, 0)
    gtot = {n: int(g.sum()) for n, g in gsurf.items()}
    found = set(); mixed = 0; n_inst = 0
    for pid in np.unique(lab_s):
        if pid == 0:
            continue
        pm = lab_s == pid
        tot = int(pm.sum())
        if tot == 0:
            continue
        n_inst += 1
        inters = {n: int((pm & g).sum()) for n, g in gsurf.items()}
        best = max(inters, key=inters.get)
        if inters[best] / tot < PURITY:
            mixed += 1
        if inters[best] / gtot[best] > COVER_THR:
            found.add(best)
    return {"n_gt": len(gsurf), "found": len(found), "mixed": mixed, "n_inst": n_inst}


def resolve(t):
    if not t:
        return sorted(Path(p).parent.name for p in glob.glob(str(HULL / "*_scene*/hull.npz")))
    out = []
    for a in t:
        if "scene" in a:
            out.append(a)
        else:
            out += [Path(p).parent.name for p in glob.glob(str(HULL / f"{a}_scene*/hull.npz"))]
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", required=True, help="逗號分隔多方法 root,一次對比")
    ap.add_argument("targets", nargs="*")
    args = ap.parse_args()
    roots = args.roots.split(",")
    scenes = resolve(args.targets)
    # stat[root][group] = [n_gt, found, mixed, n_inst, n_scene]
    stat = {r: defaultdict(lambda: [0, 0, 0, 0, 0]) for r in roots}
    for sc in scenes:
        prep = prep_gt(sc)
        if prep is None:
            continue
        g = sc.split("_")[0].rstrip("0123456789")
        for r in roots:
            res = eval_labels(sc, r, prep)
            if res is None:
                continue
            s = stat[r][g]
            s[0] += res["n_gt"]; s[1] += res["found"]; s[2] += res["mixed"]; s[3] += res["n_inst"]; s[4] += 1
    print(f"表面空間評估 (COVER>{COVER_THR} PURITY<{PURITY});recall=找到GT/總GT, mixed=跨物體inst數\n")
    for r in roots:
        print(f"── {r}")
        print(f"   {'組':6}{'場景':>5}{'GT':>6}{'找到':>6}{'recall':>8}{'mixed':>7}{'inst':>6}")
        tot = [0, 0, 0, 0, 0]
        for g in sorted(stat[r]):
            s = stat[r][g]
            for i in range(5): tot[i] += s[i]
            print(f"   {g:6}{s[4]:>5}{s[0]:>6}{s[1]:>6}{s[1]/max(s[0],1):>8.2f}{s[2]:>7}{s[3]:>6}")
        print(f"   {'全部':6}{tot[4]:>5}{tot[0]:>6}{tot[1]:>6}{tot[1]/max(tot[0],1):>8.2f}{tot[2]:>7}{tot[3]:>6}\n")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
