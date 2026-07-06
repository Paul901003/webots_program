#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""per_obj_matrix.py — 物體 × 場景組 的端到端失敗率矩陣(列=物體,欄=各組,末欄=總計)。

每 (物體, 組):找到 = 被某 instance 覆蓋 ≥COV 的場景數;失敗率 = 1 − 找到/出現。
輸出表 + CSV(data/eval/_diag/per_obj_matrix.csv)。
用法: ./srp/stage4_probe/per_obj_matrix.py
"""
import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import eval_mesh as EM   # noqa: E402

HULL = REPO / "data" / "eval" / "srp_hull"
OUT = REPO / "data" / "eval" / "_diag"
COV = 0.5
GROUPS = ["n3", "n4", "n5", "occ3", "occ4", "occ5", "stack3", "stack4", "stack5"]


def process(scene, g, cell):
    hp = HULL / scene / "hull.npz"; ip = HULL / scene / "instances.npz"
    z = np.load(hp); gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = z["occupancy"].shape
    gt = EM.solid_mesh_occ(scene, gm, vs, shape)
    if not gt:
        return
    labels = np.load(ip)["labels"] if ip.is_file() else np.zeros(shape, np.int32)
    insts = [labels == k for k in range(1, int(labels.max()) + 1) if (labels == k).any()]
    for name, g_occ in gt.items():
        gn = int(g_occ.sum())
        cov = max((int(np.logical_and(g_occ, ins).sum()) / gn for ins in insts), default=0.0) if gn else 0.0
        c = cell[name][g]
        c[0] += 1; c[1] += int(cov >= COV)


def main():
    cell = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # name -> group -> [出現, 找到]
    for g in GROUPS:
        for sc in sorted(Path(p).name for p in glob.glob(str(HULL / f"{g}_scene*"))):
            try:
                process(sc, g, cell)
            except Exception as e:
                print(f"[err] {sc}: {e}")
        print(f"  {g} 完成")

    def tot(name):
        a = sum(cell[name][g][0] for g in GROUPS); f = sum(cell[name][g][1] for g in GROUPS)
        return a, f
    names = sorted(cell, key=lambda n: -(1 - (tot(n)[1] / tot(n)[0])) if tot(n)[0] else 0)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "per_obj_matrix.csv", "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp); w.writerow(["object"] + GROUPS + ["total_fail", "total_n"])
        for n in names:
            r = [n]
            for g in GROUPS:
                a, f = cell[n][g]
                r.append(f"{1-f/a:.2f}" if a else "-")
            a, f = tot(n); r.append(f"{1-f/a:.3f}"); r.append(a)
            w.writerow(r)

    hdr = f"{'物體':<26}" + "".join(f"{g:>7}" for g in GROUPS) + f"{'總計':>8}{'n':>5}"
    print("\n" + hdr)
    for n in names:
        line = f"{n:<26}"
        for g in GROUPS:
            a, f = cell[n][g]
            line += f"{(1-f/a):>7.2f}" if a else f"{'-':>7}"
        a, f = tot(n); line += f"{(1-f/a):>8.3f}{a:>5}"
        print(line)
    print(f"\n→ {OUT/'per_obj_matrix.csv'}  (COV={COV};欄=各組失敗率,總計=跨組失敗率)")


if __name__ == "__main__":
    main()
