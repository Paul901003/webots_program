#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""per_obj_found.py — 各物體端到端「找到率 / 失敗率」(跨所有場景)。

對每場景:GT 實心 mesh 佔據 vs instance 佔據,每 GT 物體取最佳 instance 覆蓋 = max|gt∩inst|/|gt|;
覆蓋 ≥ COV 即「找到」。逐物體跨所有出現場景彙整:找到率 = 找到次數/出現次數,失敗率 = 1−找到率。
列出失敗率 > 0.5 的物體。
用法: ./srp/stage4_probe/per_obj_found.py [groups...]
"""
import glob
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import eval_mesh as EM   # noqa: E402

HULL = REPO / "data" / "eval" / "srp_hull"
COV = 0.5


def process(scene, perobj):
    hp = HULL / scene / "hull.npz"; ip = HULL / scene / "instances.npz"
    z = np.load(hp); gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = z["occupancy"].shape
    gt = EM.solid_mesh_occ(scene, gm, vs, shape)
    if not gt:
        return
    labels = np.load(ip)["labels"] if ip.is_file() else np.zeros(shape, np.int32)
    insts = [labels == k for k in range(1, int(labels.max()) + 1) if (labels == k).any()]
    for name, g in gt.items():
        gn = int(g.sum())
        cov = max((int(np.logical_and(g, ins).sum()) / gn for ins in insts), default=0.0) if gn else 0.0
        perobj[name][0] += 1
        perobj[name][1] += int(cov >= COV)


def main():
    groups = sys.argv[1:] or ["n1", "n3", "n4", "n5", "stack3", "stack4", "stack5",
                              "occ3", "occ4", "occ5"]
    perobj = defaultdict(lambda: [0, 0])   # name: [出現次數, 找到次數]
    for g in groups:
        for sc in sorted(Path(p).name for p in glob.glob(str(HULL / f"{g}_scene*"))):
            try:
                process(sc, perobj)
            except Exception as e:
                print(f"[err] {sc}: {e}")
        print(f"  {g} 完成")
    rows = [(n, ap, fd, fd / ap) for n, (ap, fd) in perobj.items()]
    print(f"\n=== 各物體失敗率(失敗率高→低;COV 門檻={COV})===")
    print(f"{'物體':<28}{'出現':>5}{'找到':>5}{'找到率':>8}{'失敗率':>8}")
    for n, ap, fd, fr in sorted(rows, key=lambda x: -(1 - x[3])):
        print(f"{n:<28}{ap:>5}{fd:>5}{fr:>8.3f}{1-fr:>8.3f}")
    print(f"\n(全 {len(rows)} 種物體)")


if __name__ == "__main__":
    main()
