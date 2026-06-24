#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""sam_recall.py — SAM 遮罩品質:每個物體在每個視角,SAM 有沒有找到對應遮罩。

對每 (場景, 視角, GT 物體):取該物 GT modal 遮罩,與該視角所有 class-agnostic SAM 遮罩比,
best IoU ≥ THR 即「找到」。指標:
  SAM 物-視角 recall = 找到的 (物,視角) 數 / 全部 (物,視角) 數;另報 mean best IoU。
分場景組(n1/n3/n4/n5/stack/occ)+ 小計 + 總體。
用法: ./srp/stage4_probe/sam_recall.py [groups...]   預設 n1 n3 n4 n5 stack3.. occ3..
"""
import glob
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import eval_reproj2d as RP   # noqa: E402  (load_modal_by_view)

SAM = REPO / "data" / "eval" / "sam_only"
THR = 0.5


def load_sam(scene, view):
    d = SAM / scene / view / "masks"
    return [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) > 127 for f in sorted(d.glob("mask_*.png"))]


def iou(a, b):
    i = np.logical_and(a, b).sum(); u = np.logical_or(a, b).sum()
    return i / u if u else 0.0


def process(scene, acc, g):
    modal, _ = RP.load_modal_by_view(scene)
    if not modal:
        return
    for view, objs in modal.items():
        sams = load_sam(scene, view)
        for name, m in objs.items():
            if m.sum() == 0:
                continue
            best = max((iou(s, m) for s in sams), default=0.0)
            acc[g][0] += 1
            acc[g][1] += int(best >= THR)
            acc[g][2] += best


def main():
    groups = sys.argv[1:] or ["n1", "n3", "n4", "n5", "stack3", "stack4", "stack5",
                              "occ3", "occ4", "occ5"]
    acc = defaultdict(lambda: [0, 0, 0.0])   # group: [物視角數, 找到數, sum best IoU]
    for g in groups:
        scenes = sorted(Path(p).name for p in glob.glob(str(REPO / "data" / "eval" / "sam_only" / f"{g}_scene*")))
        for sc in scenes:
            try:
                process(sc, acc, g)
            except Exception as e:
                print(f"[err] {sc}: {e}")
        print(f"  {g} 完成")

    def row(label, n, f, s):
        print(f"{label:<12}{n:>9}{f/n if n else 0:>10.3f}{s/n if n else 0:>10.3f}")
    print(f"\n{'組':<12}{'物-視角數':>9}{'recall@.5':>10}{'meanIoU':>10}")
    T = [0, 0, 0.0]
    for g in groups:
        n, f, s = acc[g]; row(g, n, f, s)
        T[0] += n; T[1] += f; T[2] += s
    print("-" * 41)
    for label, gs in [("小計 multi_n", ["n1", "n3", "n4", "n5"]),
                      ("小計 stack", ["stack3", "stack4", "stack5"]),
                      ("小計 occ", ["occ3", "occ4", "occ5"])]:
        n = sum(acc[g][0] for g in gs); f = sum(acc[g][1] for g in gs); s = sum(acc[g][2] for g in gs)
        if n:
            row(label, n, f, s)
    print("-" * 41)
    row("總計", *T)


if __name__ == "__main__":
    main()
