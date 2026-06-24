#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""diag_stack_sep.py — 診斷:SAM 2D 遮罩有沒有把「堆疊上下物」分開。

決定 on 有沒有救:
  - SAM 有分開(上/底物各有自己的 SAM 遮罩) → hull 融合是「關聯層」造成 → 改進分離可救回 on。
  - SAM 也沒分(同一 SAM 遮罩涵蓋上+底) → 免深度 2D 分割本身分不開 → on 是硬限制。

對每個 stack 場景:由 relations.json 取 on 對(x=上物 T, y=底物 B);逐視角用 GT modal 遮罩定位 T/B,
看 SAM 遮罩集合裡 T 與 B 的最佳匹配是否為「不同遮罩」。
用法: ./srp/stage4_probe/diag_stack_sep.py [scenes...]  (預設全 stack)
"""
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import eval_reproj2d as RP   # noqa: E402  (load_modal_by_view)

SAM = REPO / "data" / "eval" / "sam_only"
LABELS = REPO / "data" / "labels"
IOU_MIN = 0.3   # 最佳匹配需 IoU≥此才算「SAM 有抓到該物」


def iou(a, b):
    i = np.logical_and(a, b).sum(); u = np.logical_or(a, b).sum()
    return i / u if u else 0.0


def best_mask(sams, m):
    best, bi = 0.0, -1
    for i, s in enumerate(sams):
        v = iou(s, m)
        if v > best:
            best, bi = v, i
    return bi, best


def load_sam(scene, view):
    d = SAM / scene / view / "masks"
    out = []
    for f in sorted(d.glob("mask_*.png")):
        out.append(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) > 127)
    return out


def process(scene):
    rel = LABELS / scene / "relations.json"
    if not rel.is_file():
        return None
    ons = [(r["x"], r["y"]) for r in json.loads(rel.read_text())["relations"] if r["type"] == "on"]
    if not ons:
        return None
    modal, _ = RP.load_modal_by_view(scene)
    if not modal:
        return None
    T, B = ons[0]                      # 上物, 底物
    sep = merged = miss = 0
    iouT = iouB = 0.0; nv = 0
    for v, objs in modal.items():
        if T not in objs or B not in objs:
            continue
        sams = load_sam(scene, v)
        if not sams:
            continue
        it, vt = best_mask(sams, objs[T])
        ib, vb = best_mask(sams, objs[B])
        nv += 1; iouT += vt; iouB += vb
        if vt < IOU_MIN or vb < IOU_MIN:
            miss += 1                  # SAM 沒抓到其一(漏)
        elif it != ib:
            sep += 1                   # 不同 SAM 遮罩 → 分開
        else:
            merged += 1                # 同一 SAM 遮罩 → 併了
    if nv == 0:
        return None
    return {"scene": scene, "pair": f"{T} on {B}", "nv": nv,
            "sep": sep, "merged": merged, "miss": miss,
            "iouT": iouT / nv, "iouB": iouB / nv}


def main():
    scenes = sys.argv[1:] or sorted(Path(p).name for p in glob.glob(str(SAM / "stack*")))
    rows = [r for r in (process(s) for s in scenes) if r]
    if not rows:
        print("無資料"); return
    Sep = sum(r["sep"] for r in rows); Mer = sum(r["merged"] for r in rows)
    Mis = sum(r["miss"] for r in rows); NV = sum(r["nv"] for r in rows)
    print(f"{'scene':<18}{'pair':<42}{'視角':>4}{'分開':>5}{'併':>4}{'漏':>4}{'IoU上':>7}{'IoU底':>7}")
    for r in sorted(rows, key=lambda x: -x["merged"])[:20]:
        print(f"{r['scene']:<18}{r['pair']:<42}{r['nv']:>4}{r['sep']:>5}{r['merged']:>4}"
              f"{r['miss']:>4}{r['iouT']:>7.2f}{r['iouB']:>7.2f}")
    print("-" * 90)
    print(f"全 {len(rows)} 堆疊場 / {NV} 視角:分開 {Sep}({Sep/NV:.0%}) | 併 {Mer}({Mer/NV:.0%}) | "
          f"漏 {Mis}({Mis/NV:.0%})")
    print(f"\n判讀:分開率高 → 融合在關聯層,on 可救;併率高 → SAM 2D 分不開,on 為硬限制。")


if __name__ == "__main__":
    main()
