#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""reproj_iou.py — 把每個 pipeline instance 的 voxel 重投影回各視角,算 2D IoU vs GT modal 遮罩,報每場平均。

★ 2D 逐視角版(補 3D gtlabel 版之外的直接重投影比較)。定義:
  每視角:對 hull occupancy 做 zbuffer_visible → 每像素=最前 voxel 的「群 label」→ pred 群影像。
  每個 instance i:投影遮罩 = (pred==i);對各 GT 非-GEX modal 遮罩算 2D IoU,取最佳 = 該 instance 該視角 IoU。
  視角分數 = 該視角所有出現 instance 的最佳 IoU 平均;每場 IoU = 各視角(≥1 instance)平均。
  分 n/occ/stack;排 GEX(GT 遮罩不含 GEX 物體)。

拿什麼:hull occupancy(--hull-root)+ instances labels(--inst-root)同網格;GT modal 遮罩(labels/<sc>/actual)。
用法: SAM 無關;CAPTURES_ROOT 需 fast。 ./reproj_iou.py --inst-root srp_hull_divB_t50_reNNcSd_am1 [--hull-root srp_hull_mv2_v12_am1]
輸出: RESULT_reproj_iou_<inst-root>.md + reproj_iou_<inst-root>_perscene.csv
"""
import sys
import csv
import glob
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "io"))
import camera as cam            # noqa: E402
import viewpoints as VP         # noqa: E402
import cg_associate as CG       # noqa: E402
from labels import label_dir    # noqa: E402
from pycocotools import mask as RLE  # noqa: E402

REPO = HERE.parent.parent
EVAL = REPO / "data" / "eval"
CAP = REPO / "data" / "captures_fast"
GEX = {"skillet_lid", "windex_bottle", "colored_wood_blocks", "dice"}


def iou2d(a, b):
    inter = int((a & b).sum()); uni = int((a | b).sum())
    return inter / uni if uni else 0.0


def scene_iou(inst_root, hull_root, sc):
    hp = EVAL / hull_root / sc / "hull.npz"
    ip = EVAL / inst_root / sc / "instances.npz"
    if not (hp.is_file() and ip.is_file()):
        return None
    occ = np.load(hp)["occupancy"].astype(bool)
    z = np.load(ip); labels = z["labels"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
    if labels.shape != occ.shape:
        return None
    ovox = np.argwhere(occ); oP = gm + (ovox + 0.5) * vs
    lab_of_o = labels[ovox[:, 0], ovox[:, 1], ovox[:, 2]]     # 每 occ voxel 的群 label
    d = label_dir(sc); af = d / "actual" / "annotations.json"
    if not af.is_file():
        return None
    import json
    ann = json.load(open(af))
    id2n = {c["id"]: c["name"].split("_", 1)[-1] for c in ann["categories"] if c["name"] != "ur5e"}
    fn2img = {Path(im["file_name"]).stem: im["id"] for im in ann["images"]}
    seg = {(a2["image_id"], a2["category_id"]): a2["segmentation"] for a2 in ann["annotations"]}
    g = sc.split("_")[0]
    view_scores = []
    for vn in sorted(VP.selected_view_names(12)):
        img = fn2img.get(vn); pf = CAP / f"multi_{g}" / sc / f"{vn}_pose.json"
        if img is None or not pf.is_file():
            continue
        C, Rb = cam.load_pose(pf)
        va = CG.zbuffer_visible(oP, C, Rb, 1280, 720, vs)     # 每像素:最前 occ voxel 的 local idx(-1=無)
        pred = np.full(720 * 1280, 0, np.int32)
        m = va >= 0
        pred[m] = lab_of_o[va[m]]
        pred = pred.reshape(720, 1280)
        gts = []
        for cid, nm in id2n.items():
            if nm in GEX or (img, cid) not in seg:
                continue
            gts.append(RLE.decode(seg[(img, cid)]).astype(bool))
        if not gts:
            continue
        ious = []
        for i in np.unique(pred):
            if i <= 0:
                continue
            pm = (pred == i)
            best = max((iou2d(pm, gm2) for gm2 in gts), default=0.0)
            ious.append(best)
        if ious:
            view_scores.append(float(np.mean(ious)))
    if not view_scores:
        return None
    return float(np.mean(view_scores))


def grp_of(sc):
    return "stack" if sc.startswith(("stack", "stkb")) else ("occ" if sc.startswith("occ") else "n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst-root", required=True)
    ap.add_argument("--hull-root", default="srp_hull_mv2_v12_am1")
    a = ap.parse_args()
    scenes = sorted(Path(p).parent.name for p in glob.glob(str(EVAL / a.inst_root / "*_scene*" / "instances.npz")))
    scenes = [s for s in scenes if not s.startswith("n1_")]
    rows = []; G = {"n": [], "occ": [], "stack": [], "all": []}
    for sc in scenes:
        iou = scene_iou(a.inst_root, a.hull_root, sc)
        if iou is None:
            continue
        rows.append((sc, iou)); G[grp_of(sc)].append(iou); G["all"].append(iou)
    md = [f"# 重投影 2D IoU(每 instance vs GT modal 遮罩,每場平均):{a.inst_root}\n",
          f"- 建檔 2026-09-23;程式 `reproj_iou.py`;hull=`{a.hull_root}`;排 GEX;303 多物;可復現。",
          "- 每視角 zbuffer→pred群影像;每 instance 取對 GT 遮罩最佳 IoU;視角=instance 平均;每場=視角平均。\n",
          "## 分組平均 IoU\n", "| 組 | 場數 | 平均每場 IoU |", "|---|---|---|"]
    for gname in ("n", "occ", "stack", "all"):
        if G[gname]:
            md.append(f"| {gname} | {len(G[gname])} | {np.mean(G[gname]):.3f} |")
    out_md = HERE / f"RESULT_reproj_iou_{a.inst_root}.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    out_csv = HERE / f"reproj_iou_{a.inst_root}_perscene.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene", "mean_iou"])
        for sc, iou in rows:
            w.writerow([sc, f"{iou:.4f}"])
    print(f"[存檔] {out_md}\n[存檔] {out_csv}\n場數={len(rows)}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
