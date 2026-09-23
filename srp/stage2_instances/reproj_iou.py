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


def surface_of(m):
    from scipy import ndimage
    return m & ~ndimage.binary_erosion(m, ndimage.generate_binary_structure(3, 1))


def footprint_mask(Pw, K, Rwc, t, W, H, vs, rmax=8):
    """整群直接投影:把 voxel(世界座標 Pw)的 footprint 聯集成 2D 遮罩(不做 z-buffer 遮擋)。"""
    X = Pw @ Rwc.T + t; z = X[:, 2]; ok = z > 1e-9
    zz = np.where(ok, z, 1.0)
    u = np.round(K[0, 0] * X[:, 0] / zz + K[0, 2]).astype(np.int64)
    v = np.round(K[1, 1] * X[:, 1] / zz + K[1, 2]).astype(np.int64)
    r = np.zeros(len(Pw), np.int64)
    r[ok] = np.clip(np.round(K[0, 0] * (vs * 0.5) / zz[ok]).astype(np.int64), 0, rmax)
    R = int(r.max()) if len(r) else 0
    mask = np.zeros((H, W), bool)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            sel = ok & (np.abs(dx) <= r) & (np.abs(dy) <= r)
            if not sel.any():
                continue
            uu = u[sel] + dx; vv = v[sel] + dy
            inb = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            mask[vv[inb], uu[inb]] = True
    return mask


def scene_iou(inst_root, hull_root, sc, mode="zbuffer"):
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
    kids = [int(k) for k in np.unique(labels) if k > 0]
    surf_Pw = {}                                             # direct 模式:每群表面 voxel 世界座標
    if mode == "direct":
        for k in kids:
            sv = np.argwhere(surface_of(labels == k))
            surf_Pw[k] = gm + (sv + 0.5) * vs
    d = label_dir(sc); af = d / "actual" / "annotations.json"
    if not af.is_file():
        return None
    import json
    ann = json.load(open(af))
    id2n = {c["id"]: c["name"].split("_", 1)[-1] for c in ann["categories"] if c["name"] != "ur5e"}
    fn2img = {Path(im["file_name"]).stem: im["id"] for im in ann["images"]}
    seg = {(a2["image_id"], a2["category_id"]): a2["segmentation"] for a2 in ann["annotations"]}
    g = sc.split("_")[0]
    V = {"iou": [], "prec": [], "other": [], "bg": []}       # 逐視角平均

    def instance_metrics(pm, gts, gexs):
        """回 (iou, precision, 溢別物, 溢背景)。gts=非GEX GT遮罩;gexs=GEX遮罩(當中性)。"""
        if not pm.any() or not gts:
            return None
        ious = [iou2d(pm, gt) for gt in gts]
        bi = int(np.argmax(ious)); gtb = gts[bi]
        npm = int(pm.sum())
        allnon = np.zeros_like(pm)
        for gt in gts:
            allnon |= gt
        allobj = allnon.copy()
        for gt in gexs:
            allobj |= gt
        prec = int((pm & gtb).sum()) / npm
        other = int((pm & allnon & ~gtb).sum()) / npm        # 溢到別的非GEX物體
        bg = int((pm & ~allobj).sum()) / npm                 # 溢到「所有物體(含GEX)之外」=背景/鬼影
        return ious[bi], prec, other, bg

    for vn in sorted(VP.selected_view_names(12)):
        img = fn2img.get(vn); pf = CAP / f"multi_{g}" / sc / f"{vn}_pose.json"
        if img is None or not pf.is_file():
            continue
        C, Rb = cam.load_pose(pf)
        gts, gexs = [], []
        for cid, nm in id2n.items():
            if (img, cid) not in seg:
                continue
            mm = RLE.decode(seg[(img, cid)]).astype(bool)
            (gexs if nm in GEX else gts).append(mm)
        if not gts:
            continue
        pms = []
        if mode == "zbuffer":
            va = CG.zbuffer_visible(oP, C, Rb, 1280, 720, vs)
            pred = np.full(720 * 1280, 0, np.int32); m = va >= 0
            pred[m] = lab_of_o[va[m]]; pred = pred.reshape(720, 1280)
            pms = [(pred == i) for i in np.unique(pred) if i > 0]
        else:
            Rwc, t = cam.pose_to_w2c(C, Rb); K = cam.intrinsics(1280, 720)
            pms = [footprint_mask(surf_Pw[k], K, Rwc, t, 1280, 720, vs) for k in kids]
        rows = [instance_metrics(pm, gts, gexs) for pm in pms]
        rows = [r for r in rows if r is not None]
        if rows:
            arr = np.array(rows)
            for j, key in enumerate(("iou", "prec", "other", "bg")):
                V[key].append(float(arr[:, j].mean()))
    if not V["iou"]:
        return None
    return {key: float(np.mean(vals)) for key, vals in V.items()}


def grp_of(sc):
    return "stack" if sc.startswith(("stack", "stkb")) else ("occ" if sc.startswith("occ") else "n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst-root", required=True)
    ap.add_argument("--hull-root", default="srp_hull_mv2_v12_am1")
    ap.add_argument("--mode", default="zbuffer", choices=["zbuffer", "direct"],
                    help="zbuffer=遮擋後可見(modal);direct=整群footprint直接投影(不遮擋)")
    a = ap.parse_args()
    scenes = sorted(Path(p).parent.name for p in glob.glob(str(EVAL / a.inst_root / "*_scene*" / "instances.npz")))
    scenes = [s for s in scenes if not s.startswith("n1_")]
    KEYS = ("iou", "prec", "other", "bg")
    rows = []; G = {gn: {k: [] for k in KEYS} for gn in ("n", "occ", "stack", "all")}
    for sc in scenes:
        r = scene_iou(a.inst_root, a.hull_root, sc, a.mode)
        if r is None:
            continue
        rows.append((sc, r)); gn = grp_of(sc)
        for k in KEYS:
            G[gn][k].append(r[k]); G["all"][k].append(r[k])
    proj = "zbuffer(modal)" if a.mode == "zbuffer" else "direct(整群footprint直接投影,不遮擋)"
    md = [f"# 重投影 2D IoU + 溢出(mode={a.mode}):{a.inst_root}\n",
          f"- 建檔 2026-09-23;程式 `reproj_iou.py --mode {a.mode}`;hull=`{a.hull_root}`;排 GEX;303 多物;可復現。",
          f"- 投影={proj};每 instance 對 GT modal 遮罩:IoU(最佳)、precision=|pm∩gt*|/|pm|、",
          "  溢別物=溢到其他非GEX物體、溢背景=溢到所有物體(含GEX)之外。視角=instance 平均、每場=視角平均。\n",
          "## 分組平均\n", "| 組 | 場數 | IoU | precision | 溢別物% | 溢背景% |", "|---|---|---|---|---|---|"]
    for gn in ("n", "occ", "stack", "all"):
        if G[gn]["iou"]:
            md.append(f"| {gn} | {len(G[gn]['iou'])} | {np.mean(G[gn]['iou']):.3f} | "
                      f"{np.mean(G[gn]['prec'])*100:.1f} | {np.mean(G[gn]['other'])*100:.1f} | "
                      f"{np.mean(G[gn]['bg'])*100:.1f} |")
    tag = f"_{a.mode}" if a.mode != "zbuffer" else ""
    out_md = HERE / f"RESULT_reproj_iou_{a.inst_root}{tag}.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    out_csv = HERE / f"reproj_iou_{a.inst_root}{tag}_perscene.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene"] + list(KEYS))
        for sc, r in rows:
            w.writerow([sc] + [f"{r[k]:.4f}" for k in KEYS])
    print(f"[存檔] {out_md}\n[存檔] {out_csv}\n場數={len(rows)}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
