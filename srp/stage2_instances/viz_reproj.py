#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""viz_reproj.py — 各物體殼重投影 vs 對應 modal 遮罩 的可視化 + IoU 差異排名。

每場景每物體:把 instance 殼重投影到各拍攝視角,與該物體 modal(整場景含遮擋)遮罩比 2D IoU。
- 可視化:在 RGB 上疊色 —— 綠=兩者皆有(吻合)、紅=殼有但遮罩無(超出可見:遮擋恢復/過估)、
  藍=遮罩有但殼無(殼漏)。每物體取 IoU 最低的數個視角(差異最明顯處)成一張場景圖。
- 排名:輸出每場景的平均/最低 2D IoU,挑出「差異明顯(IoU 低)」的場景清單。

對應:instance↔物名 用 3D 實心 mesh IoU 配對(重用 eval_mesh)。
輸出:data/eval/srp_sweep/viz_reproj/<scene>.png + ranking.csv
需 webots_visual_hull(matplotlib/cv2/pycocotools)。
用法: ./srp/stage2_instances/viz_reproj.py <scenes> --root srp_sweep --tag am0_cvlarge_ag50 [--nworst 3]
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera as cam              # noqa: E402
import eval_mesh as EM            # noqa: E402
import eval_reproj2d as RP        # noqa: E402

CAPTURES = REPO / "data" / "captures"
EVAL = REPO / "data" / "eval"
OUT = EVAL / "srp_sweep" / "viz_reproj"
DS = 2   # RGB/遮罩下採樣倍數(顯示用)


def panel(ax, rgb, reproj, modal, title):
    ov = rgb.astype(float)
    both = reproj & modal; ho = reproj & ~modal; mo = modal & ~reproj
    for m, col in ((both, [0, 230, 0]), (ho, [240, 0, 0]), (mo, [0, 90, 255])):
        ov[m] = 0.45 * ov[m] + 0.55 * np.array(col)
    ax.imshow(np.clip(ov, 0, 255).astype(np.uint8)); ax.set_title(title, fontsize=7)
    ax.axis("off")


def process(scene, tag, root, nworst, viz=True):
    ip = EVAL / root / scene / f"instances{('_'+tag) if tag else ''}.npz"
    if not ip.is_file():
        return None
    z = np.load(ip); labels = z["labels"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
    shape = labels.shape
    k_ids = [k for k in range(1, int(labels.max()) + 1) if (labels == k).any()]
    hulls = [labels == k for k in k_ids]
    gt = EM.solid_mesh_occ(scene, gm, vs, shape)
    if not gt:
        return None
    names = list(gt); meshes = [gt[n] for n in names]
    M = np.array([[EM.iou3(h, m) for m in meshes] for h in hulls]) if hulls else np.zeros((0, 0))
    inst2name = {}
    if len(hulls) and len(names):
        ri, cj = linear_sum_assignment(-M)
        for i, j in zip(ri, cj):
            if M[i, j] > 0:
                inst2name[i] = names[j]
    modal, sdir = RP.load_modal_by_view(scene)
    if not modal:
        return None
    group = scene.split("_")[0]
    cdir = CAPTURES / f"multi_{group}" / scene

    per_obj = {}          # name -> list[(view, iou, reproj, modal)]
    centers = {}
    for i, h in enumerate(hulls):
        gi, gj, gk = np.nonzero(h)
        centers[i] = gm + (np.stack([gi, gj, gk], 1) + 0.5) * vs
    for i, name in inst2name.items():
        recs = []
        for v, objs in modal.items():
            if name not in objs:
                continue
            mod = objs[name]; H, W = mod.shape
            C, R_body = cam.load_pose(sdir / f"{v}_pose.json")
            R_w2c, t = cam.pose_to_w2c(C, R_body)
            rp = RP.reproject(centers[i], cam.intrinsics(W, H), R_w2c, t, H, W)
            recs.append((v, EM.iou3(rp, mod), rp, mod))
        if recs:
            per_obj[name] = sorted(recs, key=lambda x: x[1])   # IoU 升序
    if not per_obj:
        return None
    all_iou = [r[1] for recs in per_obj.values() for r in recs]
    summ = {"scene": scene, "n_obj": len(per_obj),
            "mean_iou": round(float(np.mean(all_iou)), 3),
            "min_iou": round(float(min(all_iou)), 3),
            "worst_obj": min(per_obj, key=lambda n: per_obj[n][0][1]),
            "worst_obj_iou": round(min(per_obj[n][0][1] for n in per_obj), 3)}

    if viz:
        rows = len(per_obj)
        fig, axes = plt.subplots(rows, nworst, figsize=(nworst * 3.2, rows * 2.0), squeeze=False)
        rgb_cache = {}
        for r, (name, recs) in enumerate(per_obj.items()):
            for c in range(nworst):
                ax = axes[r][c]
                if c >= len(recs):
                    ax.axis("off"); continue
                v, iou, rp, mod = recs[c]
                if v not in rgb_cache:
                    img = cv2.imread(str(cdir / f"{v}.png"))
                    rgb_cache[v] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
                rgb = rgb_cache[v]
                if rgb is None:
                    ax.axis("off"); continue
                rgb_d = rgb[::DS, ::DS]; rp_d = rp[::DS, ::DS]; mod_d = mod[::DS, ::DS]
                panel(ax, rgb_d, rp_d, mod_d, f"{name}\n{v} IoU={iou:.2f}")
        fig.suptitle(f"{scene}  (綠=吻合 紅=殼超出可見 藍=殼漏)  tag={tag}", fontsize=9)
        fig.tight_layout()
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / f"{scene}.png", dpi=90); plt.close(fig)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--root", default="srp_sweep")
    ap.add_argument("--tag", default="am0_cvlarge_ag50")
    ap.add_argument("--nworst", type=int, default=3, help="每物體取 IoU 最低的幾個視角")
    ap.add_argument("--no-viz", action="store_true", help="只算排名不畫圖")
    args = ap.parse_args()
    rows = []
    for i, sc in enumerate(args.scenes, 1):
        try:
            s = process(sc, args.tag, args.root, args.nworst, viz=not args.no_viz)
            if s:
                rows.append(s)
            if i % 20 == 0:
                print(f"  ...{i}/{len(args.scenes)}")
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")
    if not rows:
        print("無資料"); return
    rows.sort(key=lambda r: r["mean_iou"])
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "ranking.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n=== 重投影 IoU 差異最明顯(mean_iou 最低)的場景 前15 ===")
    print(f"{'scene':<16}{'物數':>4}{'mean_iou':>9}{'min_iou':>8}  最差物體")
    for r in rows[:15]:
        print(f"{r['scene']:<16}{r['n_obj']:>4}{r['mean_iou']:>9.3f}{r['min_iou']:>8.3f}  "
              f"{r['worst_obj']}({r['worst_obj_iou']:.2f})")
    print(f"\n圖 → {OUT}/<scene>.png  | 排名 → {OUT}/ranking.csv")


if __name__ == "__main__":
    main()
