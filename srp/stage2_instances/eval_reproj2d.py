#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""eval_reproj2d.py — 殼重投影 vs 「整場景 modal 各物體遮罩」(含遮擋)2D IoU。

與 amodal(單物完整輪廓)不同:這裡比的是 actual/annotations.json 的 **modal 遮罩**——整個場景
一起渲染、帶遮擋(被擋物只剩可見部分)。把 per-instance 殼重投影回各拍攝視角,和對應物體的
modal 遮罩比 2D IoU/recall/precision:
  - 2D recall  = |reproj ∩ modal| / |modal|   殼是否蓋住「可見」部分(應≈1)
  - 2D precision = |reproj ∩ modal| / |reproj| 殼超出可見的比例低=遮擋恢復/過估(遮擋越多越低)
  - 2D IoU     = 兩者交聯比

對應:instance ↔ GT 物體 用 3D 實心 mesh IoU 配對(重用 eval_mesh)。
modal 遮罩視角 = actual annotations 的相機,對到最近拍攝視角(~5mm),用拍攝 pose 重投影。
需 webots_visual_hull(pycocotools/scipy/trimesh)。

用法: ./srp/stage2_instances/eval_reproj2d.py <scenes> --root srp_sweep --tag am0_cvlarge_ag50
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera as cam            # noqa: E402
import eval_mesh as EM          # noqa: E402  (solid_mesh_occ, iou3)

CAPTURES = REPO / "data" / "captures"
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "srp" / "io")); from labels import LABELS  # data/labels 分層(類別/數量/場景)
EVAL = REPO / "data" / "eval"


def load_modal_by_view(scene):
    """回傳 {view_name: {obj_name: modal_mask(bool)}};actual 相機對到最近拍攝視角。"""
    ann = LABELS / scene / "actual" / "annotations.json"
    if not ann.is_file():
        return None, None
    d = json.loads(ann.read_text())
    cat = {c["id"]: c["name"] for c in d["categories"]}
    group = scene.split("_")[0]
    sdir = CAPTURES / f"multi_{group}" / scene
    # 拍攝視角相機位置
    vcam = {}
    for pf in sorted(sdir.glob("view_*_pose.json")):
        C, _ = cam.load_pose(pf)
        vcam[pf.name.split("_pose")[0]] = C
    vnames = list(vcam); vpos = np.array([vcam[v] for v in vnames])
    # 每 image → 最近視角
    img_view = {}
    for im in d["images"]:
        c = np.array(im["camera_pos_m"])
        img_view[im["id"]] = vnames[int(np.argmin(np.linalg.norm(vpos - c, axis=1)))]
    out = {}
    for a in d["annotations"]:
        name = cat[a["category_id"]]
        if name == "ur5e":
            continue
        v = img_view[a["image_id"]]
        m = mask_utils.decode(a["segmentation"]).astype(bool)
        if m.sum() == 0:
            continue
        out.setdefault(v, {})[name] = m
    return out, sdir


def reproject(centers, K, R_w2c, t, H, W, dilate=2):
    X = centers @ R_w2c.T + t
    z = X[:, 2]; ok = z > 1e-9; zz = np.where(ok, z, 1.0)
    u = np.round(K[0, 0] * X[:, 0] / zz + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * X[:, 1] / zz + K[1, 2]).astype(int)
    inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    m = np.zeros((H, W), bool); m[v[inb], u[inb]] = True
    return ndimage.binary_dilation(m, iterations=dilate) if dilate else m


def process(scene, tag, root, rows):
    ip = EVAL / root / scene / f"instances{('_'+tag) if tag else ''}.npz"
    if not ip.is_file():
        return
    z = np.load(ip); labels = z["labels"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
    shape = labels.shape
    k_ids = [k for k in range(1, int(labels.max()) + 1) if (labels == k).any()]
    hulls = [labels == k for k in k_ids]

    # 對應:instance ↔ GT 物名(3D 實心 mesh IoU 配對)
    gt = EM.solid_mesh_occ(scene, gm, vs, shape)
    if not gt:
        return
    names = list(gt); meshes = [gt[n] for n in names]
    M = np.array([[EM.iou3(h, m) for m in meshes] for h in hulls]) if hulls else np.zeros((0, 0))
    inst2name = {}
    if len(hulls) and len(names):
        ri, cj = linear_sum_assignment(-M)
        for i, j in zip(ri, cj):
            if M[i, j] > 0:
                inst2name[i] = names[j]

    modal, sdir = load_modal_by_view(scene)
    if not modal:
        return
    # 預算各 instance 的佔據世界座標
    centers = {}
    for i, h in enumerate(hulls):
        gi, gj, gk = np.nonzero(h)
        centers[i] = gm + (np.stack([gi, gj, gk], 1) + 0.5) * vs

    for i, name in inst2name.items():
        ious, recs, precs, nv = [], [], [], 0
        for v, objs in modal.items():
            if name not in objs:
                continue
            mod = objs[name]; H, W = mod.shape
            C, R_body = cam.load_pose(sdir / f"{v}_pose.json")
            R_w2c, t = cam.pose_to_w2c(C, R_body)
            K = cam.intrinsics(W, H)
            rp = reproject(centers[i], K, R_w2c, t, H, W)
            inter = int(np.logical_and(rp, mod).sum())
            uni = int(np.logical_or(rp, mod).sum())
            if mod.sum() == 0:
                continue
            ious.append(inter / uni if uni else 0)
            recs.append(inter / int(mod.sum()))
            precs.append(inter / int(rp.sum()) if rp.sum() else 0)
            nv += 1
        if nv:
            rows.append({"scene": scene, "object": name, "n_view": nv,
                         "iou2d": round(float(np.mean(ious)), 3),
                         "recall2d": round(float(np.mean(recs)), 3),
                         "prec2d": round(float(np.mean(precs)), 3)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--root", default="srp_hull")
    ap.add_argument("--tag", default="")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    rows = []
    for sc in args.scenes:
        try:
            process(sc, args.tag, args.root, rows)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")
    if not rows:
        print("無資料"); return
    m = lambda k: round(float(np.mean([r[k] for r in rows])), 3)
    print(f"== tag={args.tag or '(baseline)'} | {len(rows)} 物體 | "
          f"2D IoU {m('iou2d')} | recall {m('recall2d')}(殼蓋住可見) | "
          f"prec {m('prec2d')}(1−prec=殼超出可見:遮擋恢復/過估) ==")
    csv_path = args.csv or str(EVAL / args.root / f"reproj2d{('_'+args.tag) if args.tag else ''}.csv")
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"→ {csv_path}")


if __name__ == "__main__":
    main()
