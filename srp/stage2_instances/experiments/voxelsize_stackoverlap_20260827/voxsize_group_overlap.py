#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""voxsize_group_overlap.py — voxel 大小 vs 語意群間表面 voxel 重疊率(堆疊物 vs 同物體)。

實驗存證用(2026-08-27)。針對一場景,對 voxel 10/7/5/3/1mm 各自:
  1. 讀 srp_hull_vox{MM}mm/<scene>/hull.npz 的 surface 表面 voxel。
  2. 讀 srp_semdonut_vox{MM}mm/<scene>/instances.json 的 mask_clusters(語意群)。
  3. 每視角遮罩去大遮罩(donut)後 z-buffer 投影 → 「多數決歸屬」把每 voxel 指派給投影像素
     最多的遮罩(去邊界假重疊);各語意群 = 其成員遮罩的 voxel 聯集。
  4. GT 貼標各群(modal cover>0.5),群兩兩算 Jaccard;水平<5cm=堆疊相觸。
  5. 分「同物體群對(該合)」vs「堆疊物群對(該分)」印中位/max Jaccard。

不用深度。只用 SAM 遮罩(donut)+ 位姿(hull)+ GT(僅貼標分類,不進演算法)。
執行:見同目錄 run_voxsize_full.sh(先建 hull+surface+分群,再跑本腳本)。
"""
import sys, os, json, numpy as np
sys.path.insert(0, "srp/io"); sys.path.insert(0, "srp/stage2_instances")
os.environ.setdefault("ARM_MASK_ROOT", os.path.abspath("data/eval/srp_arm_masks"))
import camera as cam, masks as MK, viewpoints as VP, labels as L
import cg_associate as CG
from voxel_sem_cluster_donut import donut_masks
from collections import defaultdict, Counter
from pathlib import Path
from pycocotools import mask as cocomask

SCENES = sys.argv[1:] or ["stack3_scene0001"]
SIZES = [10, 7, 5, 3, 1]
MV2 = Path("data/eval/mobilesamv2_fast"); CAP = Path("data/captures_fast")
# hull/分群 root 已隨實驗一起存進本資料夾的 data/(不在 data/eval);EXP 指向本檔所在目錄。
EXP = Path(__file__).resolve().parent


def gt_info(sc):
    ann = json.loads((L.label_dir(sc) / "actual" / "annotations.json").read_text())
    cat = {c["id"]: c["name"] for c in ann["categories"]}
    id2v = {im["id"]: Path(im["file_name"]).stem for im in ann["images"]}
    modal = defaultdict(dict)
    for a in ann["annotations"]:
        if a["category_id"] == 1:
            continue
        s = a["segmentation"]
        counts = s["counts"].encode() if isinstance(s["counts"], str) else s["counts"]
        modal[id2v[a["image_id"]]][cat[a["category_id"]]] = cocomask.decode(
            {"size": s["size"], "counts": counts}).astype(bool)
    pos = {o["name"]: np.array(o["position_m"][:2]) for o in ann["images"][0]["objects"]}
    return modal, pos


def groups_for(sc, MM, modal):
    HB = EXP / "data" / f"srp_hull_vox{MM}mm"; SEM = EXP / "data" / f"srp_semdonut_vox{MM}mm"
    mc = json.loads((SEM / sc / "instances.json").read_text()).get("mask_clusters", {})
    z = np.load(HB / sc / "hull.npz"); surf = z["surface"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
    vox = np.array(np.nonzero(surf)).T; P = gm + (vox + 0.5) * vs
    group = sc.split("_")[0]; sdir = CAP / f"multi_{group}" / sc
    gvox = defaultdict(set); gobj = defaultdict(Counter)
    for vn in sorted(VP.selected_view_names(12)):
        vd = MV2 / sc / vn; pf = sdir / f"{vn}_pose.json"
        if not (vd.is_dir() and pf.is_file()):
            continue
        km = MK.kept_object_masks(vd); names = [n for _, n in km]; ms0 = [m for m, _ in km]
        if not ms0:
            continue
        ms = donut_masks(ms0); C, Rb = cam.load_pose(pf); H, W = ms0[0].shape
        vox_at = CG.zbuffer_visible(P, C, Rb, W, H, vs).reshape(H, W)
        vmc = defaultdict(Counter)
        for mi, m in enumerate(ms):
            ys, xs = np.where(m); vv = vox_at[ys, xs]
            for v in vv[vv >= 0]:
                vmc[int(v)][mi] += 1
        mvx = defaultdict(set)
        for v, cnt in vmc.items():
            mvx[cnt.most_common(1)[0][0]].add(v)   # 多數決歸屬
        cl = mc.get(vn, {}); gm_v = modal.get(vn, {})
        for mi, nm in enumerate(names):
            cid = cl.get(nm)
            if cid is not None:
                gvox[cid] |= mvx.get(mi, set())
            m = ms[mi]; a = int(m.sum()); best = None; bc = 0.5
            for on, om in gm_v.items():
                cov = (m & om).sum() / max(a, 1)
                if cov > bc:
                    bc = cov; best = on
            if best and cid is not None:
                gobj[cid][best] += 1
    return {c: (gvox[c], gobj[c].most_common(1)[0][0]) for c in gvox if gobj[c] and gvox[c]}


def main():
    print(f"語意群間 表面voxel重疊率(Jaccard)  場景數={len(SCENES)}")
    print(f"{'voxel':>6} {'同物體群對(該合)':>26} {'堆疊物群對(該分)':>26}")
    for MM in SIZES:
        same, stk = [], []
        for sc in SCENES:
            modal, pos = gt_info(sc)
            gd = groups_for(sc, MM, modal); cids = list(gd)
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    (vi, oi), (vj, oj) = gd[cids[i]], gd[cids[j]]
                    jac = len(vi & vj) / max(len(vi | vj), 1)
                    if oi == oj:
                        same.append(jac)
                    else:
                        hd = np.linalg.norm(pos[oi] - pos[oj]) if oi in pos and oj in pos else 9
                        if hd < 0.05:
                            stk.append(jac)
        ss = f"n={len(same)} 中位{np.median(same):.3f} max{np.max(same):.3f}" if same else "n=0"
        ks = f"n={len(stk)} 中位{np.median(stk):.3f} max{np.max(stk):.3f}" if stk else "n=0"
        print(f"{MM:>4}mm {ss:>26} {ks:>26}")


if __name__ == "__main__":
    main()
