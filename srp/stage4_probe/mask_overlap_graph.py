#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""mask_overlap_graph.py — 驗證:在【遮罩層級】的 voxel 重疊圖上,堆疊相觸的上下物會不會落在不同連通元件?

★ 新檔。這是「語意分群完,群內再依遮罩空間相連切開」這個構想的前置驗證,不動任何既有管線。

為什麼要問:現行流程的 3D 連通跑在【voxel】上(ndimage.label,26鄰接)——相觸物的 voxel 本來就挨著,
必然同一連通塊,機制上救不了(voxel_sem_cluster_connected.py 的 docstring 也自承「相觸/堆疊 hull 融成
一塊→不拆」)。但【遮罩】層級不同:投票時每個 voxel 在每個視角只被【一個】遮罩認領
(voxel_sem_cluster_reassign_soliddrop.py:128-131 的 break),所以不同物體的遮罩認領的是不同區域的 voxel。
→ 遮罩重疊圖有機會把相觸物切開。本腳本量的就是「有沒有機會」。

做什麼(每場):
  1. 重算 12 視角 z-buffer,得 owner[view][voxel] = 該 voxel 在此視角被哪個 kept 遮罩認領(-1=無)
     (與管線同源:kept_object_masks + DROP_ARM;遮罩順序 = sorted(masks/*.png) 經 kept 過濾後的順序)
  2. 每個遮罩 = 一個節點,其 voxel 集合 = {p | owner[view][p]==mask}
  3. 用 GT modal 遮罩替每個遮罩貼「真實物體」標籤(IoU 最大者,需 >= --gt-thr)
  4. 建圖:兩遮罩的 voxel 集合交集 >= --min-ov 就連邊 → 取連通元件
  5. 對每個 on 對 (T,B):T 的遮罩與 B 的遮罩是否落在【不同】元件?(同元件=切不開)
     同時量每個物體的遮罩被切成幾個元件(過切風險)

⚠ 本測試用【純空間重疊圖】,不含語意。若純空間就能把 T/B 分開,則在任何語意群內也分得開
   (子圖的不連通性保持)。物體被切成多塊的數量則是語意需要補回的上界。
排除:GEX(skillet_lid/windex_bottle/colored_wood_blocks/dice)的 on 對不計。
用法: ./mask_overlap_graph.py [scenes|(空=60場stack)] [--min-ov 1] [--gt-thr 0.5]
env : SAM_ROOT(mobilesamv2_fast) HULL_ROOT(srp_hull_mv2_v12_am1) CAPTURES_ROOT ARM_MASK_ROOT
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "srp" / "io")); sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import camera as cam           # noqa: E402
import masks as MK             # noqa: E402
import viewpoints as VP        # noqa: E402
import cg_associate as CG      # noqa: E402
from labels import label_dir   # noqa: E402
from stack_leak_nosep import on_pairs   # noqa: E402
from pycocotools import mask as RLE     # noqa: E402

SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "mobilesamv2_fast")))
HULL_ROOT = Path(os.environ.get("HULL_ROOT", str(REPO / "data" / "eval" / "srp_hull_mv2_v12_am1")))
CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures_fast")))
ARM = Path(os.environ.get("ARM_MASK_ROOT", str(REPO / "data" / "eval" / "srp_arm_masks")))
ARM_DROP_THR = float(os.environ.get("ARM_DROP_THR", "0.5"))
GEX = {"skillet_lid", "windex_bottle", "colored_wood_blocks", "dice"}
W, H = 1280, 720


def gt_modal(sc):
    f = label_dir(sc) / "actual" / "annotations.json"
    if not f.is_file():
        return {}
    d = json.loads(f.read_text())
    id2n = {c["id"]: c["name"].split("_", 1)[-1] for c in d["categories"] if c["name"] != "ur5e"}
    vof = {Path(im["file_name"]).stem: im["id"] for im in d["images"]}
    seg = {(a["image_id"], a["category_id"]): a["segmentation"] for a in d["annotations"]}
    per = {}
    for vn, iid in vof.items():
        for cid, nm in id2n.items():
            if (iid, cid) in seg:
                m = RLE.decode(seg[(iid, cid)]).astype(bool)
                if m.any():
                    per.setdefault(vn, {})[nm] = m
    return per


def scene_graph(sc, min_ov, gt_thr):
    """回 (nodes, comp) ; nodes=[(view,mask_idx,真實物體,voxel集合)], comp=每節點的連通元件 id。"""
    hp = HULL_ROOT / sc / "hull.npz"
    if not hp.is_file():
        return None
    z = np.load(hp)
    surf = z["surface"]; occ = z["occupancy"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
    vox = np.array(np.nonzero(surf)).T; M = len(vox); P = gm + (vox + 0.5) * vs
    ovox = np.array(np.nonzero(occ)).T; oP = gm + (ovox + 0.5) * vs
    sg = np.full(surf.shape, -1, np.int64); sg[tuple(vox.T)] = np.arange(M)
    soo = sg[tuple(ovox.T)]
    per = gt_modal(sc)
    sdir = CAPTURES / f"multi_{sc.split('_')[0]}" / sc
    nodes = []                                   # (vname, mi, obj, set(voxel idx))
    for vn in sorted(VP.selected_view_names(12)):
        vd = SAM_ROOT / sc / vn; pf = sdir / f"{vn}_pose.json"
        if not (vd / "masks").is_dir() or not pf.is_file():
            continue
        km = MK.kept_object_masks(vd); ms = [m for m, _ in km]
        if not ms:
            continue
        ap = ARM / sc / f"{vn}_arm.png"
        arm = (cv2.imread(str(ap), 0) > 127) if ap.is_file() else None
        if arm is not None:                       # 與管線同:手臂遮罩丟掉
            keep = [k for k, m in enumerate(ms)
                    if (m & arm).sum() / max(int(m.sum()), 1) < ARM_DROP_THR]
            ms = [ms[k] for k in keep]
        if not ms:
            continue
        C, Rb = cam.load_pose(pf); Rwc, t = cam.pose_to_w2c(C, Rb); K = cam.intrinsics(W, H)
        va = CG.zbuffer_visible(oP, C, Rb, W, H, vs)
        fl = np.full(len(va), -1, np.int64); mm = va >= 0; fl[mm] = soo[va[mm]]
        vox_at = fl.reshape(H, W)
        X = P @ Rwc.T + t; zc = X[:, 2]; ok = zc > 1e-9; zz = np.where(ok, zc, 1.0)
        u = np.round(K[0, 0] * X[:, 0] / zz + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * X[:, 1] / zz + K[1, 2]).astype(int)
        inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        cand = np.where(inb)[0]
        if not len(cand):
            continue
        uu, vv = u[cand], v[cand]
        vis = vox_at[vv, uu] == cand              # 中心像素最前的是自己
        if arm is not None:
            vis &= ~arm[vv, uu]
        cand = cand[vis]; uu, vv = u[cand], v[cand]
        rem = np.ones(len(cand), bool)
        gtv = per.get(vn) or {}
        for mi, m in enumerate(ms):
            hit = rem & m[vv, uu]                 # 與管線同:第一個含該像素的遮罩認領(順序 break)
            rem &= ~hit
            pset = set(cand[hit].tolist())
            if not pset:
                continue
            obj, best = None, 0.0                 # 用 GT modal 遮罩貼標籤
            for nm, g in gtv.items():
                i = float((m & g).sum()) / max(float((m | g).sum()), 1.0)
                if i > best:
                    obj, best = nm, i
            nodes.append((vn, mi, obj if best >= gt_thr else None, pset))
    n = len(nodes)
    par = list(range(n))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if len(nodes[i][3] & nodes[j][3]) >= min_ov:
                a, b = find(i), find(j)
                if a != b:
                    par[a] = b
    comp = [find(i) for i in range(n)]
    return nodes, comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--min-ov", type=int, default=1, dest="min_ov")
    ap.add_argument("--gt-thr", type=float, default=0.5, dest="gt_thr")
    a = ap.parse_args()
    scenes = a.scenes or sorted(Path(p).parent.name for p in
                                glob.glob(str(HULL_ROOT / "stack*_scene*/hull.npz")))
    rows = []; sep_ok = 0; sep_no = 0; ncomp_all = []
    for sc in scenes:
        r = scene_graph(sc, a.min_ov, a.gt_thr)
        if r is None:
            continue
        nodes, comp = r
        by = defaultdict(list)
        for i, (vn, mi, obj, ps) in enumerate(nodes):
            if obj:
                by[obj].append(i)
        for o, idx in by.items():
            if o not in GEX:
                ncomp_all.append(len({comp[i] for i in idx}))
        for (T, B) in on_pairs(sc):
            if T in GEX or B in GEX or T not in by or B not in by:
                continue
            cT = {comp[i] for i in by[T]}; cB = {comp[i] for i in by[B]}
            same = bool(cT & cB)
            sep_no += same; sep_ok += (not same)
            rows.append((sc, T, B, len(by[T]), len(by[B]), len(cT), len(cB),
                         "同元件(切不開)" if same else "不同元件(可切開)"))
    nc = np.array(ncomp_all) if ncomp_all else np.array([0])
    md = [f"# 遮罩層級 voxel 重疊圖:相觸堆疊物能不能被切開(min_ov={a.min_ov}, gt_thr={a.gt_thr})\n",
          "- 建檔 2026-09-25;程式 `srp/stage4_probe/mask_overlap_graph.py`;可復現。",
          f"- SAM=`{SAM_ROOT.name}`(kept_object_masks + DROP_ARM,與管線同源);hull=`{HULL_ROOT.name}`;12 視角 A-3。",
          "- 節點=遮罩,邊=兩遮罩認領的 voxel 集合交集 ≥ min_ov;取連通元件。**純空間,不含語意**。",
          f"- 母體:{len(scenes)} 場、**{len(rows)} 個 on 對**(排 GEX);物體標籤由 GT modal 遮罩 IoU≥{a.gt_thr} 決定。\n",
          "## 核心結果:on 對能不能被切開\n",
          "| 判定 | 對數 | 佔比 |", "|---|---|---|",
          f"| **不同元件(可切開)** | {sep_ok} | {sep_ok/max(sep_ok+sep_no,1)*100:.1f}% |",
          f"| 同元件(切不開) | {sep_no} | {sep_no/max(sep_ok+sep_no,1)*100:.1f}% |", "",
          "## 過切風險:同一物體的遮罩被切成幾個元件\n",
          "| 統計 | 值 |", "|---|---|",
          f"| 物體數(非GEX) | {len(nc)} |",
          f"| 平均元件數 | {nc.mean():.2f} |",
          f"| 中位 | {int(np.median(nc))} |",
          f"| =1(完整一團) | {int((nc==1).sum())} ({(nc==1).mean()*100:.1f}%) |",
          f"| ≥3 | {int((nc>=3).sum())} ({(nc>=3).mean()*100:.1f}%) |",
          f"| 最大 | {int(nc.max())} |", "",
          "## 逐對明細\n",
          "| 場景 | 上物 T | 下物 B | T遮罩數 | B遮罩數 | T元件數 | B元件數 | 判定 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append("| " + " | ".join(str(x) for x in r) + " |")
    out = HERE / f"RESULT_mask_overlap_graph_ov{a.min_ov}.md"
    out.write_text("\n".join(md), encoding="utf-8")
    with open(HERE / f"mask_overlap_graph_ov{a.min_ov}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene", "upper", "lower", "nT", "nB", "compT", "compB", "verdict"])
        for r in rows:
            w.writerow(list(r))
    print(f"[存檔] {out}\n\n" + "\n".join(md[:30]))


if __name__ == "__main__":
    main()
