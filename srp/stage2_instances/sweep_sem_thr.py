#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""sweep_sem_thr.py — 一次掃多個 sem_thr,重用所有「與門檻無關」的計算。

★ 另開新檔,完全不動 voxel_sem_cluster_reassign_soliddrop.py。

動機(實測 profile,單場 n5_scene0025 共 9.6s):
  zbuffer_visible 2.71s + argsort 1.81s + 形態學 1.99s + 讀遮罩/imread 1.24s + import 2.02s
  = 超過 8s 與門檻【完全無關】;真正受門檻影響的 pdist/linkage/fcluster 連 profile 前 14 名都排不進(<0.1s)。
  照原腳本每個門檻重跑 = 把那 8s 重做 N 次。

關鍵觀察(讀 voxel_sem_cluster_reassign_soliddrop.py:129-134):
  中心投票時,voxel p 在視角 vi 投給「第一個含有該像素、且有特徵的遮罩」——
  這個【遮罩歸屬 owner[vi][p]】只取決於 z-buffer 可見性與遮罩幾何,【與門檻無關】。
  門檻只決定「遮罩 -> 群」的對應(mlabel)。故可:
    每場算一次: owner[vi][p]、wt[vi][p]、特徵 F
    每個門檻只做: fcluster -> 依新 mlabel 把 per-mask 票重新加總 -> 3D 連通 -> 寫 root

一致性檢查(--verify):thr=0.4 產出的 labels 必須與既有 root 逐 voxel 完全相同,否則掃描結果不可信。

參數與 baseline 對齊:VOTE=center、DONUT=1、NEST_THR=0.8、MIN_VOX=50、DROP_ARM=1、ARM_DROP_THR=0.5、n_views=12。
env: FEAT(clip|dino) DEBIAS SAM_ROOT HULL_ROOT CAPTURES_ROOT ARM_MASK_ROOT OUT_PREFIX THRS
用法: ./sweep_sem_thr.py [scenes...] [--verify-root <既有root> --verify-thr 0.4]
輸出 root: <OUT_PREFIX>_t{20,25,30,35,40}  (OUT_PREFIX 預設 srp_hull_semcluster_reNNcSd_am1_dino_sw)
"""
import argparse
import os
import sys
import json
import glob
import datetime as _dt
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "srp" / "io")); sys.path.insert(0, str(HERE))
import camera as cam, masks as MK, mask_clip_cluster as MC, viewpoints as VP   # noqa: E402
import cg_associate as CG                                                       # noqa: E402
from voxel_sem_cluster_donut import donut_masks, donut_feats                     # noqa: E402
import dino_mask_feats as DF                                                     # noqa: E402

CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures_fast")))
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "mobilesamv2_fast")))
HULL_ROOT = Path(os.environ.get("HULL_ROOT", str(REPO / "data" / "eval" / "srp_hull_mv2_v12_am1")))
ARM = Path(os.environ.get("ARM_MASK_ROOT", str(REPO / "data" / "eval" / "srp_arm_masks")))
EVAL = REPO / "data" / "eval"
OUT_PREFIX = os.environ.get("OUT_PREFIX", "srp_hull_semcluster_reNNcSd_am1_dino_sw")
THRS = [float(x) for x in os.environ.get("THRS", "0.20,0.25,0.30,0.35,0.40").split(",")]
FEAT = os.environ.get("FEAT", "dino")
DEBIAS = os.environ.get("DEBIAS", "0") == "1"
DONUT = os.environ.get("DONUT", "1") == "1"
NEST_THR = float(os.environ.get("NEST_THR", "0.8"))
MIN_VOX = int(os.environ.get("MIN_VOX", "50"))
DROP_ARM = os.environ.get("DROP_ARM", "1") == "1"
ARM_DROP_THR = float(os.environ.get("ARM_DROP_THR", "0.5"))
_BG = MC.F_BG.astype(np.float64) if DEBIAS else None


def debias_feats(F):
    """與 voxel_sem_cluster_reassign_soliddrop.debias_feats 完全相同。"""
    F = F.astype(np.float64)
    if DEBIAS and _BG is not None:
        F = F - (F @ _BG)[:, None] * _BG[None, :]
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


def precompute(sc, n_views=12):
    """算完所有【與門檻無關】的東西:特徵 F、每視角每 voxel 的遮罩歸屬 owner 與票重 wt。"""
    group = sc.split("_")[0]; sdir = CAPTURES / f"multi_{group}" / sc
    z = np.load(HULL_ROOT / sc / "hull.npz")
    occ = z["surface"]; gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = occ.shape
    vox = np.array(np.nonzero(occ)).T; P = gm + (vox + 0.5) * vs; M = len(vox)
    SOLID = z["occupancy"]; ovox = np.array(np.nonzero(SOLID)).T; oP = gm + (ovox + 0.5) * vs
    surf_grid = np.full(shape, -1, np.int64); surf_grid[tuple(vox.T)] = np.arange(M)
    surf_of_occ = surf_grid[tuple(ovox.T)]

    def zbuf_surf(C_, Rb_, W_, H_):
        va = CG.zbuffer_visible(oP, C_, Rb_, W_, H_, vs)
        out = np.full(len(va), -1, np.int64); m = va >= 0; out[m] = surf_of_occ[va[m]]
        return out

    want = set(VP.selected_view_names(n_views)) if n_views else None
    views = []; allf = []; ref = []
    for vd in sorted((SAM_ROOT / sc).glob("view_*")):
        if want is not None and vd.name not in want:
            continue
        pf = sdir / f"{vd.name}_pose.json"
        if not pf.is_file():
            continue
        km = MK.kept_object_masks(vd); ms0 = [m for m, _ in km]; names = [nm for _, nm in km]
        if not ms0:
            continue
        ap = ARM / sc / f"{vd.name}_arm.png"
        arm = (cv2.imread(str(ap), 0) > 127) if ap.is_file() else None
        if DROP_ARM and arm is not None:
            keep = [k for k, m in enumerate(ms0)
                    if (m & arm).sum() / max(int(m.sum()), 1) < ARM_DROP_THR]
            ms0 = [ms0[k] for k in keep]; names = [names[k] for k in keep]
            if not ms0:
                continue
        C, Rb = cam.load_pose(pf); Rwc, t = cam.pose_to_w2c(C, Rb)
        K = cam.intrinsics(ms0[0].shape[1], ms0[0].shape[0])
        rgb = cv2.cvtColor(cv2.imread(str(sdir / f"{vd.name}.png")), cv2.COLOR_BGR2RGB)
        vi = len(views)
        if DONUT:
            ms = donut_masks(ms0, thr=NEST_THR)
            fmap = (donut_feats(vd, rgb, ms, names) if FEAT == "clip"
                    else DF.dino_donut_feats(vd, rgb, ms, names))
        else:
            ms = ms0
            fmap = (MK.mask_feats(vd) if FEAT == "clip"
                    else DF.dino_raw_feats(vd, rgb, ms, names))
        hasfeat = np.zeros(len(names), bool)
        for mi, nm in enumerate(names):
            f = fmap.get(nm)
            if f is not None:
                allf.append(f); ref.append((vi, mi)); hasfeat[mi] = True
        # ---- 與門檻無關:算 owner[p] = 該 voxel 在此視角投給哪個遮罩(-1=不投)
        H, W = ms[0].shape
        vox_at = zbuf_surf(C, Rb, W, H).reshape(H, W)
        X = P @ Rwc.T + t; zc = X[:, 2]; ok = zc > 1e-9; zz = np.where(ok, zc, 1.0)
        u = np.round(K[0, 0] * X[:, 0] / zz + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * X[:, 1] / zz + K[1, 2]).astype(int)
        inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        wt = 1.0 / np.maximum(np.linalg.norm(P - C, axis=1), 1e-3)
        owner = np.full(M, -1, np.int32)
        cand = np.where(inb)[0]
        if len(cand):
            uu, vv = u[cand], v[cand]
            okvis = vox_at[vv, uu] == cand                       # 中心像素最前的是自己
            if arm is not None:
                okvis &= ~arm[vv, uu]
            cand = cand[okvis]; uu, vv = u[cand], v[cand]
            # 第一個「含該像素且有特徵」的遮罩(與原程式 for mi ... break 同義)
            rem = np.ones(len(cand), bool)
            for mi in range(len(ms)):
                if not hasfeat[mi] or not rem.any():
                    continue
                hit = rem & ms[mi][vv, uu]
                owner[cand[hit]] = mi
                rem &= ~hit
        views.append(dict(vi=vi, vname=vd.name, names=names, owner=owner, wt=wt))
    return dict(sc=sc, shape=shape, gm=gm, vs=vs, vox=vox, M=M, views=views,
                F=(debias_feats(np.array(allf)) if len(allf) >= 2 else None), ref=ref)


def labels_for(pre, thr):
    """只做與門檻有關的部分:分群 -> 重新加總票 -> argmax -> 3D 連通 -> labels。"""
    shape, M, vox, views = pre["shape"], pre["M"], pre["vox"], pre["views"]
    labels = np.zeros(shape, np.int32)
    if pre["F"] is None:
        return labels, {}, {}
    cl = fcluster(linkage(pdist(pre["F"], "cosine"), "average"), t=thr, criterion="distance")
    mlabel = {r: int(c) for r, c in zip(pre["ref"], cl)}
    mask_cluster = defaultdict(dict)
    for (vi, mi), c in mlabel.items():
        mask_cluster[views[vi]["vname"]][views[vi]["names"][mi]] = int(c)
    # 每視角:遮罩index -> 群id(無特徵者 = -1)
    m2g = []
    for V in views:
        a = np.full(len(V["names"]), -1, np.int64)
        for mi in range(len(V["names"])):
            if (V["vi"], mi) in mlabel:
                a[mi] = mlabel[(V["vi"], mi)]
        m2g.append(a)
    # 群 id 的插入順序(復現原程式 votes defaultdict 的 key 順序:視角外層、voxel 由小到大)
    gl = []; seen = set()
    for V, a in zip(views, m2g):
        sel = np.where(V["owner"] >= 0)[0]
        if not len(sel):
            continue
        for g in a[V["owner"][sel]]:
            gi = int(g)
            if gi not in seen:
                seen.add(gi); gl.append(gi)
    if not gl:
        return labels, {}, {v: dict(d) for v, d in mask_cluster.items()}
    gpos = {g: i for i, g in enumerate(gl)}
    Vt = np.zeros((M, len(gl)))
    for V, a in zip(views, m2g):
        sel = np.where(V["owner"] >= 0)[0]
        if not len(sel):
            continue
        cols = np.array([gpos[int(g)] for g in a[V["owner"][sel]]], np.int64)
        np.add.at(Vt, (sel, cols), V["wt"][sel])
    assign = Vt.argmax(1); has = Vt.max(1) > 0
    st = ndimage.generate_binary_structure(3, 3)
    big, small = [], []
    for gi in range(len(gl)):
        sel = has & (assign == gi)
        if not sel.any():
            continue
        sel_p = np.where(sel)[0]
        m3 = np.zeros(shape, bool); m3[tuple(vox[sel_p].T)] = True
        lab, n = ndimage.label(m3, st)
        labs_at = lab[tuple(vox[sel_p].T)]
        for c in range(1, n + 1):
            comp_p = sel_p[labs_at == c]
            (big if len(comp_p) >= MIN_VOX else small).append((comp_p, gl[gi]))

    def imask(comp_p, gid):
        md = defaultdict(set)
        cs = set(comp_p.tolist())
        for V, a in zip(views, m2g):
            own = V["owner"]
            for p in comp_p:
                mi = own[p]
                if mi >= 0 and a[mi] == gid:
                    md[V["vname"]].add(V["names"][mi])
        return {vw: sorted(fs) for vw, fs in sorted(md.items())}

    inst_masks = {}; nid = 0
    for comp_p, gid in big:
        nid += 1; labels[tuple(vox[comp_p].T)] = nid; inst_masks[nid] = imask(comp_p, gid)
    big_snapshot = labels.copy()
    for comp_p, gid in small:
        cm = np.zeros(shape, bool); cm[tuple(vox[comp_p].T)] = True
        dil = ndimage.binary_dilation(cm, st)
        neigh = big_snapshot[dil & (big_snapshot > 0)]
        if neigh.size > 0:
            u_, c_ = np.unique(neigh, return_counts=True)
            labels[tuple(vox[comp_p].T)] = int(u_[c_.argmax()])
    return labels, inst_masks, {v: dict(d) for v, d in mask_cluster.items()}


def write_root(pre, thr, labels, inst_masks, mask_cluster, root):
    out = EVAL / root / pre["sc"]; out.mkdir(parents=True, exist_ok=True)
    meta = {"script": "sweep_sem_thr.py", "vote": "center", "occluder": "solid",
            "reassign": "connected_or_drop", "surface": True, "zbuffer": True, "donut": DONUT,
            "nest_thr": NEST_THR, "feat_src": FEAT, "debias": DEBIAS, "sem_thr": thr,
            "n_views": 12, "min_vox": MIN_VOX, "drop_arm": DROP_ARM, "arm_drop_thr": ARM_DROP_THR,
            "hull_root": HULL_ROOT.name, "sam_root": SAM_ROOT.name, "captures_root": CAPTURES.name,
            "arm_root": ARM.name, "built": _dt.datetime.now().isoformat(timespec="seconds")}
    np.savez_compressed(out / "instances.npz", labels=labels, grid_min=pre["gm"],
                        voxel_size=pre["vs"], build_meta=json.dumps(meta, ensure_ascii=False))
    insts = [{"instance": i, "n_vox": int((labels == i).sum()), "masks": inst_masks.get(i, {})}
             for i in range(1, int(labels.max()) + 1) if (labels == i).any()]
    (out / "instances.json").write_text(json.dumps(
        {"scene": pre["sc"], "voxel": pre["vs"], "n_instances": len(insts), "meta": meta,
         "mask_clusters": mask_cluster, "instances": insts}, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--verify-root", default=None, help="與此既有 root 逐 voxel 比對(驗證重現)")
    ap.add_argument("--verify-thr", type=float, default=0.40)
    a = ap.parse_args()
    scenes = []
    for x in (a.scenes or []):
        if "scene" in x:
            scenes.append(x)
        else:
            scenes += [Path(p).parent.name for p in glob.glob(str(HULL_ROOT / f"{x}_scene*/hull.npz"))]
    if not scenes:
        scenes = sorted(Path(p).parent.name for p in glob.glob(str(HULL_ROOT / "*_scene*/hull.npz")))
        scenes = [s for s in scenes if not s.startswith("n1_")]
    scenes = sorted(set(scenes))
    tags = {t: f"{OUT_PREFIX}_t{int(round(t*100)):02d}" for t in THRS}
    print(f"場景={len(scenes)}  FEAT={FEAT} DEBIAS={DEBIAS} DONUT={DONUT}  門檻={THRS}", flush=True)
    vmax, vcnt, vdiff = 0, 0, 0
    for i, sc in enumerate(scenes, 1):
        if not (HULL_ROOT / sc / "hull.npz").is_file():
            continue
        pre = precompute(sc)
        for t in THRS:
            lb, im, mc = labels_for(pre, t)
            write_root(pre, t, lb, im, mc, tags[t])
            if a.verify_root and abs(t - a.verify_thr) < 1e-9:
                p = EVAL / a.verify_root / sc / "instances.npz"
                if p.is_file():
                    ref = np.load(p)["labels"]; vcnt += 1
                    d = int((ref != lb).sum())
                    vdiff += (d > 0); vmax = max(vmax, d)
        if i % 20 == 0 or i == len(scenes):
            print(f"  {i}/{len(scenes)}", flush=True)
    if a.verify_root:
        print(f"[verify] 與 {a.verify_root} 在 thr={a.verify_thr} 比對 {vcnt} 場;"
              f"有差異的場數={vdiff};單場最大不同 voxel 數={vmax}  "
              f"({'完全重現' if vdiff == 0 else '★不一致,掃描結果不可信'})")


if __name__ == "__main__":
    main()
