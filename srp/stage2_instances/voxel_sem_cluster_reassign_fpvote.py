#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""voxel_sem_cluster_reassign_fpvote.py — reassign 版的「整顆 voxel 投票」變體(複製自 voxel_sem_cluster_reassign.py)。

★ 與 reassign 唯一差別:投票不再用 voxel 中心那 1 像素查遮罩,改用 footprint zbuffer(vox_at)——
  每個被某 voxel 認領(最前)的像素,都讓那顆 voxel 依該像素的遮罩投票。目的:救回「中心取整落遮罩外、
  但 footprint 有碰到遮罩」的邊界 voxel,減少無標籤/散點。其餘(表面 voxel、可見性、分群、reNN、div 下游)全同。
  不改原檔。用法/env 與 reassign 相同。

--- 原 docstring ---
= voxel_sem_cluster_surf.py(表面 voxel + z-buffer 投票) × donut(去大遮罩) 組合版。

= voxel_sem_cluster_surf.py(表面 voxel + z-buffer 投票) + 去除大遮罩(挖洞/甜甜圈)+ 重算 CLIP。
挖洞邏輯與特徵快取(clip_donut_feats.npy)沿用 voxel_sem_cluster_donut(donut 全跑已存 367 場,直接讀不重算)。
其餘(表面 voxel、z-buffer、分群、投票 argmax、3D 連通、來源遮罩)全同 surf。
可視化: SRP_VIZ_ARGS="<scene> 1 srp_hull_semcluster_surf_donut cubes" webots worlds/hull_viz.wbt
用法: ./voxel_sem_cluster_surf_donut.py [scene|group|(空=全部)] [--n-views 12] [--sem-thr 0.4]
env: SAM_ROOT HULL_ROOT CAPTURES_ROOT ARM_MASK_ROOT OUT_ROOT NEST_THR(0.8)
"""
import argparse, os, sys, json, glob, datetime as _dt
_here = __import__("pathlib").Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_here / "srp" / "io"))
sys.path.insert(0, str(_here / "srp" / "stage2_instances"))
import numpy as np, cv2
from pathlib import Path
from collections import defaultdict
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
import camera as cam, masks as MK, mask_clip_cluster as MC, viewpoints as VP
import cg_associate as CG   # 免深度 z-buffer
from voxel_sem_cluster_donut import donut_masks, donut_feats   # 去大遮罩挖洞 + 甜甜圈特徵快取

REPO = Path(__file__).resolve().parents[2]
CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures_fast")))
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "sam_only_fast")))
HULL_ROOT = Path(os.environ.get("HULL_ROOT", str(REPO / "data" / "eval" / "srp_hull_v12")))
ARM = Path(os.environ.get("ARM_MASK_ROOT", str(REPO / "data" / "eval" / "srp_arm_masks")))
OUT_ROOT = REPO / "data" / "eval" / os.environ.get("OUT_ROOT", "srp_hull_semcluster_surf_donut")
MIN_VOX = int(os.environ.get("MIN_VOX", "50"))
NEST_THR = float(os.environ.get("NEST_THR", "0.8"))
DROP_ARM = os.environ.get("DROP_ARM", "1") == "1"          # ★分群前去掉手臂+夾爪遮罩(用 srp_arm_masks,含夾爪)
ARM_DROP_THR = float(os.environ.get("ARM_DROP_THR", "0.5"))  # 遮罩 ≥此比例落在手臂剪影內 → 視為手臂/夾爪遮罩,丟
DEBIAS = os.environ.get("DEBIAS", "1") == "1"
_BG = MC.F_BG.astype(np.float64) if DEBIAS else None


def debias_feats(F):
    F = F.astype(np.float64)
    if DEBIAS and _BG is not None:
        F = F - (F @ _BG)[:, None] * _BG[None, :]
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


def semantic_cluster(sc, n_views, sem_thr):
    group = sc.split("_")[0]; sdir = CAPTURES / f"multi_{group}" / sc
    z = np.load(HULL_ROOT / sc / "hull.npz")
    occ = z["surface"]; gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = occ.shape   # ★表面 voxel
    vox = np.array(np.nonzero(occ)).T; P = gm + (vox + 0.5) * vs; M = len(vox)
    want = set(VP.selected_view_names(n_views)) if n_views else None
    vdata = []; allf = []; ref = []
    for vd in sorted((SAM_ROOT / sc).glob("view_*")):
        if want is not None and vd.name not in want: continue
        pf = sdir / f"{vd.name}_pose.json"
        if not pf.is_file(): continue
        km = MK.kept_object_masks(vd); ms0 = [m for m, _ in km]; names = [nm for _, nm in km]
        if not ms0: continue
        ap = ARM / sc / f"{vd.name}_arm.png"
        arm = (cv2.imread(str(ap), 0) > 127) if ap.is_file() else None
        if DROP_ARM and arm is not None:         # ★分群前:落在手臂+夾爪剪影 ≥ARM_DROP_THR 的遮罩,丟(不進特徵/分群/投票)
            keep = [k for k, m in enumerate(ms0)
                    if (m & arm).sum() / max(int(m.sum()), 1) < ARM_DROP_THR]
            ms0 = [ms0[k] for k in keep]; names = [names[k] for k in keep]
            if not ms0: continue
        ms = donut_masks(ms0, thr=NEST_THR)      # ★去大遮罩:含子遮罩的父遮罩挖成甜甜圈
        C, Rb = cam.load_pose(pf); Rwc, t = cam.pose_to_w2c(C, Rb)
        K = cam.intrinsics(ms[0].shape[1], ms[0].shape[0])
        rgb = cv2.cvtColor(cv2.imread(str(sdir / f"{vd.name}.png")), cv2.COLOR_BGR2RGB)
        vi = len(vdata)
        fmap = donut_feats(vd, rgb, ms, names)   # ★甜甜圈 CLIP(讀 clip_donut_feats.npy 快取,donut 已存)
        for mi, nm in enumerate(names):
            f = fmap.get(nm)
            if f is not None: allf.append(f); ref.append((vi, mi))
        vdata.append((ms, C, Rwc, t, K, arm, names, vd.name, Rb))
    labels = np.zeros(shape, np.int32)
    if len(allf) < 2:
        return labels, gm, vs, {}, {}
    F = debias_feats(np.array(allf))
    cl = fcluster(linkage(pdist(F, "cosine"), "average"), t=sem_thr, criterion="distance")
    mlabel = {r: int(c) for r, c in zip(ref, cl)}
    mask_cluster = defaultdict(dict)
    for (vi, mi), c in mlabel.items():
        mask_cluster[vdata[vi][7]][vdata[vi][6][mi]] = int(c)
    if os.environ.get("OBS_CAND", "0") == "1":     # ★候選只留「≥1視角 zbuffer 最前(可觀測)」,排除從來不可見(遮擋)A
        zf = np.zeros(M, int)
        for (ms_, C_, Rwc_, t_, K_, arm_, names_, vname_, Rb_) in vdata:
            H_, W_ = ms_[0].shape
            va = CG.zbuffer_visible(P, C_, Rb_, W_, H_, vs).reshape(H_, W_)
            X_ = P @ Rwc_.T + t_; zc_ = X_[:, 2]; ok_ = zc_ > 1e-9; zz_ = np.where(ok_, zc_, 1.0)
            u_ = np.round(K_[0, 0] * X_[:, 0] / zz_ + K_[0, 2]).astype(int)
            v_ = np.round(K_[1, 1] * X_[:, 1] / zz_ + K_[1, 2]).astype(int)
            inb_ = ok_ & (u_ >= 0) & (u_ < W_) & (v_ >= 0) & (v_ < H_); ii_ = np.where(inb_)[0]
            zf[ii_[va[v_[ii_], u_[ii_]] == ii_]] += 1
        obs = zf > 0
        vox = vox[obs]; P = P[obs]; M = len(vox)
    votes = defaultdict(lambda: np.zeros(M))
    vmask = defaultdict(lambda: defaultdict(set))
    for vi, (ms, C, Rwc, t, K, arm, names, vname, Rb) in enumerate(vdata):
        H, W = ms[0].shape
        vox_at = CG.zbuffer_visible(P, C, Rb, W, H, vs).reshape(H, W)   # ★z-buffer
        X = P @ Rwc.T + t; zc = X[:, 2]; ok = zc > 1e-9; zz = np.where(ok, zc, 1.0)
        u = np.round(K[0, 0] * X[:, 0] / zz + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * X[:, 1] / zz + K[1, 2]).astype(int)
        inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        wt = 1.0 / np.maximum(np.linalg.norm(P - C, axis=1), 1e-3)
        # ★整顆 voxel 投票:不再只看中心像素,改用 footprint zbuffer(vox_at)——
        #   每個被某 voxel 認領(最前)的像素,都讓那顆 voxel 依該像素的遮罩投一票。
        #   救回「中心取整落遮罩外、但 footprint 有碰到遮罩」的邊界 voxel(減少無標籤/散點)。
        ys, xs = np.where(vox_at >= 0)
        pown = vox_at[ys, xs]
        for y, x, p in zip(ys.tolist(), xs.tolist(), pown.tolist()):
            if arm is not None and arm[y, x]: continue
            for mi, m in enumerate(ms):      # ms=挖洞後遮罩
                if (vi, mi) in mlabel and m[y, x]:
                    votes[mlabel[(vi, mi)]][p] += wt[p]
                    vmask[int(p)][vname].add(names[mi]); break
    gl = list(votes.keys()); inst_masks = {}
    if gl:
        Vt = np.stack([votes[g] for g in gl], 1); assign = Vt.argmax(1); has = Vt.max(1) > 0
        st = ndimage.generate_binary_structure(3, 3)
        big = []; small = []                                       # (comp_p, gid)
        for gi in range(len(gl)):
            sel = has & (assign == gi)
            if not sel.any(): continue
            sel_p = np.where(sel)[0]
            m3 = np.zeros(shape, bool); m3[tuple(vox[sel_p].T)] = True
            lab, n = ndimage.label(m3, st)
            labs_at = lab[tuple(vox[sel_p].T)]
            for c in range(1, n + 1):
                comp_p = sel_p[labs_at == c]
                (big if len(comp_p) >= MIN_VOX else small).append((comp_p, gl[gi]))

        def imask(comp_p, gid):                                    # 該群來源遮罩
            md = defaultdict(set)
            for p in comp_p:
                for vw, fs in vmask.get(int(p), {}).items():
                    for fn in fs:
                        if mask_cluster.get(vw, {}).get(fn) == gid:
                            md[vw].add(fn)
            return {vw: sorted(fs) for vw, fs in sorted(md.items())}

        nid = 0
        for comp_p, gid in big:                                    # 大塊各自成實例
            nid += 1; labels[tuple(vox[comp_p].T)] = nid; inst_masks[nid] = imask(comp_p, gid)
        # ★ 小塊(<MIN_VOX)不丟:併進「3D 最近的大實例」(佔據可達,撐高該物 div);無大實例才自成一群
        if small:
            if (labels > 0).any():
                idx = ndimage.distance_transform_edt(labels == 0, return_distances=False, return_indices=True)
                nearest = labels[tuple(idx)]                        # 每 voxel 的最近大實例 label
                for comp_p, gid in small:
                    labels[tuple(vox[comp_p].T)] = nearest[tuple(vox[comp_p].T)]
            else:
                for comp_p, gid in small:
                    nid += 1; labels[tuple(vox[comp_p].T)] = nid; inst_masks[nid] = imask(comp_p, gid)
    return labels, gm, vs, inst_masks, {v: dict(d) for v, d in mask_cluster.items()}


def process(sc, n_views, sem_thr):
    if not (HULL_ROOT / sc / "hull.npz").is_file():
        print(f"[skip] {sc}"); return
    if os.environ.get("FORCE", "") != "1" and (OUT_ROOT / sc / "instances.npz").is_file():
        return
    labels, gm, vs, inst_masks, mask_cluster = semantic_cluster(sc, n_views, sem_thr)
    out = OUT_ROOT / sc; out.mkdir(parents=True, exist_ok=True)
    meta = {"script": "voxel_sem_cluster_reassign_fpvote.py", "vote": "footprint", "surface": True, "zbuffer": True, "donut": True,
            "nest_thr": NEST_THR, "feat": "recomputed_clip_donut",
            "built": _dt.datetime.now().isoformat(timespec="seconds"),
            "hull_root": HULL_ROOT.name, "sam_root": SAM_ROOT.name, "captures_root": CAPTURES.name,
            "arm_root": ARM.name, "debias": DEBIAS, "sem_thr": sem_thr, "n_views": n_views, "min_vox": MIN_VOX,
            "drop_arm": DROP_ARM, "arm_drop_thr": ARM_DROP_THR}
    np.savez_compressed(out / "instances.npz", labels=labels, grid_min=gm, voxel_size=vs,
                        build_meta=json.dumps(meta, ensure_ascii=False))
    insts = [{"instance": i, "n_vox": int((labels == i).sum()), "masks": inst_masks.get(i, {})}
             for i in range(1, int(labels.max()) + 1) if (labels == i).any()]
    (out / "instances.json").write_text(json.dumps(
        {"scene": sc, "voxel": vs, "n_instances": len(insts), "meta": meta,
         "mask_clusters": mask_cluster, "instances": insts},
        indent=2, ensure_ascii=False))
    print(f"[{sc}] surf+去大遮罩 → {len(insts)} instance", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--n-views", type=int, default=12, dest="n_views")
    ap.add_argument("--sem-thr", type=float, default=0.4, dest="sem_thr")
    args = ap.parse_args()
    if not args.targets:
        scenes = sorted(Path(p).parent.name for p in glob.glob(str(HULL_ROOT / "*_scene*/hull.npz")))
    else:
        scenes = []
        for a in args.targets:
            if "scene" in a: scenes.append(a)
            else: scenes += [Path(p).parent.name for p in glob.glob(str(HULL_ROOT / f"{a}_scene*/hull.npz"))]
        scenes = sorted(set(scenes))
    for sc in scenes:
        try: process(sc, args.n_views, args.sem_thr)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")


if __name__ == "__main__":
    main()
