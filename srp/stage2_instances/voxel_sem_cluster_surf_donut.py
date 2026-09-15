#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""voxel_sem_cluster_surf_donut.py — surf(表面 voxel+z-buffer) × donut(去大遮罩) 組合版。

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
        for p in np.where(inb)[0]:
            if arm is not None and arm[v[p], u[p]]: continue
            if vox_at[v[p], u[p]] != p: continue                        # ★z-buffer:只讓該視角看得到的表面 voxel 投票
            for mi, m in enumerate(ms):      # ms=挖洞後遮罩
                if (vi, mi) in mlabel and m[v[p], u[p]]:
                    votes[mlabel[(vi, mi)]][p] += wt[p]
                    vmask[int(p)][vname].add(names[mi]); break
    gl = list(votes.keys()); inst_masks = {}
    if gl:
        Vt = np.stack([votes[g] for g in gl], 1); assign = Vt.argmax(1); has = Vt.max(1) > 0
        st = ndimage.generate_binary_structure(3, 3); nid = 0
        for gi in range(len(gl)):
            sel = has & (assign == gi)
            if not sel.any(): continue
            sel_p = np.where(sel)[0]
            m3 = np.zeros(shape, bool); m3[tuple(vox[sel_p].T)] = True
            lab, n = ndimage.label(m3, st)
            labs_at = lab[tuple(vox[sel_p].T)]
            for c in range(1, n + 1):
                comp_p = sel_p[labs_at == c]
                if len(comp_p) < MIN_VOX: continue
                nid += 1; labels[lab == c] = nid
                gid = gl[gi]
                md = defaultdict(set)
                for p in comp_p:
                    for vw, fs in vmask.get(int(p), {}).items():
                        for fn in fs:
                            if mask_cluster.get(vw, {}).get(fn) == gid:
                                md[vw].add(fn)
                inst_masks[nid] = {vw: sorted(fs) for vw, fs in sorted(md.items())}
    return labels, gm, vs, inst_masks, {v: dict(d) for v, d in mask_cluster.items()}


def process(sc, n_views, sem_thr):
    if not (HULL_ROOT / sc / "hull.npz").is_file():
        print(f"[skip] {sc}"); return
    if os.environ.get("FORCE", "") != "1" and (OUT_ROOT / sc / "instances.npz").is_file():
        return
    labels, gm, vs, inst_masks, mask_cluster = semantic_cluster(sc, n_views, sem_thr)
    out = OUT_ROOT / sc; out.mkdir(parents=True, exist_ok=True)
    meta = {"script": "voxel_sem_cluster_surf_donut.py", "surface": True, "zbuffer": True, "donut": True,
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
