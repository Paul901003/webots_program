#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""voxel_sem_cluster_donut.py — semcluster 的【去除大遮罩(挖洞/甜甜圈)】版。

與 voxel_sem_cluster.py 唯一差別:讀遮罩後,對「像素上包含其他更小遮罩」的父遮罩,
【挖掉子遮罩區域(甜甜圈)】→ 再【重算 CLIP 特徵】(甜甜圈遮罩,不能沿用預存 clip_mean_feats,
否則挖洞等於沒做,特徵仍含子物體)→ 去偏凝聚分群 → voxel 投票(用挖洞後遮罩)→ 3D 連通。
其餘(分群/argmax/連通/輸出)全同 semcluster。目的:去掉 SAM 巢狀遮罩污染,看分群是否變乾淨。

判定父遮罩=像素級:子遮罩實際像素 ≥ NEST_THR 落在另一(更大)遮罩內 → 該大遮罩對此子遮罩挖洞。
可視化: SRP_VIZ_ARGS="<scene> 1 srp_hull_semcluster_donut cubes" webots worlds/hull_viz.wbt
用法: ./voxel_sem_cluster_donut.py [scene|group|(空=全部)] [--n-views 12] [--sem-thr 0.4]
env: SAM_ROOT HULL_ROOT CAPTURES_ROOT ARM_MASK_ROOT OUT_ROOT NEST_THR(0.8)
"""
import argparse, os, sys, json, glob, datetime as _dt
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "srp" / "io"))
import numpy as np, cv2
from pathlib import Path
from collections import defaultdict
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
import camera as cam, masks as MK, mask_clip_cluster as MC, viewpoints as VP

REPO = Path(__file__).resolve().parents[2]
CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures_fast")))
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "sam_only_fast")))
HULL_ROOT = Path(os.environ.get("HULL_ROOT", str(REPO / "data" / "eval" / "srp_hull_v12")))
ARM = Path(os.environ.get("ARM_MASK_ROOT", str(REPO / "data" / "eval" / "srp_arm_masks")))
OUT_ROOT = REPO / "data" / "eval" / os.environ.get("OUT_ROOT", "srp_hull_semcluster_donut")
MIN_VOX = 50
NEST_THR = float(os.environ.get("NEST_THR", "0.8"))   # 子遮罩 ≥ thr 面積落在更大遮罩內 → 該大遮罩挖洞
DEBIAS = os.environ.get("DEBIAS", "1") == "1"
_BG = MC.F_BG.astype(np.float64) if DEBIAS else None


def debias_feats(F):
    F = F.astype(np.float64)
    if DEBIAS and _BG is not None:
        F = F - (F @ _BG)[:, None] * _BG[None, :]
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


DONUT_FEAT = "clip_donut_feats.npy"   # 甜甜圈遮罩的重算 CLIP 特徵(預存,分群+報告共用,免每次重跑 CLIP)


def donut_feats(view_dir, rgb, ms_donut, names):
    """回 {遮罩檔名: 甜甜圈 CLIP 特徵 or None}。存/讀 clip_donut_feats.npy(依 sorted(masks/*.png) 全遮罩對齊,
    與 clip_mean_feats 同存法)。首次算了就存,之後直接讀 → 分群和報告共用、不重跑 CLIP。
    FORCE_FEAT=1 可強制重算特徵。"""
    vd = Path(view_dir)
    fp = vd / DONUT_FEAT
    allnames = [q.name for q in sorted((vd / "masks").glob("mask_*.png"))]   # 全遮罩(與 clip_mean 對齊)
    if fp.is_file() and os.environ.get("FORCE_FEAT", "") != "1":
        arr = np.load(fp)
        if len(arr) == len(allnames):
            d = {allnames[i]: (None if np.isnan(arr[i]).any() else arr[i]) for i in range(len(allnames))}
            return {n: d.get(n) for n in names}
    # 需重算:對「全遮罩」都算 donut 特徵存檔(kept 只是子集,但存全套與 clip_mean 對齊、可被檔名查)
    all_km = MK.kept_object_masks(vd)                       # kept 的 (mask, name)
    name2donut = {names[i]: ms_donut[i] for i in range(len(names))}
    out_arr = np.full((len(allnames), 512), np.nan, np.float32)
    idx = {n: i for i, n in enumerate(allnames)}
    kept_ms = []; kept_pos = []
    for n in names:
        if n in idx:
            kept_ms.append(name2donut[n]); kept_pos.append(idx[n])
    if kept_ms:
        feats = MC.clip_feats(rgb, kept_ms, "mean")
        for pos, f in zip(kept_pos, feats):
            if f is not None: out_arr[pos] = np.asarray(f, np.float32)
    np.save(fp, out_arr)
    return {n: (None if (n not in idx or np.isnan(out_arr[idx[n]]).any()) else out_arr[idx[n]]) for n in names}


def donut_masks(masks, thr=NEST_THR):
    """像素級『去除大遮罩』:對每對(i,j),若 i 更大且 j 有 ≥thr 面積落在 i 內 → 從 i 挖掉 j(甜甜圈)。
    回傳挖洞後遮罩(list bool)。O(n²),n~15。與 cg mask_subtract_contained 同挖洞、但改像素判定(較準)。"""
    n = len(masks)
    areas = [int(m.sum()) for m in masks]
    out = [m.copy() for m in masks]
    for i in range(n):
        for j in range(n):
            if i == j or areas[j] == 0:
                continue
            if areas[i] > areas[j] and (masks[i] & masks[j]).sum() / areas[j] >= thr:
                out[i] = out[i] & (~masks[j])   # 大遮罩 i 挖掉小遮罩 j
    return out


def semantic_cluster(sc, n_views, sem_thr):
    group = sc.split("_")[0]; sdir = CAPTURES / f"multi_{group}" / sc
    z = np.load(HULL_ROOT / sc / "hull.npz")
    occ = z["occupancy"]; gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = occ.shape
    vox = np.array(np.nonzero(occ)).T; P = gm + (vox + 0.5) * vs; M = len(vox)
    want = set(VP.selected_view_names(n_views)) if n_views else None
    vdata = []; allf = []; ref = []
    for vd in sorted((SAM_ROOT / sc).glob("view_*")):
        if want is not None and vd.name not in want: continue
        pf = sdir / f"{vd.name}_pose.json"
        if not pf.is_file(): continue
        km = MK.kept_object_masks(vd); ms0 = [m for m, _ in km]; names = [nm for _, nm in km]
        if not ms0: continue
        ms = donut_masks(ms0)               # ★去除大遮罩:含子遮罩的父遮罩挖成甜甜圈
        C, Rb = cam.load_pose(pf); Rwc, t = cam.pose_to_w2c(C, Rb)
        K = cam.intrinsics(ms[0].shape[1], ms[0].shape[0])
        rgb = cv2.cvtColor(cv2.imread(str(sdir / f"{vd.name}.png")), cv2.COLOR_BGR2RGB)
        ap = ARM / sc / f"{vd.name}_arm.png"
        arm = (cv2.imread(str(ap), 0) > 127) if ap.is_file() else None
        vi = len(vdata)
        fmap = donut_feats(vd, rgb, ms, names)   # ★甜甜圈 CLIP(存 clip_donut_feats.npy,分群+報告共用)
        for mi, nm in enumerate(names):
            f = fmap.get(nm)
            if f is not None: allf.append(f); ref.append((vi, mi))
        vdata.append((ms, C, Rwc, t, K, arm, names, vd.name))
    labels = np.zeros(shape, np.int32)
    if len(allf) < 2:
        return labels, gm, vs, {}, {}
    F = debias_feats(np.array(allf))
    cl = fcluster(linkage(pdist(F, "cosine"), "average"), t=sem_thr, criterion="distance")
    mlabel = {r: int(c) for r, c in zip(ref, cl)}
    mask_cluster = defaultdict(dict)   # view名 → {遮罩檔: 群id}
    for (vi, mi), c in mlabel.items():
        mask_cluster[vdata[vi][7]][vdata[vi][6][mi]] = int(c)
    votes = defaultdict(lambda: np.zeros(M))
    vmask = defaultdict(lambda: defaultdict(set))   # voxel p → {view: set(遮罩檔)}
    for vi, (ms, C, Rwc, t, K, arm, names, vname) in enumerate(vdata):
        H, W = ms[0].shape
        X = P @ Rwc.T + t; zc = X[:, 2]; ok = zc > 1e-9; zz = np.where(ok, zc, 1.0)
        u = np.round(K[0, 0] * X[:, 0] / zz + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * X[:, 1] / zz + K[1, 2]).astype(int)
        inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        wt = 1.0 / np.maximum(np.linalg.norm(P - C, axis=1), 1e-3)
        for p in np.where(inb)[0]:
            if arm is not None and arm[v[p], u[p]]: continue
            for mi, m in enumerate(ms):      # ms 是挖洞後遮罩 → 幾何歸屬也乾淨
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
                md = defaultdict(set)
                for p in comp_p:
                    for vw, fs in vmask.get(int(p), {}).items():
                        md[vw] |= fs
                inst_masks[nid] = {vw: sorted(fs) for vw, fs in sorted(md.items())}
    return labels, gm, vs, inst_masks, {v: dict(d) for v, d in mask_cluster.items()}


def process(sc, n_views, sem_thr):
    if not (HULL_ROOT / sc / "hull.npz").is_file():
        print(f"[skip] {sc}"); return
    if os.environ.get("FORCE", "") != "1" and (OUT_ROOT / sc / "instances.npz").is_file():
        return
    labels, gm, vs, inst_masks, mask_cluster = semantic_cluster(sc, n_views, sem_thr)
    out = OUT_ROOT / sc; out.mkdir(parents=True, exist_ok=True)
    meta = {"script": "voxel_sem_cluster_donut.py", "donut": True, "nest_thr": NEST_THR,
            "feat": "recomputed_clip_donut", "built": _dt.datetime.now().isoformat(timespec="seconds"),
            "hull_root": HULL_ROOT.name, "sam_root": SAM_ROOT.name, "captures_root": CAPTURES.name,
            "arm_root": ARM.name, "debias": DEBIAS, "sem_thr": sem_thr, "n_views": n_views, "min_vox": MIN_VOX}
    np.savez_compressed(out / "instances.npz", labels=labels, grid_min=gm, voxel_size=vs,
                        build_meta=json.dumps(meta, ensure_ascii=False))
    insts = [{"instance": i, "n_vox": int((labels == i).sum()), "masks": inst_masks.get(i, {})}
             for i in range(1, int(labels.max()) + 1) if (labels == i).any()]
    (out / "instances.json").write_text(json.dumps(
        {"scene": sc, "voxel": vs, "n_instances": len(insts), "meta": meta,
         "mask_clusters": mask_cluster, "instances": insts},
        indent=2, ensure_ascii=False))
    print(f"[{sc}] 去大遮罩+特徵分群 → {len(insts)} instance", flush=True)


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
