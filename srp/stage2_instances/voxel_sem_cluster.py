#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""voxel_sem_cluster.py — 語意「特徵分群」版:全場景遮罩 CLIP 去偏特徵凝聚分群 → 群標籤
→ voxel 距離加權投票 → 3D 連通分實例。輸出 data/eval/srp_hull_semcluster/<scene>/instances.npz。
可視化: SRP_VIZ_ARGS="<scene> 1 srp_hull_semcluster" webots worlds/hull_viz.wbt
用法: ./voxel_sem_cluster.py [scene|group|(空=全部)] [--n-views 12] [--sem-thr 0.3]
env: SAM_ROOT HULL_ROOT(srp_hull_v12) CAPTURES_ROOT ARM_MASK_ROOT
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
OUT_ROOT = REPO / "data" / "eval" / os.environ.get("OUT_ROOT", "srp_hull_semcluster")
MIN_VOX = 50
# 特徵來源可切換(基準一致:除特徵外全相同)。FEAT_FILE=siglip_b16_feats.npy 換 SigLIP2;
# DEBIAS=0 關去偏;BG_FILE=SigLIP2 bg 向量路徑(空=用 CLIP 的 MC.F_BG)。
FEAT_FILE = os.environ.get("FEAT_FILE", "clip_mean_feats.npy")
DEBIAS = os.environ.get("DEBIAS", "1") == "1"
BG_FILE = os.environ.get("BG_FILE", "")
_BG = None
if DEBIAS:
    if BG_FILE:
        _bg = np.load(BG_FILE).astype(np.float64)
        _BG = _bg / (np.linalg.norm(_bg) + 1e-9)
    else:
        _BG = MC.F_BG.astype(np.float64)   # CLIP 純填充色特徵


def debias_feats(F):
    """去偏(投影掉 bg 方向)+ L2;DEBIAS=0 時只 L2。與 MC.debias 同式,但 bg 可換成 SigLIP2 的。"""
    F = F.astype(np.float64)
    if DEBIAS and _BG is not None:
        F = F - (F @ _BG)[:, None] * _BG[None, :]
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


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
        km = MK.kept_object_masks(vd); ms = [m for m, _ in km]; names = [nm for _, nm in km]
        if not ms: continue
        C, Rb = cam.load_pose(pf); Rwc, t = cam.pose_to_w2c(C, Rb)
        K = cam.intrinsics(ms[0].shape[1], ms[0].shape[0])
        rgb = cv2.cvtColor(cv2.imread(str(sdir / f"{vd.name}.png")), cv2.COLOR_BGR2RGB)
        ap = ARM / sc / f"{vd.name}_arm.png"
        arm = (cv2.imread(str(ap), 0) > 127) if ap.is_file() else None
        vi = len(vdata)
        for mi, f in enumerate(MK.feats_list(vd, names, feat_file=FEAT_FILE)):
            if f is not None: allf.append(f); ref.append((vi, mi))
        vdata.append((ms, C, Rwc, t, K, arm, names, vd.name))
    labels = np.zeros(shape, np.int32)
    if len(allf) < 2:
        return labels, gm, vs, {}, {}
    F = debias_feats(np.array(allf))
    cl = fcluster(linkage(pdist(F, "cosine"), "average"), t=sem_thr, criterion="distance")
    mlabel = {r: int(c) for r, c in zip(ref, cl)}
    # 遮罩→群 標籤(cl):報告用來區分「切 hull 的主群」vs「幾何蓋到的跨群零星」
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
            for mi, m in enumerate(ms):
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
        return   # 續跑保護:已算過就跳過(FORCE=1 強制重算)
    labels, gm, vs, inst_masks, mask_cluster = semantic_cluster(sc, n_views, sem_thr)
    out = OUT_ROOT / sc; out.mkdir(parents=True, exist_ok=True)
    # 記錄「這批 instance 怎麼來的」— 用哪個 hull/遮罩來源、特徵、門檻、視角數
    meta = {"script": "voxel_sem_cluster.py", "built": _dt.datetime.now().isoformat(timespec="seconds"),
            "hull_root": HULL_ROOT.name, "sam_root": SAM_ROOT.name, "captures_root": CAPTURES.name,
            "arm_root": ARM.name, "feat_file": FEAT_FILE, "debias": DEBIAS,
            "bg_file": (BG_FILE or "clip_F_BG"), "sem_thr": sem_thr, "n_views": n_views, "min_vox": MIN_VOX}
    np.savez_compressed(out / "instances.npz", labels=labels, grid_min=gm, voxel_size=vs,
                        build_meta=json.dumps(meta, ensure_ascii=False))
    insts = [{"instance": i, "n_vox": int((labels == i).sum()), "masks": inst_masks.get(i, {})}
             for i in range(1, int(labels.max()) + 1) if (labels == i).any()]
    (out / "instances.json").write_text(json.dumps(
        {"scene": sc, "voxel": vs, "n_instances": len(insts), "meta": meta,
         "mask_clusters": mask_cluster, "instances": insts},   # 每遮罩的群id(cl),報告用
        indent=2, ensure_ascii=False))
    print(f"[{sc}] 特徵分群 → {len(insts)} instance", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--n-views", type=int, default=12, dest="n_views")
    ap.add_argument("--sem-thr", type=float, default=0.3, dest="sem_thr")
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
