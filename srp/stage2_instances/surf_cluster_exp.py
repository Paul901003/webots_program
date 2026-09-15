#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""surf_cluster_exp.py — 表面 voxel 分群實驗:{CLIP純語意 / 共現+幾何} × {凝聚 / DBSCAN}。

只對 hull 表面 voxel 分群(不是整體 occupancy),z-buffer 判可見(被擋 voxel 該視角不算)。
一次 z-buffer 迴圈同時建每個表面 voxel 的:
  · 共現向量 f∈{0,1}^M:可見時落在哪些遮罩(不靠外觀,用遮罩 instance 身份)。
  · CLIP:落入遮罩的 clip_mean_feats(同 semcluster 那份,不重算)累加平均 → 去偏。
距離矩陣(兩演算法吃同一個):
  · feature=clip   → D = 1-cos(voxel CLIP)          （純語意,無幾何）
  · feature=cooccur→ D = α·d_geo + (1-α)·(1-cos(f))  （幾何+共現）
演算法:agg=average-linkage 閾值切(同 semcluster 作法,作用在 voxel);dbscan=precomputed。
輸出【表面 labels】instances.npz(不填實心),交 gen_viz_objs 看:
  SRP_VIZ_ARGS="<scene> 1 <out_root>" webots worlds/hull_viz.wbt
用法: ./surf_cluster_exp.py <scene...> --feature clip|cooccur --algo agg|dbscan --out-root <root>
      [--alpha .25 --thresh .4 --eps .15 --min 5 --min-vox 50 --n-views 12]
env: HULL_ROOT(srp_hull_mv2_v12_am1) SAM_ROOT(mobilesamv2_fast) CAPTURES_ROOT(captures_fast) ARM_MASK_ROOT(srp_arm_masks)
"""
import argparse, os, sys, json, datetime as _dt
from pathlib import Path
import numpy as np, cv2
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import camera as cam, masks as MK, viewpoints as VP  # noqa: E402
import cg_associate as CG                             # noqa: E402  重用免深度 zbuffer_visible
import mask_clip_cluster as MC                        # noqa: E402  取 F_BG(去偏用,同 semcluster)

EVAL = REPO / "data" / "eval"
CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures_fast")))
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(EVAL / "mobilesamv2_fast")))
HULL_ROOT = Path(os.environ.get("HULL_ROOT", str(EVAL / "srp_hull_mv2_v12_am1")))
ARM = Path(os.environ.get("ARM_MASK_ROOT", str(EVAL / "srp_arm_masks")))
FEAT_FILE = "clip_mean_feats.npy"
_BG = MC.F_BG.astype(np.float64)


def _debias(F):
    """投影掉背景方向 + L2(與 semcluster 同式)。"""
    F = F.astype(np.float64)
    F = F - (F @ _BG)[:, None] * _BG[None, :]
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


def build_feats(sc, n_views):
    z = np.load(HULL_ROOT / sc / "hull.npz")
    if "surface" not in z.files:
        raise RuntimeError(f"{sc}: hull.npz 無 surface")
    surf = z["surface"]; gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = surf.shape
    vox = np.array(np.nonzero(surf)).T; P = gm + (vox + 0.5) * vs; Ns = len(vox)
    group = sc.split("_")[0]; sdir = CAPTURES / f"multi_{group}" / sc
    want = set(VP.selected_view_names(n_views)) if n_views else None
    cooc_cols = []                          # 每欄=一個遮罩的 Ns bool 共現
    clip_sum = None; clip_cnt = np.zeros(Ns)
    for vd in sorted((SAM_ROOT / sc).glob("view_*")):
        if want is not None and vd.name not in want:
            continue
        pf = sdir / f"{vd.name}_pose.json"
        if not pf.is_file():
            continue
        km = MK.kept_object_masks(vd); ms = [m for m, _ in km]; names = [nm for _, nm in km]
        if not ms:
            continue
        feats = list(MK.feats_list(vd, names, feat_file=FEAT_FILE))   # 對應 names,可能含 None
        C, Rb = cam.load_pose(pf); H, W = ms[0].shape
        vox_at = CG.zbuffer_visible(P, C, Rb, W, H, vs).reshape(H, W)  # 每像素最近表面 voxel local idx(-1)
        ap = ARM / sc / f"{vd.name}_arm.png"
        arm = (cv2.imread(str(ap), 0) > 127) if ap.is_file() else None
        for mi, m in enumerate(ms):
            sel = m > 0
            if arm is not None:
                sel = sel & ~arm
            hit = vox_at[sel]; hit = hit[hit >= 0]
            if len(hit) == 0:
                continue
            uh = np.unique(hit)
            f = np.zeros(Ns, bool); f[uh] = True; cooc_cols.append(f)
            fe = feats[mi] if mi < len(feats) else None
            if fe is not None:
                fe = np.asarray(fe, np.float64)
                if clip_sum is None:
                    clip_sum = np.zeros((Ns, len(fe)))
                clip_sum[uh] += fe; clip_cnt[uh] += 1
    F_cooc = np.stack(cooc_cols, 1).astype(np.float32) if cooc_cols else np.zeros((Ns, 1), np.float32)
    F_clip = None
    if clip_sum is not None:
        F_clip = np.zeros_like(clip_sum)
        nz = clip_cnt > 0
        F_clip[nz] = clip_sum[nz] / clip_cnt[nz, None]   # voxel = 落入遮罩 CLIP 平均
        F_clip = _debias(F_clip)                         # 平均後去偏(無票 voxel 仍是 0 向量)
    return vox, P, gm, vs, shape, F_cooc, F_clip, clip_cnt


def run(sc, feature, algo, alpha, thresh, eps, msamp, min_vox, n_views):
    vox, P, gm, vs, shape, F_cooc, F_clip, clip_cnt = build_feats(sc, n_views)
    Ns = len(vox)
    if feature == "clip":
        if F_clip is None:
            return np.zeros(shape, np.int32), 0, Ns
        D = cosine_distances(F_clip).astype(np.float32)
    else:
        Dmax = float(np.linalg.norm(P.max(0) - P.min(0))) + 1e-6
        d_geo = euclidean_distances(P).astype(np.float32) / Dmax
        d_sem = cosine_distances(F_cooc).astype(np.float32)
        D = alpha * d_geo + (1.0 - alpha) * d_sem
    np.fill_diagonal(D, 0.0)
    if algo == "agg":
        Z = linkage(squareform(D, checks=False), method="average")
        lab = fcluster(Z, t=thresh, criterion="distance")
    else:
        lab = DBSCAN(eps=eps, min_samples=msamp, metric="precomputed").fit_predict(D)
    out = np.zeros(shape, np.int32); newid = 0
    for c in np.unique(lab):
        if c == -1:                                       # dbscan noise 丟棄
            continue
        selv = np.where(lab == c)[0]
        if len(selv) < min_vox:
            continue
        newid += 1
        out[vox[selv, 0], vox[selv, 1], vox[selv, 2]] = newid
    return out, newid, Ns


def process(sc, args):
    if not (HULL_ROOT / sc / "hull.npz").is_file():
        print(f"[skip] {sc}"); return
    outdir = EVAL / args.out_root / sc
    if os.environ.get("FORCE", "") != "1" and (outdir / "instances.npz").is_file():
        return
    labels, ninst, Ns = run(sc, args.feature, args.algo, args.alpha, args.thresh,
                            args.eps, args.min, args.min_vox, args.n_views)
    z = np.load(HULL_ROOT / sc / "hull.npz")
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"script": "surf_cluster_exp.py", "built": _dt.datetime.now().isoformat(timespec="seconds"),
            "feature": args.feature, "algo": args.algo, "alpha": args.alpha, "thresh": args.thresh,
            "eps": args.eps, "min_samples": args.min, "min_vox": args.min_vox, "n_views": args.n_views,
            "hull_root": HULL_ROOT.name, "sam_root": SAM_ROOT.name, "surface_labels": True, "n_surface_vox": Ns}
    np.savez_compressed(outdir / "instances.npz", labels=labels, grid_min=z["grid_min"],
                        voxel_size=float(z["voxel_size"]), build_meta=json.dumps(meta, ensure_ascii=False))
    (outdir / "instances.json").write_text(json.dumps({"scene": sc, "n_instances": ninst, "meta": meta},
                                                       indent=2, ensure_ascii=False))
    print(f"[{sc}] {args.feature}/{args.algo} → {ninst} 群 (表面 {Ns} vox)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--feature", required=True, choices=["clip", "cooccur"])
    ap.add_argument("--algo", required=True, choices=["agg", "dbscan"])
    ap.add_argument("--out-root", required=True, dest="out_root")
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--thresh", type=float, default=0.4)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--min", type=int, default=5)
    ap.add_argument("--min-vox", type=int, default=50, dest="min_vox")
    ap.add_argument("--n-views", type=int, default=12, dest="n_views")
    args = ap.parse_args()
    scenes = []
    for a in args.scenes:
        scenes.append(a) if "scene" in a else scenes.extend(
            p.parent.name for p in (HULL_ROOT).glob(f"{a}_scene*/hull.npz"))
    fails = 0
    for sc in sorted(set(scenes)):
        try:
            process(sc, args)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[ERR] {sc}: {e}", flush=True)
            fp = EVAL / args.out_root / "_failed.log"       # 失敗一定留檔(不靠 shell log)
            fp.parent.mkdir(parents=True, exist_ok=True)
            with open(fp, "a") as f:
                f.write(f"=== {sc} ({args.feature}/{args.algo}): {e} ===\n{tb}\n")
            fails += 1
    if fails:
        print(f"[!] {fails} 場失敗 → 見 {EVAL / args.out_root / '_failed.log'}", flush=True)


if __name__ == "__main__":
    main()
