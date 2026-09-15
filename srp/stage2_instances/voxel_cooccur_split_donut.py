#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""voxel_cooccur_split_donut.py — voxel 共現分群 × donut(去大遮罩)。同 voxel_cooccur_split,但遮罩先挖洞(父遮罩不再蓋子物體voxel→共現向量乾淨)。cooccur 不用 CLIP,故 donut 只挖遮罩、不重算特徵。

不走「遮罩分群→投票」(semcluster)也不走「偵測配對成圖」(cg),而是直接對每個【表面 voxel】
建二元共現向量,再聯合幾何+共現距離做 DBSCAN 把 voxel 分成物體:
  ① 每視角:cg 的 zbuffer_visible(免深度,hull 自己當遮擋體)取每像素最近 voxel。
  ② 每 voxel 的共現向量 f∈{0,1}^M:跨全視角落在哪些 SAM 物件遮罩內就 =1(z-buffer 判可見)。
  ③ d_geo = ‖Pi−Pj‖/對角線(直徑歸一);d_sem = 1−cos(fi,fj);D = α·d_geo + (1−α)·d_sem。
  ④ DBSCAN(metric=precomputed, eps, min_samples) 分 voxel;noise 各自成群、<MIN_VOX 濾掉。
輸出表面 labels(需 fill_solid 才做 3D-IoU 評估)。與 semcluster/cg 同基準可比。
用法: ./voxel_cooccur_split.py [scene|group|(空=全部)] [--n-views 12] [--alpha 0.25] [--eps 0.08] [--min 20]
env: HULL_ROOT SAM_ROOT CAPTURES_ROOT OUT_ROOT
"""
import argparse, os, sys, json, glob, datetime as _dt, time as _time
from pathlib import Path
import numpy as np, cv2
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import camera as cam, masks as MK, viewpoints as VP  # noqa: E402
import cg_associate as CG                             # noqa: E402  重用免深度 zbuffer_visible
from voxel_sem_cluster_donut import donut_masks       # noqa: E402  去大遮罩(像素級挖洞)

CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures_fast")))
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "sam_only_fast")))
HULL_ROOT = Path(os.environ.get("HULL_ROOT", str(REPO / "data" / "eval" / "srp_hull_v12")))
OUT_ROOT = REPO / "data" / "eval" / os.environ.get("OUT_ROOT", "srp_hull_cooccur_donut")
MIN_VOX = int(os.environ.get("MIN_VOX", "50"))   # 群小於此丟掉;MIN_VOX=0=不濾(全留)


def split_scene(sc, n_views, alpha, eps, min_samples):
    z = np.load(HULL_ROOT / sc / "hull.npz")
    if "surface" not in z.files:
        raise RuntimeError(f"{sc}: hull.npz 無 surface(先跑 add_surface_mask)")
    surf = z["surface"]; gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = surf.shape
    vox = np.array(np.nonzero(surf)).T                      # (Ns,3) 表面 voxel 的 grid 索引
    P = gm + (vox + 0.5) * vs                               # 世界座標
    Ns = len(vox)
    out = np.zeros(shape, np.int32)
    if Ns < min_samples:
        return out, 0, Ns
    group = sc.split("_")[0]; sdir = CAPTURES / f"multi_{group}" / sc
    want = set(VP.selected_view_names(n_views))
    cols = []                                               # 每欄=一個 mask 的 Ns bool 共現
    for vd in sorted((SAM_ROOT / sc).glob("view_*")):
        if vd.name not in want:
            continue
        pf = sdir / f"{vd.name}_pose.json"
        if not pf.is_file():
            continue
        km = MK.kept_object_masks(vd); ms = [m for m, _ in km]   # 物件遮罩(去背景/桌面/邊界)
        if not ms:
            continue
        ms = donut_masks(ms)   # ★去大遮罩:父遮罩挖成甜甜圈→共現向量不再蓋到子物體 voxel
        H, W = ms[0].shape
        C, Rb = cam.load_pose(pf)
        vox_at = CG.zbuffer_visible(P, C, Rb, W, H, vs).reshape(H, W)   # 每像素最近 voxel local idx(-1)
        for m in ms:
            f = np.zeros(Ns, bool)
            hit = vox_at[m > 0]                             # mask 內像素的 winning voxel
            hit = hit[hit >= 0]
            if len(hit):
                f[np.unique(hit)] = True
            cols.append(f)
    if len(cols) < 1:
        return out, 0, Ns
    F = np.stack(cols, 1).astype(np.float32)               # Ns × M 二元共現
    Dmax = float(np.linalg.norm(P.max(0) - P.min(0))) + 1e-6
    d_geo = euclidean_distances(P).astype(np.float32) / Dmax
    d_sem = cosine_distances(F).astype(np.float32)         # 空向量→距離1(只靠 geo 連,通常變 noise)
    D = alpha * d_geo + (1.0 - alpha) * d_sem
    lab = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(D)
    nid = int(lab.max()) + 1 if (lab >= 0).any() else 0    # noise(-1)各自獨立成群
    for i in range(len(lab)):
        if lab[i] == -1:
            lab[i] = nid; nid += 1
    newid = 0                                              # 寫回 grid(0=背景),濾 <MIN_VOX
    for c in np.unique(lab):
        sel = lab == c
        if int(sel.sum()) < MIN_VOX:
            continue
        newid += 1
        out[tuple(vox[sel].T)] = newid
    return out, newid, Ns


def process(sc, n_views, alpha, eps, min_samples):
    if not (HULL_ROOT / sc / "hull.npz").is_file():
        print(f"[skip] {sc}"); return
    if os.environ.get("FORCE", "") != "1" and (OUT_ROOT / sc / "instances.npz").is_file():
        return
    z = np.load(HULL_ROOT / sc / "hull.npz")
    _t0 = _time.perf_counter()
    labels, ninst, Ns = split_scene(sc, n_views, alpha, eps, min_samples)
    _elapsed = _time.perf_counter() - _t0        # 每場計算耗時(秒)
    out = OUT_ROOT / sc; out.mkdir(parents=True, exist_ok=True)
    meta = {"script": "voxel_cooccur_split.py", "built": _dt.datetime.now().isoformat(timespec="seconds"),
            "hull_root": HULL_ROOT.name, "sam_root": SAM_ROOT.name, "captures_root": CAPTURES.name,
            "alpha": alpha, "eps": eps, "min_samples": min_samples, "n_views": n_views,
            "min_vox": MIN_VOX, "surface_labels": True, "n_surface_vox": Ns, "elapsed_s": round(_elapsed, 3)}
    np.savez_compressed(out / "instances.npz", labels=labels, grid_min=z["grid_min"],
                        voxel_size=float(z["voxel_size"]), build_meta=json.dumps(meta, ensure_ascii=False))
    (out / "instances.json").write_text(json.dumps({"scene": sc, "n_instances": ninst, "meta": meta},
                                                    indent=2, ensure_ascii=False))
    print(f"[{sc}] 共現分群 → {ninst} instance (表面 {Ns} vox) [{_elapsed:.2f}s]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--n-views", type=int, default=12, dest="n_views")
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--eps", type=float, default=0.08)
    ap.add_argument("--min", type=int, default=20, dest="min_samples")
    args = ap.parse_args()
    if not args.targets:
        scenes = sorted(Path(p).parent.name for p in glob.glob(str(HULL_ROOT / "*_scene*/hull.npz")))
    else:
        scenes = []
        for a in args.targets:
            if "scene" in a:
                scenes.append(a)
            else:
                scenes += [Path(p).parent.name for p in glob.glob(str(HULL_ROOT / f"{a}_scene*/hull.npz"))]
        scenes = sorted(set(scenes))
    for sc in scenes:
        try:
            process(sc, args.n_views, args.alpha, args.eps, args.min_samples)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")


if __name__ == "__main__":
    main()
