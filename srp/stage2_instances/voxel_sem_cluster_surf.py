#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""voxel_sem_cluster.py — 語意「特徵分群」版:全場景遮罩 CLIP 去偏特徵凝聚分群 → 群標籤
→ voxel 距離加權投票 → 3D 連通分實例。輸出 data/eval/srp_hull_semcluster/<scene>/instances.npz。
可視化: SRP_VIZ_ARGS="<scene> 1 srp_hull_semcluster" webots worlds/hull_viz.wbt
用法: ./voxel_sem_cluster.py [scene|group|(空=全部)] [--n-views 12] [--sem-thr 0.3]
env: SAM_ROOT HULL_ROOT(srp_hull_mv2_v12_am1) CAPTURES_ROOT ARM_MASK_ROOT
★★ 本檔 = voxel_sem_cluster 的【表面 voxel + z-buffer】版:遮罩 CLIP 分群 / argmax / 3D 連通【全同 semcluster】,
   只把投票 voxel 從「整體 occupancy」換成「hull surface」,且投票加 z-buffer(只讓該視角真正看得到的表面 voxel 投票)。
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
import cg_associate as CG   # 免深度 z-buffer(zbuffer_visible)

REPO = Path(__file__).resolve().parents[2]
CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures_fast")))
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "sam_only_fast")))
HULL_ROOT = Path(os.environ.get("HULL_ROOT", str(REPO / "data" / "eval" / "srp_hull_v12")))
ARM = Path(os.environ.get("ARM_MASK_ROOT", str(REPO / "data" / "eval" / "srp_arm_masks")))
OUT_ROOT = REPO / "data" / "eval" / os.environ.get("OUT_ROOT", "srp_hull_semcluster_surf")
MIN_VOX = int(os.environ.get("MIN_VOX", "50"))   # 表面 voxel 版總數較少,可調低
# 巢狀父遮罩降權(opt-in,解「小物體貼大物體被吞併」):一張遮罩若「包含其他更小的遮罩」(父遮罩,如海綿把方塊框住),
# 其投票 ×NESTED_ALPHA;不含任何其他遮罩的葉遮罩投完整一票。NESTED_ALPHA=1.0=完全等同原行為(不影響 baseline)。
NESTED_ALPHA = float(os.environ.get("NESTED_ALPHA", "1.0"))   # <1 開啟降權;=1 關閉(預設)
NESTED_THR = float(os.environ.get("NESTED_THR", "0.8"))       # 子遮罩 ≥ thr 面積落在某更大遮罩內 → 該大遮罩=父遮罩
# 實心模式(opt-in):SOLID=1 → 用整體 occupancy(實心) 且不做 z-buffer(每個 voxel 投影落進遮罩就投,同原 semcluster);
# 預設 SOLID=0 = 表面 voxel + z-buffer(surf)。用來做「表面 vs 實心」受控比較(其餘全相同)。
SOLID = os.environ.get("SOLID", "0") == "1"
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


def parent_flags(ms, thr):
    """回傳每張遮罩 is_parent:面積比它小、且 ≥thr 落在它內部的其他遮罩存在 → 它是父遮罩(包含小遮罩)。O(n²),n~15。"""
    areas = [int(m.sum()) for m in ms]; n = len(ms); par = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j or areas[j] == 0: continue
            if areas[i] > areas[j] and (ms[i] & ms[j]).sum() / areas[j] >= thr:
                par[i] = True; break
    return par


def semantic_cluster(sc, n_views, sem_thr):
    group = sc.split("_")[0]; sdir = CAPTURES / f"multi_{group}" / sc
    z = np.load(HULL_ROOT / sc / "hull.npz")
    occ = z["occupancy"] if SOLID else z["surface"]   # ★SOLID=實心 occupancy;否則表面 voxel
    gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = occ.shape
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
        ispar = parent_flags(ms, NESTED_THR) if NESTED_ALPHA < 1.0 else None
        vdata.append((ms, C, Rwc, t, K, arm, names, vd.name, Rb, ispar))
    labels = np.zeros(shape, np.int32)
    if len(allf) < 2:
        return labels, gm, vs, {}, {}
    F = debias_feats(np.array(allf))
    _pf = os.environ.get("PCA_FILE")            # opt-in:設了才降維(實驗用,不影響 baseline)
    if _pf:
        _pz = np.load(_pf)
        F = (F - _pz["mean"]) @ _pz["components"].T
        F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)   # L2 讓 cosine 有意義
    _cm = os.environ.get("CLUSTER_METHOD", "agg")   # opt-in:dbscan(實驗);預設 agg 不變,不影響 baseline/主管線
    if _cm == "dbscan":
        from sklearn.cluster import DBSCAN
        eps = float(os.environ.get("DBSCAN_EPS", "0.15")); msamp = int(os.environ.get("DBSCAN_MIN", "1"))
        cl = DBSCAN(eps=eps, min_samples=msamp, metric="cosine").fit_predict(F)
        nid = int(cl.max()) + 1 if (cl >= 0).any() else 0   # noise(-1)各自獨立成群,不併成一團
        cl = cl.copy()
        for i in range(len(cl)):
            if cl[i] == -1:
                cl[i] = nid; nid += 1
    else:
        cl = fcluster(linkage(pdist(F, "cosine"), "average"), t=sem_thr, criterion="distance")
    mlabel = {r: int(c) for r, c in zip(ref, cl)}
    # 遮罩→群 標籤(cl):報告用來區分「切 hull 的主群」vs「幾何蓋到的跨群零星」
    mask_cluster = defaultdict(dict)   # view名 → {遮罩檔: 群id}
    for (vi, mi), c in mlabel.items():
        mask_cluster[vdata[vi][7]][vdata[vi][6][mi]] = int(c)
    votes = defaultdict(lambda: np.zeros(M))
    vmask = defaultdict(lambda: defaultdict(set))   # voxel p → {view: set(遮罩檔)}
    for vi, (ms, C, Rwc, t, K, arm, names, vname, Rb, ispar) in enumerate(vdata):
        H, W = ms[0].shape
        vox_at = None if SOLID else CG.zbuffer_visible(P, C, Rb, W, H, vs).reshape(H, W)   # ★z-buffer(SOLID時不用)
        X = P @ Rwc.T + t; zc = X[:, 2]; ok = zc > 1e-9; zz = np.where(ok, zc, 1.0)
        u = np.round(K[0, 0] * X[:, 0] / zz + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * X[:, 1] / zz + K[1, 2]).astype(int)
        inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        wt = 1.0 / np.maximum(np.linalg.norm(P - C, axis=1), 1e-3)
        for p in np.where(inb)[0]:
            if arm is not None and arm[v[p], u[p]]: continue
            if not SOLID and vox_at[v[p], u[p]] != p: continue          # ★z-buffer:只讓該視角真正看得到的表面 voxel 投票(SOLID時不過濾)
            if ispar is None:                                           # baseline:第一張含此像素的遮罩投完整一票即 break(原行為)
                for mi, m in enumerate(ms):
                    if (vi, mi) in mlabel and m[v[p], u[p]]:
                        votes[mlabel[(vi, mi)]][p] += wt[p]
                        vmask[int(p)][vname].add(names[mi]); break
            else:                                                       # 巢狀降權:含此像素的每張遮罩都投,父遮罩×α(不 break,子遮罩也能投)
                for mi, m in enumerate(ms):
                    if (vi, mi) in mlabel and m[v[p], u[p]]:
                        votes[mlabel[(vi, mi)]][p] += wt[p] * (NESTED_ALPHA if ispar[mi] else 1.0)
                        vmask[int(p)][vname].add(names[mi])
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
                gid = gl[gi]                          # 這個 instance 所屬的群
                md = defaultdict(set)                 # ★來源遮罩=這些 hull voxel 實際投票落在的、且屬該群的遮罩
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
        return   # 續跑保護:已算過就跳過(FORCE=1 強制重算)
    labels, gm, vs, inst_masks, mask_cluster = semantic_cluster(sc, n_views, sem_thr)
    out = OUT_ROOT / sc; out.mkdir(parents=True, exist_ok=True)
    # 記錄「這批 instance 怎麼來的」— 用哪個 hull/遮罩來源、特徵、門檻、視角數
    meta = {"script": "voxel_sem_cluster_surf.py", "surface": not SOLID, "solid": SOLID, "zbuffer": not SOLID,
            "built": _dt.datetime.now().isoformat(timespec="seconds"),
            "hull_root": HULL_ROOT.name, "sam_root": SAM_ROOT.name, "captures_root": CAPTURES.name,
            "arm_root": ARM.name, "feat_file": FEAT_FILE, "debias": DEBIAS,
            "bg_file": (BG_FILE or "clip_F_BG"), "sem_thr": sem_thr, "n_views": n_views, "min_vox": MIN_VOX,
            "nested_alpha": NESTED_ALPHA, "nested_thr": NESTED_THR}
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
