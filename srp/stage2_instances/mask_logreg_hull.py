#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""節點=各遮罩,用邏輯迴歸最佳分界連邊→連通元件→instance。
分界(60場學得): 44.10·voxel重疊 − 3.32·語意距離 + 1.45 > 0 → 連邊(該同物體)。
每連通元件遮罩投票鋪表面voxel(多數決)→hull。
用法: ./mask_logreg_hull.py [scene|group...] env: OUT_ROOT W_OV W_SEM BIAS MIN_VOX
"""
import sys, os, json, argparse, glob
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "srp" / "io"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "srp" / "stage2_instances"))
import camera as cam, masks as MK, viewpoints as VP
import cg_associate as CG, mask_clip_cluster as MC
from voxel_sem_cluster_donut import donut_masks, donut_feats
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from scipy import ndimage
REPO = Path(__file__).resolve().parents[2]
HB = REPO/"data/eval/srp_hull_mv2_v12_am1"; SAM = REPO/"data/eval/mobilesamv2_fast"; CAP = REPO/"data/captures_fast"
BG = MC.F_BG.astype(np.float64)
W_OV = float(os.environ.get("W_OV", "44.10"))     # 邏輯迴歸權重(voxel重疊)
W_SEM = float(os.environ.get("W_SEM", "-3.32"))   # (語意距離)
BIAS = float(os.environ.get("BIAS", "1.45"))
MIN_VOX = int(os.environ.get("MIN_VOX", "0"))
def deb(f): f = f.astype(np.float64); f = f-(f@BG)*BG; return f/(np.linalg.norm(f)+1e-9)


def build(scene):
    z = np.load(HB/scene/"hull.npz"); surf = z["surface"]; gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = surf.shape
    voxarr = np.array(np.nonzero(surf)).T; P = gm+(voxarr+0.5)*vs
    group = scene.split("_")[0]; sdir = CAP/f"multi_{group}"/scene
    md = []   # (voxel集合, 去偏特徵)
    for vn in sorted(VP.selected_view_names(12)):
        vd = SAM/scene/vn; pf = sdir/f"{vn}_pose.json"
        if not (vd.is_dir() and pf.is_file()): continue
        km = MK.kept_object_masks(vd); names = [n for _, n in km]; ms0 = [m for m, _ in km]
        if not ms0: continue
        ms = donut_masks(ms0); rgb = cv2.cvtColor(cv2.imread(str(sdir/f"{vn}.png")), cv2.COLOR_BGR2RGB)
        fmap = donut_feats(vd, rgb, ms, names)
        C, Rb = cam.load_pose(pf); H, W = ms0[0].shape; vox_at = CG.zbuffer_visible(P, C, Rb, W, H, vs).reshape(H, W)
        vmc = defaultdict(Counter)
        for mi, m in enumerate(ms):
            ys, xs = np.where(m); vv = vox_at[ys, xs]
            for v in vv[vv >= 0]: vmc[int(v)][mi] += 1
        mvx = defaultdict(set)
        for v, cnt in vmc.items(): mvx[cnt.most_common(1)[0][0]].add(v)
        for mi, nm in enumerate(names):
            hit = frozenset(mvx.get(mi, set())); f = fmap.get(nm)
            if hit and f is not None: md.append((hit, deb(f)))
    n = len(md)
    labels = np.zeros(shape, np.int32)
    if n < 2: return labels, gm, vs, 0
    F = np.stack([m[1] for m in md]); sem = 1-(F@F.T)
    ov = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            jac = len(md[i][0] & md[j][0])/max(len(md[i][0] | md[j][0]), 1); ov[i, j] = ov[j, i] = jac
    edge = (W_OV*ov + W_SEM*sem + BIAS) > 0        # 邏輯迴歸決策:>0 連邊
    np.fill_diagonal(edge, False)
    if edge.sum() > 0:
        r, c = np.where(edge); A = csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
        _, comp = connected_components(A, directed=False)
    else:
        comp = np.arange(n)
    # 每連通元件遮罩投票鋪voxel
    votes = defaultdict(lambda: np.zeros(len(voxarr)))
    lidx = {tuple(v): i for i, v in enumerate(voxarr.tolist())}
    for (hit, f), g in zip(md, comp):
        for v in hit: votes[int(g)][v] += 1
    gl = list(votes.keys()); Vt = np.stack([votes[g] for g in gl], 1); assign = Vt.argmax(1); has = Vt.max(1) > 0
    st = ndimage.generate_binary_structure(3, 3); newid = 0
    for gi in range(len(gl)):
        sel = has & (assign == gi)
        if not sel.any(): continue
        sp = np.where(sel)[0]; m3 = np.zeros(shape, bool); m3[tuple(voxarr[sp].T)] = True
        lb, ncc = ndimage.label(m3, st); labs = lb[tuple(voxarr[sp].T)]
        for c in range(1, ncc+1):
            cp = sp[labs == c]
            if len(cp) < MIN_VOX: continue
            newid += 1; labels[lb == c] = newid
    return labels, gm, vs, newid


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("targets", nargs="+"); a = ap.parse_args()
    out_root = os.environ.get("OUT_ROOT", "srp_hull_logreg")
    scenes = []
    for t in a.targets:
        if "scene" in t: scenes.append(t)
        else: scenes += [Path(p).name for p in glob.glob(str(SAM/f"{t}_scene*"))]
    for sc in sorted(set(scenes)):
        try:
            labels, gm, vs, ninst = build(sc)
            out = REPO/"data/eval"/out_root/sc; out.mkdir(parents=True, exist_ok=True)
            meta = {"script": "mask_logreg_hull", "w_ov": W_OV, "w_sem": W_SEM, "bias": BIAS, "min_vox": MIN_VOX}
            np.savez_compressed(out/"instances.npz", labels=labels, grid_min=gm, voxel_size=vs, build_meta=json.dumps(meta, ensure_ascii=False))
            (out/"instances.json").write_text(json.dumps({"scene": sc, "n_instances": ninst, "meta": meta}, ensure_ascii=False))
            print(f"[{sc}] → {ninst} instance", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")


if __name__ == "__main__":
    main()
