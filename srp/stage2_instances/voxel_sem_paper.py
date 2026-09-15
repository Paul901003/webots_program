#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""voxel_sem_paper.py — 貼合論文版:CLIP 命名 + 可見性(z-buffer) + 三權重(信心cos×中心距離×相機距離)
→ 加權多數決 → 3D 連通。內部無票 voxel 用最近有票 voxel 類別填充。
輸出 data/eval/srp_hull_sempaper/<scene>/instances.npz。
可視化: SRP_VIZ_ARGS="<scene> 1 srp_hull_sempaper" webots worlds/hull_viz.wbt
用法: ./voxel_sem_paper.py [scene|group|(空=全部)] [--n-views 12]
env: SAM_ROOT HULL_ROOT(srp_hull_v12) CAPTURES_ROOT ARM_MASK_ROOT
"""
import argparse, os, sys, json, glob
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "srp" / "io"))
import numpy as np, cv2
from pathlib import Path
from collections import defaultdict
from scipy import ndimage
from scipy.spatial import cKDTree
import camera as cam, masks as MK, mask_clip_cluster as MC, viewpoints as VP

REPO = Path(__file__).resolve().parents[2]
CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures_fast")))
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "sam_only_fast")))
HULL_ROOT = Path(os.environ.get("HULL_ROOT", str(REPO / "data" / "eval" / "srp_hull_v12")))
ARM = Path(os.environ.get("ARM_MASK_ROOT", str(REPO / "data" / "eval" / "srp_arm_masks")))
OUT_ROOT = REPO / "data" / "eval" / os.environ.get("OUT_ROOT", "srp_hull_sempaper")
MIN_VOX = 50
Z = np.load(REPO / "data" / "eval" / "clip_text_feats.npz", allow_pickle=True)
TPH = [str(x) for x in Z["phrases"]]
TF = Z["feats"].astype(np.float32); TF = TF / np.linalg.norm(TF, axis=1, keepdims=True)


def semantic_paper(sc, n_views):
    group = sc.split("_")[0]; sdir = CAPTURES / f"multi_{group}" / sc
    z = np.load(HULL_ROOT / sc / "hull.npz")
    occ = z["occupancy"]; gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = occ.shape
    vox = np.array(np.nonzero(occ)).T; P = gm + (vox + 0.5) * vs; M = len(vox)
    want = set(VP.selected_view_names(n_views)) if n_views else None
    votes = defaultdict(lambda: np.zeros(M))
    vmask = defaultdict(lambda: defaultdict(set))   # voxel p → {view: set(遮罩檔)}(可見投票落到的遮罩)
    for vd in sorted((SAM_ROOT / sc).glob("view_*")):
        if want is not None and vd.name not in want: continue
        pf = sdir / f"{vd.name}_pose.json"
        if not pf.is_file(): continue
        km = MK.kept_object_masks(vd); ms = [m for m, _ in km]; names = [nm for _, nm in km]
        if not ms: continue
        C, Rb = cam.load_pose(pf); Rwc, t = cam.pose_to_w2c(C, Rb)
        H, W = ms[0].shape; K = cam.intrinsics(W, H); cx, cy = W / 2, H / 2
        rgb = cv2.cvtColor(cv2.imread(str(sdir / f"{vd.name}.png")), cv2.COLOR_BGR2RGB)
        ap = ARM / sc / f"{vd.name}_arm.png"
        arm = (cv2.imread(str(ap), 0) > 127) if ap.is_file() else None
        mph = []
        for f in MK.feats_list(vd, names):
            if f is None: mph.append(None); continue
            sim = TF @ f; j = int(sim.argmax()); mph.append((TPH[j], float(sim[j])))
        X = P @ Rwc.T + t; zc = X[:, 2]; ok = zc > 1e-9; zz = np.where(ok, zc, 1.0)
        u = np.round(K[0, 0] * X[:, 0] / zz + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * X[:, 1] / zz + K[1, 2]).astype(int)
        inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.where(inb)[0]
        depth = np.full((H, W), np.inf); owner = np.full((H, W), -1, int)
        for p in idx[np.argsort(zc[idx])]:
            if zc[p] < depth[v[p], u[p]]:
                depth[v[p], u[p]] = zc[p]; owner[v[p], u[p]] = p
        vis = np.unique(owner[owner >= 0])
        dist = np.linalg.norm(P - C, axis=1)
        for p in vis:
            up, vp = u[p], v[p]
            if arm is not None and arm[vp, up]: continue
            for mi, m in enumerate(ms):
                if mph[mi] is not None and m[vp, up]:
                    ph, S = mph[mi]
                    cw = 1.0 / max(np.hypot(up - cx, vp - cy), 1.0)
                    dw = 1.0 / max(dist[p], 1e-3)
                    votes[ph][p] += S * cw * dw
                    vmask[int(p)][vd.name].add(names[mi]); break
    phrases = list(votes.keys())
    labels = np.zeros(shape, np.int32); inst_masks = {}
    if not phrases:
        return labels, gm, vs, inst_masks
    Vt = np.stack([votes[p] for p in phrases], 1); voted = Vt.max(1) > 0
    assign = np.full(M, -1, int); assign[voted] = Vt[voted].argmax(1)
    if voted.any() and (~voted).any():
        _, nn = cKDTree(vox[voted]).query(vox[~voted]); assign[~voted] = assign[voted][nn]
    st = ndimage.generate_binary_structure(3, 3); nid = 0
    for pi in range(len(phrases)):
        sel = assign == pi
        if not sel.any(): continue
        sel_p = np.where(sel)[0]
        m3 = np.zeros(shape, bool); m3[tuple(vox[sel_p].T)] = True
        lab, n = ndimage.label(m3, st)
        labs_at = lab[tuple(vox[sel_p].T)]
        for c in range(1, n + 1):
            comp_p = sel_p[labs_at == c]
            if len(comp_p) < MIN_VOX: continue
            nid += 1; labels[lab == c] = nid
            md = defaultdict(set)            # 只有「可見有票」voxel 有遮罩;填充的無票 voxel 自然略過
            for p in comp_p:
                for vw, fs in vmask.get(int(p), {}).items():
                    md[vw] |= fs
            inst_masks[nid] = {vw: sorted(fs) for vw, fs in sorted(md.items())}
    return labels, gm, vs, inst_masks


def process(sc, n_views):
    if not (HULL_ROOT / sc / "hull.npz").is_file():
        print(f"[skip] {sc}"); return
    labels, gm, vs, inst_masks = semantic_paper(sc, n_views)
    out = OUT_ROOT / sc; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "instances.npz", labels=labels, grid_min=gm, voxel_size=vs)
    insts = [{"instance": i, "n_vox": int((labels == i).sum()), "masks": inst_masks.get(i, {})}
             for i in range(1, int(labels.max()) + 1) if (labels == i).any()]
    (out / "instances.json").write_text(json.dumps(
        {"scene": sc, "voxel": vs, "n_instances": len(insts), "instances": insts}, indent=2, ensure_ascii=False))
    print(f"[{sc}] 論文版 → {len(insts)} instance", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--n-views", type=int, default=12, dest="n_views")
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
        try: process(sc, args.n_views)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")


if __name__ == "__main__":
    main()
