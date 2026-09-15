#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""voxel_group_geo.py — 以【語意群 voxel】為單位進行幾何關聯:
 ① 語意分群(surf_donut mask_clusters) → 每 voxel 跨視角多數決認領語意群
 ② DISPUTE=keep:爭議voxel(多群認領)用多數決歸最常認領群;DISPUTE=drop:排除爭議voxel
 ③ 每語意群 voxel 聯集 → 群間 voxel 重疊(Jaccard)>JAC 則合併(union-find,幾何關聯)
 ④ 合併後每組=一個 instance(語意群為單位,不切 3D 連通)
輸出 data/eval/<OUT_ROOT>/<scene>/instances.{npz,json}。
用法: ./voxel_group_geo.py [scene|group...] [--jac 0.05]  env: DISPUTE(keep/drop) OUT_ROOT MIN_VOX
"""
import sys, os, json, argparse, glob
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "srp" / "io"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "srp" / "stage2_instances"))
import camera as cam, masks as MK, viewpoints as VP
import cg_associate as CG
from voxel_sem_cluster_donut import donut_masks
REPO = Path(__file__).resolve().parents[2]
HB = REPO / "data/eval/srp_hull_mv2_v12_am1"; SAM = REPO / "data/eval/mobilesamv2_fast"
CAP = REPO / "data/captures_fast"; SEM = REPO / "data/eval/srp_hull_semcluster_surf_donut"
DISPUTE = os.environ.get("DISPUTE", "keep")            # keep=爭議多數決歸屬; drop=排除爭議
MIN_VOX = int(os.environ.get("MIN_VOX", "0"))


def build(scene, jac):
    ij = SEM / scene / "instances.json"
    if not ij.is_file(): return None
    mc = json.loads(ij.read_text()).get("mask_clusters", {})
    z = np.load(HB / scene / "hull.npz"); surf = z["surface"]; gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = surf.shape
    voxarr = np.array(np.nonzero(surf)).T; P = gm + (voxarr + 0.5) * vs
    group = scene.split("_")[0]; sdir = CAP / f"multi_{group}" / scene
    vox_groups = defaultdict(Counter)                  # voxel → {語意群: 認領次數}
    for vn in sorted(VP.selected_view_names(12)):
        vd = SAM / scene / vn; pf = sdir / f"{vn}_pose.json"
        if not (vd.is_dir() and pf.is_file()): continue
        km = MK.kept_object_masks(vd); names = [n for _, n in km]; ms0 = [m for m, _ in km]
        if not ms0: continue
        ms = donut_masks(ms0)
        C, Rb = cam.load_pose(pf); H, W = ms0[0].shape
        vox_at = CG.zbuffer_visible(P, C, Rb, W, H, vs).reshape(H, W)
        vmc = defaultdict(Counter)
        for mi, m in enumerate(ms):
            ys, xs = np.where(m); vv = vox_at[ys, xs]
            for v in vv[vv >= 0]: vmc[int(v)][mi] += 1
        cl = mc.get(vn, {})
        for v, cnt in vmc.items():
            cid = cl.get(names[cnt.most_common(1)[0][0]])
            if cid is not None: vox_groups[v][cid] += 1
    # 每語意群 voxel 集合
    gvox = defaultdict(set)
    for v, gc in vox_groups.items():
        if DISPUTE == "drop" and len(gc) > 1: continue  # 排除爭議voxel
        gvox[gc.most_common(1)[0][0]].add(v)            # keep:多數決歸屬
    cids = [c for c in gvox if gvox[c]]
    # 群間 voxel 重疊(Jaccard)>jac → 合併(幾何關聯,union-find)
    n = len(cids); parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            a, b = gvox[cids[i]], gvox[cids[j]]
            if len(a & b) / max(len(a | b), 1) > jac: parent[find(i)] = find(j)
    merged = defaultdict(set)
    for i in range(n): merged[find(i)] |= gvox[cids[i]]
    labels = np.zeros(shape, np.int32); newid = 0
    for root, vset in sorted(merged.items(), key=lambda x: -len(x[1])):
        if len(vset) < MIN_VOX: continue
        li = np.fromiter(vset, np.int64); newid += 1
        labels[tuple(voxarr[li].T)] = newid
    return labels, gm, vs, newid


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("targets", nargs="+"); ap.add_argument("--jac", type=float, default=0.05)
    a = ap.parse_args()
    out_root = os.environ.get("OUT_ROOT", f"srp_hull_grpgeo_{DISPUTE}")
    scenes = []
    for t in a.targets:
        if "scene" in t: scenes.append(t)
        else: scenes += [Path(p).name for p in glob.glob(str(SEM / f"{t}_scene*"))]
    for sc in sorted(set(scenes)):
        try:
            r = build(sc, a.jac)
            if r is None: print(f"[skip] {sc}"); continue
            labels, gm, vs, ninst = r
            out = REPO / "data/eval" / out_root / sc; out.mkdir(parents=True, exist_ok=True)
            meta = {"script": "voxel_group_geo", "dispute": DISPUTE, "jac": a.jac, "unit": "semantic_group_voxel", "min_vox": MIN_VOX}
            np.savez_compressed(out / "instances.npz", labels=labels, grid_min=gm, voxel_size=vs, build_meta=json.dumps(meta, ensure_ascii=False))
            (out / "instances.json").write_text(json.dumps({"scene": sc, "n_instances": ninst, "meta": meta}, ensure_ascii=False))
            print(f"[{sc}] {DISPUTE} jac={a.jac} → {ninst} instance", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")


if __name__ == "__main__":
    main()
