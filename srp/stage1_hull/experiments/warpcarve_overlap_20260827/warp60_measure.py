#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""60 場:量原始 vs 雕後 的同物體/堆疊物群間 voxel 重疊率,彙總。"""
import sys, os, json
import numpy as np
sys.path.insert(0, "srp/io"); sys.path.insert(0, "srp/stage2_instances")
os.environ.setdefault("ARM_MASK_ROOT", os.path.abspath("data/eval/srp_arm_masks"))
import camera as cam, masks as MK, viewpoints as VP, labels as L
import cg_associate as CG
from voxel_sem_cluster_donut import donut_masks
from collections import defaultdict, Counter
from pathlib import Path
from pycocotools import mask as cocomask

MV2 = Path("data/eval/mobilesamv2_fast"); CAP = Path("data/captures_fast")
scenes = Path("/tmp/claude-1000/-home-cho-webots-program/338e7fee-7000-4d61-a82a-92a2c5d8b46c/scratchpad/warp60_scenes.txt").read_text().split()

def gt(sc):
    ann = json.loads((L.label_dir(sc) / "actual" / "annotations.json").read_text())
    cat = {c["id"]: c["name"] for c in ann["categories"]}
    id2v = {im["id"]: Path(im["file_name"]).stem for im in ann["images"]}
    modal = defaultdict(dict)
    for a in ann["annotations"]:
        if a["category_id"] == 1: continue
        s = a["segmentation"]; c = s["counts"].encode() if isinstance(s["counts"], str) else s["counts"]
        modal[id2v[a["image_id"]]][cat[a["category_id"]]] = cocomask.decode({"size": s["size"], "counts": c}).astype(bool)
    pos = {o["name"]: np.array(o["position_m"][:2]) for o in ann["images"][0]["objects"]}
    return modal, pos

def groups(sc, hroot, sroot, modal):
    ij = Path(f"data/eval/{sroot}") / sc / "instances.json"
    hp = Path(f"data/eval/{hroot}") / sc / "hull.npz"
    if not (ij.is_file() and hp.is_file()): return {}
    mc = json.loads(ij.read_text()).get("mask_clusters", {})
    z = np.load(hp); surf = z["surface"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
    vox = np.array(np.nonzero(surf)).T; Pw = gm + (vox + 0.5) * vs
    sdir = CAP / f"multi_{sc.split('_')[0]}" / sc; gvox = defaultdict(set); gobj = defaultdict(Counter)
    for vn in sorted(VP.selected_view_names(12)):
        vd = MV2 / sc / vn; pf = sdir / f"{vn}_pose.json"
        if not (vd.is_dir() and pf.is_file()): continue
        km = MK.kept_object_masks(vd); names = [n for _, n in km]; ms0 = [m for m, _ in km]
        if not ms0: continue
        ms = donut_masks(ms0); C, Rb = cam.load_pose(pf); H, Wd = ms0[0].shape
        va = CG.zbuffer_visible(Pw, C, Rb, Wd, H, vs).reshape(H, Wd)
        vmc = defaultdict(Counter)
        for mi, m in enumerate(ms):
            ys, xs = np.where(m); vv = va[ys, xs]
            for v in vv[vv >= 0]: vmc[int(v)][mi] += 1
        mvx = defaultdict(set)
        for v, cnt in vmc.items(): mvx[cnt.most_common(1)[0][0]].add(v)
        cl = mc.get(vn, {}); gm_v = modal.get(vn, {})
        for mi, nm in enumerate(names):
            cid = cl.get(nm)
            if cid is not None: gvox[cid] |= mvx.get(mi, set())
            m = ms[mi]; a = int(m.sum()); best = None; bc = 0.5
            for on, om in gm_v.items():
                cov = (m & om).sum() / max(a, 1)
                if cov > bc: bc = cov; best = on
            if best and cid is not None: gobj[cid][best] += 1
    return {c: (gvox[c], gobj[c].most_common(1)[0][0]) for c in gvox if gobj[c] and gvox[c]}

agg = {"原始": {"same": [], "stk": []}, "雕後": {"same": [], "stk": []}}
per_scene = []
for sc in scenes:
    modal, pos = gt(sc)
    row = {"scene": sc}
    for tag, hr, sr in [("原始", "srp_hull_warp_orig", "semdonut_warp_orig"),
                        ("雕後", "srp_hull_warp_carved", "semdonut_warp_carved")]:
        gd = groups(sc, hr, sr, modal); cids = list(gd)
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                (vi, oi), (vj, oj) = gd[cids[i]], gd[cids[j]]
                jac = len(vi & vj) / max(len(vi | vj), 1)
                if oi == oj: agg[tag]["same"].append(jac)
                else:
                    hd = np.linalg.norm(pos[oi] - pos[oj]) if oi in pos and oj in pos else 9
                    if hd < 0.05: agg[tag]["stk"].append(jac)
    per_scene.append(row)

print(f"\n===== 60 場彙總({len(scenes)}場)同物體/堆疊物群間 voxel 重疊率 =====")
print(f"{'hull':>6} {'同物體群對(該合)':>30} {'堆疊物群對(該分)':>30}")
for tag in ("原始", "雕後"):
    sm = agg[tag]["same"]; st = agg[tag]["stk"]
    ss = f"n={len(sm)} 中位{np.median(sm):.3f} 均{np.mean(sm):.3f}" if sm else "n=0"
    ks = f"n={len(st)} 中位{np.median(st):.3f} 均{np.mean(st):.3f}" if st else "n=0"
    print(f"{tag:>6} {ss:>30} {ks:>30}")
o_stk = np.median(agg["原始"]["stk"]); c_stk = np.median(agg["雕後"]["stk"])
o_sm = np.median(agg["原始"]["same"]); c_sm = np.median(agg["雕後"]["same"])
print(f"\n堆疊物重疊(該分,越低越好): {o_stk:.3f} → {c_stk:.3f} (降 {(o_stk-c_stk)/o_stk*100:.0f}%)")
print(f"同物體重疊(該合): {o_sm:.3f} → {c_sm:.3f}")
print(f"堆疊/同物 比值(>1=方向反,分不開): 原始 {o_stk/max(o_sm,1e-9):.1f} → 雕後 {c_stk/max(c_sm,1e-9):.1f}")
