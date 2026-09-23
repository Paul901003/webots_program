"""物理正確 eval — 四條線在同一次、各用自己 hull 的可見表面當宇宙、mesh-3D containment 當 GT、Hungarian。
複製自上一 session scratchpad/eval_solid_final.py,唯一差別:每條線用「自己的 hull」算可見表面(vis)與 surface voxel,
不再全部寫死 am1_fp(這樣 am1 thin 線才是拿自己的可見表面比,基準一致)。GEX 與原 eval 完全相同。
#4/#5(soliddrop)與新的 soliderode 兩線一起重算,確保 apples-to-apples。
"""
import numpy as np, glob, sys
from pathlib import Path
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, "srp/stage2_instances"); sys.path.insert(0, "srp/io")
import eval_mesh as EM, camera as cam, viewpoints as VP
import cg_associate as CG
E = Path("data/eval"); CAP = Path("data/captures_fast")
GEX = {"024_bowl", "028_skillet_lid", "022_windex_bottle", "070-a_colored_wood_blocks", "062_dice"}
# (顯示名, div root, hull root)
LINES = [
    ("#4 fp  drop 胖am1fp", "srp_hull_divB_t50_reNNfpSd_am1fp", "srp_hull_mv2_v12_am1_fp"),
    ("#5 fp  drop 瘦am1  ", "srp_hull_divB_t50_reNNfpSd_am1",   "srp_hull_mv2_v12_am1"),
    ("#6 中心 drop 瘦am1 ", "srp_hull_divB_t50_reNNcSd_am1",     "srp_hull_mv2_v12_am1"),
    ("#8 fp  侵蝕 胖am1fp", "srp_hull_divB_t50_reNNfpSe_am1fp", "srp_hull_mv2_v12_am1_fp"),
    ("#8 fp  侵蝕 瘦am1  ", "srp_hull_divB_t50_reNNfpSe_am1",   "srp_hull_mv2_v12_am1"),
]


def solid_vis(sc, occ, gm, vs, shape, vox):
    ovox = np.argwhere(occ); oP = gm + (ovox + 0.5) * vs
    sg = np.full(shape, -1, np.int64); sg[tuple(vox.T)] = np.arange(len(vox))
    s_of_o = sg[tuple(ovox.T)]
    g = sc.split("_")[0]; vis = np.zeros(len(vox), bool)
    for vn in sorted(VP.selected_view_names(12)):
        pf = CAP / f"multi_{g}" / sc / f"{vn}_pose.json"
        if not pf.is_file(): continue
        C, Rb = cam.load_pose(pf); va = CG.zbuffer_visible(oP, C, Rb, 1280, 720, vs)
        si = s_of_o[va[va >= 0]]; vis[si[si >= 0]] = True
    return vis


# 場景集:用 #4 div 的產出(303 多物場)
scenes = sorted(Path(p).parent.name for p in glob.glob(str(E / "srp_hull_divB_t50_reNNfpSd_am1fp/*_scene*/instances.npz")))
agg = {k: {"h5": 0, "h7": 0, "tot": 0, "iou": 0.0, "miss": 0} for k, _, _ in LINES}
# 每條線用自己 hull 的宇宙 → cache 每 hull 的 (vis,vox,mlab)
hull_cache = {}
for sc in scenes:
    for name, root, hull in LINES:
        divf = E / root / sc / "instances.npz"
        if not divf.is_file():
            continue
        key = (sc, hull)
        if key not in hull_cache:
            z = np.load(E / hull / sc / "hull.npz")
            occ = z["occupancy"].astype(bool); surf = z["surface"].astype(bool)
            gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = surf.shape
            vox = np.argwhere(surf)
            vis = solid_vis(sc, occ, gm, vs, shape, vox)
            gt = EM.solid_mesh_occ(sc, gm, vs, shape); onames = [o for o in gt if o not in GEX]
            mlab = np.full(len(vox), -1, int)
            for oi, o in enumerate(onames):
                mlab[gt[o].astype(bool)[tuple(vox.T)]] = oi
            hull_cache[key] = (vox, vis, mlab, onames)
        vox, vis, mlab, onames = hull_cache[key]
        if not onames:
            continue
        lab = np.load(divf)["labels"]; vlab = lab[tuple(vox.T)]
        sel = vis & (mlab >= 0)
        gtm = [(mlab == oi) & sel for oi in range(len(onames))]
        ks = [k for k in np.unique(vlab) if k > 0]; predm = [(vlab == k) & sel for k in ks]
        Mx = np.zeros((len(gtm), len(predm)))
        for i, gg in enumerate(gtm):
            if gg.sum() == 0: continue
            for j, pp in enumerate(predm):
                it = int((gg & pp).sum())
                if it: Mx[i, j] = it / int((gg | pp).sum())
        ri, cj = linear_sum_assignment(-Mx); best = {i: Mx[i, j] for i, j in zip(ri, cj)}
        a = agg[name]
        for i in range(len(onames)):
            if gtm[i].sum() == 0: continue
            io = best.get(i, 0); a["tot"] += 1; a["iou"] += io
            if io >= 0.5: a["h5"] += 1
            if io >= 0.7: a["h7"] += 1

print(f"物理正確 eval(各線自己 hull 的可見表面當宇宙,mesh-GT,GEX 排除;{len(scenes)} 場)")
print(f"{'line':16s} {'found@0.5':>9} {'found@0.7':>9} {'mIoU':>6} {'obj數':>6}")
for name, _, _ in LINES:
    a = agg[name]; t = max(a['tot'], 1)
    print(f"{name:16s} {a['h5']/t:9.3f} {a['h7']/t:9.3f} {a['iou']/t:6.3f} {a['tot']:6d}")
