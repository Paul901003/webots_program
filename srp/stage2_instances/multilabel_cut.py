#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""multilabel_cut.py — 群 voxel 集重疊合併(每群自己的 silhouette 佔據)。

核心(使用者提):每個語意群各自對應一組 voxel(用該群出現的視角做 silhouette 交集,=該群迷你 hull);
量群對 voxel 集重疊,重疊高 → 同物體該合。不是先給 voxel 貼標籤再回頭數。

修掉舊版 bug:frac 分母用「全 12 視角」,少視角群(只在 4 視角有遮罩)被壓掉;正確分母=該群自己出現的視角。

群 g:出現視角 P_g=mask_clusters 有指派到 g 的視角;
frac_g(voxel)=(P_g 中 voxel in-bounds 且落 g 遮罩聯集內的視角數)/(P_g 中 voxel in-bounds 的視角數);
S_g={voxel:frac_g≥θ}(θ=0.8,該群 silhouette 交集,容忍少量漏)。
群對重疊 IoU=|S_A∩S_B|/|S_A∪S_B|、含入率=|S_A∩S_B|/min(|S_A|,|S_B|);含入率≥σ 的群 union-find 合成一實例。

輸入:hull(srp_hull_mv2_v12_am1_photo)、群指派(srp_hull_semcluster_surf_am1photo/instances.json mask_clusters)、
      原遮罩(mobilesamv2_fast,不挖洞)、手臂剪影(srp_arm_masks)、位姿(captures_fast)、視角(selected_view_names 12)。
輸出:data/eval/solidcut_ml_am1photo/<scene>/{instances.npz(合併labels+occupancy+build_meta), grpov.json}。
env:SAM_ROOT CAPTURES_ROOT。用法: ./multilabel_cut.py stack4_scene0001 [--theta 0.8]
"""
import os, sys, json, argparse
import numpy as np, cv2
from collections import defaultdict
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/"srp/io")); sys.path.insert(0, str(REPO/"srp/stage2_instances"))
import camera as cam, masks as MK, viewpoints as VP
EVAL = REPO/"data"/"eval"
MV2 = EVAL/"mobilesamv2_fast"; CAP = REPO/"data"/"captures_fast"; ARM = EVAL/"srp_arm_masks"
INST = "srp_hull_semcluster_surf_am1photo"; HULL = "srp_hull_mv2_v12_am1_photo"; OUT = "solidcut_ml_am1photo"


def uf_merge(n, edges):
    p = list(range(n))
    def f(x):
        while p[x] != x: p[x] = p[p[x]]; x = p[x]
        return x
    for a, b in edges: p[f(a)] = f(b)
    return [f(i) for i in range(n)]


def run(scene, theta, sigmas):
    z = np.load(EVAL/HULL/scene/"hull.npz")
    occ = z["occupancy"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
    vidx = np.argwhere(occ); Pw = gm+(vidx+0.5)*vs; N = len(vidx)
    mc = json.loads((EVAL/INST/scene/"instances.json").read_text()).get("mask_clusters", {})
    sdir = CAP/f"multi_{scene.split('_')[0]}"/scene
    gids = set()
    hit = defaultdict(lambda: np.zeros(N, np.int16))      # gid -> voxel 落在 g 內的視角數(僅算 g 出現視角)
    pres_inb = defaultdict(lambda: np.zeros(N, np.int16)) # gid -> voxel 在「g 出現視角」中 in-bounds 的視角數
    for vn in sorted(VP.selected_view_names(12)):
        vd = MV2/scene/vn; pf = sdir/f"{vn}_pose.json"
        if not (vd.is_dir() and pf.is_file()): continue
        km = MK.kept_object_masks(vd); names = [n for _, n in km]; ms0 = [m for m, _ in km]
        ap = ARM/scene/f"{vn}_arm.png"; arm = (cv2.imread(str(ap), 0) > 127) if ap.is_file() else None
        if arm is not None:
            keep = [k for k, m in enumerate(ms0) if (m & arm).sum()/max(int(m.sum()), 1) < 0.5]
            names = [names[k] for k in keep]; ms0 = [ms0[k] for k in keep]
        if not ms0: continue
        H, W = ms0[0].shape
        cl = mc.get(vn, {})
        gsil = defaultdict(lambda: np.zeros((H, W), bool))  # gid -> 該視角群遮罩聯集
        for k, nm in enumerate(names):
            gid = cl.get(nm)
            if gid is not None: gsil[gid] |= ms0[k]; gids.add(gid)
        C, Rb = cam.load_pose(pf); Rwc, t = cam.pose_to_w2c(C, Rb); K = cam.intrinsics(W, H)
        X = Pw@Rwc.T+t; zc = X[:, 2]; ok = zc > 1e-6; zz = np.where(ok, zc, 1.0)
        px = np.round(K[0, 0]*X[:, 0]/zz+K[0, 2]).astype(int); py = np.round(K[1, 1]*X[:, 1]/zz+K[1, 2]).astype(int)
        inb = ok & (px >= 0) & (px < W) & (py >= 0) & (py < H); ii = np.where(inb)[0]
        yy = py[ii]; xx = px[ii]
        for gid, sil in gsil.items():                       # 這視角有出現的群才計(分母=g 出現視角)
            pres_inb[gid][ii] += 1
            hit[gid][ii] += sil[yy, xx]
    gids = sorted(gids)
    # 每群自己的 voxel 集 S_g:在 g 出現視角中 ≥θ 落在 g 內(silhouette 交集,容忍漏)
    Sg = {}
    for g in gids:
        den = np.maximum(pres_inb[g], 1)
        Sg[g] = (pres_inb[g] > 0) & (hit[g]/den >= theta)
    cnt = {g: int(Sg[g].sum()) for g in gids}
    # 群對重疊:IoU 與 含入率(inter/min)
    ov = {}
    for i, a in enumerate(gids):
        for b in gids[i+1:]:
            if cnt[a] == 0 or cnt[b] == 0: continue
            inter = int((Sg[a] & Sg[b]).sum())
            if inter == 0: continue
            uni = cnt[a]+cnt[b]-inter
            ov[(a, b)] = (inter, inter/uni, inter/min(cnt[a], cnt[b]))
    idx = {g: i for i, g in enumerate(gids)}
    merged_counts = {}; chosen_labels = None
    mid = sigmas[len(sigmas)//2]
    for sg in sigmas:
        edges = [(idx[a], idx[b]) for (a, b), (_, _, cr) in ov.items() if cr >= sg]
        comp = uf_merge(len(gids), edges)
        merged_counts[sg] = len(set(comp))
        if abs(sg-mid) < 1e-9:
            g2c = {g: comp[idx[g]]+1 for g in gids}
            labels = np.zeros(occ.shape, np.int32)
            best = np.full(N, -1, int); bestv = np.zeros(N)  # voxel→frac 最高的群所屬元件(僅視覺化)
            for g in gids:
                den = np.maximum(pres_inb[g], 1); r = np.where(pres_inb[g] > 0, hit[g]/den, 0.0)
                sel = r > bestv; best[sel] = g2c[g]; bestv[sel] = r[sel]
            pos = best > 0
            labels[vidx[pos, 0], vidx[pos, 1], vidx[pos, 2]] = best[pos]
            chosen_labels = labels
    d = EVAL/OUT/scene; d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d/"instances.npz", labels=chosen_labels, occupancy=occ, grid_min=gm,
                        voxel_size=np.float64(vs),
                        build_meta=json.dumps({"src": "群voxel集重疊合併", "hull": HULL, "theta": theta,
                                               "sigma_for_labels": mid, "ngroups": len(gids)}))
    (d/"grpov.json").write_text(json.dumps({
        "ngroups": len(gids), "theta": theta, "group_vox": cnt,
        "overlap": [{"a": a, "b": b, "inter": iv, "iou": round(io, 3), "contain": round(cr, 3)}
                    for (a, b), (iv, io, cr) in sorted(ov.items(), key=lambda kv: -kv[1][2])],
        "merged_count_by_sigma": {str(s): merged_counts[s] for s in sigmas}}, ensure_ascii=False, indent=1))
    return len(gids), ov, merged_counts, cnt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--theta", type=float, default=0.8)
    a = ap.parse_args()
    sigmas = [0.3, 0.5, 0.7]
    for sc in a.scenes:
        ng, ov, mcnt, cnt = run(sc, a.theta, sigmas)
        print(f"\n{sc}: {ng} 群 | 群 voxel 數中位 {int(np.median(list(cnt.values()))) if cnt else 0}")
        top = sorted(ov.items(), key=lambda kv: -kv[1][2])[:6]
        print("  top 群重疊(a,b: iou/含入):", ", ".join(f"{a}-{b}:{io:.2f}/{cr:.2f}" for (a, b), (iv, io, cr) in top))
        print("  合併後群數 @σ(含入率):", ", ".join(f"{s_}→{mcnt[s_]}" for s_ in sigmas))
