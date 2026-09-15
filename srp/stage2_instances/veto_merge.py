#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""veto_merge.py — 語意群 + 表面深度 veto 式合併(把 CLIP 過切的同物併回)。

思想:CLIP 只會過切、不會亂併 → 預設把「相鄰 instance 對」MERGE,除非任一 veto 命中就 KEEP-SEPARATE:
  ① 空間不連通:兩 instance voxel 3D 不相接 → 各自獨立(cups+cups 並排同深度)。
  ② z 高度分離:高度區間重疊比例 < zov_thr → 上下相疊(gelatin/foam)。
  ③ 深度牆:平面階梯 min_step(跨視角最小)> step_thr → 持續深度牆(並排有間隙異物)。
其餘 → MERGE(救 CLIP 過切)。用 min_step 讓自遮擋(只掠角跳、正面連續)不被當牆。

輸入:<inst-root>/<scene>/{instances.json(instances[].masks、instance==npz label), instances.npz(labels)}、
      hull(前表面深度)、mobilesamv2 遮罩、captures 位姿。GT mesh 評分(排 GLOBAL_EXCLUDE)。
輸出:<out-root>/<scene>/{instances.npz(合併labels+build_meta), merge_reasons.json}。並印 clean/lost/mixed/split。
env:SAM_ROOT CAPTURES_ROOT。用法: ./veto_merge.py <scenes...> [--out-root ...] [--step-thr 40] [--zov-thr 0.3]
"""
import sys, json, argparse
import numpy as np, cv2
from collections import defaultdict, Counter
from pathlib import Path
from scipy import ndimage
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/"srp/io")); sys.path.insert(0, str(REPO/"srp/stage2_instances"))
import camera as cam, masks as MK, viewpoints as VP
import eval_mesh as EM
EVAL = REPO/"data"/"eval"; MV2 = EVAL/"mobilesamv2_fast"; CAP = REPO/"data"/"captures_fast"
INST = "srp_hull_semcluster_surf_am1photo"; HULL = "srp_hull_mv2_v12_am1_photo"
GEX = {"skillet_lid", "windex_bottle", "colored_wood_blocks", "dice"}
KB = 7


def robust_plane(xs, ys, zs, it=2):
    keep = np.ones(len(zs), bool); coef = None
    for _ in range(it+1):
        if keep.sum() < 5: break
        A = np.c_[xs[keep], ys[keep], np.ones(keep.sum())]; coef, *_ = np.linalg.lstsq(A, zs[keep], rcond=None)
        r = zs-(coef[0]*xs+coef[1]*ys+coef[2]); m = np.median(r[keep]); mad = np.median(np.abs(r[keep]-m))+1e-6
        keep = np.abs(r-m) < 2.5*1.4826*mad
    return coef


def surface(o):
    s = np.zeros_like(o)
    s[1:-1, 1:-1, 1:-1] = o[1:-1, 1:-1, 1:-1] & ~(o[:-2, 1:-1, 1:-1] & o[2:, 1:-1, 1:-1] & o[1:-1, :-2, 1:-1] &
                                                  o[1:-1, 2:, 1:-1] & o[1:-1, 1:-1, :-2] & o[1:-1, 1:-1, 2:])
    return s


def zov(a, b):
    lo = max(a[0], b[0]); hi = min(a[1], b[1]); inter = max(0, hi-lo)
    return inter/max(min(a[1]-a[0], b[1]-b[0]), 1e-6)


def process(sc, step_thr, zov_thr, offs, ker):
    d = json.loads((EVAL/INST/sc/"instances.json").read_text())
    z = np.load(EVAL/INST/sc/"instances.npz", allow_pickle=True); labels = z["labels"].copy()
    zz = np.load(EVAL/HULL/sc/"hull.npz"); occ = zz["occupancy"]; gm = zz["grid_min"]; vs = float(zz["voxel_size"])
    surf = surface(occ); sidx = np.argwhere(surf); Pw = gm+(sidx+0.5)*vs
    insts = [it["instance"] for it in d["instances"]]
    imasks = {it["instance"]: it["masks"] for it in d["instances"]}     # {inst:{view:[names]}}
    # 每 instance voxel + z 區間
    ivox = {i: np.argwhere(labels == i) for i in insts}
    zint = {i: (ivox[i][:, 2].min(), ivox[i][:, 2].max()) if len(ivox[i]) else (0, 0) for i in insts}
    # ① 空間連通:局部 bbox 3D dilation 判相接(省記憶體)
    def connected(i, j):
        vi, vj = ivox[i], ivox[j]
        if len(vi) == 0 or len(vj) == 0: return False
        lo = np.minimum(vi.min(0), vj.min(0))-1; hi = np.maximum(vi.max(0), vj.max(0))+1
        sh = hi-lo+1
        a = np.zeros(sh, bool); a[tuple((vi-lo).T)] = True
        b = np.zeros(sh, bool); b[tuple((vj-lo).T)] = True
        return bool((ndimage.binary_dilation(a, iterations=1) & b).any())
    # 逐視角:instance 群遮罩 + 前表面深度 → 相鄰對 plane-step
    sdir = CAP/f"multi_{sc.split('_')[0]}"/sc
    steps = defaultdict(list); adj2d = set()
    for vn in sorted(VP.selected_view_names(12)):
        pf = sdir/f"{vn}_pose.json"
        if not pf.is_file(): continue
        km = MK.kept_object_masks(MV2/sc/vn); n2m = {n: m for m, n in km}
        gmask = {}
        for i in insts:
            names = imasks[i].get(vn, [])
            mm = None
            for nm in names:
                m = n2m.get(nm)
                if m is not None: mm = m if mm is None else (mm | m)
            if mm is not None and int(mm.sum()) > 30: gmask[i] = mm
        gids = list(gmask)
        if len(gids) < 2: continue
        H, W = gmask[gids[0]].shape
        C, Rb = cam.load_pose(pf); Rwc, t = cam.pose_to_w2c(C, Rb); K = cam.intrinsics(W, H)
        X = Pw@Rwc.T+t; zc = X[:, 2]; ok = zc > 1e-6; zzc = np.where(ok, zc, 1.0)
        px = np.round(K[0, 0]*X[:, 0]/zzc+K[0, 2]).astype(int); py = np.round(K[1, 1]*X[:, 1]/zzc+K[1, 2]).astype(int)
        inb = ok & (px >= 0) & (px < W) & (py >= 0) & (py < H); ii = np.where(inb)[0]
        dn = np.full((H, W), np.inf); yy = py[ii]; xx = px[ii]; zv = zc[ii]
        for dx, dy in offs:
            np.minimum.at(dn, (np.clip(yy+dy, 0, H-1), np.clip(xx+dx, 0, W-1)), zv)
        vld = np.isfinite(dn); mu = {i: gmask[i].astype(np.uint8) for i in gids}
        for ai in range(len(gids)):
            for bi in range(ai+1, len(gids)):
                a, b = gids[ai], gids[bi]
                if (cv2.dilate(mu[a], ker).astype(bool) & gmask[b]).sum() < 30: continue
                adj2d.add((min(a, b), max(a, b)))
                ao = gmask[a] & ~gmask[b] & vld; bo = gmask[b] & ~gmask[a] & vld
                if ao.sum() < 50 or bo.sum() < 50: continue
                bz = cv2.dilate(mu[a], ker).astype(bool) & cv2.dilate(mu[b], ker).astype(bool)
                byy, bxx = np.where(bz)
                if len(bxx) < 5: continue
                ay, ax = np.where(ao); by, bx = np.where(bo)
                pa = robust_plane(ax.astype(float), ay.astype(float), dn[ao]); pb = robust_plane(bx.astype(float), by.astype(float), dn[bo])
                if pa is None or pb is None: continue
                cx, cy = bxx.mean(), byy.mean()
                s = abs((pa[0]*cx+pa[1]*cy+pa[2])-(pb[0]*cx+pb[1]*cy+pb[2]))*1000
                steps[(min(a, b), max(a, b))].append(s)
    # veto 決策
    edges = []; reasons = {}
    for key in adj2d:
        i, j = key
        conn = connected(i, j)
        zo = zov(zint[i], zint[j])
        mn = min(steps[key]) if steps.get(key) else None
        v1 = not conn; v2 = (zo < zov_thr); v3 = (mn is not None and mn > step_thr)
        merge = not (v1 or v2 or v3)
        if merge: edges.append((i, j))
        reasons[f"{i}-{j}"] = {"merge": merge, "connected": conn, "zov": round(zo, 2),
                               "min_step": round(mn, 1) if mn is not None else None,
                               "veto": ("不連通" if v1 else "") + ("z分離" if v2 else "") + ("深度牆" if v3 else "") or "合"}
    # union-find
    par = {i: i for i in insts}
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in edges: par[find(a)] = find(b)
    remap = {}; new = np.zeros_like(labels)
    for i in insts:
        r = find(i); remap.setdefault(r, len(remap)+1)
    for i in insts:
        new[labels == i] = remap[find(i)]
    n_after = len(set(remap.values()))
    return new, occ, gm, vs, reasons, len(insts), n_after


def evaluate(sc, labels, gm, vs, shape):
    """回 (n_gt, clean, split, lost, mixed, n_inst_effective)。排 GLOBAL_EXCLUDE。"""
    gtocc = EM.solid_mesh_occ(sc, gm, vs, shape)
    gtocc = {k: v for k, v in gtocc.items() if k.split("_", 1)[-1] not in GEX}
    labs = [int(x) for x in np.unique(labels) if x > 0]
    inst_obj = {}
    for L in labs:
        m = (labels == L)
        ov = {k: int((m & v).sum()) for k, v in gtocc.items()}
        best = max(ov, key=ov.get) if ov else None
        inst_obj[L] = best if (best and ov[best] > 0) else None
    obj_insts = defaultdict(list)
    for L, o in inst_obj.items():
        if o: obj_insts[o].append(L)
    clean = split = lost = 0
    for o in gtocc:
        ins = obj_insts.get(o, [])
        if len(ins) == 0: lost += 1
        elif len(ins) == 1: clean += 1
        else: split += 1
    mixed = sum(1 for L in labs if inst_obj[L] is None)   # 對不到任何 GT(鬼影/混)
    return len(gtocc), clean, split, lost, mixed, len(labs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--out-root", default="srp_hull_veto_merge")
    ap.add_argument("--step-thr", type=float, default=40.0)
    ap.add_argument("--zov-thr", type=float, default=0.3)
    ap.add_argument("--save", action="store_true", help="存 instances.npz + merge_reasons.json")
    a = ap.parse_args()
    R = 3; offs = [(dx, dy) for dx in range(-R, R+1) for dy in range(-R, R+1) if dx*dx+dy*dy <= R*R]
    ker = np.ones((KB, KB), np.uint8)
    tot = Counter(); n_gt_sum = 0; before_sum = after_sum = 0
    for si, sc in enumerate(a.scenes):
        outd = EVAL/a.out_root/sc; outp = outd/"instances.npz"
        try:
            if a.save and outp.is_file():                       # 已存→讀存檔評分(可續跑)
                zz = np.load(outp, allow_pickle=True); new = zz["labels"]; gm = zz["grid_min"]; vs = float(zz["voxel_size"])
                zh = np.load(EVAL/INST/sc/"instances.npz"); nbefore = len([x for x in np.unique(zh["labels"]) if x > 0])
                nafter = len([x for x in np.unique(new) if x > 0])
            else:
                new, occ, gm, vs, reasons, nbefore, nafter = process(sc, a.step_thr, a.zov_thr, offs, ker)
                if a.save:
                    outd.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(outp, labels=new, occupancy=occ, grid_min=gm, voxel_size=np.float64(vs),
                                        build_meta=json.dumps({"script": "veto_merge.py", "src": INST, "step_thr": a.step_thr, "zov_thr": a.zov_thr}))
                    (outd/"merge_reasons.json").write_text(json.dumps(reasons, ensure_ascii=False, indent=1))
            ngt, clean, split, lost, mixed, neff = evaluate(sc, new, gm, vs, new.shape)
        except Exception as e:
            print(f"[fail] {sc}: {e}"); continue
        tot["clean"] += clean; tot["split"] += split; tot["lost"] += lost; tot["mixed"] += mixed
        n_gt_sum += ngt; before_sum += nbefore; after_sum += nafter
        if (si+1) % 40 == 0: print(f"  ..{si+1}/{len(a.scenes)}", flush=True)
    print(f"\n===== veto_merge n={len(a.scenes)}場, step_thr={a.step_thr} zov_thr={a.zov_thr} =====")
    print(f"instance 數:合併前 {before_sum} → 後 {after_sum} (GT 物體 {n_gt_sum}) 過切比 {after_sum/max(n_gt_sum,1):.2f}")
    print(f"物體判定(分母=GT物體 {n_gt_sum}):clean {tot['clean']}({100*tot['clean']/max(n_gt_sum,1):.0f}%) "
          f"split {tot['split']}({100*tot['split']/max(n_gt_sum,1):.0f}%) lost {tot['lost']}({100*tot['lost']/max(n_gt_sum,1):.0f}%)")
    print(f"對不到GT的實例(鬼影/混):{tot['mixed']}")


if __name__ == "__main__":
    main()
