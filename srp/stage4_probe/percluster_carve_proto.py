#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""per-cluster carving v2:三態雕殼 + 雜訊群過濾 + 過切 blob 合併。"""
import sys, json, os
import numpy as np, cv2
from scipy import ndimage
from collections import Counter
sys.path.insert(0, "srp/io"); sys.path.insert(0, "srp/stage2_instances")
import camera as cam, masks as MK, viewpoints as VP, mask_clip_cluster as MC
from build_hull_gt import reproject, iou2

EVAL = "data/eval"; SC = "occ5_scene0006"
SAM = f"{EVAL}/mobilesamv2_fast/{SC}"; CAP = f"data/captures_fast/multi_occ5/{SC}"; ARM = f"{EVAL}/srp_arm_masks/{SC}"
BOX_MIN = np.array([0.0, -0.35, 0.0]); BOX_MAX = np.array([0.7, 0.35, 0.35]); VOX = 0.005
MIN_SUP = 2; ALLOW_MISS = 1
MIN_MASKS = 3        # 群遮罩數門檻(雜訊群過濾)
MIN_VOX_HULL = 250   # blob 最小體素(雜訊過濾)
OVERLAP = 0.3        # blob 間重疊>此(對較小者)
SEM_MERGE = 0.55     # 且群平均特徵 cos>此 才合併(語意 gate:同物過切 vs 不同物)

dims = np.round((BOX_MAX - BOX_MIN) / VOX).astype(int)
gi, gj, gk = np.meshgrid(*[np.arange(x) for x in dims], indexing="ij")
idx = np.stack([gi.ravel(), gj.ravel(), gk.ravel()], 1); P = BOX_MIN + (idx + 0.5) * VOX; M = len(P)

d = json.load(open(f"{EVAL}/srp_hull_semcluster_clip_am1/{SC}/instances.json")); mc = d["mask_clusters"]
cl_nmask = Counter(c for vn in mc for c in mc[vn].values())   # 每群遮罩數
sel = sorted(set(VP.selected_view_names(12))); views = []
for vn in sel:
    pf = f"{CAP}/{vn}_pose.json"
    if not os.path.isfile(pf): continue
    km = MK.kept_object_masks(f"{SAM}/{vn}")
    if not km: continue
    H, W = km[0][0].shape; L = np.zeros((H, W), np.int16)
    for m, nm in km:
        c = mc.get(vn, {}).get(nm)
        if c is not None: L[(L == 0) & m] = c
    ap = f"{ARM}/{vn}_arm.png"
    if os.path.isfile(ap):
        arm = cv2.imread(ap, 0) > 127
        if arm.shape == L.shape: L[arm] = -9
    C, Rb = cam.load_pose(pf); Rwc, t = cam.pose_to_w2c(C, Rb)
    views.append((L, Rwc, t, cam.intrinsics(W, H), C, Rb, vn, H, W))

V = len(views); labmat = np.full((M, V), -1, np.int16)
for j, (L, Rwc, t, K, C, Rb, vn, H, W) in enumerate(views):
    X = P @ Rwc.T + t; zc = X[:, 2]; ok = zc > 1e-9; zz = np.where(ok, zc, 1.0)
    u = np.round(K[0, 0] * X[:, 0] / zz + K[0, 2]).astype(int); v = np.round(K[1, 1] * X[:, 1] / zz + K[1, 2]).astype(int)
    inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H); labmat[inb, j] = L[v[inb], u[inb]]

bg = (labmat == 0).sum(1)
clusters = [c for c in sorted(cl_nmask) if cl_nmask[c] >= MIN_MASKS]   # ① 雜訊群過濾:遮罩數
dropped = [c for c in sorted(cl_nmask) if cl_nmask[c] < MIN_MASKS]
print(f"雕的群 {clusters}（遮罩<{MIN_MASKS} 丟棄: {dropped}）")

# 每群三態雕殼 → 連通 → blob(過小丟)
st = ndimage.generate_binary_structure(3, 3); blobs = []
for c in clusters:
    sup = (labmat == c).sum(1); keep = (sup >= MIN_SUP) & (bg <= ALLOW_MISS)
    if keep.sum() < MIN_VOX_HULL: continue
    lab, n = ndimage.label(keep.reshape(tuple(dims)), st); flat = lab.ravel()
    for ci in range(1, n + 1):
        vset = np.flatnonzero(flat == ci)
        if len(vset) >= MIN_VOX_HULL:          # ② 雜訊過濾:blob 太小丟
            blobs.append([set(vset.tolist()), c])
print(f"blob 數(過小過濾後): {len(blobs)}")

# 群平均去偏特徵(語意 gate 用)
_bg=MC.F_BG.astype(np.float64)
def _deb(F):
    F=F.astype(np.float64); F=F-(F@_bg)[:,None]*_bg[None,:]; return F/(np.linalg.norm(F,axis=1,keepdims=True)+1e-9)
cl_feats={}
for vn in sel:
    fm=MK.mask_feats(f"{SAM}/{vn}","clip_mean_feats.npy")
    for nm,ft in fm.items():
        c=mc.get(vn,{}).get(nm)
        if c is not None and ft is not None: cl_feats.setdefault(c,[]).append(ft)
cl_mean={c:_deb(np.array(v)).mean(0) for c,v in cl_feats.items() if v}
for c in list(cl_mean): cl_mean[c]=cl_mean[c]/(np.linalg.norm(cl_mean[c])+1e-9)
def clcos(a,b):
    if a not in cl_mean or b not in cl_mean: return 0.0
    return float(cl_mean[a]@cl_mean[b])

# ③ 過切合併:blob 間重疊>OVERLAP(較小者)→ union-find
par = list(range(len(blobs)))
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for a in range(len(blobs)):
    for b in range(a + 1, len(blobs)):
        inter = len(blobs[a][0] & blobs[b][0]); mn = min(len(blobs[a][0]), len(blobs[b][0]))
        if mn and inter/mn>OVERLAP:
            co=clcos(blobs[a][1],blobs[b][1])
            print(f'  重疊 群{blobs[a][1]}~群{blobs[b][1]}: overlap={inter/mn:.2f} cos={co:.2f} → {"合併" if co>SEM_MERGE else "不合(語意異)"}')
            if co>SEM_MERGE: par[find(a)]=find(b)
merged = {}
for i, (vs, c) in enumerate(blobs):
    r = find(i); merged.setdefault(r, [set(), set()])
    merged[r][0] |= vs; merged[r][1].add(c)

# GT 標(重投影 IoU)
gj_ = json.load(open(f"{EVAL}/gt_reproj/{SC}/gt.json")); z = np.load(f"{EVAL}/gt_reproj/{SC}/gt.npz")
modal = {k[6:].rsplit('__', 1)[0] + '\x00' + k[6:].rsplit('__', 1)[1]: z[k].astype(bool) for k in z.files if k.startswith("modal_")}
modal = {(k.split('\x00')[0], k.split('\x00')[1]): v for k, v in modal.items()}
unocc = gj_["unoccluded_views"]; H0, W0 = next(iter(modal.values())).shape
print(f"\n=== v2 結果:{len(merged)} instances(合併/過濾後)===")
best_by_gt = {}
for r, (vs, cs) in sorted(merged.items(), key=lambda x: -len(x[1][0])):
    vox = idx[np.array(sorted(vs))]; Pw = BOX_MIN + (vox + 0.5) * VOX
    bg_g, bi = None, 0
    for g in gj_["gt_objects"]:
        ious = [iou2(reproject(Pw, C, Rb, W0, H0, VOX), modal[(g, vn)])
                for (L, Rwc, t, K, C, Rb, vn, H, W) in views if vn in unocc.get(g, []) and (g, vn) in modal]
        a = float(np.mean(ious)) if ious else 0
        if a > bi: bi, bg_g = a, g
    print(f"  群{sorted(cs)}: {len(vs)} vox → 最像 {bg_g.split('_',1)[-1] if bg_g else '?'} (IoU {bi:.3f})")
    if bg_g: best_by_gt[bg_g] = max(best_by_gt.get(bg_g, 0), bi)

print("\n=== 每 GT 物體最佳 IoU:am1 vs v2 ===")
hj = json.load(open(f"{EVAL}/srp_hull_semcluster_clip_am1/{SC}/hull_gt.json"))
am1 = {}
for k, info in hj["hulls"].items():
    for g, dd in info["per_gt"].items(): am1[g] = max(am1.get(g, 0), dd.get("avg", 0))
for g in gj_["gt_objects"]:
    a = am1.get(g, 0); b = best_by_gt.get(g, 0)
    fa = "找到" if a >= 0.6 else "漏"; fb = "找到" if b >= 0.6 else "漏"
    print(f"  {g.split('_',1)[-1]:<18} am1={a:.2f}({fa})  v2={b:.2f}({fb})")
print(f"\n  am1 recall={sum(1 for g in gj_['gt_objects'] if am1.get(g,0)>=0.6)}/{len(gj_['gt_objects'])}"
      f"  v2 recall={sum(1 for g in gj_['gt_objects'] if best_by_gt.get(g,0)>=0.6)}/{len(gj_['gt_objects'])}"
      f"  | v2 instances={len(merged)}")
