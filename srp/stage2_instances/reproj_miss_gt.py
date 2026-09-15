#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""reproj_miss_gt.py — 「沒對上自己群語意遮罩」的 voxel,有多少是過估(不在GT mesh)vs 真mesh。

怎麼生 / 設置:
- instances(--inst-root)的 labels(語意群 voxel)+ instances.json 每 instance 的 per-view 群遮罩。
- 每個語意 voxel 投影各視角,zbuffer(中心)判可見;「超出」= voxel 立方體 8 角投影的像素 bbox 完全不含自己群遮罩像素(整個 footprint 都在遮罩外)。累積每 voxel 沒對上的視角數。
- GT 實心 mesh(eval_mesh.solid_mesh_occ,真值)判每 voxel 在不在真 mesh 內。
- 報:miss voxel 中 過估(不在mesh)/ 真mesh(在mesh)各佔比;對照 hit voxel 同樣分類。
用法: SAM_ROOT=$PWD/data/eval/mobilesamv2_fast CAPTURES_ROOT=$PWD/data/captures_fast \
  ./reproj_miss_gt.py [scene|group|(空=舊367)] --inst-root srp_hull_semcluster_surf_am1photo
"""
import argparse, os, sys, json
from pathlib import Path
import numpy as np, cv2
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io")); sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import camera as cam, viewpoints as VP, labels as L
import cg_associate as CG
import eval_mesh as EM
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data/eval/mobilesamv2_fast")))
CAP = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data/captures_fast")))
EVAL = REPO / "data" / "eval"

def scene_counts(sc, inst_root, nviews=12):
    ip = EVAL / inst_root / sc / "instances.npz"
    if not ip.is_file(): return None
    z = np.load(ip); labels = z["labels"]; gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = labels.shape
    ij = json.loads((EVAL / inst_root / sc / "instances.json").read_text())
    inst_masks = {it["instance"]: it.get("masks", {}) for it in ij["instances"]}
    vidx = np.argwhere(labels > 0); lab = labels[labels > 0]; N = len(vidx)
    if N == 0: return None
    Pw = gm + (vidx + 0.5) * vs
    gt = EM.solid_mesh_occ(sc, gm, vs, shape)
    if not gt: return None
    ugt = np.zeros(shape, bool)
    for g in gt.values(): ugt |= g
    real = ugt[vidx[:, 0], vidx[:, 1], vidx[:, 2]]        # 每 voxel 是否在真 mesh
    vis_cnt = np.zeros(N, int); out_cnt = np.zeros(N, int)
    sdir = CAP / f"multi_{sc.split('_')[0]}" / sc
    for vn in sorted(VP.selected_view_names(nviews)):
        vd = SAM_ROOT / sc / vn; pf = sdir / f"{vn}_pose.json"
        if not (vd.is_dir() and pf.is_file()): continue
        C, Rb = cam.load_pose(pf); Rwc, t = cam.pose_to_w2c(C, Rb)
        # 影像尺寸
        anymask = next(iter((vd / "masks").glob("mask_*.png")), None)
        if anymask is None: continue
        H, W = cv2.imread(str(anymask), 0).shape
        K = cam.intrinsics(W, H)
        X = Pw @ Rwc.T + t; zc = X[:, 2]; ok = zc > 1e-6
        zz = np.where(ok, zc, 1.0)
        px = np.round(K[0, 0] * X[:, 0] / zz + K[0, 2]).astype(int)
        py = np.round(K[1, 1] * X[:, 1] / zz + K[1, 2]).astype(int)
        inb = ok & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        va = CG.zbuffer_visible(Pw, C, Rb, W, H, vs).reshape(H, W)
        vis = np.zeros(N, bool)
        idx = np.where(inb)[0]
        vis[idx] = va[py[idx], px[idx]] == idx            # 該 voxel 中心是此像素 zbuffer 最前 → 可見
        # voxel 立方體 8 角投影 → 每 voxel 的像素 bbox(footprint)
        corn = np.array([[dx, dy, dz] for dx in (-.5, .5) for dy in (-.5, .5) for dz in (-.5, .5)]) * vs
        Xc = (Pw[:, None, :] + corn[None]) @ Rwc.T + t     # (N,8,3)
        zcn = np.clip(Xc[:, :, 2], 1e-6, None)
        cpx = K[0, 0] * Xc[:, :, 0] / zcn + K[0, 2]; cpy = K[1, 1] * Xc[:, :, 1] / zcn + K[1, 2]
        x0 = np.clip(np.floor(cpx.min(1)).astype(int), 0, W - 1); x1 = np.clip(np.ceil(cpx.max(1)).astype(int), 0, W - 1)
        y0 = np.clip(np.floor(cpy.min(1)).astype(int), 0, H - 1); y1 = np.clip(np.ceil(cpy.max(1)).astype(int), 0, H - 1)
        # 每 instance 群遮罩聯集 + 積分圖(供 bbox O(1) 求和)
        inst_ii = {}
        for k, mv in inst_masks.items():
            names = mv.get(vn, [])
            if not names: continue
            u = np.zeros((H, W), bool)
            for nm in names:
                m = cv2.imread(str(vd / "masks" / nm), 0)
                if m is not None: u |= (m > 127)
            inst_ii[k] = np.pad(u.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        visi = np.where(vis)[0]
        vis_cnt[visi] += 1
        for i in visi:
            II = inst_ii.get(int(lab[i]))
            if II is None:
                out_cnt[i] += 1; continue                  # 自己群此視角無遮罩 → footprint 全在外
            a, b, c, d = y0[i], y1[i] + 1, x0[i], x1[i] + 1
            s = II[b, d] - II[a, d] - II[b, c] + II[a, c]   # footprint bbox 內遮罩像素數
            if s == 0: out_cnt[i] += 1                      # 整個 footprint 都在自己群遮罩外
    seen = vis_cnt > 0
    # 回每 voxel 的 (沒對上視角數, 可見視角數, 是否真mesh) — 供門檻掃描
    return {"N": N, "out_cnt": out_cnt[seen].astype(np.int16),
            "vis_cnt": vis_cnt[seen].astype(np.int16), "real": real[seen]}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("targets", nargs="*")
    ap.add_argument("--inst-root", default="srp_hull_semcluster_surf_am1photo"); ap.add_argument("--nviews", type=int, default=12)
    args = ap.parse_args()
    root = EVAL / args.inst_root
    if not args.targets:
        groups = ("n1", "n3", "n4", "n5", "occ3", "occ4", "occ5", "stack3", "stack4", "stack5")
        scenes = sorted(p.name for g in groups for p in root.glob(f"{g}_scene*"))
    else:
        scenes = []
        for a in args.targets:
            scenes += [a] if "scene" in a else sorted(p.name for p in root.glob(f"{a}_scene*"))
    OUT = []; VIS = []; REAL = []
    for i, sc in enumerate(scenes):
        try:
            r = scene_counts(sc, args.inst_root, args.nviews)
            if r is None: continue
            OUT.append(r["out_cnt"]); VIS.append(r["vis_cnt"]); REAL.append(r["real"])
            if (i + 1) % 40 == 0: print(f"  {i+1}/{len(scenes)}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}", flush=True)
    out = np.concatenate(OUT); vis = np.concatenate(VIS); real = np.concatenate(REAL)
    HERE = REPO / "srp/stage2_instances/experiments/reproj_miss_gt_20260828"; HERE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(HERE / f"pervoxel_{args.inst_root}.npz", out_cnt=out, vis_cnt=vis, real=real)
    ghost = ~real; NG = int(ghost.sum()); NR = int(real.sum())
    print(f"\n===== {len(scenes)}場 門檻掃描 (inst={args.inst_root}) =====")
    print(f"可見語意 voxel {len(out)}: 真mesh {NR} / 過估(鬼影) {NG}")
    print(f"規則:沒對上視角數 ≥ k → 判鬼影去掉。看去掉的鬼影/真mesh。")
    print(f"{'k(≥沒對上)':>10}{'去掉voxel':>10}{'其中鬼影':>10}{'其中真mesh':>10}{'鬼影去除率':>10}{'真mesh誤刪':>10}{'去除純度':>9}")
    for k in range(1, 13):
        sel = out >= k; n = int(sel.sum()); g = int((sel & ghost).sum()); rr = n - g
        print(f"{k:>10}{n:>10}{g:>10}{rr:>10}{g/max(NG,1)*100:>9.1f}%{rr/max(NR,1)*100:>9.1f}%{g/max(n,1)*100:>8.1f}%")
    print(f"→ 每 voxel 明細存 {HERE}/pervoxel_{args.inst_root}.npz")

if __name__ == "__main__": main()
