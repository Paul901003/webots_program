#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""reproj_overlap.py — 把「跨語意群共用的重疊 voxel」標記後,重投影回各視角疊在 RGB 上。

重疊 voxel = 在不同視角的多數決歸屬中被指派給 ≥2 個語意群的 voxel(群邊界的模糊處)。
每視角:半透明畫出 hull 可見表面 voxel(藍),重疊 voxel 疊紅;輸出 12 視角 montage。
用途:目視「語意群重疊發生在影像哪裡」。

用法: SAM_ROOT=$PWD/data/eval/mobilesamv2_fast \
  ./reproj_overlap.py stack3_scene0001 --hull-root srp_hull_warp_carved --sem-root semdonut_warp_carved
"""
import argparse, os, sys, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np, cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io")); sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import camera as cam, masks as MK, viewpoints as VP
import cg_associate as CG
from voxel_sem_cluster_donut import donut_masks

SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data/eval/mobilesamv2_fast")))
CAP = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data/captures_fast")))
EVAL = REPO / "data" / "eval"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--hull-root", default="srp_hull_warp_carved")
    ap.add_argument("--sem-root", default="semdonut_warp_carved")
    ap.add_argument("--nviews", type=int, default=12)
    ap.add_argument("--min-dom", type=float, default=0.7, dest="min_dom",
                    help="主群佔比 < 此才算『真重疊』(過濾少數視角誤歸的假重疊);1.0=舊行為(曾≥2群就算)")
    ap.add_argument("--tile-w", type=int, default=640, dest="tile_w",
                    help="montage 每格寬度(px);0=原尺寸不縮(最清晰但檔案大)")
    args = ap.parse_args()
    sc = args.scene

    z = np.load(EVAL / args.hull_root / sc / "hull.npz")
    surf = z["surface"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
    vox = np.array(np.nonzero(surf)).T; Pw = gm + (vox + 0.5) * vs
    mc = json.loads((EVAL / args.sem_root / sc / "instances.json").read_text()).get("mask_clusters", {})

    sdir = CAP / f"multi_{sc.split('_')[0]}" / sc
    views = sorted(VP.selected_view_names(args.nviews))
    # 第一遍:每 voxel 各語意群拿到幾個視角(跨視角多數決歸屬計數)
    vgroups = defaultdict(Counter); per_view = {}
    for vn in views:
        vd = SAM_ROOT / sc / vn; pf = sdir / f"{vn}_pose.json"
        if not (vd.is_dir() and pf.is_file()): continue
        km = MK.kept_object_masks(vd); names = [n for _, n in km]; ms0 = [m for m, _ in km]
        if not ms0: continue
        ms = donut_masks(ms0); C, Rb = cam.load_pose(pf); H, W = ms0[0].shape
        va = CG.zbuffer_visible(Pw, C, Rb, W, H, vs).reshape(H, W)
        vmc = defaultdict(Counter)
        for mi, m in enumerate(ms):
            ys, xs = np.where(m); vv = va[ys, xs]
            for v in vv[vv >= 0]: vmc[int(v)][mi] += 1
        cl = mc.get(vn, {})
        mvx = {}
        for v, cnt in vmc.items():
            mi = cnt.most_common(1)[0][0]
            gid = cl.get(names[mi])
            mvx[v] = gid
            if gid is not None: vgroups[v][gid] += 1
        per_view[vn] = (va, C, Rb, W, H)
    # 真重疊 = 被 ≥2 群歸屬 且 主群佔比 < min_dom(沒有明確主群 → 真的分不清)
    overlap = {v for v, c in vgroups.items()
               if len(c) >= 2 and max(c.values()) / sum(c.values()) < args.min_dom}
    print(f"{sc}: 表面 {len(Pw)} voxel,真重疊(主群<{args.min_dom:.0%}) {len(overlap)} 個 "
          f"({len(overlap)/max(len(Pw),1)*100:.1f}%)")

    tiles = []
    for vn in views:
        if vn not in per_view: continue
        va, C, Rb, W, H = per_view[vn]
        img = cv2.cvtColor(cv2.imread(str(sdir / f"{vn}.png")), cv2.COLOR_BGR2RGB).copy()
        hull_layer = np.zeros_like(img); ov_mask = np.zeros((H, W), bool)
        vis = va.reshape(H, W)
        ys, xs = np.where(vis >= 0)
        vids = vis[ys, xs]
        ov_set = overlap
        for y, x, vid in zip(ys, xs, vids):
            if vid in ov_set: ov_mask[y, x] = True                   # 重疊
            else: hull_layer[y, x] = (0, 128, 255)                   # hull=藍(照舊)
        out = cv2.addWeighted(img, 1.0, hull_layer, 0.35, 0)         # hull 藍半透明照舊
        out[ov_mask] = (0, 0, 0)                                     # 重疊=黑(直接塗黑,非加法)
        cv2.putText(out, vn, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        tiles.append(out)
    # montage 3×4
    if tiles:
        if args.tile_w > 0:
            h, w = tiles[0].shape[:2]; sc_f = args.tile_w / w
            tiles = [cv2.resize(t, (int(w * sc_f), int(h * sc_f))) for t in tiles]
        while len(tiles) < 12: tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[r*4:(r+1)*4]) for r in range(3)]
        mont = np.vstack(rows)
        out_dir = EVAL / args.hull_root / sc; out_dir.mkdir(parents=True, exist_ok=True)
        outp = out_dir / "reproj_overlap.png"
        cv2.imwrite(str(outp), cv2.cvtColor(mont, cv2.COLOR_RGB2BGR))
        print(f"→ {outp}")


if __name__ == "__main__":
    main()
