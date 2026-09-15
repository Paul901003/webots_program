#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""reproj_outmask.py — hull 表面 voxel 重投影回各視角,標「投影落在物體遮罩外」的 voxel(=ghost/過估)。

怎麼生 / 設置:
- hull 表面 voxel(--hull-root 的 hull.npz["surface"])→ 每視角 zbuffer 取「該視角看得到的表面 voxel」。
- 物體遮罩聯集 = kept_object_masks 去掉手臂+夾爪(≥ARM_THR 落在 srp_arm_masks 剪影者剔除)。
- 可見 voxel 投影像素:在遮罩聯集內=綠(半透明);在聯集外=紅(整個投影超出遮罩 → ghost)。
- 輸出 12 視角 montage + 印各視角/整體「超出遮罩 voxel 比例」。
用法: SAM_ROOT=$PWD/data/eval/mobilesamv2_fast CAPTURES_ROOT=$PWD/data/captures_fast \
  ./reproj_outmask.py <scene> --hull-root srp_hull_mv2_v12_am1_photo [--tile-w 640]
"""
import argparse, os, sys, json
from pathlib import Path
import numpy as np, cv2
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io")); sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import camera as cam, masks as MK, viewpoints as VP
import cg_associate as CG
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data/eval/mobilesamv2_fast")))
CAP = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data/captures_fast")))
ARM = Path(os.environ.get("ARM_MASK_ROOT", str(REPO / "data/eval/srp_arm_masks")))
EVAL = REPO / "data" / "eval"; ARM_THR = 0.5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--hull-root", default="srp_hull_mv2_v12_am1_photo")
    ap.add_argument("--nviews", type=int, default=12)
    ap.add_argument("--tile-w", type=int, default=640, dest="tile_w")
    args = ap.parse_args(); sc = args.scene
    z = np.load(EVAL / args.hull_root / sc / "hull.npz")
    surf = z["surface"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
    vox = np.array(np.nonzero(surf)).T; Pw = gm + (vox + 0.5) * vs
    sdir = CAP / f"multi_{sc.split('_')[0]}" / sc
    tiles = []; tot_vis = tot_out = 0
    for vn in sorted(VP.selected_view_names(args.nviews)):
        vd = SAM_ROOT / sc / vn; pf = sdir / f"{vn}_pose.json"
        if not (vd.is_dir() and pf.is_file()): continue
        km = MK.kept_object_masks(vd); ms0 = [m for m, _ in km]
        if not ms0: continue
        C, Rb = cam.load_pose(pf); H, W = ms0[0].shape
        ap_ = ARM / sc / f"{vn}_arm.png"; arm = (cv2.imread(str(ap_), 0) > 127) if ap_.is_file() else None
        obj = np.zeros((H, W), bool)                 # 物體遮罩聯集(去手臂夾爪)
        for m in ms0:
            if arm is not None and (m & arm).sum() / max(int(m.sum()), 1) >= ARM_THR: continue
            obj |= m
        va = CG.zbuffer_visible(Pw, C, Rb, W, H, vs).reshape(H, W)
        img = cv2.cvtColor(cv2.imread(str(sdir / f"{vn}.png")), cv2.COLOR_BGR2RGB).copy()
        layer = np.zeros_like(img)
        ys, xs = np.where(va >= 0)                    # 可見表面 voxel 的像素
        inside = obj[ys, xs]
        layer[ys[inside], xs[inside]] = (0, 200, 0)          # 綠=投影落遮罩內
        out_px = ~inside
        layer[ys[out_px], xs[out_px]] = (255, 0, 0)          # 紅=投影超出遮罩(ghost)
        blended = cv2.addWeighted(img, 1.0, layer, 0.45, 0)
        nvis = len(ys); nout = int(out_px.sum()); tot_vis += nvis; tot_out += nout
        cv2.putText(blended, f"{vn} out={nout/max(nvis,1)*100:.1f}%", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        tiles.append(blended)
    print(f"{sc}: 可見表面 voxel-pixel {tot_vis}, 超出遮罩 {tot_out} ({tot_out/max(tot_vis,1)*100:.1f}%)")
    if tiles:
        if args.tile_w > 0:
            h, w = tiles[0].shape[:2]; f = args.tile_w / w
            tiles = [cv2.resize(t, (int(w*f), int(h*f))) for t in tiles]
        while len(tiles) < 12: tiles.append(np.zeros_like(tiles[0]))
        mont = np.vstack([np.hstack(tiles[r*4:(r+1)*4]) for r in range(3)])
        outp = EVAL / args.hull_root / sc / "reproj_outmask.png"
        cv2.imwrite(str(outp), cv2.cvtColor(mont, cv2.COLOR_RGB2BGR))
        print(f"→ {outp}")

if __name__ == "__main__": main()
