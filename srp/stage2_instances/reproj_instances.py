#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""reproj_instances.py — 把 instances.npz 的 instance labels 重投影上色疊在 RGB 上(12 視角 montage),供目視。
每個 instance 一色(可見表面 voxel);輸出 <inst-root>/<scene>/reproj_instances.png。
用法: SAM_ROOT=.. CAPTURES_ROOT=.. ./reproj_instances.py <scene> --inst-root srp_hull_semcluster_surf_am1photo_merged
"""
import argparse, os, sys
from pathlib import Path
import numpy as np, cv2
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io")); sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import camera as cam, viewpoints as VP
import cg_associate as CG
CAP = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data/captures_fast"))); EVAL = REPO / "data" / "eval"
PAL = [(230,60,60),(60,160,230),(60,200,90),(230,200,50),(200,80,220),(50,210,210),(240,140,40),(150,110,220),
       (120,200,60),(230,100,150),(90,90,230),(180,180,60)]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("scene")
    ap.add_argument("--inst-root", default="srp_hull_semcluster_surf_am1photo_merged")
    ap.add_argument("--nviews", type=int, default=12); ap.add_argument("--tile-w", type=int, default=700)
    a = ap.parse_args(); sc = a.scene
    z = np.load(EVAL / a.inst_root / sc / "instances.npz")
    labels = z["labels"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
    vidx = np.argwhere(labels > 0); lab = labels[labels > 0]; Pw = gm + (vidx + 0.5) * vs
    sdir = CAP / f"multi_{sc.split('_')[0]}" / sc; tiles = []
    for vn in sorted(VP.selected_view_names(a.nviews)):
        pf = sdir / f"{vn}_pose.json"; img = cv2.cvtColor(cv2.imread(str(sdir / f"{vn}.png")), cv2.COLOR_BGR2RGB).copy()
        if not pf.is_file(): tiles.append(img); continue
        H, W = img.shape[:2]; C, Rb = cam.load_pose(pf)
        va = CG.zbuffer_visible(Pw, C, Rb, W, H, vs).reshape(H, W)
        layer = np.zeros_like(img); ys, xs = np.where(va >= 0); vids = va[ys, xs]
        for y, x, vid in zip(ys, xs, vids): layer[y, x] = PAL[(int(lab[vid]) - 1) % len(PAL)]
        out = cv2.addWeighted(img, 1.0, layer, 0.55, 0)
        cv2.putText(out, f"{vn} ({int(labels.max())}inst)", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        tiles.append(out)
    h, w = tiles[0].shape[:2]; f = a.tile_w / w
    tiles = [cv2.resize(t, (int(w*f), int(h*f))) for t in tiles]
    while len(tiles) < 12: tiles.append(np.zeros_like(tiles[0]))
    mont = np.vstack([np.hstack(tiles[r*4:(r+1)*4]) for r in range(3)])
    outp = EVAL / a.inst_root / sc / "reproj_instances.png"
    cv2.imwrite(str(outp), cv2.cvtColor(mont, cv2.COLOR_RGB2BGR)); print(f"→ {outp}")

if __name__ == "__main__": main()
