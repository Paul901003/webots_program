#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""reproj_selfmask.py — 每個語意群的 voxel 重投影,檢查是否落在「該群在該視角的語意遮罩」上。

怎麼生 / 設置:
- instances(--inst-root)的 instances.npz 提供 voxel 網格 labels(語意群);instances.json 的每 instance
  "masks"={view:[遮罩檔名]} 提供該群在各視角的語意遮罩。
- hull(--hull-root)提供 surface(可選,預設用 labels>0 的 voxel)。
- 每視角:對每個 instance 建其遮罩聯集(mobilesamv2_fast/<sc>/<view>/masks/<name>)。
  zbuffer 取可見 voxel → 依其 label 查該群遮罩:落在自己群遮罩內=綠、落在別群或無遮罩=紅。
- 輸出 12 視角 montage + 印每群/整體「重投影對上自己群遮罩」的比例。
用法: SAM_ROOT=$PWD/data/eval/mobilesamv2_fast CAPTURES_ROOT=$PWD/data/captures_fast \
  ./reproj_selfmask.py <scene> --inst-root srp_hull_semcluster_surf_am1photo [--tile-w 700]
"""
import argparse, os, sys, json
from pathlib import Path
import numpy as np, cv2
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io")); sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import camera as cam, viewpoints as VP
import cg_associate as CG
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data/eval/mobilesamv2_fast")))
CAP = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data/captures_fast")))
EVAL = REPO / "data" / "eval"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene"); ap.add_argument("--inst-root", default="srp_hull_semcluster_surf_am1photo")
    ap.add_argument("--nviews", type=int, default=12); ap.add_argument("--tile-w", type=int, default=700, dest="tile_w")
    args = ap.parse_args(); sc = args.scene
    z = np.load(EVAL / args.inst_root / sc / "instances.npz")
    labels = z["labels"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
    ij = json.loads((EVAL / args.inst_root / sc / "instances.json").read_text())
    inst_masks = {it["instance"]: it.get("masks", {}) for it in ij["instances"]}   # {k: {view:[names]}}
    vidx = np.argwhere(labels > 0)                       # 有語意群的 voxel
    lab = labels[labels > 0]                             # 對應 label
    Pw = gm + (vidx + 0.5) * vs
    sdir = CAP / f"multi_{sc.split('_')[0]}" / sc
    tiles = []; tot_vis = tot_hit = 0
    for vn in sorted(VP.selected_view_names(args.nviews)):
        vd = SAM_ROOT / sc / vn; pf = sdir / f"{vn}_pose.json"
        if not (vd.is_dir() and pf.is_file()): continue
        img = cv2.cvtColor(cv2.imread(str(sdir / f"{vn}.png")), cv2.COLOR_BGR2RGB).copy()
        H, W = img.shape[:2]; C, Rb = cam.load_pose(pf)
        # 每 instance 在此視角的遮罩聯集
        inst_union = {}
        for k, mv in inst_masks.items():
            names = mv.get(vn, [])
            if not names: continue
            u = np.zeros((H, W), bool)
            for nm in names:
                p = vd / "masks" / nm
                m = cv2.imread(str(p), 0)
                if m is not None: u |= (m > 127)
            inst_union[k] = u
        va = CG.zbuffer_visible(Pw, C, Rb, W, H, vs).reshape(H, W)
        layer = np.zeros_like(img)
        ys, xs = np.where(va >= 0); vids = va[ys, xs]
        labs = lab[vids]
        for y, x, k in zip(ys, xs, labs):
            u = inst_union.get(int(k))
            if u is not None and u[y, x]:
                layer[y, x] = (0, 200, 0); tot_hit += 1        # 綠=落在自己群遮罩
            else:
                layer[y, x] = (255, 0, 0)                       # 紅=沒對上自己群遮罩
        tot_vis += len(ys)
        blended = cv2.addWeighted(img, 1.0, layer, 0.5, 0)
        hit_v = int(sum(1 for y, x, k in zip(ys, xs, labs) if inst_union.get(int(k)) is not None and inst_union[int(k)][y, x]))
        cv2.putText(blended, f"{vn} hit={hit_v/max(len(ys),1)*100:.0f}%", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        tiles.append(blended)
    print(f"{sc}: 可見語意 voxel-pixel {tot_vis}, 對上自己群遮罩 {tot_hit} ({tot_hit/max(tot_vis,1)*100:.1f}%),"
          f" 沒對上 {tot_vis-tot_hit} ({(tot_vis-tot_hit)/max(tot_vis,1)*100:.1f}%)")
    if tiles:
        if args.tile_w > 0:
            h, w = tiles[0].shape[:2]; f = args.tile_w / w
            tiles = [cv2.resize(t, (int(w*f), int(h*f))) for t in tiles]
        while len(tiles) < 12: tiles.append(np.zeros_like(tiles[0]))
        mont = np.vstack([np.hstack(tiles[r*4:(r+1)*4]) for r in range(3)])
        outp = EVAL / args.inst_root / sc / "reproj_selfmask.png"
        cv2.imwrite(str(outp), cv2.cvtColor(mont, cv2.COLOR_RGB2BGR)); print(f"→ {outp}")

if __name__ == "__main__": main()
