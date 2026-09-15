#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""analyze_reproj_labels.py — 只讀 store_reproj_labels.py 存的 reproj_labels.npz,算 miss 門檻掃描 + a/b/c 拆解。
不重投影。旗標事件 = 某可見視角 footprint 完全在自己群遮罩外(vis & ~in_own)。
  (b)同物體別群 = proj_obj==gt_obj;(a)別的物體 = proj_obj>=0 且 !=gt_obj;(c)真鬼影區 = proj_obj<0(footprint全在物體遮罩外)。
用法: ./analyze_reproj_labels.py --inst-root srp_hull_semcluster_surf_am1photo
"""
import argparse, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; EVAL = REPO / "data" / "eval"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inst-root", default="srp_hull_semcluster_surf_am1photo")
    args = ap.parse_args(); root = EVAL / args.inst_root
    OUT = []; COUT = []; REAL = []
    ba = {"real": np.zeros(3, np.int64), "ghost": np.zeros(3, np.int64)}   # [b,a,c] 旗標事件
    for f in sorted(root.glob("*_scene*/reproj_labels.npz")):
        z = np.load(f)
        gt = z["gt_obj"].astype(int); vis = z["vis"]; inown = z["in_own"]; proj = z["proj_obj"].astype(int)
        real = gt >= 0
        flag = vis & ~inown                       # (N,V) 「在自己群遮罩外」旗標
        cflag = vis & (proj < 0)                  # (N,V) 「footprint 全在物體遮罩外」
        OUT.append(flag.sum(1)); COUT.append(cflag.sum(1)); REAL.append(real)
        # a/b/c 拆解(逐事件)
        gtb = np.broadcast_to(gt[:, None], flag.shape)
        c = flag & (proj < 0)                              # footprint 全在物體遮罩外(真鬼影區)
        b = flag & (proj >= 0) & (proj == gtb)             # 落在自己 GT 物體(需 gt>=0;鬼影 gt=-1 恆不成立)
        a = flag & (proj >= 0) & (proj != gtb)             # 落在別的物體
        for grp, sel in [("real", real), ("ghost", ~real)]:
            ba[grp][0] += int(b[sel].sum()); ba[grp][1] += int(a[sel].sum()); ba[grp][2] += int(c[sel].sum())
    out = np.concatenate(OUT); cout = np.concatenate(COUT); real = np.concatenate(REAL); ghost = ~real
    NG = int(ghost.sum()); NR = int(real.sum())
    print(f"===== 讀 {args.inst_root} 存好的標籤 =====")
    print(f"可見語意 voxel {len(out)}: 真mesh {NR} / 鬼影 {NG}\n")
    def sweep(cnt, title):
        print(title)
        print(f"{'k':>3}{'去掉':>9}{'鬼影':>9}{'真mesh':>9}{'鬼影去除率':>10}{'真mesh誤刪':>10}{'純度':>8}")
        for k in range(1, 8):
            sel = cnt >= k; n = int(sel.sum()); g = int((sel & ghost).sum()); rr = n - g
            print(f"{k:>3}{n:>9}{g:>9}{rr:>9}{g/max(NG,1)*100:>9.1f}%{rr/max(NR,1)*100:>9.1f}%{g/max(n,1)*100:>7.1f}%")
    sweep(out, "判準A:在自己群遮罩外視角數 ≥ k → 去掉")
    print()
    sweep(cout, "判準C:footprint全在物體遮罩外視角數 ≥ k → 去掉")
    print("\n『整個 footprint 在自己群遮罩外』旗標事件 a/b/c 拆解:")
    for grp in ("real", "ghost"):
        d = ba[grp]; tot = int(d.sum()); lab = "真mesh voxel" if grp == "real" else "鬼影 voxel"
        print(f"[{lab}] 事件 {tot}: (b)同物體別群 {d[0]} ({d[0]/max(tot,1)*100:.1f}%) | "
              f"(a)別的物體 {d[1]} ({d[1]/max(tot,1)*100:.1f}%) | (c)真鬼影區 {d[2]} ({d[2]/max(tot,1)*100:.1f}%)")

if __name__ == "__main__": main()
