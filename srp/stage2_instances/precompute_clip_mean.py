#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""precompute_clip_mean.py — 預存每場景 12 視角每遮罩的 CLIP 填均值特徵。
存 SAM_ROOT/<scene>/<view>/clip_mean_feats.npy (N×512,對應 sorted(masks/*.png) **全部遮罩**,無效遮罩=nan)。
與 border_frac 背景過濾解耦:後續一律用 masks.mask_feats(view_dir) 以「遮罩檔名」查特徵,不靠 kept 順序。
cg/三方法(voxel_sem_vote/cluster/paper)讀這份 cache、不重算 CLIP。
用法: ./precompute_clip_mean.py [scene|group|(空=全部)] [--n-views 12] [FORCE=1 重算]
env: SAM_ROOT HULL_ROOT(srp_hull_v12) CAPTURES_ROOT
"""
import argparse, os, sys, glob
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "srp" / "io"))
import numpy as np, cv2
from pathlib import Path
import masks as MK, mask_clip_cluster as MC, viewpoints as VP

REPO = Path(__file__).resolve().parents[2]
CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures_fast")))
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "sam_only_fast")))
HULL_ROOT = Path(os.environ.get("HULL_ROOT", str(REPO / "data" / "eval" / "srp_hull_v12")))
FORCE = os.environ.get("FORCE", "") == "1"


def process(sc, n_views):
    group = sc.split("_")[0]; sdir = CAPTURES / f"multi_{group}" / sc
    want = set(VP.selected_view_names(n_views)) if n_views else None
    done = 0
    for vd in sorted((SAM_ROOT / sc).glob("view_*")):
        if want is not None and vd.name not in want: continue
        out = vd / "clip_mean_feats.npy"
        if out.is_file() and not FORCE: done += 1; continue
        img = sdir / f"{vd.name}.png"
        if not img.is_file(): continue
        mpaths = sorted((vd / "masks").glob("mask_*.png"))   # 全部遮罩(不濾背景),與 masks.mask_feats 對齊
        ms = [(cv2.imread(str(p), 0) > 127) for p in mpaths]
        if not ms: continue
        rgb = cv2.cvtColor(cv2.imread(str(img)), cv2.COLOR_BGR2RGB)
        feats = MC.clip_feats(rgb, ms, "mean")
        arr = np.stack([f if f is not None else np.full(512, np.nan, np.float32) for f in feats]).astype(np.float32)
        np.save(out, arr); done += 1
    print(f"[{sc}] 預存 {done} 視角 clip_mean_feats", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--n-views", type=int, default=12, dest="n_views")
    args = ap.parse_args()
    if not args.targets:
        scenes = sorted(Path(p).parent.name for p in glob.glob(str(HULL_ROOT / "*_scene*/hull.npz")))
    else:
        scenes = []
        for a in args.targets:
            if "scene" in a: scenes.append(a)
            else: scenes += [Path(p).parent.name for p in glob.glob(str(HULL_ROOT / f"{a}_scene*/hull.npz"))]
        scenes = sorted(set(scenes))
    for sc in scenes:
        try: process(sc, args.n_views)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")


if __name__ == "__main__":
    main()
