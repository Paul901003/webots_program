#!/home/cho/.pyenv/versions/siglip/bin/python3
"""precompute_siglip_mean.py — 預存每場景每視角每遮罩的 SigLIP2 影像塔特徵(pipeline 用)。

格式對齊 CLIP 版(precompute_clip_mean):存 SAM_ROOT/<scene>/<view>/siglip_<tag>_feats.npy
  (N×768,對應 sorted(masks/mask_*.png) **全部遮罩**,無效=nan),供 masks.mask_feats 用檔名查。
crop 幾何用共用 crop_util.sqcrop_geom(與判別力驗證/CLIP 完全一致):遮罩外填灰127 → resize256 → value_range。
另存去偏用 bg 向量(純灰 crop 的 zimg)到 data/eval/_diag/siglip_probe/siglip_<tag>_bg.npy。
只跑影像塔(text=None,不需 tokenizer)。兩 model 同 process 會 GPU OOM → 預設逐一 subprocess。
用法: ./precompute_siglip_mean.py [scene|group ...] [--only b16|b32]
      B16 記憶體大,子行程會用 XLA_PYTHON_CLIENT_PREALLOCATE=false + SIGLIP_BATCH=16。
env: SAM_ROOT(mobilesamv2_fast) CAPTURES_ROOT(captures_fast)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage4_probe" / "siglip_probe"))
import viewpoints as VP          # noqa: E402
from crop_util import sqcrop_geom   # noqa: E402
from feat_siglip import build, embed, MODELS, RES, FILL   # noqa: E402  復用載入/前處理

SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "mobilesamv2_fast")))
CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures_fast")))
DIAG = REPO / "data" / "eval" / "_diag" / "siglip_probe"
FORCE = os.environ.get("FORCE", "") == "1"


def resolve(targets):
    if not targets:
        return []
    out = []
    for a in targets:
        if "scene" in a:
            out.append(a)
        else:
            out += [p.name for p in SAM_ROOT.glob(f"{a}_scene*") if p.is_dir()]
    return sorted(set(out))


def run_one(tag, scenes):
    variant, ckpt = next((v, c) for t, v, c in MODELS if t == tag)
    model, params = build(variant, ckpt)
    # 去偏 bg:純灰 crop 的 zimg(所有 crop 的共同背景成分)
    DIAG.mkdir(parents=True, exist_ok=True)
    gray = np.full((1, RES, RES, 3), 127, np.uint8)
    bg = embed(model, params, gray)[0]
    np.save(DIAG / f"siglip_{tag}_bg.npy", bg)
    print(f"[{tag}] bg 向量存 {DIAG/('siglip_'+tag+'_bg.npy')}", flush=True)

    sel = set(VP.selected_view_names(12))
    for sc in scenes:
        group = sc.split("_")[0]
        sdir = CAPTURES / f"multi_{group}" / sc
        done = 0
        for vd in sorted((SAM_ROOT / sc).glob("view_*")):
            if vd.name not in sel:
                continue
            out = vd / f"siglip_{tag}_feats.npy"
            if out.is_file() and not FORCE:
                done += 1; continue
            rp = sdir / f"{vd.name}.png"
            if not rp.is_file():
                continue
            rgb = cv2.cvtColor(cv2.imread(str(rp)), cv2.COLOR_BGR2RGB)
            mpaths = sorted((vd / "masks").glob("mask_*.png"))
            crops, valid = [], []
            for i, mp in enumerate(mpaths):
                seg = cv2.imread(str(mp), 0) > 127
                c = sqcrop_geom(rgb, seg, FILL)
                if c is not None:
                    crops.append(cv2.resize(c, (RES, RES))); valid.append(i)
            feats = np.full((len(mpaths), 768), np.nan, np.float32)
            if crops:
                feats[valid] = embed(model, params, np.stack(crops).astype(np.uint8))
            np.save(out, feats)
            done += 1
        print(f"[{tag}][{sc}] {done} 視角 siglip_{tag}_feats", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--only", choices=["b16", "b32"], default=None)
    args = ap.parse_args()
    scenes = resolve(args.targets)
    if not scenes:
        print("需指定場景/組(如 n3 stack3)"); return
    if args.only:
        run_one(args.only, scenes)
        return
    for tag, _, _ in MODELS:                     # 逐 model 子行程,結束即釋放 GPU
        print(f"\n===== 子行程 {tag} ({len(scenes)} 場) =====", flush=True)
        env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
        if tag == "b16":
            env["SIGLIP_BATCH"] = "16"
        subprocess.run([sys.executable, __file__, "--only", tag, *scenes], check=True, env=env)


if __name__ == "__main__":
    main()
