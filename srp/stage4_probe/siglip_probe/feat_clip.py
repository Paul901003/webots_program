#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""feat_clip.py — 讀 manifest,對每個遮罩抽 CLIP-B32(open_clip ViT-B-32 openai)特徵。

crop 幾何用共用 crop_util.sqcrop_geom(與 SigLIP2 三組完全一致);CLIP 標準前處理:
  遮罩外填 CLIP mean 色[123,117,104] → resize 224 → (x/255 - CMEAN)/CSTD → encode_image → L2。
輸出 data/eval/_diag/siglip_probe/clip_b32.npy (N×512,順序對齊 manifest.items,無效=nan)。
用法: ./feat_clip.py   (需 webots_visual_hull env:torch+open_clip)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import open_clip

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crop_util import sqcrop_geom   # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data" / "eval" / "_diag" / "siglip_probe"
CAPTURES = REPO / "data" / "captures_fast"
FILL = np.array([123, 117, 104], np.uint8)   # CLIP mean 色(填後 normalize≈0)
CMEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CSTD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 256


@torch.no_grad()
def main():
    man = json.loads((OUT / "manifest.json").read_text())
    items = man["items"]
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(DEV).eval()

    rgb_cache = {}
    feats = np.full((len(items), 512), np.nan, np.float32)
    buf, idx = [], []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(DEV)
        f = model.encode_image(x).float()
        f = (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()
        for k, i in enumerate(idx):
            feats[i] = f[k]
        buf.clear(); idx.clear()

    for i, it in enumerate(items):
        key = (it["group"], it["scene"], it["view"])
        if key not in rgb_cache:
            rp = CAPTURES / f"multi_{it['group']}" / it["scene"] / f"{it['view']}.png"
            rgb_cache[key] = cv2.cvtColor(cv2.imread(str(rp)), cv2.COLOR_BGR2RGB) if rp.is_file() else None
        rgb = rgb_cache[key]
        if rgb is None:
            continue
        seg = cv2.imread(str(REPO / it["mask_rel"]), 0) > 127
        c = sqcrop_geom(rgb, seg, FILL)
        if c is None:
            continue
        im = cv2.resize(c, (224, 224))
        t = (torch.from_numpy(im).permute(2, 0, 1).float() / 255 - CMEAN) / CSTD
        buf.append(t); idx.append(i)
        if len(buf) >= BATCH:
            flush()
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(items)}", flush=True)
    flush()
    np.save(OUT / "clip_b32.npy", feats)
    ok = int(np.isfinite(feats[:, 0]).sum())
    print(f"CLIP-B32: {ok}/{len(items)} 有效 → {OUT/'clip_b32.npy'}")


if __name__ == "__main__":
    main()
