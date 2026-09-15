#!/home/cho/.pyenv/versions/siglip/bin/python3
"""feat_siglip.py — 讀 manifest,對每個遮罩抽 SigLIP2 影像塔特徵(B/32 與 B/16 兩組)。

只跑影像塔(model.apply(..., image, None) → 只算 zimg,文字塔完全不碰,不需 tokenizer/tf-text)。
crop 幾何用共用 crop_util.sqcrop_geom(與 CLIP 完全一致);SigLIP2 標準前處理:
  遮罩外填中性灰[127,127,127] → resize 256 → value_range(-1,1)=x/127.5-1 → 影像塔 → L2(內建)。
輸出 data/eval/_diag/siglip_probe/siglip_b32.npy、siglip_b16.npy (N×768,對齊 manifest,無效=nan)。
用法: ./feat_siglip.py [--only b32|b16]   (兩個 model 同 process 會 GPU OOM,預設逐一 subprocess 跑)
       需 siglip env:jax+flax;checkpoint 在 ~/Downloads/siglip2_{b32,b16}_256.npz
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import ml_collections

REPO = Path(__file__).resolve().parents[3]
BV = REPO / "big_vision"
sys.path.insert(0, str(BV))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import big_vision.models.proj.image_text.two_towers as model_mod   # noqa: E402
from crop_util import sqcrop_geom   # noqa: E402

OUT = REPO / "data" / "eval" / "_diag" / "siglip_probe"
CAPTURES = REPO / "data" / "captures_fast"
CKPT_DIR = Path.home() / "Downloads"
FILL = np.array([127, 127, 127], np.uint8)   # value_range 後≈0 的中性灰
RES = 256
BATCH = int(os.environ.get("SIGLIP_BATCH", "128"))   # b16(256 patches)activation 大,可用 SIGLIP_BATCH 降
MODELS = [("b32", "B/32", CKPT_DIR / "siglip2_b32_256.npz"),
          ("b16", "B/16", CKPT_DIR / "siglip2_b16_256.npz")]


def build(variant, ckpt):
    txtv = variant.split("/")[0]
    embdim = {"B": 768, "L": 1024, "So400m": 1152}[txtv]
    cfg = ml_collections.ConfigDict(dict(
        image_model="vit",
        image=dict(pool_type="map", scan=True, variant=variant),
        text_model="proj.image_text.text_transformer",
        text=dict(scan=True, variant=txtv, vocab_size=256_000),
        out_dim=[None, embdim], bias_init=-10))
    model = model_mod.Model(**cfg)
    params = model_mod.load(None, str(ckpt), cfg)
    return model, params


def load_crops(items):
    """回傳 (valid_idx, crops uint8 (M,RES,RES,3))。crop 幾何三組一致,只填色/尺寸依 SigLIP。"""
    rgb_cache = {}
    idx, crops = [], []
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
        idx.append(i)
        crops.append(cv2.resize(c, (RES, RES)))
    return np.array(idx), np.stack(crops).astype(np.uint8)


def embed(model, params, crops):
    """crops (M,RES,RES,3) uint8 → zimg (M,768) L2-normalized。"""
    out = np.zeros((len(crops), 768), np.float32)
    for s in range(0, len(crops), BATCH):
        x = crops[s:s + BATCH].astype(np.float32) / 127.5 - 1.0   # value_range(-1,1)
        zimg, _, _ = model.apply({"params": params}, x, None)
        out[s:s + BATCH] = np.asarray(zimg)
        print(f"  {min(s+BATCH,len(crops))}/{len(crops)}", flush=True)
    return out


def run_one(tag):
    man = json.loads((OUT / "manifest.json").read_text())
    items = man["items"]
    variant, ckpt = next((v, c) for t, v, c in MODELS if t == tag)
    idx, crops = load_crops(items)
    print(f"[{tag}] 有效 crop {len(idx)}/{len(items)},載入 {variant} {ckpt.name} ...", flush=True)
    model, params = build(variant, ckpt)
    z = embed(model, params, crops)
    feats = np.full((len(items), 768), np.nan, np.float32)
    feats[idx] = z
    np.save(OUT / f"siglip_{tag}.npy", feats)
    print(f"[{tag}] {len(idx)} 有效 → {OUT/('siglip_'+tag+'.npy')}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["b32", "b16"], default=None)
    args = ap.parse_args()
    if args.only:
        run_one(args.only)
        return
    # 預設:每個 model 各自一個 subprocess 跑,結束即釋放 GPU,避免同 process 兩 model OOM
    for tag, _, _ in MODELS:
        print(f"\n===== 子行程跑 {tag} =====", flush=True)
        subprocess.run([sys.executable, __file__, "--only", tag], check=True)


if __name__ == "__main__":
    main()
