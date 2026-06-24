#!/home/cho/.pyenv/versions/grounded_sam/bin/python3
"""precompute_clip.py — 預計算並存檔 CLIP 特徵(影像 + 文字),供評估程式查表。

A1 影像特徵:每場景/視角/每塊 SAM 遮罩 → crop(正方形 bbox + ImageNet 均值填背景)
            → CLIP ViT-B/32 encode_image → L2 norm。
   存: data/eval/sam_only/<scene>/<view>/clip_feats.npy(M×512)
       + clip_feats_files.json(對應的 mask 檔名順序)
A2 文字特徵:所有 YCB 物體名 → PROMPT_TABLE 片語 → "a photo of a {片語}"
            → encode_text → L2 norm。 存: data/eval/clip_text_feats.npz

需 grounded_sam 環境(clip/torch/cv2)。
用法: ./instance_hull/precompute_clip.py 1 3 4 5     (影像特徵;同時會更新文字特徵)
       ./instance_hull/precompute_clip.py --text-only  (只更新文字特徵)
       FORCE=1 重算已存在的影像特徵
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import clip

REPO = Path(__file__).resolve().parents[1]
CAPTURES = REPO / "data" / "captures"
SAM_ROOT = REPO / "data" / "eval" / "sam_only"
TEXT_OUT = REPO / "data" / "eval" / "clip_text_feats.npz"
CLIP_MODEL = "ViT-B/32"
CLIP_MEAN = np.array([123, 116, 103], dtype=np.uint8)

sys.path.insert(0, str(REPO / "controllers" / "ycb_supervisor"))
try:
    from config import PROMPT_TABLE
except Exception:
    PROMPT_TABLE = {}
_GEO = json.loads((REPO / "controllers" / "ycb_supervisor" / "ycb_geometries.json").read_text(encoding="utf-8"))


def square_mean_crop(rgb, seg):
    ys, xs = np.nonzero(seg)
    if xs.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = rgb[y0:y1, x0:x1].copy()
    crop[~seg[y0:y1, x0:x1]] = CLIP_MEAN
    h, w = crop.shape[:2]
    side = max(h, w)
    canvas = np.empty((side, side, 3), dtype=np.uint8); canvas[:] = CLIP_MEAN
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop
    return Image.fromarray(canvas)


def ycb_to_phrase(name):
    if name in PROMPT_TABLE:
        return PROMPT_TABLE[name]
    parts = name.split("_")
    start = 1 if parts[0][0].isdigit() else 0
    return " ".join(parts[start:]).replace("-", " ")


def resolve_scenes(targets):
    out = []
    for a in targets:
        if "scene" in a:
            out.append(a)
        else:
            out += [d.name for d in sorted((CAPTURES / f"multi_n{a}").glob(f"n{a}_scene*"))]
    return out


@torch.no_grad()
def precompute_images(scenes, model, prep, device, force):
    for si, scene in enumerate(scenes, 1):
        g = scene.split("_")[0]
        sdir = CAPTURES / f"multi_{g}" / scene
        sam = SAM_ROOT / scene
        if not sam.is_dir():
            print(f"[skip] {scene}: 無 SAM"); continue
        done = 0
        for vdir in sorted(sam.glob("view_*")):
            out_npy = vdir / "clip_feats.npy"
            if out_npy.is_file() and not force:
                continue
            img = cv2.imread(str(sdir / f"{vdir.name}.png"))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            files, crops = [], []
            for mp in sorted((vdir / "masks").glob("mask_*.png")):
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                c = square_mean_crop(rgb, m > 127)
                if c is None:
                    continue
                files.append(mp.name); crops.append(prep(c))
            if not crops:
                feats = np.zeros((0, 512), np.float32)
            else:
                feats = []
                for i in range(0, len(crops), 256):
                    b = torch.stack(crops[i:i + 256]).to(device)
                    f = model.encode_image(b).float()
                    f = f / f.norm(dim=-1, keepdim=True)
                    feats.append(f.cpu().numpy())
                feats = np.concatenate(feats, 0).astype(np.float32)
            np.save(out_npy, feats)
            (vdir / "clip_feats_files.json").write_text(json.dumps(files), encoding="utf-8")
            done += 1
        print(f"[{si}/{len(scenes)}] {scene}: 更新 {done} views")


@torch.no_grad()
def precompute_text(model, device):
    names = sorted(set(_GEO.keys()) | set(PROMPT_TABLE.keys()))
    # 補上各場景 manifest 出現的名字(以防 geo 沒涵蓋)
    for mp in CAPTURES.glob("multi_n*/n*_scene*/scene_manifest.json"):
        try:
            m = json.loads(mp.read_text())
            for o in m["actual"]["viewpoints"][0]["objects"]:
                names.append(o["name"])
        except Exception:
            pass
    names = sorted(set(names))
    phrases = [f"a photo of a {ycb_to_phrase(n)}" for n in names]
    tok = clip.tokenize(phrases).to(device)
    tf = model.encode_text(tok).float()
    tf = (tf / tf.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)
    np.savez(TEXT_OUT, names=np.array(names), feats=tf,
             phrases=np.array(phrases))
    print(f"文字特徵: {len(names)} 個物體名 → {TEXT_OUT}")
    for n, p in list(zip(names, phrases))[:6]:
        print(f"    {n} → \"{p}\"")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=[])
    ap.add_argument("--text-only", action="store_true", dest="text_only")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, prep = clip.load(CLIP_MODEL, device=device); model.eval()
    precompute_text(model, device)
    if not args.text_only:
        scenes = resolve_scenes(args.scenes or ["3"])
        precompute_images(scenes, model, prep, device, force=bool(os.environ.get("FORCE")))


if __name__ == "__main__":
    main()
