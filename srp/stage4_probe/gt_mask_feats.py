#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""gt_mask_feats.py — 用 GT modal 遮罩(遮擋下每物體真實可見輪廓),對場景每物體各視角
同時算 CLIP + DINOv2 特徵並儲存。排除 SAM 過分割誤差,看純語意特徵對同物體的一致性。

CLIP:  square_mean_crop(摳遮罩填灰)→ open_clip ViT-B-32(openai 權重)→ 512d(L2 norm)。
DINO:  原圖 dense → 遮罩 region 內 patch token 平均 → dinov2_vitb14 768d(L2 norm)。
輸出:  data/eval/gt_mask_feats/<scene>.npz  {names(物體), views, clip(N,512), dino(N,768)}。
需 webots_visual_hull(open_clip + 本機 DINOv2 權重)。
用法: ./srp/stage4_probe/gt_mask_feats.py stack3_scene0001 [stack3 stack4 stack5]
env: CAPTURES_ROOT(captures_fast)
"""
import json, os, sys, glob
import numpy as np, cv2, torch, open_clip
from pathlib import Path
from PIL import Image
from pycocotools import mask as mask_utils

REPO = Path(__file__).resolve().parents[2]
CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures")))
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "srp" / "io")); from labels import LABELS  # data/labels 分層(類別/數量/場景)
OUT = REPO / "data" / "eval" / "gt_mask_feats"
CLIP_MEAN = np.array([123, 116, 103], dtype=np.uint8)
dev = "cuda" if torch.cuda.is_available() else "cpu"

print("載入 CLIP(open_clip ViT-B-32/openai) + DINOv2 vitb14 ...")
clip_model, _, clip_prep = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model = clip_model.to(dev).eval()
dino = torch.hub.load('/home/cho/.cache/torch/hub/facebookresearch_dinov2_main',
                      'dinov2_vitb14', source='local').to(dev).eval()
DMEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
DSTD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def square_mean_crop(rgb, seg):
    ys, xs = np.nonzero(seg)
    if xs.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = rgb[y0:y1, x0:x1].copy()
    crop[~seg[y0:y1, x0:x1]] = CLIP_MEAN
    h, w = crop.shape[:2]; side = max(h, w)
    canvas = np.empty((side, side, 3), np.uint8); canvas[:] = CLIP_MEAN
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop
    return Image.fromarray(canvas)


@torch.no_grad()
def clip_feats(rgb, segs):
    crops, valid = [], []
    for i, s in enumerate(segs):
        c = square_mean_crop(rgb, s)
        if c is not None:
            crops.append(clip_prep(c)); valid.append(i)
    out = [None] * len(segs)
    if crops:
        f = clip_model.encode_image(torch.stack(crops).to(dev)).float()
        f = (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()
        for k, i in enumerate(valid):
            out[i] = f[k].astype(np.float32)
    return out


@torch.no_grad()
def dino_feats(rgb, segs):
    H, W = rgb.shape[:2]; H14, W14 = (H // 14) * 14, (W // 14) * 14
    im = cv2.resize(rgb, (W14, H14))
    x = (torch.from_numpy(im).permute(2, 0, 1).float() / 255 - DMEAN) / DSTD
    f = dino.forward_features(x[None].to(dev))['x_norm_patchtokens'][0]
    ph, pw = H14 // 14, W14 // 14; fmap = f.reshape(ph, pw, -1)
    out = []
    for s in segs:
        mm = cv2.resize(s.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST) > 0
        if mm.sum() == 0:
            out.append(None); continue
        v = fmap[torch.from_numpy(mm).to(dev)].mean(0); v = v / (v.norm() + 1e-9)
        out.append(v.cpu().numpy().astype(np.float32))
    return out


def gt_modal(scene):
    ann = LABELS / scene / "actual" / "annotations.json"
    if not ann.is_file():
        return None
    d = json.loads(ann.read_text())
    cat = {c["id"]: c["name"] for c in d["categories"]}
    vof = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
    mo = {}
    for a in d["annotations"]:
        nm = cat[a["category_id"]]
        if nm == "ur5e":
            continue
        m = mask_utils.decode(a["segmentation"]).astype(bool)
        if m.sum() == 0:
            continue
        mo.setdefault(vof[a["image_id"]], {})[nm] = m
    return mo


def resolve(targets):
    out = []
    for a in targets:
        if "scene" in a:
            out.append(a)
        else:
            out += [Path(p).parent.parent.name for p in glob.glob(str(LABELS / f"{a}_scene*/actual/annotations.json"))]
    return sorted(set(out))


def process(scene):
    mo = gt_modal(scene)
    if not mo:
        print(f"[skip] {scene}: 無 GT modal"); return
    g = scene.split("_")[0]; sdir = CAPTURES / f"multi_{g}" / scene
    names, views, clips, dinos = [], [], [], []
    for vn, objs in sorted(mo.items()):
        img = cv2.imread(str(sdir / f"{vn}.png"))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        olist = list(objs); segs = [objs[o] for o in olist]
        cf = clip_feats(rgb, segs); df = dino_feats(rgb, segs)
        for o, c, d in zip(olist, cf, df):
            if c is None or d is None:
                continue
            names.append(o); views.append(vn); clips.append(c); dinos.append(d)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{scene}.npz", names=np.array(names), views=np.array(views),
                        clip=np.array(clips), dino=np.array(dinos))
    print(f"[{scene}] {len(names)} 筆(物體×視角) → {OUT / f'{scene}.npz'}  物體: {sorted(set(names))}")


def main():
    for sc in resolve(sys.argv[1:] or ["stack3", "stack4", "stack5"]):
        process(sc)


if __name__ == "__main__":
    main()
