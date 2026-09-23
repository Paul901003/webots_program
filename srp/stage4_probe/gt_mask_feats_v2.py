#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""gt_mask_feats_v2.py — 同 gt_mask_feats.py,但修正兩個問題並一次產出三版 DINO 特徵。

★ 另開新檔,原 gt_mask_feats.py 與 data/eval/gt_mask_feats/ 完全不動(不覆蓋既有成果)。

與原版(gt_mask_feats.py)的差異:
  1. 【視角基準】原版跑 annotations.json 裡【全部 34 視角】,與 srp 管線的 12 視角(A-3 selected)不一致
     → 本版限 srp/io/viewpoints.selected_view_names(12),與管線/SigLIP2 比較同基準。
  2. 【patch 選取與聚合】原版 dino_feats 行 73 用 INTER_NEAREST:每個 14x14=196px 的 patch 只檢查 1 個取樣點,
     選中後一律等權平均 → 實測(RESULT_dino_patch_coverage.md,60場8090筆):19.0% 的特徵權重來自覆蓋<50%
     的 patch、與 cov>=0.5 選法不一致率 29.6%、細長物(刀/叉/湯匙/夾鉗)達 33~41%。
     → 本版改用【真實覆蓋面積】決定納入,產出兩版,【納入規則相同、只差加權】(單一變因=加權):
        dino_cw : 有覆蓋(cov>0)+ 面積加權   v = sum(cov_p * f_p) / sum(cov_p)
        dino_ce : 有覆蓋(cov>0)+ 等權平均   v = mean(f_p), p in {cov>0}
     覆蓋率 cov_p = 該 patch 的 196 個像素落在遮罩內的比例,在 714x1274(模型實際輸入尺寸)上算,與 patch 網格對齊。
     ⚠ cov>0 納入會讓交界 patch 被相鄰兩物體同時取用(實測 stack3 前10場:6.7% 的 patch 屬此情形);
       原版 NEAREST 因中央點只有一個歸屬則不會。對相觸堆疊物是優點或缺點【待驗證】,正是本次對照要回答的。

CLIP 側與原版完全相同(square_mean_crop 填 CLIP mean -> ViT-B-32/openai -> 512d L2),作為比較基準。

輸出: data/eval/gt_mask_feats_v2/<scene>.npz
       {names, views, clip(N,512), dino_cw(N,768), dino_ce(N,768)}
一致性檢查: --verify 比對 CLIP 側與原版 npz(CLIP 程式未改,應逐筆相同)。
需 webots_visual_hull(open_clip + DINOv2 torch.hub)。
用法: ./gt_mask_feats_v2.py [stack3 stack4 stack5] [--verify]
env : CAPTURES_ROOT(預設 captures_fast)
"""
import json
import os
import sys
import glob
from pathlib import Path

import numpy as np
import cv2
import torch
import open_clip
from PIL import Image
from pycocotools import mask as mask_utils

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "srp" / "io"))
from labels import LABELS       # noqa: E402
import viewpoints as VP         # noqa: E402

CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures_fast")))
OUT = REPO / "data" / "eval" / "gt_mask_feats_v2"
OLD = REPO / "data" / "eval" / "gt_mask_feats"
CLIP_MEAN = np.array([123, 116, 103], dtype=np.uint8)
PATCH = 14
dev = "cuda" if torch.cuda.is_available() else "cpu"

print("載入 CLIP(open_clip ViT-B-32/openai) + DINOv2 vitb14 ...", flush=True)
clip_model, _, clip_prep = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model = clip_model.to(dev).eval()
dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', verbose=False).to(dev).eval()
DMEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
DSTD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def square_mean_crop(rgb, seg):
    """與原版 gt_mask_feats.square_mean_crop 完全相同。"""
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
    """與原版完全相同。"""
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
def dino_feats2(rgb, segs):
    """整圖跑一次 DINOv2,對每個遮罩回兩版 768d 特徵。

    兩版【納入規則完全相同】= 所有與遮罩有交集的 patch(cov>0),只差聚合權重 → 單一變因是加權:
      cw(面積加權): v = sum(cov_p * f_p) / sum(cov_p)     邊界 patch 依覆蓋比例降權
      ce(等權平均): v = mean(f_p),  p in {cov>0}          每個有交集的 patch 權重相同
    覆蓋率 cov_p = 該 patch 的 196 個像素落在遮罩內的比例,在 H14xW14(模型實際輸入尺寸)上算。
    """
    H, W = rgb.shape[:2]
    H14, W14 = (H // PATCH) * PATCH, (W // PATCH) * PATCH
    ph, pw = H14 // PATCH, W14 // PATCH
    im = cv2.resize(rgb, (W14, H14))
    x = (torch.from_numpy(im).permute(2, 0, 1).float() / 255 - DMEAN) / DSTD
    f = dino.forward_features(x[None].to(dev))['x_norm_patchtokens'][0]   # (ph*pw, 768)
    fmap = f.reshape(ph, pw, -1)
    out = []
    for s in segs:
        u8 = s.astype(np.uint8)
        s14 = cv2.resize(u8, (W14, H14), interpolation=cv2.INTER_NEAREST)
        cov = s14.reshape(ph, PATCH, pw, PATCH).sum(axis=(1, 3)) / float(PATCH * PATCH)
        hit = cov > 0
        if not hit.any():
            out.append((None, None)); continue
        covt = torch.from_numpy(cov).float().to(dev)

        def norm(v):
            return (v / (v.norm() + 1e-9)).cpu().numpy().astype(np.float32)

        v_cw = norm((fmap * covt[:, :, None]).sum((0, 1)) / covt.sum())
        v_ce = norm(fmap[torch.from_numpy(hit).to(dev)].mean(0))
        out.append((v_cw, v_ce))
    return out


def gt_modal(scene):
    """與原版 gt_mask_feats.gt_modal 完全相同(排 ur5e、排全空遮罩)。"""
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


def process(scene, sel_views):
    mo = gt_modal(scene)
    if not mo:
        print(f"[skip] {scene}: 無 GT modal"); return
    g = scene.split("_")[0]; sdir = CAPTURES / f"multi_{g}" / scene
    names, views, clips, dcw, dce = [], [], [], [], []
    for vn in sorted(sel_views):                     # ★ 只跑 A-3 selected 12 視角
        objs = mo.get(vn)
        if not objs:
            continue
        img = cv2.imread(str(sdir / f"{vn}.png"))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        olist = list(objs); segs = [objs[o] for o in olist]
        cf = clip_feats(rgb, segs); df = dino_feats2(rgb, segs)
        for o, c, (a, b) in zip(olist, cf, df):
            if c is None or a is None or b is None:
                continue                              # 三者皆有才收(保持逐筆可對齊比較)
            names.append(o); views.append(vn); clips.append(c); dcw.append(a); dce.append(b)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{scene}.npz", names=np.array(names), views=np.array(views),
                        clip=np.array(clips), dino_cw=np.array(dcw), dino_ce=np.array(dce))
    print(f"[{scene}] {len(names)} 筆(物體×視角,{len(set(views))} 視角) → {OUT / f'{scene}.npz'}", flush=True)


def verify():
    """一致性檢查:CLIP 側程式與原版【完全相同】,故共同 (物體,視角) 的 clip 向量應逐筆相同。

    差 ~0 → 影像來源/遮罩讀法/視角命名全部對齊,重跑無偏差。
    差很大 → 兩次跑的 CAPTURES_ROOT 不同(原版預設 data/captures 有雜訊,本版用 captures_fast 無雜訊),
             此時 DINO 側亦不可與原版逐筆比較,只能比同一批資料內的三種聚合法。
    """
    tot, mx, cnt = 0, 0.0, 0
    for p in sorted(OUT.glob("*.npz")):
        op = OLD / p.name
        if not op.is_file():
            continue
        a = np.load(p); b = np.load(op)
        ka = {(n, v): i for i, (n, v) in enumerate(zip(a["names"], a["views"]))}
        kb = {(n, v): i for i, (n, v) in enumerate(zip(b["names"], b["views"]))}
        for k in set(ka) & set(kb):
            mx = max(mx, float(np.abs(a["clip"][ka[k]] - b["clip"][kb[k]]).max())); cnt += 1
        tot += 1
    print(f"[verify] 比對 {tot} 場、{cnt} 筆共同 (物體,視角);clip 最大逐元素差 = {mx:.3e}"
          f"  ({'對齊' if mx < 1e-4 else '★不對齊→影像來源不同,見 docstring'})")


def main():
    args = [a for a in sys.argv[1:] if a != "--verify"]
    targets = args or ["stack3", "stack4", "stack5"]
    scenes = []
    for a in targets:
        if "scene" in a:
            scenes.append(a)
        else:
            scenes += [Path(p).parent.parent.name
                       for p in glob.glob(str(LABELS / f"{a}_scene*/actual/annotations.json"))]
    scenes = sorted(set(scenes))
    sel = set(VP.selected_view_names(12))
    print(f"場景 {len(scenes)} 場;視角 {len(sel)}(A-3 selected)", flush=True)
    for sc in scenes:
        process(sc, sel)
    if "--verify" in sys.argv:
        verify()


if __name__ == "__main__":
    main()
