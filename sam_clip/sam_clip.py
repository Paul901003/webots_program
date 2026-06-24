#!/home/cho/.pyenv/versions/grounded_sam/bin/python3
"""sam_clip.py — SAM 全自動分割 + CLIP 分類(不需文字框、不需位置)。

流程:SamAutomaticMaskGenerator 切出整張圖所有遮罩 → 逐遮罩裁切 → CLIP 與
候選類別(+ 干擾類)比對 → 指派類別;每類別取分數最高的遮罩。

回傳 {class_name: 二值遮罩(H,W,uint8)},命名/用途與 evaluate_masks 的預測 mask 對齊,
可直接餵 build_torchhull 建 visual hull。

需在 grounded_sam 環境執行(torch + segment_anything(repo) + clip)。
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
GSA = REPO / "Grounded-Segment-Anything"
sys.path.insert(0, str(GSA / "segment_anything"))

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator  # noqa: E402
import clip  # noqa: E402

SAM_CHECKPOINT = str(GSA / "sam_vit_b_01ec64.pth")
SAM_ENCODER    = "vit_b"
CLIP_MODEL     = "ViT-B/32"
# 干擾類:讓桌面/手臂/背景等非物體遮罩被分到這些類而被丟棄
# DISTRACTORS = ["background", "table", "floor", "robot arm", "robot gripper", "wall", "shadow"]

DISTRACTORS = []

PROB_THRESHOLD = 0.3


def weight_dirname(clip_model: str = CLIP_MODEL, prob: float = PROB_THRESHOLD) -> str:
    """輸出資料夾名 sam_clip_<clip模型>_<prob>(與 grounded_sam 權重分層對稱)。
    例:ViT-B/32, 0.3 → sam_clip_vitb32_0.3"""
    m = clip_model.lower().replace("/", "").replace("-", "")
    return f"sam_clip_{m}_{prob:g}"


def load_models(device: torch.device, clip_model: str = CLIP_MODEL,
                points_per_side: int = 32):
    sam = sam_model_registry[SAM_ENCODER](checkpoint=SAM_CHECKPOINT).to(device)
    amg = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        min_mask_region_area=400,   # 濾掉太碎的小遮罩
    )
    cmodel, preprocess = clip.load(clip_model, device=device)
    return {"amg": amg, "clip": cmodel, "preprocess": preprocess, "device": device}


def _masked_crop(image_rgb: np.ndarray, seg: np.ndarray, bbox, pad: int = 12) -> Image.Image:
    x, y, w, h = [int(v) for v in bbox]
    H, W = image_rgb.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    
    crop = image_rgb[y0:y1, x0:x1].copy()
    m = seg[y0:y1, x0:x1]
    
    # 修正: 100% 塗成純灰色，完全抹除框內任何屬於背景的紋理
    crop[~m] = np.array([255, 255, 255], dtype=np.uint8)
    
    return Image.fromarray(crop)


def segment_and_classify(image_bgr, class_names, models,
                         prob_thresh: float = 0.3,
                         min_area_frac: float = 0.0005,
                         max_area_frac: float = 0.6):
    """回傳 ({class_name: mask}, records)。records=[(class, prob, area), ...] 供除錯。"""
    device = models["device"]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]
    out = {c: np.zeros((H, W), dtype=np.uint8) for c in class_names}

    masks = models["amg"].generate(image_rgb)
    lo, hi = min_area_frac * H * W, max_area_frac * H * W
    masks = [m for m in masks if lo <= m["area"] <= hi]
    if not masks:
        return out, []

    # CLIP 文字特徵:候選類別 + 干擾類
    prompts = list(class_names) + DISTRACTORS
    text_tok = clip.tokenize([f"a photo of a {p}" for p in prompts]).to(device)
    crops = [models["preprocess"](_masked_crop(image_rgb, m["segmentation"], m["bbox"])) for m in masks]
    img_in = torch.stack(crops).to(device)
    with torch.no_grad():
        tfeat = models["clip"].encode_text(text_tok)
        ifeat = models["clip"].encode_image(img_in)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
        ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
        logit_scale = models["clip"].logit_scale.exp()
        probs = (logit_scale * ifeat @ tfeat.T).softmax(dim=-1).cpu().numpy()

    n_cls = len(class_names)
    per_class_best = {}            # class -> (prob, seg)
    records = []
    for i, m in enumerate(masks):
        ci = int(probs[i].argmax())
        p = float(probs[i, ci])
        if ci >= n_cls:            # 被判為干擾類(桌面/手臂…)→ 丟棄
            continue
        if p < prob_thresh:        # 信心不足 → 丟棄
            continue
        cname = class_names[ci]
        records.append((cname, round(p, 3), int(m["area"])))
        for i, m in enumerate(masks):
            ci = int(probs[i].argmax())
            p = float(probs[i, ci])
            if ci >= n_cls or p < prob_thresh:
                continue
            cname = class_names[ci]
            records.append((cname, round(p, 3), int(m["area"])))
            
            # 直接用 bitwise_or 或 logical_or 將遮罩疊加，支援複數相同物體或被切割的物體
            out[cname] = cv2.bitwise_or(out[cname], m["segmentation"].astype(np.uint8))

    for cname, (p, seg) in per_class_best.items():
        out[cname] = seg.astype(np.uint8)
    return out, records
