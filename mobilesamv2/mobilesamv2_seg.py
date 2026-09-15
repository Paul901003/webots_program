#!/home/cho/.pyenv/versions/grounded_sam/bin/python3
"""mobilesamv2_seg.py — 用 MobileSAMv2 產 class-agnostic 遮罩(對照 sam_only)。

MobileSAMv2 = ObjectAwareModel(YOLO 出物件框) → efficientvit_l2 編碼 → PromptGuidedDecoder(框→遮罩)。
輸出結構**與 sam_only 完全相同**,故下游(sam_recall / hull / associate)用 SAM_ROOT 覆寫即可比較:
  overlay.png          所有遮罩以不同顏色疊在原圖上(也是 skip-done marker)
  masks/mask_000.png … 每張遮罩二值圖(依面積大→小)
  meta.txt             每張遮罩 area / bbox

輸出(SAM_OUT_ROOT;預設 data/eval/mobilesamv2_fast/<scene>/<view>/)。需 grounded_sam 環境。

用法(與 sam_only 相同):
  ./mobilesamv2/mobilesamv2_seg.py n3_scene0001        # 整個場景
  ./mobilesamv2/mobilesamv2_seg.py 3                   # 整組 n3
  ./mobilesamv2/mobilesamv2_seg.py occ3 stack3         # 多組
  ./mobilesamv2/mobilesamv2_seg.py --input-image <img> # 單張
  FORCE=1 ...                                          # 重做
env: CAPTURES_ROOT(輸入,預設 data/captures) SAM_OUT_ROOT(輸出,預設 data/eval/mobilesamv2_fast) FORCE
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np
import torch

MV2_SEG_DIR = Path(__file__).resolve().parent
REPO = MV2_SEG_DIR.parent
MV2_DIR = REPO / "MobileSAM" / "MobileSAMv2"
sys.path.insert(0, str(MV2_DIR))

from mobilesamv2 import sam_model_registry, SamPredictor                 # noqa: E402
from mobilesamv2.promt_mobilesamv2 import ObjectAwareModel               # noqa: E402

CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures")))
OUT_ROOT = Path(os.environ.get("SAM_OUT_ROOT", str(REPO / "data" / "eval" / "mobilesamv2_fast")))
FORCE = os.environ.get("FORCE") == "1"

# 權重(絕對路徑,可從任意 cwd 跑)
OBJ_MODEL_PT = str(MV2_DIR / "weight" / "ObjectAwareModel.pt")
DECODER_PT = str(MV2_DIR / "PromptGuidedDecoder" / "Prompt_guided_Mask_Decoder.pt")
# 影像編碼器可用 env MV2_ENCODER 切換(efficientvit_l2 官方預設 / tiny_vit / sam_vit_h)
ENCODER_WEIGHTS = {"efficientvit_l2": "l2.pt", "tiny_vit": "mobile_sam.pt", "sam_vit_h": "sam_vit_h.pt"}
ENCODER_TYPE = os.environ.get("MV2_ENCODER", "efficientvit_l2")
ENCODER_PT = str(MV2_DIR / "weight" / ENCODER_WEIGHTS[ENCODER_TYPE])


def resolve_views(targets):
    """targets: 場景名(n3_scene0001)或組號(3/occ3/stack3)。回傳 [view_XX.png 路徑, ...]。"""
    views = []
    for a in targets:
        if "scene" in a:
            g = a.split("_")[0]
            d = CAPTURES / f"multi_{g}" / a
            if not d.is_dir():
                print(f"[warn] 找不到場景: {d}"); continue
            views += [v for v in sorted(d.glob("view_*.png")) if "_depth" not in v.name]
        else:
            g = f"n{a}" if a.isdigit() else a   # "3"→n3;"occ3"/"stack3"/"n3" 直接當組名
            for d in sorted((CAPTURES / f"multi_{g}").glob(f"{g}_scene*")):
                views += [v for v in sorted(d.glob("view_*.png")) if "_depth" not in v.name]
    return views


def default_out_dir(img_path: Path) -> Path:
    return OUT_ROOT / img_path.parent.name / img_path.stem


def batch_iterator(batch_size: int, *args):
    assert len(args) > 0 and all(len(a) == len(args[0]) for a in args)
    n = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n):
        yield [arg[b * batch_size:(b + 1) * batch_size] for arg in args]


def create_model(device):
    """建 MobileSAMv2(vit_h 骨架 + PromptGuided 的 prompt/mask decoder + efficientvit_l2 encoder)+ ObjectAwareModel。"""
    ObjAwareModel = ObjectAwareModel(OBJ_MODEL_PT)
    pg = sam_model_registry["PromptGuidedDecoder"](DECODER_PT)
    m = sam_model_registry["vit_h"]()
    m.prompt_encoder = pg["PromtEncoder"]
    m.mask_decoder = pg["MaskDecoder"]
    m.image_encoder = sam_model_registry[ENCODER_TYPE](ENCODER_PT)
    m.to(device=device)
    m.eval()
    return m, ObjAwareModel


def infer_masks(model, ObjAwareModel, predictor, image_rgb, device) -> torch.Tensor:
    """回傳 [N,H,W] bool 遮罩(N=偵測框數;0 框則回空 tensor)。"""
    obj = ObjAwareModel(image_rgb, device=device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
    boxes_xyxy = obj[0].boxes.xyxy
    if boxes_xyxy is None or boxes_xyxy.shape[0] == 0:
        return torch.zeros((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=torch.bool)
    predictor.set_image(image_rgb)
    boxes = predictor.transform.apply_boxes(boxes_xyxy.cpu().numpy(), predictor.original_size)
    boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device)
    img_emb_all = predictor.features
    pe_all = model.prompt_encoder.get_dense_pe()
    sam_mask = []
    for (b,) in batch_iterator(320, boxes):
        with torch.no_grad():
            img_emb = img_emb_all.repeat_interleave(b.shape[0], dim=0) if img_emb_all.shape[0] == 1 \
                else img_emb_all[:b.shape[0]]
            pe = pe_all.repeat_interleave(b.shape[0], dim=0) if pe_all.shape[0] == 1 else pe_all[:b.shape[0]]
            sparse, dense = model.prompt_encoder(points=None, boxes=b, masks=None)
            low_res, _ = model.mask_decoder(
                image_embeddings=img_emb, image_pe=pe,
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=False, simple_type=True)
            low_res = predictor.model.postprocess_masks(low_res, predictor.input_size, predictor.original_size)
            sam_mask.append((low_res > model.mask_threshold).squeeze(1))
    return torch.cat(sam_mask, dim=0).bool().cpu()


def process_image(model, ObjAwareModel, predictor, img_path: Path, out_dir: Path,
                  alpha: float, device) -> int:
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        print(f"  [warn] 讀不到 {img_path}"); return -1
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    masks_t = infer_masks(model, ObjAwareModel, predictor, image_rgb, device)  # [N,H,W] bool
    segs = [masks_t[i].numpy() for i in range(masks_t.shape[0])]
    segs.sort(key=lambda s: int(s.sum()), reverse=True)   # 依面積大→小(與 sam_only 一致)

    rng = np.random.default_rng(0)
    overlay = image_bgr.astype(np.float32)
    meta = []
    for i, seg in enumerate(segs):
        if seg.sum() == 0:
            continue
        color = rng.integers(0, 255, size=3).astype(np.float32)
        overlay[seg] = overlay[seg] * (1 - alpha) + color * alpha
        cv2.imwrite(str(mask_dir / f"mask_{i:03d}.png"), (seg.astype(np.uint8) * 255))
        ys, xs = np.where(seg)
        x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        meta.append(f"mask_{i:03d}  area={int(seg.sum()):8d}  bbox=({x},{y},{w},{h})")
    # overlay 一律寫出(含 0 框時=原圖)→ 當 skip-done marker
    cv2.imwrite(str(out_dir / "overlay.png"), np.clip(overlay, 0, 255).astype(np.uint8))
    (out_dir / "meta.txt").write_text("\n".join(meta), encoding="utf-8")
    return len(segs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", help="場景名(n3_scene0001)或組號(3/occ3/stack3)")
    ap.add_argument("--input-image", default=None, help="單張影像(優先於 targets)")
    ap.add_argument("--output-dir", default=None, help="輸出目錄(單張時可指定;批次時忽略)")
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.input_image:
        jobs = [(Path(args.input_image),
                 Path(args.output_dir) if args.output_dir else default_out_dir(Path(args.input_image)))]
    else:
        if not args.targets:
            sys.exit("請給場景名/組號(如 n3_scene0001 或 3),或用 --input-image")
        jobs = [(v, default_out_dir(v)) for v in resolve_views(args.targets)]
    if not jobs:
        sys.exit("沒有可處理的影像")

    print(f"載入 MobileSAMv2 ({ENCODER_TYPE}, {args.device}) ... 影像數 {len(jobs)}")
    model, ObjAwareModel = create_model(torch.device(args.device))
    predictor = SamPredictor(model)

    for i, (img_path, out_dir) in enumerate(jobs, 1):
        if out_dir.exists() and (out_dir / "overlay.png").exists() and not FORCE:
            print(f"  [{i}/{len(jobs)}] {img_path.parent.name}/{img_path.stem} 已存在,跳過")
            continue
        n = process_image(model, ObjAwareModel, predictor, img_path, out_dir, args.alpha,
                          torch.device(args.device))
        if n >= 0:
            print(f"  [{i}/{len(jobs)}] {img_path.parent.name}/{img_path.stem}: {n} 張遮罩 → {out_dir}")

    print(f"\n完成。輸出根目錄: {OUT_ROOT}")


if __name__ == "__main__":
    main()
