#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""grounded_sam.py — Grounded-SAM 產遮罩核心(GroundingDINO 文字找框 → SAM 分割)。

pipeline A 的「產遮罩」部分,從舊 evaluate_masks 拆出,與 sam_clip 對稱。
回傳 {class_name: 二值遮罩},命名/用途與 sam_clip 一致(view_XX_mask_<class>.png),
供 evaluate_masks(評估)與 build_torchhull(建殼)使用。

需在 webots_visual_hull 環境執行(GroundingDINO + SAM)。
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
GSA = REPO / "Grounded-Segment-Anything"
for pkg in (GSA / "GroundingDINO", GSA / "segment_anything"):
    if pkg.exists() and str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
sys.path.insert(0, str(REPO / "controllers" / "ycb_supervisor"))

from groundingdino.util.inference import Model as GroundingDINOModel  # noqa: E402
from segment_anything import SamPredictor, sam_model_registry  # noqa: E402
try:
    from config import PROMPT_TABLE
except Exception:
    PROMPT_TABLE = {}

DINO_CONFIG     = str(GSA / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
DINO_CHECKPOINT = str(GSA / "groundingdino_swint_ogc.pth")
SAM_CHECKPOINT  = str(GSA / "sam_vit_b_01ec64.pth")
SAM_ENCODER     = "vit_b"

BOX_THRESHOLD  = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD  = 0.8


def weight_dirname(box=None, text=None, nms=None) -> str:
    """輸出權重資料夾名 grounded_sam_<box>_<text>_<nms>。"""
    b = BOX_THRESHOLD if box is None else box
    t = TEXT_THRESHOLD if text is None else text
    n = NMS_THRESHOLD if nms is None else nms
    return f"grounded_sam_{b:g}_{t:g}_{n:g}"


def set_thresholds(box: float, text: float, nms: float) -> None:
    global BOX_THRESHOLD, TEXT_THRESHOLD, NMS_THRESHOLD
    BOX_THRESHOLD, TEXT_THRESHOLD, NMS_THRESHOLD = box, text, nms


def sanitize_mask_name(value: str) -> str:
    out = []
    for ch in value.strip().lower():
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
    return "".join(out).strip("_")


def ycb_name_to_class(name: str) -> str:
    """物體名 → SAM prompt。優先查 config.PROMPT_TABLE,缺則回退舊規則。"""
    if name in PROMPT_TABLE:
        return PROMPT_TABLE[name]
    parts = name.split("_")
    start = 1 if parts[0].isdigit() else 0
    return " ".join(parts[start:])


def classes_to_prompt(classes: list[str]) -> str:
    return " . ".join(classes)


def load_models(device: torch.device):
    print(f"[GroundedSAM] 載入 GroundingDINO ({device})...")
    dino = GroundingDINOModel(model_config_path=DINO_CONFIG,
                              model_checkpoint_path=DINO_CHECKPOINT, device=device)
    print(f"[GroundedSAM] 載入 SAM ({SAM_ENCODER})...")
    sam = sam_model_registry[SAM_ENCODER](checkpoint=SAM_CHECKPOINT).to(device)
    return {"dino": dino, "sam": SamPredictor(sam)}


def predict_masks_per_class(models, image_bgr, prompt_classes: list[str]) -> dict[str, np.ndarray]:
    """文字找框 → NMS → SAM 分割,回傳 {prompt類別: 二值遮罩};未偵測為全零。"""
    dino, sam_predictor = models["dino"], models["sam"]
    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    class_masks = {cls: np.zeros((H, W), dtype=np.uint8) for cls in prompt_classes}

    detections = dino.predict_with_classes(
        image=image_rgb, classes=prompt_classes,
        box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD)
    if len(detections.xyxy) == 0:
        return class_masks

    import torchvision
    ids = detections.class_id.tolist() if detections.class_id is not None else []
    sanitized = np.array([-1 if c is None else int(c) for c in ids], dtype=np.int64)
    nms_idx = torchvision.ops.batched_nms(
        torch.from_numpy(detections.xyxy).float(),
        torch.from_numpy(detections.confidence).float(),
        torch.from_numpy(sanitized), NMS_THRESHOLD).numpy()
    detections = detections[nms_idx]

    sam_predictor.set_image(image_rgb)
    for box, class_id in zip(detections.xyxy, detections.class_id):
        if class_id is None:
            continue
        with torch.inference_mode():
            masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
        mask = masks[np.argmax(scores)].astype(bool)
        cls = prompt_classes[int(class_id)]
        class_masks[cls] = (class_masks[cls].astype(bool) | mask).astype(np.uint8)
    return class_masks
