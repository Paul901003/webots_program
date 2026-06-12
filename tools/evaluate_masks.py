#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""evaluate_masks.py — 純評估(產遮罩已拆到 grounded_sam/ 或 sam_clip/)。

讀現成的預測遮罩 view_XX_mask_<class>.png(由 grounded_sam 或 sam_clip 產生)與
GT(generate_labels 的 annotations.json),計算每物體每視角的 IoU/像素正確率,
輸出 results.csv、summary.json、visualizations/。

可評估任一條 pipeline 的遮罩:--pred-dir 指向該 pipeline 的場景遮罩資料夾即可。

用法:
  python tools/evaluate_masks.py \
    --labels data/labels/n3_scene0001/actual/annotations.json \
    --pred-dir data/eval/grounded_sam_0.25_0.25_0.8/multi_n3/n3_scene0001
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

TOOLS_DIR   = Path(__file__).resolve().parent
PROJECT_DIR = TOOLS_DIR.parent
LABELS_ROOT = PROJECT_DIR / "data" / "labels"
EVAL_ROOT   = PROJECT_DIR / "data" / "eval"

sys.path.insert(0, str(PROJECT_DIR / "controllers" / "ycb_supervisor"))
try:
    from config import PROMPT_TABLE
except Exception:
    PROMPT_TABLE = {}


def sanitize_mask_name(value: str) -> str:
    out = []
    for ch in value.strip().lower():
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
    return "".join(out).strip("_")


def ycb_name_to_class(name: str) -> str:
    if name in PROMPT_TABLE:
        return PROMPT_TABLE[name]
    parts = name.split("_")
    start = 1 if parts[0].isdigit() else 0
    return " ".join(parts[start:])


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    gt_b, pred_b = gt.astype(bool), pred.astype(bool)
    tp = (gt_b & pred_b).sum(); tn = (~gt_b & ~pred_b).sum()
    fp = (~gt_b & pred_b).sum(); fn = (gt_b & ~pred_b).sum()
    total = gt_b.size
    accuracy  = (tp + tn) / total
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp)      if (tp + fp) > 0      else 0.0
    recall    = tp / (tp + fn)      if (tp + fn) > 0      else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"iou": round(float(iou), 4), "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4), "recall": round(float(recall), 4),
            "f1": round(float(f1), 4), "gt_px": int(gt_b.sum()), "pred_px": int(pred_b.sum())}


def make_comparison_image(image_bgr, gt_mask, pred_mask, metrics) -> np.ndarray:
    vis = image_bgr.astype(np.float32)
    gt_b, pred_b = gt_mask.astype(bool), pred_mask.astype(bool)
    only_gt, only_pred, overlap = gt_b & ~pred_b, pred_b & ~gt_b, gt_b & pred_b
    a = 0.5
    vis[only_gt]   = vis[only_gt]   * (1 - a) + np.array([128, 0, 0], np.float32) * a
    vis[only_pred] = vis[only_pred] * (1 - a) + np.array([0, 0, 200], np.float32) * a
    vis[overlap]   = vis[overlap]   * (1 - a) + np.array([0, 180, 0], np.float32) * a
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    for i, line in enumerate([f"IoU: {metrics['iou']:.3f}", f"Acc: {metrics['accuracy']:.3f}",
                              f"P: {metrics['precision']:.3f}", f"R: {metrics['recall']:.3f}"]):
        y = 24 + i * 22
        cv2.putText(vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def process_annotations(labels_path: Path, pred_dir: Path,
                        output_dir: Path | None = None, category: str | None = None) -> dict:
    """讀 pred_dir 的預測遮罩 + labels_path 的 GT → 算指標。結果預設寫回 pred_dir。"""
    labels_path = Path(labels_path).resolve()
    pred_dir    = Path(pred_dir).resolve()
    output_dir  = Path(output_dir).resolve() if output_dir else pred_dir
    vis_dir     = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    with open(labels_path, encoding="utf-8") as f:
        coco = json.load(f)
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    images     = {img["id"]: img for img in coco["images"]}

    ycb_names = [c["name"] for c in coco["categories"]
                 if c["name"] != "ur5e" and (category is None or c["name"] == category)]
    ycb_to_class = {n: ycb_name_to_class(n) for n in ycb_names}

    ann_by_image: dict[int, dict[str, np.ndarray]] = {}
    for ann in coco["annotations"]:
        cat = categories[ann["category_id"]]
        if cat not in ycb_to_class:
            continue
        rle = ann["segmentation"]
        if isinstance(rle["counts"], str):
            rle = {"counts": rle["counts"].encode(), "size": rle["size"]}
        ann_by_image.setdefault(ann["image_id"], {})[cat] = mask_utils.decode(rle).astype(np.uint8)

    # 視覺化背景用的 Webots 拍攝圖
    scene_name  = labels_path.parent.parent.name
    group       = scene_name.split("_")[0]
    capture_dir = PROJECT_DIR / "data" / "captures" / f"multi_{group}" / scene_name

    rows, iou_list, acc_list = [], [], []
    total_count = detected_count = 0

    for img_id, obj_gt in sorted(ann_by_image.items()):
        img_info = images[img_id]
        W, H = img_info["width"], img_info["height"]
        cap = capture_dir / f"view_{int(img_id):02d}.png"
        bg = cv2.imread(str(cap)) if cap.exists() else np.full((H, W, 3), 128, np.uint8)

        for ycb_name, gt_mask in obj_gt.items():
            cls = ycb_to_class[ycb_name]
            pred_path = pred_dir / f"view_{int(img_id):02d}_mask_{sanitize_mask_name(cls)}.png"
            if pred_path.exists():
                pm = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
                pred_mask = (pm > 127).astype(np.uint8) if pm is not None else np.zeros((H, W), np.uint8)
            else:
                pred_mask = np.zeros((H, W), np.uint8)

            total_count += 1
            detected = int(pred_mask.sum()) > 0
            detected_count += int(detected)
            metrics = compute_metrics(gt_mask, pred_mask)
            iou_list.append(metrics["iou"]); acc_list.append(metrics["accuracy"])
            cv2.imwrite(str(vis_dir / f"eval_{img_id:04d}_{cls.replace(' ', '_')}.png"),
                        make_comparison_image(bg, gt_mask, pred_mask, metrics))
            rows.append({"image_id": img_id, "object_name": cls, "detected": int(detected), **metrics})

    summary = _summarize(rows, iou_list, acc_list, total_count, detected_count, output_dir)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[{scene_name}] total={summary['total_count']} det={summary['detection_rate']:.1%} "
          f"mIoU={summary['mean_iou']:.4f} → {output_dir}")
    return summary


def _summarize(rows, iou_list, acc_list, total_count, detected_count, output_dir) -> dict:
    if not rows:
        return {"total_count": 0, "detected_count": 0, "detection_rate": 0.0,
                "mean_iou": 0.0, "mean_accuracy": 0.0,
                "iou_ge_0.5": 0, "iou_ge_0.7": 0, "iou_ge_0.9": 0, "per_object": {}}
    with open(output_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    per_object = {}
    for obj in sorted({r["object_name"] for r in rows}):
        rr = [r for r in rows if r["object_name"] == obj]
        ious = [r["iou"] for r in rr]
        per_object[obj] = {
            "total_count": len(rr), "detected_count": sum(r["detected"] for r in rr),
            "detection_rate": round(sum(r["detected"] for r in rr) / len(rr), 4),
            "mean_iou": round(float(np.mean(ious)), 4),
            "mean_accuracy": round(float(np.mean([r["accuracy"] for r in rr])), 4),
            "iou_ge_0.5": sum(v >= 0.5 for v in ious), "iou_ge_0.7": sum(v >= 0.7 for v in ious),
            "iou_ge_0.9": sum(v >= 0.9 for v in ious)}
    return {"total_count": total_count, "detected_count": detected_count,
            "detection_rate": round(detected_count / total_count, 4) if total_count else 0.0,
            "mean_iou": round(float(np.mean(iou_list)), 4),
            "mean_accuracy": round(float(np.mean(acc_list)), 4),
            "iou_ge_0.5": sum(v >= 0.5 for v in iou_list), "iou_ge_0.7": sum(v >= 0.7 for v in iou_list),
            "iou_ge_0.9": sum(v >= 0.9 for v in iou_list), "per_object": per_object}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="generate_labels 的 annotations.json")
    parser.add_argument("--pred-dir", required=True,
                        help="預測遮罩資料夾(含 view_XX_mask_<class>.png),由 grounded_sam/sam_clip 產生")
    parser.add_argument("--output", default=None, help="結果輸出目錄(預設=pred-dir)")
    parser.add_argument("--category", default=None)
    args = parser.parse_args()
    process_annotations(Path(args.labels), Path(args.pred_dir),
                        Path(args.output) if args.output else None, args.category)


if __name__ == "__main__":
    main()
