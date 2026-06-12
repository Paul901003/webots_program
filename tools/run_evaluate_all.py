#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""run_evaluate_all.py — 批次評估(純讀遮罩,不載模型)。

對每個場景,用 --weight-dir 指定的 pipeline 遮罩資料夾(grounded_sam_<...> 或 sam_clip)
與 GT 比對。已有 summary.json 的場景自動跳過(--force 重算)。

用法:
  python tools/run_evaluate_all.py                                    # 預設 grounded_sam_0.25_0.25_0.8
  python tools/run_evaluate_all.py --weight-dir sam_clip             # 評估 SAM+CLIP 的遮罩
  python tools/run_evaluate_all.py --weight-dir grounded_sam_0.3_0.3_0.7
"""

import argparse
from pathlib import Path

from evaluate_masks import LABELS_ROOT, EVAL_ROOT, process_annotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-dir", default="grounded_sam_0.25_0.25_0.8",
                        help="data/eval/ 下的遮罩資料夾名(pipeline 輸出)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    weight_root = EVAL_ROOT / args.weight_dir
    if not weight_root.is_dir():
        raise SystemExit(f"找不到遮罩資料夾: {weight_root}（請先用對應 pipeline 產遮罩）")

    done = skipped = nomask = failed = 0
    for ann_path in sorted(LABELS_ROOT.glob("*/actual/annotations.json")):
        scene = ann_path.parent.parent.name
        group = scene.split("_")[0]
        pred_dir = weight_root / f"multi_{group}" / scene
        if not (pred_dir.is_dir() and any(pred_dir.glob("view_*_mask_*.png"))):
            nomask += 1
            continue
        if not args.force and (pred_dir / "summary.json").exists():
            skipped += 1
            continue
        try:
            process_annotations(ann_path, pred_dir)
            done += 1
        except Exception as e:
            print(f"  [錯誤] {scene}: {e}")
            failed += 1

    print(f"\n完成。評估 {done}  跳過(已完成) {skipped}  無遮罩 {nomask}  失敗 {failed}")


if __name__ == "__main__":
    main()
