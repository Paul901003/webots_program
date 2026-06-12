#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""run_grounded_sam.py — Grounded-SAM 批次產遮罩(pipeline A,與 run_sam_clip 對稱)。

對每張 Webset 拍攝圖 view_XX.png,用場景物體(經 PROMPT_TABLE → prompt)做
GroundingDINO→SAM,輸出 view_XX_mask_<class>.png 到
data/eval/grounded_sam_<box>_<text>_<nms>/multi_n{N}/<scene>/。
之後交給 evaluate_masks(評估)或 build_torchhull(建殼)。

需在 webots_visual_hull 環境執行。

用法:
  ./grounded_sam/run_grounded_sam.py n3_scene0001
  ./grounded_sam/run_grounded_sam.py 1 3 4 5
  ./grounded_sam/run_grounded_sam.py 3 --box-threshold 0.3 --text-threshold 0.3 --nms-threshold 0.7
  FORCE=1 ./grounded_sam/run_grounded_sam.py 3
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import torch

GS_DIR = Path(__file__).resolve().parent
REPO = GS_DIR.parent
sys.path.insert(0, str(GS_DIR))

import grounded_sam as G  # noqa: E402

CAPTURES = REPO / "data" / "captures"
EVAL = REPO / "data" / "eval"
FORCE = os.environ.get("FORCE") == "1"


def resolve_scenes(tokens):
    scenes = []
    for a in tokens:
        if "scene" in a:
            g = a.split("_")[0]
            d = CAPTURES / f"multi_{g}" / a
            scenes.append((g, d)) if d.is_dir() else print(f"[warn] 找不到場景: {d}")
        else:
            g = f"n{a}"
            for d in sorted((CAPTURES / f"multi_{g}").glob(f"{g}_scene*")):
                scenes.append((g, d))
    return scenes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="*", default=["1", "3", "4", "5"],
                        help="場景名(n3_scene0001)或組號(3)")
    parser.add_argument("--box-threshold", type=float, default=G.BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=G.TEXT_THRESHOLD)
    parser.add_argument("--nms-threshold", type=float, default=G.NMS_THRESHOLD)
    args = parser.parse_args()

    G.set_thresholds(args.box_threshold, args.text_threshold, args.nms_threshold)
    weight = G.weight_dirname()
    scenes = resolve_scenes(args.targets or ["1", "3", "4", "5"])
    if not scenes:
        sys.exit("沒有可處理的場景")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"門檻 {weight}  場景數 {len(scenes)}")
    models = G.load_models(device)

    for i, (g, scene_dir) in enumerate(scenes, 1):
        scene = scene_dir.name
        out_dir = EVAL / weight / f"multi_{g}" / scene
        manifest = json.loads((scene_dir / "scene_manifest.json").read_text())
        names = [o["name"] for o in manifest["planned"]["objects"]]
        prompt_classes = [G.ycb_name_to_class(n) for n in names]

        if out_dir.exists() and any(out_dir.glob("view_*_mask_*.png")) and not FORCE:
            print(f"  [{i}/{len(scenes)}] {scene} 已存在,跳過")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        views = [v for v in sorted(scene_dir.glob("view_*.png")) if "_depth" not in v.name]
        print(f"  [{i}/{len(scenes)}] {scene}  classes={prompt_classes}  views={len(views)}")
        hit = {c: 0 for c in prompt_classes}
        for v in views:
            image_bgr = cv2.imread(str(v))
            if image_bgr is None:
                print(f"    [warn] 讀不到 {v.name}"); continue
            masks = G.predict_masks_per_class(models, image_bgr, prompt_classes)
            for cls, m in masks.items():
                cv2.imwrite(str(out_dir / f"{v.stem}_mask_{G.sanitize_mask_name(cls)}.png"),
                            (m.astype("uint8") * 255))
                if int(m.sum()) > 0:
                    hit[cls] += 1
        print("      命中視角數:", hit)

    print(f"\n完成。輸出: {EVAL / weight}")


if __name__ == "__main__":
    main()
