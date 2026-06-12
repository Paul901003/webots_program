#!/home/cho/.pyenv/versions/grounded_sam/bin/python3
"""run_sam_clip.py — 對多物體場景批次跑 SAM 全自動分割 + CLIP 分類,輸出每物體遮罩。

候選類別取自場景 manifest 的物體(經 PROMPT_TABLE → prompt)。對每張 Webots 拍攝圖
view_XX.png 產生 {類別: 遮罩},存成 view_XX_mask_<class>.png(與 build_torchhull 對齊),
可直接餵 build_torchhull 建 visual hull。

輸出: data/eval/sam_clip/multi_n{N}/<scene>/

用法:
  ./sam_clip/run_sam_clip.py n3_scene0001       # 單一場景
  ./sam_clip/run_sam_clip.py 3                   # 整組 n3
  ./sam_clip/run_sam_clip.py 1 3 4 5            # 多組
  FORCE=1 ./sam_clip/run_sam_clip.py 3          # 重做(忽略已存在)
"""

import json
import os
import sys
from pathlib import Path

import cv2
import torch

SC_DIR = Path(__file__).resolve().parent
REPO = SC_DIR.parent
sys.path.insert(0, str(SC_DIR))
sys.path.insert(0, str(REPO / "controllers" / "ycb_supervisor"))

from sam_clip import load_models, segment_and_classify, weight_dirname, CLIP_MODEL, PROB_THRESHOLD  # noqa: E402
try:
    from config import PROMPT_TABLE
except Exception:
    PROMPT_TABLE = {}

CAPTURES = REPO / "data" / "captures"
EVAL = REPO / "data" / "eval"
FORCE = os.environ.get("FORCE") == "1"


def ycb_name_to_class(name: str) -> str:
    if name in PROMPT_TABLE:
        return PROMPT_TABLE[name]
    p = name.split("_"); s = 1 if p[0].isdigit() else 0
    return " ".join(p[s:])


def sanitize(value: str) -> str:
    out = []
    for ch in value.strip().lower():
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
    return "".join(out).strip("_")


def resolve_scenes(args):
    """args 可為場景名(n3_scene0001)或組號(3)。回傳 [(group, scene_dir), ...]"""
    scenes = []
    for a in args:
        if "scene" in a:                                  # 場景名
            g = a.split("_")[0]
            d = CAPTURES / f"multi_{g}" / a
            if d.is_dir():
                scenes.append((g, d))
            else:
                print(f"[warn] 找不到場景: {d}")
        else:                                             # 組號
            g = f"n{a}"
            for d in sorted((CAPTURES / f"multi_{g}").glob(f"{g}_scene*")):
                scenes.append((g, d))
    return scenes


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="*", default=["1", "3", "4", "5"],
                        help="場景名(n3_scene0001)或組號(3)")
    parser.add_argument("--clip-model", default=CLIP_MODEL, help="CLIP 模型,如 ViT-B/32 或 ViT-B/16")
    parser.add_argument("--prob-threshold", type=float, default=PROB_THRESHOLD,
                        help="CLIP 分類信心門檻(同時決定輸出資料夾名)")
    args = parser.parse_args()

    weight = weight_dirname(args.clip_model, args.prob_threshold)
    out_root = EVAL / weight
    scenes = resolve_scenes(args.targets or ["1", "3", "4", "5"])
    if not scenes:
        sys.exit("沒有可處理的場景")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"載入 SAM + CLIP ({device}) clip={args.clip_model} prob={args.prob_threshold:g} → {weight}"
          f"  場景數 {len(scenes)}")
    models = load_models(device, clip_model=args.clip_model)

    for i, (g, scene_dir) in enumerate(scenes, 1):
        scene = scene_dir.name
        out_dir = out_root / f"multi_{g}" / scene
        manifest = json.loads((scene_dir / "scene_manifest.json").read_text())
        names = [o["name"] for o in manifest["planned"]["objects"]]
        class_names = [ycb_name_to_class(n) for n in names]

        done = out_dir.exists() and any(out_dir.glob("view_*_mask_*.png"))
        if done and not FORCE:
            print(f"  [{i}/{len(scenes)}] {scene} 已存在,跳過")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        views = sorted(scene_dir.glob("view_*.png"))
        views = [v for v in views if "_depth" not in v.name]
        print(f"  [{i}/{len(scenes)}] {scene}  classes={class_names}  views={len(views)}")
        hit = {c: 0 for c in class_names}
        for v in views:
            image_bgr = cv2.imread(str(v))
            if image_bgr is None:
                print(f"    [warn] 讀不到 {v.name}"); continue
            masks, _ = segment_and_classify(image_bgr, class_names, models,
                                            prob_thresh=args.prob_threshold)
            for cls, m in masks.items():
                cv2.imwrite(str(out_dir / f"{v.stem}_mask_{sanitize(cls)}.png"),
                            (m.astype("uint8") * 255))
                if int(m.sum()) > 0:
                    hit[cls] += 1
        print("      命中視角數:", {c: hit[c] for c in class_names})

    print(f"\n完成。輸出: {out_root}")


if __name__ == "__main__":
    main()
