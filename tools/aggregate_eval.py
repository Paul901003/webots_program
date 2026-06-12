#!/usr/bin/env python3
"""aggregate_eval.py

彙總各場景 evaluate_masks 結果,計算 n1/n3/n4/n5 + overall。
依權重儲存:掃 data/eval/grounded_sam_*/multi_n{N}/<scene>/summary.json,
每個權重資料夾各自輸出 eval_summary.json。
"""

import json
import re
from collections import defaultdict
from pathlib import Path

TOOLS_DIR   = Path(__file__).resolve().parent
PROJECT_DIR = TOOLS_DIR.parent
EVAL_ROOT   = PROJECT_DIR / "data" / "eval"

GROUPS   = ("n1", "n3", "n4", "n5")
SCENE_RE = re.compile(r"^(n\d+)_scene\d+$")


def parse_group(scene_name: str) -> str | None:
    m = SCENE_RE.match(scene_name)
    return m.group(1) if m else None


def collect_summaries(weight_dir: Path) -> dict[str, list[dict]]:
    """回傳 {group: [summary, ...]};掃 weight_dir/multi_n*/<scene>/summary.json"""
    by_group: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(weight_dir.glob("multi_n*/*/summary.json")):
        scene_name = path.parent.name
        group = parse_group(scene_name)
        if group is None:
            print(f"[warn] 無法解析 group,跳過: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
        summary["_scene"] = scene_name
        by_group[group].append(summary)
    return dict(by_group)


def aggregate_group(summaries: list[dict]) -> dict:
    total_count    = sum(s.get("total_count", 0) for s in summaries)
    detected_count = sum(s.get("detected_count", 0) for s in summaries)
    if total_count == 0:
        return {"scenes": len(summaries), "total_count": 0, "detected_count": 0,
                "detection_rate": None, "mean_iou": None, "mean_accuracy": None,
                "iou_ge_0.5": 0, "iou_ge_0.7": 0, "iou_ge_0.9": 0}
    mean_iou = sum(s["mean_iou"] * s["total_count"] for s in summaries) / total_count
    mean_acc = sum(s["mean_accuracy"] * s["total_count"] for s in summaries) / total_count
    return {
        "scenes":         len(summaries),
        "total_count":    total_count,
        "detected_count": detected_count,
        "detection_rate": round(detected_count / total_count, 4),
        "mean_iou":       round(mean_iou, 4),
        "mean_accuracy":  round(mean_acc, 4),
        "iou_ge_0.5":     sum(s["iou_ge_0.5"] for s in summaries),
        "iou_ge_0.7":     sum(s["iou_ge_0.7"] for s in summaries),
        "iou_ge_0.9":     sum(s["iou_ge_0.9"] for s in summaries),
    }


def build_result(by_group: dict[str, list[dict]]) -> dict:
    result = {}
    all_summaries = []
    for group in GROUPS:
        summaries = by_group.get(group, [])
        result[group] = aggregate_group(summaries)
        all_summaries.extend(summaries)
    result["overall"] = aggregate_group(all_summaries)
    return result


def main():
    # 任何含 multi_n*/<scene>/summary.json 的方法資料夾(grounded_sam_* / sam_clip / …)
    weight_dirs = sorted(d for d in EVAL_ROOT.iterdir()
                         if d.is_dir() and any(d.glob("multi_n*/*/summary.json")))
    if not weight_dirs:
        print(f"[錯誤] 找不到任何含 summary 的方法資料夾於 {EVAL_ROOT}")
        return

    for weight_dir in weight_dirs:
        by_group = collect_summaries(weight_dir)
        total = sum(len(v) for v in by_group.values())
        print(f"\n=== {weight_dir.name}：{total} 個場景 ===")
        if total == 0:
            print("  (無 summary,略過)")
            continue
        result = build_result(by_group)
        out_path = weight_dir / "eval_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        for key in (*GROUPS, "overall"):
            g = result[key]
            if g["scenes"] == 0:
                continue
            det = f"{g['detection_rate']:.1%}" if g["detection_rate"] is not None else "N/A"
            iou = f"{g['mean_iou']:.4f}"       if g["mean_iou"]       is not None else "N/A"
            acc = f"{g['mean_accuracy']:.4f}"  if g["mean_accuracy"]  is not None else "N/A"
            print(f"  {key:8s} scenes={g['scenes']:3d} total={g['total_count']:5d} "
                  f"det={det} iou={iou} acc={acc}")
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()
