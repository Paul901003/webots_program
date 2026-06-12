#!/usr/bin/env python3
"""run_visual_hull_multi.py

新資料 visual hull 的**批次建殼包裝器**(build_torchhull.py 的 Python 版 run_all)。
本身不跑 SAM——mask 須由 grounded_sam 步驟先產好,放在每個場景的 <mask-dir>(預設
grounded_sam_0.25_0.25_0.8)下,命名 view_XX_mask_<class>.png。

對每個場景以 subprocess 呼叫 build_torchhull.py,由它自動拆類別(新資料場景名無 '+',
build_torchhull 會 fallback 用 mask 檔名找 per-class)並輸出 visual_hull_<class>.obj 到 mask-dir。
收集 success / partial / failed 統計,行為對齊 run_all_torchhull.sh。

用法:
  python tools/run_visual_hull_multi.py 3            # n3
  python tools/run_visual_hull_multi.py 1 3 4 5      # 全部
環境變數:
  MASK_DIR_NAME (預設 grounded_sam_0.25_0.25_0.8)  每場景下的 mask 子目錄名
  DEVICE        (預設 auto)                          傳給 build_torchhull 的 --device
  MASKS_PARTIAL (設 1 才開)                          物體可能被裁切時用
  BUILD_PY      (預設 webots_visual_hull 的 python)  跑 build_torchhull 的直譯器
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO = TOOLS_DIR.parent
BUILD_SCRIPT = REPO / "Grounded-Segment-Anything" / "webots_visual_hull" / "build_torchhull.py"
CAPTURES = REPO / "data" / "captures"
EVAL = REPO / "data" / "eval"

BUILD_PY = os.environ.get(
    "BUILD_PY", "/home/cho/.pyenv/versions/webots_visual_hull/bin/python3")

# torchhull 的 JIT 編譯需 CUDA 12.1+;系統 /usr/bin/nvcc 是 12.0,改指向 12.6
_CUDA126 = "/usr/local/cuda-12.6"
BUILD_ENV = dict(os.environ)
if os.path.isdir(_CUDA126):
    BUILD_ENV.setdefault("CUDA_HOME", _CUDA126)
    BUILD_ENV.setdefault("CUDACXX", f"{_CUDA126}/bin/nvcc")
    BUILD_ENV["PATH"] = f"{_CUDA126}/bin:" + BUILD_ENV.get("PATH", "")


def scene_dirs(group: int):
    root = CAPTURES / f"multi_n{group}"
    return sorted(d for d in root.glob(f"n{group}_scene*") if d.is_dir())


def parse_failed_objects(stdout: str):
    """解析 build_torchhull 的 'Failed objects:' 區塊,回傳失敗物體名稱清單。"""
    failed = []
    in_block = False
    for line in stdout.splitlines():
        if line == "Failed objects:":
            in_block = True
            continue
        if not in_block:
            continue
        if line.startswith("  object: "):
            failed.append(line[len("  object: "):])
        elif line and not line.startswith(("- failed object", "  ")):
            in_block = False
    return failed


def main():
    parser = argparse.ArgumentParser(
        description="批次建 visual hull:依門檻找對應權重的 evaluate_masks mask 來雕殼")
    parser.add_argument("groups", nargs="*", type=int, default=[1, 3, 4, 5],
                        help="要處理的組(1/3/4/5),預設全部")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--masks-partial", action="store_true")
    args = parser.parse_args()

    groups = args.groups or [1, 3, 4, 5]
    # 依門檻組出對應的權重資料夾名(與 evaluate_masks 一致)→ 找它輸出的 mask
    weight = f"grounded_sam_{args.box_threshold:g}_{args.text_threshold:g}_{args.nms_threshold:g}"
    if not BUILD_SCRIPT.is_file():
        sys.exit(f"找不到 build_torchhull.py: {BUILD_SCRIPT}")
    if not (EVAL / weight).is_dir():
        sys.exit(f"找不到對應權重的 mask 目錄: {EVAL / weight}\n"
                 f"請先用相同門檻跑 evaluate(run_evaluate_all.py --box-threshold ... 等)")

    print(f"[INFO] groups={groups}  weight={weight}  device={args.device}  "
          f"masks_partial={args.masks_partial}")

    total = ok = failed = 0
    partial_scenes = []
    failed_scenes = []
    failed_objects = []

    for g in groups:
        scenes = scene_dirs(g)
        print(f"\n===== n{g}: {len(scenes)} 場景 =====")
        for scene in scenes:
            total += 1
            # mask/hull 在 data/eval/<weight>/multi_n{g}/<scene>/(evaluate_masks 的輸出)
            mask_dir = EVAL / weight / f"multi_n{g}" / scene.name
            cmd = [BUILD_PY, str(BUILD_SCRIPT),
                   "--scene-dir", str(scene),
                   "--mask-dir", str(mask_dir),
                   "--device", args.device]
            if args.masks_partial:
                cmd.append("--masks-partial")

            print(f"----- [{total}] {scene.name} -----")
            proc = subprocess.run(cmd, env=BUILD_ENV, capture_output=True, text=True)
            out = proc.stdout + proc.stderr
            fobjs = parse_failed_objects(proc.stdout)

            if proc.returncode == 0:
                ok += 1
                if fobjs:
                    partial_scenes.append((scene.name, fobjs))
                    failed_objects += [(scene.name, o) for o in fobjs]
                    print(f"  [PARTIAL] 失敗物體: {fobjs}")
                else:
                    print("  [OK]")
            else:
                failed += 1
                failed_scenes.append(scene.name)
                # 失敗主因印最後幾行
                tail = "\n".join(out.strip().splitlines()[-3:])
                print(f"  [FAIL] rc={proc.returncode}\n    {tail}")

    print("\n===== SUMMARY =====")
    print(f"場景總數: {total}  成功: {ok}  失敗: {failed}  部分成功: {len(partial_scenes)}")
    if partial_scenes:
        print("部分成功(有物體建殼失敗):")
        for name, objs in partial_scenes:
            print(f"  - {name}: {objs}")
    if failed_scenes:
        print("整個失敗的場景:", failed_scenes)
    print(f"失敗物體總數: {len(failed_objects)}")


if __name__ == "__main__":
    main()
