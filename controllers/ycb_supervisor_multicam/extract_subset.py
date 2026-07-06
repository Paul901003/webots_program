#!/usr/bin/env python3
"""extract_subset.py — 從 40 台多相機「全量擷取」中,按視角數抽出子集(依 el/az 檔名)。

多相機一次拍滿全部 validated 視角(view_el{el}_az{az}.png)。本腳本讀
selected_viewpoints_multi_n{count} 算出該數量要哪些 el/az,再從全量擷取把對應
的 rgb/depth/pose 連結(或複製)到子集目錄,供下游(SAM/hull/評估)使用。

用法:
  ./extract_subset.py 6                # n6 子集,全部場景
  ./extract_subset.py 12 n3            # n12,只 n3 組
  ./extract_subset.py 8 n3_scene0001   # n8,單一場景
  FORCE=1 ./extract_subset.py 6        # 重做(刪舊子集)
  MODE=copy ./extract_subset.py 6      # 用複製(預設 symlink)
"""
import math
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VIEWPOINTS_DIR = REPO / "data" / "viewpoints"
SRC_ROOT = REPO / "data" / "captures_multicam"
TARGET = [0.35, 0.0, 0.0]
SUFFIXES = (".png", "_depth.npy", "_depth.png", "_pose.json")


def el_az_name(origin, target=TARGET):
    """與 gen_multicam_world.el_az_name 同算法 → 同檔名。"""
    d = [origin[k] - target[k] for k in range(3)]
    dist = math.sqrt(sum(c * c for c in d)) or 1e-9
    el = round(math.degrees(math.asin(max(-1.0, min(1.0, d[2] / dist)))))
    if el >= 88:
        return "view_el90"
    az = round(math.degrees(math.atan2(d[1], d[0])) % 360)
    return f"view_el{el:02d}_az{az:03d}"


def load_subset_names(count):
    import json
    tag = f"x{int(round(0.35 * 100)):+04d}"
    f = VIEWPOINTS_DIR / f"selected_viewpoints_multi_n{count}_{tag}.json"
    if not f.is_file():
        sys.exit(f"找不到 {f.name}(先跑 A-3 select_counts)")
    sel = json.loads(f.read_text())["selected"]
    return [el_az_name(v["ray"]["ray_origin_m"]) for v in sel]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    count = int(sys.argv[1])
    filters = sys.argv[2:]
    names = load_subset_names(count)
    print(f"[extract] n{count} 需要 {len(names)} 個視角: {', '.join(sorted(n.replace('view_', '') for n in names))}")

    mode = os.environ.get("MODE", "symlink")
    force = os.environ.get("FORCE")
    dst_root = REPO / "data" / f"captures_multicam_n{count}"

    def want(scene_dir):
        if not filters:
            return True
        nm = scene_dir.name
        return any(nm == f or nm.startswith(f) or f in scene_dir.parts for f in filters)

    n_scene = n_file = 0
    for manifest in SRC_ROOT.rglob("scene_manifest.json"):
        scene_dir = manifest.parent
        if not want(scene_dir):
            continue
        rel = scene_dir.relative_to(SRC_ROOT)
        dst = dst_root / rel
        if dst.exists() and not force:
            continue
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        got = 0
        for name in names:
            for suf in SUFFIXES:
                src_f = scene_dir / f"{name}{suf}"
                if not src_f.exists():
                    continue
                dst_f = dst / src_f.name
                if mode == "copy":
                    shutil.copy2(src_f, dst_f)
                else:
                    os.symlink(os.path.relpath(src_f, dst), dst_f)
                if suf == ".png":
                    got += 1
                n_file += 1
        # 連 manifest 一併帶上
        (dst / "scene_manifest.json").write_bytes(manifest.read_bytes())
        n_scene += 1
        miss = len(names) - got
        print(f"  {rel}: {got}/{len(names)} 視角" + (f"  ⚠缺{miss}" if miss else ""))
    print(f"[extract] 完成 {n_scene} 場景 → {dst_root}  ({mode})")


if __name__ == "__main__":
    main()
