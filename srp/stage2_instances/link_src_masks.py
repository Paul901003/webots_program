#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""link_src_masks.py — 為每個 hull instance 建立「來源遮罩」的 symlink,方便翻看實際遮罩圖。

對每個場景讀 instances.json,對每個 instance 的來源遮罩(associate 記的 masks[inst][view]),
在 HULL_ROOT/<scene>/inst<K>_srcmasks/ 下建相對 symlink 指向
SAM_ROOT/<scene>/<view>/masks/<mask>.png。命名 <view>__<mask>.png,一眼看得出視角/遮罩。

用法:
  ./srp/stage2_instances/link_src_masks.py                  # 全部場景
  ./srp/stage2_instances/link_src_masks.py stack3_scene0001 # 單場景
  ./srp/stage2_instances/link_src_masks.py stack3 occ3      # 整組(可多組)
env: SAM_ROOT HULL_ROOT (預設 data/eval/sam_only、data/eval/srp_hull;fast 版自行 export)

每次重跑會先清掉該場景舊的 inst*_srcmasks 再建 → associate 更新 instances.json 後重跑即同步。
"""
import argparse, os, sys, json, glob, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "sam_only")))
HULL_ROOT = Path(os.environ.get("HULL_ROOT", str(REPO / "data" / "eval" / "srp_hull")))


def resolve(targets):
    if not targets:
        return sorted(p.parent.name for p in HULL_ROOT.glob("*/instances.json"))
    out = []
    for a in targets:
        if "scene" in a:
            if (HULL_ROOT / a / "instances.json").is_file():
                out.append(a)
        else:
            out += [Path(p).parent.name for p in glob.glob(str(HULL_ROOT / f"{a}_scene*/instances.json"))]
    return sorted(set(out))


def link_scene(scene):
    ijp = HULL_ROOT / scene / "instances.json"
    if not ijp.is_file():
        return 0, 0, 0
    try:
        d = json.load(open(ijp))
    except Exception as e:
        print(f"[skip] {scene}: instances.json 讀取失敗 {e}"); return 0, 0, 0
    for old in ijp.parent.glob("inst*_srcmasks"):
        shutil.rmtree(old)
    n_inst = n_link = n_missing = 0
    for it in d.get("instances", []):
        k = it.get("instance"); masks = it.get("masks", {})
        if k is None or not masks:
            continue
        outdir = ijp.parent / f"inst{k}_srcmasks"; outdir.mkdir(exist_ok=True); n_inst += 1
        for view, mlist in masks.items():
            for mk in mlist:
                src = SAM_ROOT / scene / view / "masks" / mk
                if not src.is_file():
                    n_missing += 1; continue
                link = outdir / f"{view}__{mk}"
                if link.is_symlink() or link.exists():
                    link.unlink()
                os.symlink(os.path.relpath(src, start=outdir), link)
                n_link += 1
    return n_inst, n_link, n_missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    args = ap.parse_args()
    scenes = resolve(args.targets)
    if not scenes:
        print("找不到符合的場景(需 HULL_ROOT/<scene>/instances.json)"); return
    tot_inst = tot_link = tot_missing = 0
    for sc in scenes:
        ni, nl, nm = link_scene(sc)
        tot_inst += ni; tot_link += nl; tot_missing += nm
    print(f"場景 {len(scenes)}, instance {tot_inst}, symlink {tot_link}, 來源缺失 {tot_missing}")


if __name__ == "__main__":
    main()
