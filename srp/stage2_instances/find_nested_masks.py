#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""find_nested_masks.py — 檢查各 hull 的來源遮罩是否包含重疊的 SAM 小遮罩。

對每個場景讀 instances.json,對每個 instance 的來源遮罩(associate 記的 masks[inst][view]),
逐視角在 SAM 完整輸出(kept_object_masks,保留巢狀)裡找「嚴格內含」的小遮罩:
某遮罩面積更小、且 >CONTAIN 比例落在來源遮罩內。

輸出 HULL_ROOT/<scene>/nested_masks.json:
  {"scene":..., "contain_thr":0.8,
   "instances": {"<inst>": {"<view>": {"<來源遮罩>": ["<內含小遮罩>", ...]}}}}
只記有來源遮罩對上的視角;每個來源遮罩一定記(內含空則空 list)。這是語意分群的第一步產物。

用法:
  ./srp/stage2_instances/find_nested_masks.py                  # 全部場景
  ./srp/stage2_instances/find_nested_masks.py stack3_scene0001 # 單場景
  ./srp/stage2_instances/find_nested_masks.py stack3 occ3      # 整組(可多組)
env: SAM_ROOT HULL_ROOT (主力: HULL_ROOT=data/eval/srp_hull_v12、SAM_ROOT=data/eval/sam_only_fast)
"""
import argparse, os, sys, json, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "srp" / "io"))
import masks as MK

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


def process_scene(scene, contain):
    ijp = HULL_ROOT / scene / "instances.json"
    d = json.load(open(ijp))
    view_cache = {}   # view -> {name: (mask, area)}

    def get_masks(view):
        if view not in view_cache:
            km = MK.kept_object_masks(SAM_ROOT / scene / view)
            view_cache[view] = {n: (m, int(m.sum())) for m, n in km}
        return view_cache[view]

    inst_map = {}
    n_src = n_with = n_nested = 0
    for it in d.get("instances", []):
        K = it.get("instance"); masks = it.get("masks", {})
        if K is None or not masks:
            continue
        vout = {}
        for view, anames in masks.items():
            n2m = get_masks(view)
            per_view = {}
            for a in anames:
                if a not in n2m:
                    continue
                A, aA = n2m[a]; n_src += 1
                nested = sorted(nm for nm, (m, am) in n2m.items()
                                if nm != a and 0 < am < aA and int((A & m).sum()) / am > contain)
                per_view[a] = nested
                if nested:
                    n_with += 1; n_nested += len(nested)
            if per_view:
                vout[view] = per_view
        inst_map[str(K)] = vout
    out = {"scene": scene, "contain_thr": contain, "instances": inst_map}
    (ijp.parent / "nested_masks.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return n_src, n_with, n_nested


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--contain", type=float, default=0.8,
                    help="內含判定:小遮罩 >此比例面積落在來源遮罩內(預設 0.8)")
    args = ap.parse_args()
    scenes = resolve(args.targets)
    if not scenes:
        print("找不到符合的場景(需 HULL_ROOT/<scene>/instances.json)"); return
    T = W = N = 0
    for sc in scenes:
        try:
            s, w, n = process_scene(sc, args.contain); T += s; W += w; N += n
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")
    print(f"場景 {len(scenes)}; 來源遮罩 {T}; 有內含小遮罩的 {W}; 內含小遮罩總數 {N} "
          f"→ 各場景 nested_masks.json (contain={args.contain})")


if __name__ == "__main__":
    main()
