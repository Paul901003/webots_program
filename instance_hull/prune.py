#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""prune.py — 依 phantom_probe 分數剪枝 hull(不動原始檔,只複製合格者 + 存清單)。

規則(OR):刪除 hull ⟺ self_consist < T1  或  exclusive < T2(任一給 off 即不參與)。
無分數的 hull(phantom_probe.csv 沒這列,通常 recon<2 視角)→ 視為保留(無法判斷)。

對每場景:
  讀 data/eval/<root>/<scene>/instances.json(只讀,不改)
  → 保留合格 instances,寫到新目錄 data/eval/<root>__sc{T1}_ex{T2}/<scene>/instances.json
  → 同時寫 prune_list.json:{kept:[...idx], pruned:[{idx,self_consist,exclusive,reason}]}
分數來源:data/eval/phantom_probe.csv(欄位 method,scene,hull,self_consist,exclusive,...)。

用法: ./instance_hull/prune.py 1 3 4 5 --root instance_hull --t1 0.2 --t2 0.6
       (--t1 off / --t2 off 可單用一個分數)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "data" / "eval"
CAPTURES = REPO / "data" / "captures"
PROBE = EVAL / "phantom_probe.csv"


def resolve_scenes(targets):
    out = []
    for a in targets:
        if "scene" in a:
            out.append(a)
        else:
            out += [d.name for d in sorted((CAPTURES / f"multi_n{a}").glob(f"n{a}_scene*"))]
    return out


def load_scores(method):
    """{(scene, hull_idx): (self_consist, exclusive)}"""
    sc = {}
    if not PROBE.is_file():
        sys.exit(f"找不到分數檔 {PROBE}(先跑 phantom_probe.py)")
    for r in csv.DictReader(open(PROBE)):
        if r["method"] != method:
            continue
        sc[(r["scene"], int(r["hull"]))] = (float(r["self_consist"]), float(r["exclusive"]))
    return sc


def tag(t1, t2):
    p = []
    p.append("scoff" if t1 is None else f"sc{t1:g}")
    p.append("exoff" if t2 is None else f"ex{t2:g}")
    return "_".join(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["n3_scene0001"])
    ap.add_argument("--root", default="instance_hull", help="來源方法目錄 data/eval/<root>/")
    ap.add_argument("--t1", default="0.2", help="self_consist 門檻(低於則剪);off=不用")
    ap.add_argument("--t2", default="0.6", help="exclusive 門檻(低於則剪);off=不用")
    ap.add_argument("--out-root", default=None, help="輸出目錄名(預設 <root>__<tag>)")
    args = ap.parse_args()

    T1 = None if str(args.t1).lower() == "off" else float(args.t1)
    T2 = None if str(args.t2).lower() == "off" else float(args.t2)
    if T1 is None and T2 is None:
        sys.exit("t1 與 t2 不能同時 off")
    scores = load_scores(args.root)
    out_root = args.out_root or f"{args.root}__{tag(T1, T2)}"
    scenes = resolve_scenes(args.scenes or ["n3_scene0001"])

    tot_in = tot_keep = 0
    for scene in scenes:
        src = EVAL / args.root / scene / "instances.json"
        if not src.is_file():
            print(f"[skip] {scene}: 無 {src}"); continue
        data = json.loads(src.read_text())
        insts = data.get("instances", [])
        kept_idx, pruned = [], []
        for i, inst in enumerate(insts):
            se = scores.get((scene, i))
            if se is None:                       # 無分數 → 保留
                kept_idx.append(i); continue
            s, e = se
            cut = (T1 is not None and s < T1) or (T2 is not None and e < T2)
            if cut:
                reason = []
                if T1 is not None and s < T1:
                    reason.append(f"sc<{T1:g}")
                if T2 is not None and e < T2:
                    reason.append(f"ex<{T2:g}")
                pruned.append({"idx": i, "self_consist": round(s, 4),
                               "exclusive": round(e, 4), "reason": "|".join(reason)})
            else:
                kept_idx.append(i)

        # 複製合格 instances 到新檔(原始檔不動)
        new = dict(data)
        new["instances"] = [insts[i] for i in kept_idx]
        if isinstance(data.get("centers"), list) and len(data["centers"]) == len(insts):
            new["centers"] = [data["centers"][i] for i in kept_idx]
        new["pruned_from"] = args.root
        new["prune_rule"] = {"t1_self_consist": T1, "t2_exclusive": T2}
        out_dir = EVAL / out_root / scene
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "instances.json").write_text(
            json.dumps(new, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "prune_list.json").write_text(json.dumps(
            {"scene": scene, "rule": new["prune_rule"], "n_in": len(insts),
             "n_kept": len(kept_idx), "kept": kept_idx, "pruned": pruned},
            indent=2, ensure_ascii=False), encoding="utf-8")
        tot_in += len(insts); tot_keep += len(kept_idx)
        print(f"[{scene}] {len(insts)} → 保留 {len(kept_idx)} / 剪 {len(pruned)}")

    print(f"\n== {args.root} → data/eval/{out_root}/  (規則 {tag(T1,T2)}) ==")
    print(f"hull 總數 {tot_in} → 保留 {tot_keep} / 剪 {tot_in - tot_keep}"
          f"  ({100*(tot_in-tot_keep)/tot_in:.1f}% 剪除)" if tot_in else "")


if __name__ == "__main__":
    main()
