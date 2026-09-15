#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""build_leak_tables.py — 產「胖瘦×中心/fp×三reassign」的定案指標表(表A概覽 + 表B逐on對),落地存檔可復現。

復用 stack_leak_nosep.py 的定義(已四錨點驗證):per-object 洩漏%(無門檻)、沒分開(主inst相同)、受污染inst純度。
分母=同前景hull表面voxel(瘦=am1+gtlabel_am1、胖=am1fp+gtlabel_am1fp,各自對齊);GT=gtlabel;排GEX。

判準(照使用者定案,不給單一總排名):
  單一 on 對「乾淨」= 分開(主群不同) 且 leak 低 且 受污染群純度高;三者缺一不算好。
  表A只當分布概覽(組平均 + stack 最差 on 對);真正優劣看表B逐對 + 「本對有無方法救得起來(最佳值)」。

輸出(存檔):
  srp/stage2_instances/RESULT_leak_nosep_tables.md   (表A + 表B + provenance + 判準)
  srp/stage2_instances/leak_nosep_perpair.csv         (原始:每 on 對 × 每方法 leak/純度/分開)
用法: ./build_leak_tables.py            # 全 303 多物場
"""
import sys
import csv
import glob
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "io"))
from stack_leak_nosep import scene_stats, on_pairs, grp_of, GEX, EVAL  # noqa: E402

# 12 方法:(root 後綴, gtlabel base, 顯示碼, hull, 投票, reassign)
METHODS = [
    ("reNNcS_am1",   "am1",   "瘦c-S",  "瘦", "中心", "併最近"),
    ("reNNcSd_am1",  "am1",   "瘦c-Sd", "瘦", "中心", "drop"),
    ("reNNcSe_am1",  "am1",   "瘦c-Se", "瘦", "中心", "侵蝕"),
    ("reNNfpS_am1",  "am1",   "瘦f-S",  "瘦", "fp",  "併最近"),
    ("reNNfpSd_am1", "am1",   "瘦f-Sd", "瘦", "fp",  "drop"),
    ("reNNfpSe_am1", "am1",   "瘦f-Se", "瘦", "fp",  "侵蝕"),
    ("reNNcS_am1fp", "am1fp", "胖c-S",  "胖", "中心", "併最近"),
    ("reNNcSd_am1fp","am1fp", "胖c-Sd", "胖", "中心", "drop"),
    ("reNNcSe_am1fp","am1fp", "胖c-Se", "胖", "中心", "侵蝕"),
    ("reNNfpS_am1fp","am1fp", "胖f-S",  "胖", "fp",  "併最近"),
    ("reNNfpSd_am1fp","am1fp","胖f-Sd", "胖", "fp",  "drop"),
    ("reNNfpSe_am1fp","am1fp","胖f-Se", "胖", "fp",  "侵蝕"),
]


def all_scenes(root):
    scs = sorted(Path(p).parent.name for p in
                 glob.glob(str(EVAL / root / "*_scene*" / "instances.npz")))
    return [s for s in scs if not s.startswith("n1_")]


def compute():
    # 收集每方法:per-group per-object leak 清單、per stack on 對指標
    per_group_leak = {m[2]: {"n": [], "occ": [], "stack": [], "all": []} for m in METHODS}
    nosep = {m[2]: {"n": [0, 0], "occ": [0, 0], "stack": [0, 0], "all": [0, 0]} for m in METHODS}
    # pairs[(scene,u,l)][code] = dict(maxleak,minpur,sep,leak_u,leak_l,pur_u,pur_l)
    pairs = {}
    scenes = all_scenes(f"srp_hull_divB_t50_{METHODS[0][0]}")
    for suf, gtb, code, *_ in METHODS:
        root = f"srp_hull_divB_t50_{suf}"
        gtr = f"srp_hull_gtlabel_{gtb}"
        for sc in scenes:
            st = scene_stats(root, gtr, sc)
            if st is None:
                continue
            g = grp_of(sc)
            onobjs = set()
            for (u, l) in on_pairs(sc):
                if u in GEX or l in GEX:
                    continue
                iu = st["name2idx"].get(u)
                il = st["name2idx"].get(l)
                if iu is None or il is None:
                    continue
                onobjs.update([iu, il])
                mu, ml = st["per"][iu]["main_i"], st["per"][il]["main_i"]
                sep = (mu != ml) and mu > 0 and ml > 0
                lu, ll = st["per"][iu]["leak_frac"], st["per"][il]["leak_frac"]
                pu = st["purity"](mu) if mu > 0 else 0.0
                pl = st["purity"](ml) if ml > 0 else 0.0
                for k in (g, "all"):
                    nosep[code][k][0] += int(sep is False)  # 沒分開=not sep
                    nosep[code][k][1] += 1
                pairs.setdefault((sc, u, l), {})[code] = dict(
                    maxleak=max(lu, ll), minpur=min(pu, pl), sep=sep,
                    leak_u=lu, leak_l=ll, pur_u=pu, pur_l=pl)
            for o in st["objs"]:
                if st["per"][o]["name"] in GEX:
                    continue
                if g == "stack" and o not in onobjs:
                    continue
                lf = st["per"][o]["leak_frac"]
                per_group_leak[code][g].append(lf)
                per_group_leak[code]["all"].append(lf)
    return scenes, per_group_leak, nosep, pairs


def fmt_pct(x):
    return f"{x*100:.1f}"


def main():
    scenes, pgl, nosep, pairs = compute()
    md = []
    md.append("# 定案指標結果表:胖瘦×中心/fp×三reassign(洩漏%/沒分開/純度)\n")
    md.append("- 建檔:2026-09-16。程式:`build_leak_tables.py`(復用 `stack_leak_nosep.py`,四錨點驗證過)。可復現:重跑本檔即得。")
    md.append("- 分母=同前景hull表面voxel(瘦=am1+gtlabel_am1、胖=am1fp+gtlabel_am1fp,**各自對齊、不跨hull比絕對值**);GT=gtlabel;**排GEX**(skillet_lid/windex/colored_wood_blocks/dice)。")
    md.append("- 量:per-object 洩漏%(無門檻)、沒分開率(主inst相同)、受污染inst純度。母體=303多物場;stack on對=29。")
    md.append("- 判準(不給總冠軍):單對『乾淨』=分開 且 leak低 且 純度高,三者缺一不算好;看表B逐對。\n")

    # 表 A
    md.append("## 表A — 12 方法 × 分組概覽\n")
    md.append("| hull | 投票 | reassign | n洩漏% | occ洩漏% | stack洩漏% | stack最差洩漏%(哪對) | stack沒分開% |")
    md.append("|---|---|---|---|---|---|---|---|")
    stack_pairs = [k for k in pairs]
    for suf, gtb, code, hull, vote, rea in METHODS:
        def gmean(g):
            v = pgl[code][g]
            return np.mean(v) * 100 if v else 0.0
        # stack 最差 on 對
        worst = 0.0
        worst_pair = "-"
        for (sc, u, l), d in pairs.items():
            if code in d and d[code]["maxleak"] > worst:
                worst = d[code]["maxleak"]
                worst_pair = f"{sc.replace('_scene','#')} {u}/{l}"
        ns, tot = nosep[code]["stack"]
        nsp = ns / tot * 100 if tot else 0.0
        md.append(f"| {hull} | {vote} | {rea} | {gmean('n'):.2f} | {gmean('occ'):.2f} | "
                  f"{gmean('stack'):.2f} | {worst*100:.1f} ({worst_pair}) | {nsp:.1f} |")

    # 表 B — 逐 on 對 × 方法(cell = maxleak%;沒分開標✗)
    md.append("\n## 表B — 堆疊 on 對逐對(cell = 該對最差 leak%;`✗`=沒分開;粗體=本對最佳)\n")
    codes = [m[2] for m in METHODS]
    md.append("| 場景 | 上物→下物 | " + " | ".join(codes) + " | 本對最佳(最低maxleak) | 最佳值 |")
    md.append("|---|---|" + "---|" * (len(codes) + 2))
    # 依場景排序
    for (sc, u, l) in sorted(pairs.keys()):
        d = pairs[(sc, u, l)]
        # 找最佳:優先分開者中 maxleak 最小;若無分開者取 maxleak 最小
        best_code, best_val = None, 1e9
        for c in codes:
            if c in d:
                v = d[c]["maxleak"]
                score = v + (0 if d[c]["sep"] else 10)  # 沒分開重罰
                if score < best_val:
                    best_val = score
                    best_code = c
        cells = []
        for c in codes:
            if c not in d:
                cells.append("-")
                continue
            v = d[c]["maxleak"] * 100
            mark = "" if d[c]["sep"] else "✗"
            s = f"{v:.0f}{mark}"
            if c == best_code:
                s = f"**{s}**"
            cells.append(s)
        bv = d[best_code]["maxleak"] * 100 if best_code else 0
        md.append(f"| {sc.replace('_scene','#')} | {u}→{l} | " + " | ".join(cells) +
                  f" | {best_code} | {bv:.0f}% |")

    md.append("\n### 逐對『最佳可達 maxleak』完整排序(threshold-free;不設任何門檻,直接看連續分布與自然斷層)")
    bl = []
    for (sc, u, l), d in pairs.items():
        vals = [(d[c]["maxleak"], d[c]["sep"]) for c in codes if c in d]
        sepvals = [v for v, s in vals if s]
        best = min(sepvals) if sepvals else min(v for v, _ in vals)
        bl.append((best, sc, u, l))
    bl.sort(reverse=True)
    md.append("| 名次 | 場景 | 上→下 | 最佳可達 maxleak% |")
    md.append("|---|---|---|---|")
    for i, (b, sc, u, l) in enumerate(bl, 1):
        md.append(f"| {i} | {sc.replace('_scene','#')} | {u}→{l} | {b*100:.0f} |")
    allb = [b for b, *_ in bl]
    md.append(f"\nmin={min(allb)*100:.0f}% / 中位={np.median(allb)*100:.0f}% / max={max(allb)*100:.0f}%。**不設門檻**,嚴重程度由連續值與其間斷層自行呈現。")

    out_md = HERE / "RESULT_leak_nosep_tables.md"
    out_md.write_text("\n".join(md), encoding="utf-8")

    # CSV 原始
    out_csv = HERE / "leak_nosep_perpair.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "upper", "lower", "method", "maxleak", "minpurity", "separated",
                    "leak_upper", "leak_lower", "pur_upper", "pur_lower"])
        for (sc, u, l), d in sorted(pairs.items()):
            for c in codes:
                if c not in d:
                    continue
                x = d[c]
                w.writerow([sc, u, l, c, f"{x['maxleak']:.4f}", f"{x['minpur']:.4f}", int(x['sep']),
                            f"{x['leak_u']:.4f}", f"{x['leak_l']:.4f}", f"{x['pur_u']:.4f}", f"{x['pur_l']:.4f}"])

    print(f"[存檔] {out_md}")
    print(f"[存檔] {out_csv}")
    print(f"場景={len(scenes)}  on對={len(pairs)}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
