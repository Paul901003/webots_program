#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""parse_on_direction.py — 從已存的 VLM 回答解析「誰在上」,驗證 on 關係的【方向】是否正確。

★ 新檔,不跑任何模型,純文字分析 llava_pair_relation.csv(已存 89 對 × 3 視角完整回答)。

動機:先前 RESULT_llava_pair_relation.md 的判讀只檢查「回答裡有沒有 on 類字眼」,
      【沒有驗證方向】。實際抽查 stack4_scene0007(sugar_box 在上、sponge 在下)發現:
      3 個視角有 2 個把上下講反,卻全被算成答對 → 那個 93.1% 召回率有水分。
      而專案的 on(X,Y) 定義是【有方向的】(X 在上、Y 在下),方向錯 = 關係錯。

解析規則(對每個視角的回答各判一次):
  找關係詞 → 取其【前】最後出現的物體、其【後】最先出現的物體
    "A ... on top of / placed on / resting on / sitting on / stacked on ... B"  → A 在上
    "A ... underneath / beneath / under ... B"                                  → B 在上
  代詞處理:關係詞後若只有 it/them 等,視為指向【句中更早出現的另一個物體】。
  ⚠ 解析不出(找不到兩個物體、或兩邊指到同一個)→ 標【無法判定】,不猜。

輸出: RESULT_on_direction.md + on_direction.csv
用法: ./parse_on_direction.py
"""
import argparse
import csv
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


# 物體 → 在回答文字中可能的關鍵詞(小寫)
KW = {
    "sugar_box": ["sugar"], "sponge": ["sponge"], "gelatin_box": ["gelatin", "jell"],
    "foam_brick": ["foam", "brick"], "racquetball": ["ball"], "tuna_fish_can": ["tuna"],
    "master_chef_can": ["coffee", "master", "chef"], "tomato_soup_can": ["tomato", "soup"],
    "lemon": ["lemon"], "windex_bottle": ["windex", "spray"], "cracker_box": ["cracker"],
    "pudding_box": ["pudding"], "potted_meat_can": ["potted", "meat", "spam"],
    "wood_block": ["wood", "block"], "banana": ["banana"], "bowl": ["bowl"], "mug": ["mug"],
    "power_drill": ["drill"], "scissors": ["scissor"], "mustard_bottle": ["mustard"],
    "bleach_cleanser": ["bleach"], "baseball": ["baseball"], "tennis_ball": ["tennis"],
    "apple": ["apple"], "orange": ["orange"], "peach": ["peach"], "pear": ["pear"],
    "plum": ["plum"], "strawberry": ["strawberry"], "golf_ball": ["golf"],
    "rubiks_cube": ["rubik", "cube"], "padlock": ["padlock", "lock"], "fork": ["fork"],
    "knife": ["knife"], "spoon": ["spoon"], "plate": ["plate"], "cups": ["cup"],
}
UP_FIRST = r"on top of|stacked on|stacked upon|placed on|resting on|sits on|sitting on|lying on|balanced on"
UP_SECOND = r"underneath|beneath|under(?!stand)"
PRON = re.compile(r"\b(it|them|this|that|the (box|can|object|item))\b", re.I)


def find(text, obj):
    """回該物體在文字中所有出現位置;沒有回 []。"""
    out = []
    for k in KW.get(obj, [obj.replace("_", " ")]):
        for m in re.finditer(re.escape(k), text, re.I):
            out.append(m.start())
    return sorted(out)


def parse(ans, A, B):
    """回 'A' / 'B' / None(無法判定):誰在上。"""
    t = ans
    for pat, first_is_upper in ((UP_FIRST, True), (UP_SECOND, False)):
        for m in re.finditer(pat, t, re.I):
            pre, post = t[:m.start()], t[m.end():]
            pa, pb = find(pre, A), find(pre, B)
            sa, sb = find(post, A), find(post, B)
            # 前方最後出現者 = 主語
            subj = None
            if pa or pb:
                subj = A if (pa and (not pb or pa[-1] > pb[-1])) else (B if pb else None)
            # 後方最先出現者 = 受詞;若只有代詞則取「另一個」
            objn = None
            if sa or sb:
                objn = A if (sa and (not sb or sa[0] < sb[0])) else (B if sb else None)
            elif subj and PRON.search(post[:40]):
                objn = B if subj == A else A
            if subj and objn and subj != objn:
                return subj if first_is_upper else objn
    return None


ONKW = re.compile(r"on top of|stacked on|stacked upon|placed on|resting on|sits on|sitting on|"
                  r"lying on|underneath|beneath|supporting|balanced on", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="llava_pair_relation.csv")
    ap.add_argument("--tag", default="", help="輸出檔名後綴")
    ap.add_argument("--vote", type=int, default=2, help="偵測:3視角需幾票")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(HERE / a.csv)))
    # --- 偵測(有沒有 on):逐視角關鍵詞 + 投票;與各 runner 同一套規則,確保三版可比
    det = []
    for r in rows:
        segs = [s2.split("] ", 1)[-1] for s2 in r["answer"].split(" || ")]
        k = sum(1 for s2 in segs if ONKW.search(s2))
        det.append({"is_on": r["is_on"] == "True", "n": len(segs), "k": k})
    def cm(t):
        TP = sum(1 for d in det if d["is_on"] and d["k"] >= t)
        FN = sum(1 for d in det if d["is_on"] and d["k"] < t)
        FP = sum(1 for d in det if not d["is_on"] and d["k"] >= t)
        TN = sum(1 for d in det if not d["is_on"] and d["k"] < t)
        return TP, FN, FP, TN
    out = []
    for r in rows:
        if r["is_on"] != "True":
            continue
        A, B, up = r["objA"], r["objB"], r["upper"]
        per = []
        for seg in r["answer"].split(" || "):
            body = seg.split("] ", 1)[-1]
            per.append(parse(body, A, B))
        votes = [p for p in per if p]
        maj = Counter(votes).most_common(1)[0][0] if votes else None
        out.append(dict(scene=r["scene"], objA=A, objB=B, gt_upper=up,
                        v1=per[0] if len(per) > 0 else None,
                        v2=per[1] if len(per) > 1 else None,
                        v3=per[2] if len(per) > 2 else None,
                        n_parsed=len(votes), pred_upper=maj,
                        correct=(maj == up) if maj else None))
    n = len(out)
    parsed = [r for r in out if r["pred_upper"]]
    ok = [r for r in parsed if r["correct"]]
    # 逐視角層級
    vs = [(r[f"v{i}"], r["gt_upper"]) for r in out for i in (1, 2, 3) if r[f"v{i}"]]
    vok = sum(1 for p, g in vs if p == g)
    md = [f"# 關係評分({a.csv}):偵測 + 方向\n",
          "## 偵測(有沒有 on;逐視角關鍵詞 + 投票)\n",
          "| 需幾票 | 召回率 | 假陽性率 | 精確率 | 平衡準確率 |", "|---|---|---|---|---|"] + [
          (lambda TP, FN, FP, TN: f"| ≥{t}/{max(d[chr(34)+chr(110)+chr(34)] for d in det)} |"[:-1] + " |" if False else f"| ≥{t}票 | {TP/max(TP+FN,1)*100:.1f}% | {FP/max(FP+TN,1)*100:.1f}% | "
           f"{TP/max(TP+FP,1)*100:.1f}% | **{(TP/max(TP+FN,1)+TN/max(FP+TN,1))/2*100:.1f}%** |")(*cm(t))
          for t in range(1, max(d["n"] for d in det) + 1)] + ["",
          "# on 關係的【方向】驗證:誰在上?\n",
          "- 建檔 2026-09-25;程式 `srp/stage4_probe/parse_on_direction.py`;**純文字分析,不跑模型**。",
          "- 資料:`llava_pair_relation.csv`(89 對 × 3 視角完整回答,Open3DSG 成對裁切 + 原句 prompt)。",
          "- ⚠ 先前 `RESULT_llava_pair_relation.md` 只判「有沒有 on 字眼」,**未驗證方向**;本檔補上。",
          "- 解析不出者標【無法判定】,**不猜**。\n",
          "## 對層級(29 個 GT on 對;3 視角多數決)\n",
          "| 項目 | 數量 | 佔 29 對 |", "|---|---|---|",
          f"| 能解析出方向 | {len(parsed)} | {len(parsed)/max(n,1)*100:.1f}% |",
          f"| **方向正確** | **{len(ok)}** | **{len(ok)/max(n,1)*100:.1f}%** |",
          f"| 方向錯誤 | {len(parsed)-len(ok)} | {(len(parsed)-len(ok))/max(n,1)*100:.1f}% |",
          f"| 無法判定 | {n-len(parsed)} | {(n-len(parsed))/max(n,1)*100:.1f}% |", "",
          f"- 在**能解析**的 {len(parsed)} 對中,方向正確率 = **{len(ok)/max(len(parsed),1)*100:.1f}%**",
          f"  (亂猜的基準是 50%)\n",
          "## 視角層級(每個視角各算一次)\n",
          "| 項目 | 數量 |", "|---|---|",
          f"| 可解析的視角數 | {len(vs)} |",
          f"| **方向正確** | **{vok}**({vok/max(len(vs),1)*100:.1f}%) |",
          f"| 方向錯誤 | {len(vs)-vok}({(len(vs)-vok)/max(len(vs),1)*100:.1f}%) |", "",
          "## 逐對明細(29 對全列)\n",
          "| 場景 | A | B | GT上物 | 視角1 | 視角2 | 視角3 | 多數決 | 對? |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in out:
        f = lambda x: x or "—"
        md.append(f"| {r['scene']} | {r['objA']} | {r['objB']} | **{r['gt_upper']}** | "
                  f"{f(r['v1'])} | {f(r['v2'])} | {f(r['v3'])} | {f(r['pred_upper'])} | "
                  f"{'✅' if r['correct'] else ('❌' if r['pred_upper'] else '—')} |")
    (HERE / f"RESULT_on_direction{a.tag}.md").write_text("\n".join(md), encoding="utf-8")
    with open(HERE / f"on_direction{a.tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print("\n".join(md[:28]))
    print(f"\n[存檔] {HERE}/RESULT_on_direction{a.tag}.md")


if __name__ == "__main__":
    main()
