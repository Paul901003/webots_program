#!/usr/bin/env python3
"""compare_vlm_on.py — 同一份輸入、同一判定規則,比較 LLaVA-1.6-8B 與 InstructBLIP-Vicuna-7B
在 on 關係偵測上的表現。唯一變因 = 模型。

新檔(取代上一輪未存檔的 inline 計算,那份數字無 provenance、已作廢)。
輸入(兩個 CSV 為同一批圖 pair_crops_big 產生,已核對 538 對全對得上、每對視角數 0 不一致):
  llava_pair_relation_big.csv        LLaVA-1.6-8B 4bit NF4
  instructblip_pair_relation.csv     InstructBLIP-Vicuna-7B 4bit NF4(Open3DSG 論文用的模型)
  兩者 prompt 皆 Open3DSG 原句 "Describe the relationship between A and B?"
欄位: is_on(GT)、upper(GT 上方物體)、answer(逐視角全文,以 " || " 分隔)
兩種判定,門檻 t = 1..4 票:
  A) 純偵測      : 該對有 ≥t 個視角答案命中 ON 關鍵詞 → 預測 on
  B) 偵測+可解析 : 在 A 之上,再要求能從文字解析出「誰在上」(不看對錯,可部署)
  方向正確率      : 只在 GT on 對上算(分母=B 判為 on 的 GT on 對),單獨一欄,不混進精確率
  ⚠ 不可用「方向==GT upper」當預測條件:非 on 對的 GT upper 為空 → FP 恆 0、精確率假 100%
分母: 召回率 = 29 個 GT on 對;假陽性率 = 509 個非 on 對;精確率 = 預測為 on 的對數
排除: pair_crops_big 生成時已排 b 組(nb/occb/stkb)與 n1;未排 GLOBAL_EXCLUDE(這批 on 對無 GEX 物體)
⚠ 上界模擬: 物體名/bbox/視角可見性全來自 GT annotation,非管線 class-agnostic instance
輸出: 僅印,不存檔(表格貼進 RESULT/commit)
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from parse_on_direction import ONKW, parse   # 與判讀器同一套規則


def load(fn):
    rows = []
    for r in csv.DictReader(open(HERE / fn)):
        segs = [s.split("] ", 1)[-1] for s in r["answer"].split(" || ")]
        k = sum(1 for s in segs if ONKW.search(s))
        picks = [parse(s, r["objA"], r["objB"]) for s in segs]
        picks = [p for p in picks if p]
        maj = max(set(picks), key=picks.count) if picks else None
        rows.append({"on": r["is_on"] == "True", "k": k, "kd": len(picks),
                     "parsed": maj is not None, "dir_ok": bool(maj) and maj == r["upper"]})
    return rows


def table(rows, name):
    P = sum(1 for r in rows if r["on"]); N = len(rows) - P
    print(f"\n### {name}  (GT on={P}, 非on={N}, 共{len(rows)})")
    print("| 門檻 | 標準 | TP | FP | FN | 召回率 | 精確率 | F1 | 假陽性率 | 方向正確/TP |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for t in (1, 2, 3, 4):
        for lab, ok in (("A 純偵測", lambda r, t=t: r["k"] >= t),
                        ("B 偵測+可解析", lambda r, t=t: r["k"] >= t and r["parsed"])):
            TP = sum(1 for r in rows if r["on"] and ok(r))
            FP = sum(1 for r in rows if not r["on"] and ok(r))
            FN = P - TP
            rec = TP / P if P else 0
            pre = TP / (TP + FP) if TP + FP else 0
            f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0
            dok = sum(1 for r in rows if r["on"] and ok(r) and r["dir_ok"])
            ds = f"{dok}/{TP} ({dok/TP:.0%})" if TP else "—"
            print(f"| ≥{t} | {lab} | {TP} | {FP} | {FN} | {rec:.1%} | {pre:.1%} | {f1:.1%} | {FP/N:.1%} | {ds} |")


for fn, nm in (("llava_pair_relation_big.csv", "LLaVA-1.6-8B"),
               ("instructblip_pair_relation.csv", "InstructBLIP-Vicuna-7B(論文模型)")):
    table(load(fn), nm)
