#!/home/cho/.pyenv/versions/llava/bin/python3
"""llava_stack_batch.py — 擴大測試:裁切輸入下,LLaVA 能不能正確偵測「有沒有堆疊」?

★ 新檔。承 llava_input_variants.py 的 3 案例結果(crop 3/3、orig 0/3、maskbg 0/3),擴大到 90 場並【加入對照組】。
   影像由 make_vlm_inputs_batch.py 產出(crop;視角=el30 中 kept 面積最大者,可部署不用 GT)。

母體:stack 組 60 場(29 有 on 對、31 無)+ occ 組 30 場(全無 on 對)= 90 場。
      → 實驗組 29、對照組 61。★沒有對照組就無法排除「模型不管怎樣都說有疊」。
prompt: count_stack(與先前一致,自由清點 + 問堆疊)。
判讀:程式先抽關鍵詞(stacked on / on top of / underneath / beneath / sitting on),再由人覆核。
輸出: RESULT_llava_stack_batch.md + llava_stack_batch.csv(逐場原始回答)
用法: ./llava_stack_batch.py [--max-new 140]
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
IMG = HERE / "vlm_inputs_batch"
MODEL = "llava-hf/llama3-llava-next-8b-hf"
Q = ("List every distinct physical object you can see. For each one, give its name. "
     "Then state clearly whether any object is stacked on top of another object, "
     "and which is on top of which.")
POS = re.compile(r"stacked on|on top of|underneath|beneath|sitting on|placed on top|resting on", re.I)
NEG = re.compile(r"not stacked|no object is stacked|none of the objects are stacked|"
                 r"are not stacked|no stacking|not placed on top", re.I)


def verdict(t):
    """回 'YES'(說有堆疊) / 'NO'(明確說沒有) / '?'(不明)。否定句優先。"""
    if NEG.search(t):
        return "NO"
    return "YES" if POS.search(t) else "?"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-new", type=int, default=140, dest="max_new")
    a = ap.parse_args()
    idx = json.loads((IMG / "index.json").read_text())
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    print(f"載入 {MODEL} ...", flush=True)
    proc = LlavaNextProcessor.from_pretrained(MODEL)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=q, device_map="cuda:0", dtype=torch.float16).eval()
    rows = []
    t00 = time.time()
    for i, r in enumerate(idx, 1):
        p = IMG / f"{r['scene']}.png"
        if not p.is_file():
            continue
        im = Image.open(p).convert("RGB")
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": Q}]}]
        pr = proc.apply_chat_template(conv, add_generation_prompt=True)
        inp = proc(images=im, text=pr, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            o = model.generate(**inp, max_new_tokens=a.max_new, do_sample=False)
        txt = proc.decode(o[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        rows.append({**r, "verdict": verdict(txt), "answer": txt.replace("\n", " ")})
        if i % 15 == 0:
            print(f"  {i}/{len(idx)}  {time.time()-t00:.0f}s", flush=True)
    # 統計
    TP = sum(1 for r in rows if r["has_on"] and r["verdict"] == "YES")
    FN = sum(1 for r in rows if r["has_on"] and r["verdict"] != "YES")
    FP = sum(1 for r in rows if not r["has_on"] and r["verdict"] == "YES")
    TN = sum(1 for r in rows if not r["has_on"] and r["verdict"] != "YES")
    npos = TP + FN; nneg = FP + TN
    md = ["# 擴大測試:裁切輸入下 LLaVA 的堆疊偵測(含對照組)\n",
          f"- 建檔 2026-09-25;程式 `srp/stage4_probe/llava_stack_batch.py`;模型 `{MODEL}`,4-bit NF4。",
          "- 影像:`make_vlm_inputs_batch.py` 產的 **crop**;視角=el30 中 kept 面積最大者(**可部署,不用 GT**)。",
          f"- 母體 **{len(rows)} 場**:stack 組 60(29 有 on 對 / 31 無)+ occ 組 30(全無)。",
          "- 真值 = `relations.json` 的 on 關係(排 GEX);判讀 = 關鍵詞抽取(需人工覆核)。\n",
          "## 混淆矩陣\n",
          "| | 模型說「有堆疊」 | 模型說「沒有」或不明 | 合計 |", "|---|---|---|---|",
          f"| **實際有堆疊** | **TP {TP}** | FN {FN} | {npos} |",
          f"| **實際沒堆疊** | **FP {FP}** | TN {TN} | {nneg} |", "",
          "| 指標 | 值 |", "|---|---|",
          f"| 召回率(有堆疊被抓到) | {TP/max(npos,1)*100:.1f}% |",
          f"| **假陽性率(沒堆疊卻說有)** | **{FP/max(nneg,1)*100:.1f}%** |",
          f"| 精確率(說有堆疊時是對的) | {TP/max(TP+FP,1)*100:.1f}% |",
          f"| 平衡準確率 | {(TP/max(npos,1)+TN/max(nneg,1))/2*100:.1f}% |", "",
          "⚠ **假陽性率是關鍵**:若接近召回率,代表模型只是傾向說「有堆疊」,訊號無價值。\n",
          "## 逐場(前 40)\n",
          "| 場景 | 組 | 實際有on | 判讀 | 回答(截斷) |", "|---|---|---|---|---|"]
    for r in rows[:40]:
        md.append(f"| {r['scene']} | {r['group']} | {'是' if r['has_on'] else '否'} | "
                  f"{r['verdict']} | {r['answer'][:110]} |")
    out = HERE / "RESULT_llava_stack_batch.md"
    out.write_text("\n".join(md), encoding="utf-8")
    with open(HERE / "llava_stack_batch.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "group", "view", "has_on", "on_pairs",
                                          "crop_wh", "verdict", "answer"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print(f"[存檔] {out}\n\n" + "\n".join(md[5:26]))


if __name__ == "__main__":
    main()
