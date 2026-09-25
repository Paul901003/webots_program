#!/home/cho/.pyenv/versions/llava/bin/python3
"""llava_pair_relation.py — 照 Open3DSG 的做法問【成對關係】,測 VLM 能不能說出 on。

★ 新檔,獨立 pyenv `llava`。依 Open3DSG(CVPR2024)補充材料 Sec.A 的原句 prompt:
     "Describe the relationship between [object1] and [object2]?"
   物體名當 context(論文由 CLIP 先推論;此處用 GT 名 = 上界模擬)。
   影像 = make_pair_crops.py 產的【兩物 bbox 聯集】裁切(論文 box_ij = box_ik ∪ box_jk)。

與先前失敗測試的差異(RESULT_llava_stack_batch.md 平衡準確率 52.8%):
   先前=整張場景圖 + 開放式全場清點;本版=成對裁切 + 成對提問 + 物體名 context(忠於論文)。

★ 對照組不可省:60 個【沒有 on 關係】的物體對。若它們也大量被說成 "on top of",訊號無效。
判讀:關鍵詞抽取(on top of / stacked on / resting on / underneath / supporting / beneath / sits on),
      需人工覆核。⚠ 只判「有沒有 on」,不判方向正確性(另列)。
輸出: RESULT_llava_pair_relation.md + llava_pair_relation.csv
用法: ./llava_pair_relation.py [--max-new 120]
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
IMG = HERE / "pair_crops"
MODEL = "llava-hf/llama3-llava-next-8b-hf"
NICE = {"gelatin_box": "gelatin box", "sponge": "sponge", "tuna_fish_can": "tuna fish can",
        "master_chef_can": "coffee can", "sugar_box": "sugar box", "foam_brick": "foam brick",
        "racquetball": "rubber ball", "tomato_soup_can": "tomato soup can", "lemon": "lemon",
        "windex_bottle": "spray bottle", "cracker_box": "cracker box", "pudding_box": "pudding box",
        "potted_meat_can": "potted meat can", "wood_block": "wooden block", "banana": "banana",
        "bowl": "bowl", "mug": "mug", "power_drill": "power drill", "scissors": "scissors",
        "mustard_bottle": "mustard bottle", "bleach_cleanser": "bleach bottle"}
ON = re.compile(r"on top of|stacked on|stacked upon|resting on|sits on|sitting on|placed on|"
                r"underneath|beneath|supporting|balanced on|lying on", re.I)


def nice(o):
    return NICE.get(o, o.replace("_", " "))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-new", type=int, default=120, dest="max_new")
    a = ap.parse_args()
    idx = json.loads((IMG / "index.json").read_text())
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    print(f"載入 {MODEL} ...", flush=True)
    proc = LlavaNextProcessor.from_pretrained(MODEL)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=q, device_map="cuda:0", dtype=torch.float16).eval()
    rows = []; t00 = time.time()
    for n, r in enumerate(idx, 1):
        p = IMG / r["file"]
        if not p.is_file():
            continue
        im = Image.open(p).convert("RGB")
        # ★ Open3DSG 補充材料 Sec.A 原句
        txt = f"Describe the relationship between {nice(r['objA'])} and {nice(r['objB'])}?"
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": txt}]}]
        pr = proc.apply_chat_template(conv, add_generation_prompt=True)
        inp = proc(images=im, text=pr, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            o = model.generate(**inp, max_new_tokens=a.max_new, do_sample=False)
        ans = proc.decode(o[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        rows.append({**r, "pred_on": bool(ON.search(ans)), "answer": ans.replace("\n", " ")})
        if n % 20 == 0:
            print(f"  {n}/{len(idx)}  {time.time()-t00:.0f}s", flush=True)
    TP = sum(1 for r in rows if r["is_on"] and r["pred_on"])
    FN = sum(1 for r in rows if r["is_on"] and not r["pred_on"])
    FP = sum(1 for r in rows if not r["is_on"] and r["pred_on"])
    TN = sum(1 for r in rows if not r["is_on"] and not r["pred_on"])
    P, N = TP + FN, FP + TN
    md = ["# 成對關係提問(Open3DSG 做法):VLM 能不能說出 on?\n",
          f"- 建檔 2026-09-25;程式 `srp/stage4_probe/llava_pair_relation.py`;模型 `{MODEL}`,4-bit NF4。",
          "- **忠於 Open3DSG(CVPR2024)補充材料 Sec.A**:影像=兩物 bbox 聯集裁切(`box_ij = box_ik ∪ box_jk`);",
          '  prompt=原句 `"Describe the relationship between [object1] and [object2]?"`;物體名當 context。',
          "- ⚠ 物體名用 **GT = 上界模擬**(論文由 CLIP 先推論);視角選擇用 GT modal 面積(論文用深度判遮擋,本專案不可用)。",
          f"- 母體 **{len(rows)} 對**:on **{P}**、非 on **{N}**(對照組,來自同樣 60 場 stack;排 GEX)。",
          "- 判讀:關鍵詞(on top of / stacked on / resting on / underneath / supporting …),需人工覆核。\n",
          "## 混淆矩陣(只判「有沒有 on」)\n",
          "| | 模型說有 on | 模型沒說 on | 合計 |", "|---|---|---|---|",
          f"| **GT 有 on** | **TP {TP}** | FN {FN} | {P} |",
          f"| **GT 沒 on** | **FP {FP}** | TN {TN} | {N} |", "",
          "| 指標 | 值 |", "|---|---|",
          f"| 召回率 | {TP/max(P,1)*100:.1f}% |",
          f"| **假陽性率** | **{FP/max(N,1)*100:.1f}%** |",
          f"| 精確率 | {TP/max(TP+FP,1)*100:.1f}% |",
          f"| **平衡準確率** | **{(TP/max(P,1)+TN/max(N,1))/2*100:.1f}%** |", "",
          "(對照:先前整張場景圖版 = 52.8%,`RESULT_llava_stack_batch.md`)\n",
          "## 逐對(GT 有 on 的 29 對全列)\n",
          "| 場景 | A | B | 上物 | 判讀 | 回答(截斷) |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r["is_on"]:
            md.append(f"| {r['scene']} | {r['objA']} | {r['objB']} | {r['upper']} | "
                      f"{'✅on' if r['pred_on'] else '❌無'} | {r['answer'][:100]} |")
    md += ["", "## 對照組中被誤判為 on 的(前 15)\n",
           "| 場景 | A | B | 回答(截斷) |", "|---|---|---|---|"]
    for r in [x for x in rows if not x["is_on"] and x["pred_on"]][:15]:
        md.append(f"| {r['scene']} | {r['objA']} | {r['objB']} | {r['answer'][:100]} |")
    out = HERE / "RESULT_llava_pair_relation.md"
    out.write_text("\n".join(md), encoding="utf-8")
    with open(HERE / "llava_pair_relation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "view", "objA", "objB", "is_on", "upper",
                                          "areaA", "areaB", "file", "crop_wh", "pred_on", "answer"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print(f"[存檔] {out}\n\n" + "\n".join(md[7:26]))


if __name__ == "__main__":
    main()
