#!/home/cho/.pyenv/versions/llava/bin/python3
"""llava_pair_direction.py — 封閉問句直接問「誰在上」,一次同時驗證偵測與方向。

★ 新檔,不動 llava_pair_relation.py(開放式版本,保留對照)。

動機:開放式 prompt(Open3DSG 原句 "Describe the relationship between A and B?")的問題:
  - 偵測「有堆疊」很好(3/3 票時平衡準確率 78.8%、假陽性 18.3%)
  - 但【方向】只有 69.0% 對(RESULT_on_direction.md),且 3 對因回答含糊
    (如 "two cans stacked on top of each other")完全解析不出誰在上。
  → 改用封閉問句,強迫模型直接選一個答案。

問法(三選一,同時涵蓋偵測與方向;對照組的正確答案是 NEITHER):
  "Which object is on top of the other: {X} or {Y}? If neither object is on top of the
   other, answer NEITHER. Answer with only the object name or NEITHER."

★ 位置偏誤控制:選項順序由 (場景,A,B) 的雜湊決定(確定性、可復現),避免固定順序造成偏誤;
  報告另列「模型選中第一個選項的比例」作為偏誤檢查(理想 ≈ 50%)。
影像:pair_crops(兩物 bbox 聯集裁切,top-3 視角),與開放式版完全相同 → 單一變因 = prompt。
輸出: RESULT_llava_pair_direction.md + llava_pair_direction.csv
用法: ./llava_pair_direction.py [--max-new 24] [--vote 2]
"""
import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter
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
        "mustard_bottle": "mustard bottle", "bleach_cleanser": "bleach bottle",
        "baseball": "baseball", "tennis_ball": "tennis ball", "apple": "apple",
        "orange": "orange", "peach": "peach", "pear": "pear", "plum": "plum",
        "strawberry": "strawberry", "golf_ball": "golf ball", "rubiks_cube": "rubik's cube",
        "padlock": "padlock", "fork": "fork", "knife": "knife", "spoon": "spoon",
        "plate": "plate", "cups": "cup"}


def nice(o):
    return NICE.get(o, o.replace("_", " "))


def order(sc, A, B):
    """確定性偽隨機:決定選項先後,控制位置偏誤。"""
    h = int(hashlib.md5(f"{sc}|{A}|{B}".encode()).hexdigest(), 16)
    return (A, B) if h % 2 == 0 else (B, A)


def parse(ans, A, B):
    """回 'A名'/'B名'/'NEITHER'/None。先判 NEITHER,再比對物體關鍵詞出現位置。"""
    t = ans.strip().lower()
    if re.search(r"\bneither\b|\bnone\b|\bnot .{0,12}on top\b|\bno .{0,10}stack", t):
        return "NEITHER"
    pa = [m.start() for w in nice(A).split() if len(w) > 2 for m in re.finditer(re.escape(w), t)]
    pb = [m.start() for w in nice(B).split() if len(w) > 2 for m in re.finditer(re.escape(w), t)]
    if pa and (not pb or min(pa) < min(pb)):
        return A
    if pb:
        return B
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=24, dest="max_new")
    ap.add_argument("--vote", type=int, default=2)
    a = ap.parse_args()
    idx = json.loads((IMG / "index.json").read_text())
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    print(f"載入 {MODEL} ...", flush=True)
    proc = LlavaNextProcessor.from_pretrained(MODEL)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=q, device_map="cuda:0", dtype=torch.float16).eval()
    rows = []; t0 = time.time(); first_pick = 0; n_pick = 0
    for n, r in enumerate(idx, 1):
        A, B = r["objA"], r["objB"]
        o1, o2 = order(r["scene"], A, B)
        txt = (f"Which object is on top of the other: {nice(o1)} or {nice(o2)}? "
               f"If neither object is on top of the other, answer NEITHER. "
               f"Answer with only the object name or NEITHER.")
        votes, answers = [], []
        for fr in r.get("frames", []):
            p = IMG / fr["file"]
            if not p.is_file():
                continue
            im = Image.open(p).convert("RGB")
            conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": txt}]}]
            pr = proc.apply_chat_template(conv, add_generation_prompt=True)
            inp = proc(images=im, text=pr, return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=a.max_new, do_sample=False)
            ans = proc.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            v = parse(ans, A, B)
            votes.append(v); answers.append(f"[{fr['view']}] {ans}")
            if v in (A, B):
                n_pick += 1; first_pick += (v == o1)
        vv = [v for v in votes if v]
        maj = Counter(vv).most_common(1)[0] if vv else (None, 0)
        pred = maj[0] if maj[1] >= a.vote else ("NEITHER" if maj[0] is None else maj[0] if maj[1] >= a.vote else None)
        pred = maj[0] if maj[1] >= a.vote else None
        rows.append({**r, "opt1": o1, "opt2": o2, "votes": "|".join(str(v) for v in votes),
                     "pred": pred, "answer": " || ".join(answers)})
        if n % 20 == 0:
            print(f"  {n}/{len(idx)}  {time.time()-t0:.0f}s", flush=True)
    # 評估:GT on 對 → 正解是 upper;非 on 對 → 正解是 NEITHER
    ons = [r for r in rows if r["is_on"]]
    nons = [r for r in rows if not r["is_on"]]
    dir_ok = sum(1 for r in ons if r["pred"] == r["upper"])
    det_ok = sum(1 for r in ons if r["pred"] in (r["objA"], r["objB"]))      # 有說某物在上(不論對錯)
    fp = sum(1 for r in nons if r["pred"] in (r["objA"], r["objB"]))
    tn = sum(1 for r in nons if r["pred"] == "NEITHER")
    und = sum(1 for r in nons if r["pred"] is None)
    md = ["# 封閉問句直接問「誰在上」(對照:開放式 Open3DSG 原句)\n",
          f"- 建檔 2026-09-25;程式 `srp/stage4_probe/llava_pair_direction.py`;模型 `{MODEL}`,4-bit NF4。",
          "- 影像與開放式版**完全相同**(pair_crops,兩物 bbox 聯集,top-3 視角)→ **單一變因 = prompt**。",
          f"- 判定:3 視角需 **≥{a.vote} 票**一致;GT on 對的正解 = 上物,非 on 對的正解 = NEITHER。",
          "- ⚠ 物體名用 GT(上界模擬);**位置偏誤**由 (場景,A,B) 雜湊決定選項順序來控制。\n",
          "## 位置偏誤檢查\n", "| 項目 | 值 |", "|---|---|",
          f"| 模型選了「某個物體」的次數 | {n_pick} |",
          f"| 其中選中**第一個選項**的比例 | **{first_pick/max(n_pick,1)*100:.1f}%**(理想 ≈50%) |", "",
          "## 主結果\n", "| 母體 | 數量 | 指標 | 值 |", "|---|---|---|---|",
          f"| GT 有 on | {len(ons)} | **方向完全正確** | **{dir_ok} ({dir_ok/max(len(ons),1)*100:.1f}%)** |",
          f"| GT 有 on | {len(ons)} | 有說某物在上(不論方向) | {det_ok} ({det_ok/max(len(ons),1)*100:.1f}%) |",
          f"| GT 沒 on | {len(nons)} | **正確答 NEITHER** | **{tn} ({tn/max(len(nons),1)*100:.1f}%)** |",
          f"| GT 沒 on | {len(nons)} | 誤報有堆疊(假陽性) | {fp} ({fp/max(len(nons),1)*100:.1f}%) |",
          f"| GT 沒 on | {len(nons)} | 票數不足/無法判定 | {und} |", "",
          f"**含方向的平衡準確率 = {(dir_ok/max(len(ons),1) + tn/max(len(nons),1))/2*100:.1f}%**\n",
          "### 與開放式版對照\n",
          "| 版本 | 偵測平衡準確率 | 方向正確率(29 對) |", "|---|---|---|",
          "| 開放式(Open3DSG 原句) | 78.8%(3/3票) | **69.0%** |",
          f"| **封閉問句(本次)** | — | **{dir_ok/max(len(ons),1)*100:.1f}%** |", "",
          "## 逐對(GT on 的 29 對)\n",
          "| 場景 | A | B | GT上物 | 選項順序 | 三視角投票 | 判定 | 對? |",
          "|---|---|---|---|---|---|---|---|"]
    for r in ons:
        md.append(f"| {r['scene']} | {r['objA']} | {r['objB']} | **{r['upper']}** | "
                  f"{r['opt1']}→{r['opt2']} | {r['votes']} | {r['pred']} | "
                  f"{'✅' if r['pred']==r['upper'] else '❌'} |")
    (HERE / "RESULT_llava_pair_direction.md").write_text("\n".join(md), encoding="utf-8")
    with open(HERE / "llava_pair_direction.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "objA", "objB", "is_on", "upper", "opt1",
                                          "opt2", "votes", "pred", "answer"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print("\n".join(md[5:30]))


if __name__ == "__main__":
    main()
