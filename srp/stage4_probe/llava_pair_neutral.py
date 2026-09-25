#!/home/cho/.pyenv/versions/llava/bin/python3
"""llava_pair_neutral.py — 中性句型問關係(第三種 prompt),與前兩種單一變因對照。

★ 新檔。前兩種 prompt 的問題:
  ① Open3DSG 原句 "Describe the relationship between A and B?" → 偵測好(假陽性18.3%)
     但回答常含糊("two cans stacked on top of each other"),方向僅 69.0%、3 對解析不出。
  ② 我編的封閉句 "Which object is on top of the other: X or Y?" → 方向 72.4% 略升,
     但【預設了「有一個在上」】,把模型推著挑一個 → 假陽性惡化到 41.7%(29個on對100%都挑了)。
  ③ 本版:中性句型,不預設有堆疊,但明確要求「若有,說出誰在上」。

prompt(我設計,非論文原句):
  "What is the spatial relationship between {X} and {Y}? If one is on top of the other,
   state which one is on top."

影像、物體名、視角、投票規則與前兩版【完全相同】(pair_crops, top-3) → 單一變因 = prompt。
評分由 parse_on_direction.py 統一處理(同一套解析器,確保三版可比)。
輸出: llava_pair_neutral.csv(欄位與 llava_pair_relation.csv 相同)
用法: ./llava_pair_neutral.py [--max-new 120]
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
        "mustard_bottle": "mustard bottle", "bleach_cleanser": "bleach bottle",
        "baseball": "baseball", "tennis_ball": "tennis ball", "apple": "apple", "orange": "orange",
        "peach": "peach", "pear": "pear", "plum": "plum", "strawberry": "strawberry",
        "golf_ball": "golf ball", "rubiks_cube": "rubik's cube", "padlock": "padlock",
        "fork": "fork", "knife": "knife", "spoon": "spoon", "plate": "plate", "cups": "cup"}
ON = re.compile(r"on top of|stacked on|stacked upon|placed on|resting on|sits on|sitting on|"
                r"lying on|underneath|beneath|supporting|balanced on", re.I)


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
    rows = []; t0 = time.time()
    for n, r in enumerate(idx, 1):
        txt = (f"What is the spatial relationship between {nice(r['objA'])} and {nice(r['objB'])}? "
               f"If one is on top of the other, state which one is on top.")
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
                o = model.generate(**inp, max_new_tokens=a.max_new, do_sample=False)
            ans = proc.decode(o[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            votes.append(bool(ON.search(ans))); answers.append(f"[{fr['view']}] " + ans.replace("\n", " "))
        if not votes:
            continue
        rows.append({**r, "n_view": len(votes), "n_on": sum(votes),
                     "pred_on": sum(votes) >= 2, "answer": " || ".join(answers)})
        if n % 20 == 0:
            print(f"  {n}/{len(idx)}  {time.time()-t0:.0f}s", flush=True)
    with open(HERE / "llava_pair_neutral.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "objA", "objB", "is_on", "upper", "views",
                                          "n_view", "n_on", "pred_on", "answer"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print(f"[存檔] {HERE/'llava_pair_neutral.csv'}  {len(rows)} 對")


if __name__ == "__main__":
    main()
