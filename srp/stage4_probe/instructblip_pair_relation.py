#!/home/cho/.pyenv/versions/llava/bin/python3
"""instructblip_pair_relation.py — 換成 Open3DSG 論文實際使用的模型,單一變因 = 模型。

★ 新檔,不動 llava_pair_relation.py。
模型:Salesforce/instructblip-vicuna-7b(Open3DSG 補充材料 Sec.A 註腳 1 給的連結)。
  論文選它的理由(原文):"a big drawback of deploying a generative approach is that restricting
  the output to a desired answer is not straightforward. To this end, InstructBLIP is uniquely
  designed to give more output control using prompting. The InstructBLIP model consists of a ViT
  encoder followed by a Qformer, which receives context from learnable tokens, a user prompt and
  the output of the ViT."
  → 與 LLaVA 的差別:prompt 會影響【視覺特徵抽取】那一步(instruction-aware),LLaVA 的 prompt
    只在 LLM 端進來。是否對本任務有幫助【待驗證】。

除模型外,下列全部與 llava_pair_relation.py 相同(確保單一變因):
  影像 = pair_crops_big(538 對、1975 圖;el30 側視全取、兩物 bbox 聯集 + margin30)
  prompt = Open3DSG 原句 "Describe the relationship between [A] and [B]?"
  物體名 = GT(上界模擬);判讀 = 同一組 ON 關鍵詞;評分 = parse_on_direction.py
輸出: instructblip_pair_relation.csv(欄位與 LLaVA 版相同,可直接餵同一評分器)
用法: PAIR_DIR=pair_crops_big ./instructblip_pair_relation.py [--max-new 120]
"""
import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor, BitsAndBytesConfig

HERE = Path(__file__).resolve().parent
IMG = HERE / os.environ.get("PAIR_DIR", "pair_crops_big")
MODEL = "Salesforce/instructblip-vicuna-7b"
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
    ap.add_argument("--vote", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    idx = json.loads((IMG / "index.json").read_text())
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    print(f"載入 {MODEL}(4-bit NF4)...", flush=True)
    t0 = time.time()
    proc = InstructBlipProcessor.from_pretrained(MODEL)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=q, device_map="cuda:0", dtype=torch.float16).eval()
    print(f"  載入完成 {time.time()-t0:.0f}s;VRAM {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)
    rows = []; t00 = time.time()
    if a.limit:
        idx = idx[:a.limit]
    for n, r in enumerate(idx, 1):
        txt = f"Describe the relationship between {nice(r['objA'])} and {nice(r['objB'])}?"
        votes, answers = [], []
        for fr in r.get("frames", []):
            p = IMG / fr["file"]
            if not p.is_file():
                continue
            im = Image.open(p).convert("RGB")
            inp = proc(images=im, text=txt, return_tensors="pt").to("cuda:0", torch.float16)
            with torch.inference_mode():
                o = model.generate(**inp, max_new_tokens=a.max_new, do_sample=False,
                                   num_beams=1, repetition_penalty=1.5, length_penalty=1.0)
            ans = proc.batch_decode(o, skip_special_tokens=True)[0].strip()
            votes.append(bool(ON.search(ans))); answers.append(f"[{fr['view']}] " + ans.replace("\n", " "))
        if not votes:
            continue
        rows.append({**r, "n_view": len(votes), "n_on": sum(votes),
                     "pred_on": sum(votes) >= a.vote, "answer": " || ".join(answers)})
        if n % 50 == 0 or a.limit:
            el = time.time() - t00
            print(f"  {n}/{len(idx)}  {el:.0f}s  (預計總 {el/n*len(idx)/60:.0f} 分)", flush=True)
    with open(HERE / "instructblip_pair_relation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "objA", "objB", "is_on", "upper", "views",
                                          "n_view", "n_on", "pred_on", "answer"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print(f"[存檔] instructblip_pair_relation.csv  {len(rows)} 對")


if __name__ == "__main__":
    main()
