#!/home/cho/.pyenv/versions/llava/bin/python3
"""llava_missing_probe.py — 給 VLM「已偵測物體清單」,問它畫面中還有沒有清單外的物體?

★ 新檔,獨立 pyenv 環境 `llava`。動機見 llava_scene_desc.py。
   先前用「整張圖自由清點 + 判斷堆疊」(llava_scene_desc.py)效果差:LLaVA 把垂直堆疊讀成水平並排。
   本腳本改成【封閉式差集問題】——只要它做辨識,不要它做空間關係推理。

⚠ 這是【上界模擬,不是可部署方法】:清單用 GT 物體名。真實管線的 instance 是 class-agnostic 沒有名字,
  取得名字要靠 CLIP 命名(EXPERIMENT_SUMMARY.md §A:top-1 僅 0.493)或 VLM caption,那是另一個問題。
  此處先用 GT 隔離變因,測「若命名完美,這個機制行不行」。

★ 對照組不可省:每張圖問兩次
   missing  : 清單 = 該視角可見物體【扣掉】被管線漏掉的那個 → 正確答案「有,是 X」
   complete : 清單 = 該視角【全部】可見物體                   → 正確答案「沒有」
   若 complete 也回答「有」,代表模型在亂猜,訊號無效。

可見性:GT modal 遮罩面積 >= --min-area(預設 300 px)才算「該視角可見」。
GEX 物體(skillet_lid/windex_bottle/colored_wood_blocks/dice)【列入清單】——它們確實在畫面中,
不列會製造假陽性;但評估時本來就排除,故不當作待偵測目標。
用法: ./llava_missing_probe.py [--max-new 160] [--min-area 300]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, BitsAndBytesConfig
from pycocotools import mask as RLE

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "srp" / "io"))
from labels import label_dir   # noqa: E402

CAP = REPO / "data" / "captures_fast"
MODEL = "llava-hf/llama3-llava-next-8b-hf"
CASES = [
    ("stack3_scene0005", "view_el30_az135", "gelatin_box"),
    ("stack3_scene0005", "view_el75_az225", "gelatin_box"),
    ("stack4_scene0007", "view_el45_az225", "sponge"),
    ("stack4_scene0007", "view_el30_az210", "sponge"),
    ("stack4_scene0010", "view_el75_az225", "tuna_fish_can"),
    ("stack4_scene0010", "view_el30_az195", "tuna_fish_can"),
]
NICE = {"gelatin_box": "gelatin box", "sponge": "sponge", "tuna_fish_can": "tuna fish can",
        "master_chef_can": "coffee can", "sugar_box": "sugar box", "foam_brick": "foam brick",
        "racquetball": "rubber ball", "tomato_soup_can": "tomato soup can", "lemon": "lemon",
        "windex_bottle": "spray bottle"}


def visible_objs(sc, vn, min_area):
    f = label_dir(sc) / "actual" / "annotations.json"
    d = json.loads(f.read_text())
    id2n = {c["id"]: c["name"].split("_", 1)[-1] for c in d["categories"] if c["name"] != "ur5e"}
    iid = {Path(im["file_name"]).stem: im["id"] for im in d["images"]}.get(vn)
    out = {}
    for a in d["annotations"]:
        if a["image_id"] == iid and a["category_id"] in id2n:
            ar = int(RLE.decode(a["segmentation"]).sum())
            if ar >= min_area:
                out[id2n[a["category_id"]]] = ar
    return out


def ask(model, proc, im, q, max_new):
    conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    p = proc.apply_chat_template(conv, add_generation_prompt=True)
    inp = proc(images=im, text=p, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        o = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return proc.decode(o[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=160, dest="max_new")
    ap.add_argument("--min-area", type=int, default=300, dest="min_area")
    a = ap.parse_args()
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    print(f"載入 {MODEL}(4-bit NF4)...", flush=True)
    proc = LlavaNextProcessor.from_pretrained(MODEL)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qcfg, device_map="cuda:0", dtype=torch.float16).eval()
    md = ["# VLM 差集探測:給已偵測清單,問畫面中還有沒有清單外的物體\n",
          f"- 建檔 2026-09-25;程式 `srp/stage4_probe/llava_missing_probe.py`;模型 `{MODEL}`,**4-bit NF4**。",
          f"- 可見性門檻:GT modal 遮罩面積 ≥ {a.min_area} px。GEX 物體列入清單(畫面中確實存在)。",
          "- ⚠ **清單用 GT 物體名 = 上界模擬**,非可部署方法(真實 instance 無名字)。",
          "- **對照組**:`complete`(清單完整)正確答案是「沒有」;若也答「有」則訊號無效。\n"]
    for sc, vn, missed in CASES:
        p = CAP / f"multi_{sc.split('_')[0]}" / sc / f"{vn}.png"
        vis = visible_objs(sc, vn, a.min_area)
        if not p.is_file() or missed not in vis:
            md.append(f"## {sc} / {vn}\n\n★ 跳過(影像不存在或 {missed} 在此視角不可見,面積<{a.min_area})\n")
            continue
        im = Image.open(p).convert("RGB")
        full = [NICE.get(o, o.replace("_", " ")) for o in sorted(vis)]
        part = [NICE.get(o, o.replace("_", " ")) for o in sorted(vis) if o != missed]
        md.append(f"## {sc} / {vn}\n")
        md.append(f"- 該視角可見物體(面積 px):{ {o: vis[o] for o in sorted(vis)} }")
        md.append(f"- 被管線漏掉:**{NICE.get(missed, missed)}**\n")
        for tag, lst, expect in (("missing", part, f"應答:有,{NICE.get(missed,missed)}"),
                                 ("complete", full, "應答:沒有")):
            q = (f"I have already detected these objects in this image: {', '.join(lst)}. "
                 f"Is there any other physical object visible in this image that is NOT in that list? "
                 f"Answer with YES or NO first. If YES, name the object and describe where it is.")
            t0 = time.time(); txt = ask(model, proc, im, q, a.max_new)
            md.append(f"**條件 `{tag}`** — 清單:{', '.join(lst)}  ({expect};{time.time()-t0:.0f}s)\n")
            md.append("```\n" + txt + "\n```\n")
            print(f"[{sc}/{vn}/{tag}] {time.time()-t0:.0f}s", flush=True)
    out = HERE / "RESULT_llava_missing_probe.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[存檔] {out}")


if __name__ == "__main__":
    main()
