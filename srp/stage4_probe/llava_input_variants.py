#!/home/cho/.pyenv/versions/llava/bin/python3
"""llava_input_variants.py — 單一變因測「輸入形式」對 VLM 的影響(orig / crop / maskbg)。

★ 新檔。影像由 make_vlm_inputs.py 產出(kept_object_masks + 手臂過濾,非 GT,可部署)。
   orig=原圖;crop=裁到物體外接框(去夾爪/背景牆,不塗色);maskbg=非物體像素塗中性灰。
   DISC Sec.II-C 主張遮罩去背會造成 domain shift → maskbg 預期最差,列入以驗證。

prompt 三種:
  count_stack : 自由清點 + 問堆疊(先前原圖版失敗:把垂直堆疊讀成水平並排)
  missing     : 差集探測,清單【扣掉】被漏物體 → 應答 YES 並指出它
  complete    : 差集探測,清單【完整】       → 應答 NO(★對照組,先前 5/6 失敗)
清單用 GT 名 = 上界模擬(真實 instance 無名字),與 llava_missing_probe.py 同口徑。
用法: ./llava_input_variants.py [--max-new 160]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, BitsAndBytesConfig
from pycocotools import mask as RLE

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "srp" / "io"))
from labels import label_dir   # noqa: E402

IMG = HERE / "vlm_inputs"
MODEL = "llava-hf/llama3-llava-next-8b-hf"
CASES = [("stack3_scene0005", "view_el30_az135", "gelatin_box"),
         ("stack4_scene0007", "view_el45_az225", "sponge"),
         ("stack4_scene0010", "view_el30_az195", "tuna_fish_can")]
VARIANTS = ["orig", "crop", "maskbg"]
NICE = {"gelatin_box": "gelatin box", "sponge": "sponge", "tuna_fish_can": "tuna fish can",
        "master_chef_can": "coffee can", "sugar_box": "sugar box", "foam_brick": "foam brick",
        "racquetball": "rubber ball", "tomato_soup_can": "tomato soup can", "lemon": "lemon",
        "windex_bottle": "spray bottle"}
Q_COUNT = ("List every distinct physical object you can see. For each one, give its name. "
           "Then state clearly whether any object is stacked on top of another object, "
           "and which is on top of which.")


def vis_objs(sc, vn, min_area=300):
    d = json.loads((label_dir(sc) / "actual" / "annotations.json").read_text())
    id2n = {c["id"]: c["name"].split("_", 1)[-1] for c in d["categories"] if c["name"] != "ur5e"}
    iid = {Path(im["file_name"]).stem: im["id"] for im in d["images"]}.get(vn)
    return sorted(id2n[a["category_id"]] for a in d["annotations"]
                  if a["image_id"] == iid and a["category_id"] in id2n
                  and int(RLE.decode(a["segmentation"]).sum()) >= min_area)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-new", type=int, default=160, dest="max_new")
    a = ap.parse_args()
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    print(f"載入 {MODEL}(4-bit NF4)...", flush=True)
    proc = LlavaNextProcessor.from_pretrained(MODEL)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=q, device_map="cuda:0", dtype=torch.float16).eval()

    def ask(im, txt):
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": txt}]}]
        p = proc.apply_chat_template(conv, add_generation_prompt=True)
        inp = proc(images=im, text=p, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            o = model.generate(**inp, max_new_tokens=a.max_new, do_sample=False)
        return proc.decode(o[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    md = ["# 輸入形式對 VLM 的影響:orig vs crop vs maskbg\n",
          f"- 建檔 2026-09-25;程式 `srp/stage4_probe/llava_input_variants.py`;模型 `{MODEL}`,4-bit NF4。",
          "- 影像由 `make_vlm_inputs.py` 產出:**裁切/去背都用 kept_object_masks + 手臂過濾(非 GT,可部署)**。",
          "- `maskbg` 用來驗證 DISC Sec.II-C 的主張(遮罩去背造成 domain shift)。",
          "- 差集探測的清單用 **GT 名 = 上界模擬**;`complete` 是**對照組**,正確答案為「沒有」。\n"]
    for sc, vn, missed in CASES:
        vis = vis_objs(sc, vn)
        full = [NICE.get(o, o.replace("_", " ")) for o in vis]
        part = [NICE.get(o, o.replace("_", " ")) for o in vis if o != missed]
        md.append(f"## {sc} / {vn}(漏掉:**{NICE.get(missed, missed)}**)\n")
        md.append(f"- 該視角可見:{', '.join(full)}\n")
        for var in VARIANTS:
            p = IMG / f"{sc}__{vn}__{var}.png"
            if not p.is_file():
                md.append(f"### 輸入 `{var}` — ★缺影像\n"); continue
            im = Image.open(p).convert("RGB")
            md.append(f"### 輸入 `{var}`(圖 {im.size[0]}x{im.size[1]})\n")
            for tag, txt, exp in (
                    ("count_stack", Q_COUNT, "應指出堆疊"),
                    ("missing", f"I have already detected these objects in this image: {', '.join(part)}. "
                                "Is there any other physical object visible in this image that is NOT in that "
                                "list? Answer with YES or NO first. If YES, name the object and describe "
                                "where it is.", f"應答 YES + {NICE.get(missed, missed)}"),
                    ("complete", f"I have already detected these objects in this image: {', '.join(full)}. "
                                 "Is there any other physical object visible in this image that is NOT in that "
                                 "list? Answer with YES or NO first. If YES, name the object and describe "
                                 "where it is.", "**應答 NO(對照組)**")):
                t0 = time.time(); out = ask(im, txt)
                md.append(f"**`{tag}`**({exp};{time.time()-t0:.0f}s)\n")
                md.append("```\n" + out + "\n```\n")
                print(f"[{sc}/{var}/{tag}] {time.time()-t0:.0f}s", flush=True)
    o = HERE / "RESULT_llava_input_variants.md"
    o.write_text("\n".join(md), encoding="utf-8")
    print(f"[存檔] {o}")


if __name__ == "__main__":
    main()
