#!/home/cho/.pyenv/versions/llava/bin/python3
"""ask_vlm.py — 手動把任意圖片 + 任意 prompt 丟給本機 VLM,印出回答。

用途:不跑整套實驗,只想自己看某張圖模型會怎麼答(除錯 / 試 prompt / 抽查)。
模型二選一(都是 4-bit NF4,載入後 VRAM 4~6 GiB):
  --model llava   llava-hf/llama3-llava-next-8b-hf        (本專案主力,回答較精準)
  --model iblip   Salesforce/instructblip-vicuna-7b       (Open3DSG 論文原用)
兩者權重已下載在 ~/.cache/huggingface,不需網路。

用法:
  ./ask_vlm.py 圖.png "Describe the relationship between A and B?"
  ./ask_vlm.py 圖1.png 圖2.png -p "What is on top?"            # 多張圖,同一 prompt
  ./ask_vlm.py 'pair_crops_big/stack3_scene0005*.png' -p "..."  # glob(記得加引號)
  ./ask_vlm.py 圖.png -p "..." --model iblip --max-new 200
  ./ask_vlm.py 圖.png -p "..." --csv out.csv                    # 順手存檔
  ./ask_vlm.py 圖.png -p "..." --fig fig.png                     # 出 matplotlib 圖:
                                                                 #   影像+問句+模型名+回答
  ./ask_vlm.py '圖*.png' -p "..." --fig fig.png --fig-cols 2      # 每列 2 張

--fig 會呼叫 plot_vlm_answers.py(跑在 webots_visual_hull,因為 llava 環境沒有 matplotlib);
圖若來自 pair_crops*/ 會自動附上 GT(視角名 / 是否 on / 上方物體)方便目視核對。

注意:圖片直接餵原檔,不做去背/裁切。本專案實驗用的圖是 make_pair_crops.py 產的
(已去背去手臂、兩物 bbox 聯集裁切),要重現實驗結果請用 pair_crops*/ 裡的圖。
"""
import argparse
import csv
import glob
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import BitsAndBytesConfig

MODELS = {"llava": "llava-hf/llama3-llava-next-8b-hf",
          "iblip": "Salesforce/instructblip-vicuna-7b"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="圖片路徑(可多張、可 glob)。第一個非 -p 的剩餘引數若不是檔案,視為 prompt")
    ap.add_argument("-p", "--prompt", default=None, help="提示詞;不給則取 images 最後一個非檔案的引數")
    ap.add_argument("--model", choices=list(MODELS), default="llava")
    ap.add_argument("--max-new", type=int, default=200, dest="max_new")
    ap.add_argument("--csv", default=None, help="把 (圖, prompt, 回答) 存成 CSV")
    ap.add_argument("--fig", default=None, help="輸出 matplotlib 圖:影像+問句+模型名+回答")
    ap.add_argument("--fig-cols", type=int, default=1, dest="fig_cols", help="--fig 每列幾筆")
    a = ap.parse_args()

    # 拆圖片 / prompt:展開 glob,無法展開成檔案的最後一個引數當 prompt
    imgs, leftover = [], []
    for x in a.images:
        g = sorted(glob.glob(x))
        if g:
            imgs += g
        elif Path(x).is_file():
            imgs.append(x)
        else:
            leftover.append(x)
    prompt = a.prompt or (leftover[-1] if leftover else None)
    if not imgs or not prompt:
        sys.exit(f"[錯誤] 需要至少一張存在的圖 + 一個 prompt。解析到 圖={len(imgs)} prompt={prompt!r}")

    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    mid = MODELS[a.model]
    print(f"載入 {mid}(4-bit NF4)...", file=sys.stderr, flush=True)
    if a.model == "llava":
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        proc = LlavaNextProcessor.from_pretrained(mid)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            mid, quantization_config=q, device_map="cuda:0", dtype=torch.float16).eval()
    else:
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
        proc = InstructBlipProcessor.from_pretrained(mid)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            mid, quantization_config=q, device_map="cuda:0", dtype=torch.float16).eval()
    print(f"  VRAM {torch.cuda.memory_allocated()/2**30:.1f} GiB", file=sys.stderr, flush=True)

    rows = []
    for f in imgs:
        im = Image.open(f).convert("RGB")
        if a.model == "llava":
            chat = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            txt = proc.apply_chat_template(chat, add_generation_prompt=True)
            inp = proc(images=im, text=txt, return_tensors="pt").to("cuda:0", torch.float16)
        else:
            inp = proc(images=im, text=prompt, return_tensors="pt").to("cuda:0", torch.float16)
        with torch.inference_mode():
            o = model.generate(**inp, max_new_tokens=a.max_new, do_sample=False, num_beams=1,
                               **({"repetition_penalty": 1.5} if a.model == "iblip" else {}))
        ans = proc.batch_decode(o, skip_special_tokens=True)[0].strip()
        if a.model == "llava" and "assistant" in ans:       # 去掉 chat template 前綴
            ans = ans.rsplit("assistant", 1)[-1].strip()
        print(f"\n=== {Path(f).name} ===\n{ans}")
        rows.append({"image": f, "model": mid, "prompt": prompt, "answer": ans.replace("\n", " ")})

    out_csv = a.csv or (str(Path(a.fig).with_suffix(".csv")) if a.fig else None)
    if out_csv:
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["image", "model", "prompt", "answer"])
            w.writeheader(); w.writerows(rows)
        print(f"\n[存檔] {out_csv}  {len(rows)} 筆", file=sys.stderr)

    if a.fig:       # 畫圖跑在 webots_visual_hull(llava 環境沒有 matplotlib)
        import subprocess
        cmd = [str(Path(__file__).resolve().parent / "plot_vlm_answers.py"),
               "--csv", out_csv, "--out", a.fig, "--cols", str(a.fig_cols)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip(), file=sys.stderr)
        if r.returncode:
            sys.exit(f"[錯誤] 畫圖失敗(returncode {r.returncode})")


if __name__ == "__main__":
    main()
