#!/home/cho/.pyenv/versions/llava/bin/python3
"""llava_scene_desc.py — 用 LLaVA-1.6(llama3-llava-next-8b)描述場景影像,看 VLM 能不能發現被漏掉的堆疊物。

★ 新檔,獨立 pyenv 環境 `llava`(不動 webots_visual_hull 等既有環境)。

動機:match_eval 顯示現行最佳法在 303 場漏 15 個物體、全在 stack 組。SAM 在 2D 有切開(併=0%,
      見 RESULT_sam2d_sep_mv2.md),遮罩層級 voxel 重疊圖也切不開(RESULT_mask_overlap_graph_ov1.md)
      → 幾何路走不通,改看「VLM 的語意描述」這個獨立訊號能否指出漏掉的物體。
模型:llava-hf/llama3-llava-next-8b-hf(= BBQ 論文用的 LLaVA-1.6 那代),4-bit NF4 量化
      (RTX 4070 Ti 12GB;fp16 需 16GB 放不下。NF4 品質損失待驗證)。
用法: ./llava_scene_desc.py [--max-new 256] [--out RESULT_llava_failcases.md]
"""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, BitsAndBytesConfig

REPO = Path(__file__).resolve().parents[2]
CAP = REPO / "data" / "captures_fast"
MODEL = "llava-hf/llama3-llava-next-8b-hf"

# (場景, 視角, 被管線漏掉的物體, 備註)
CASES = [
    ("stack3_scene0005", "view_el30_az135", "gelatin_box", "側視,gelatin 可見面積最大"),
    ("stack3_scene0005", "view_el75_az225", "gelatin_box", "高仰角對照"),
    ("stack4_scene0007", "view_el45_az225", "sponge", "sponge 可見面積最大"),
    ("stack4_scene0007", "view_el30_az210", "sponge", "低仰角對照"),
    ("stack4_scene0010", "view_el75_az225", "tuna_fish_can", "俯視,tuna 面積最大但疑似看不出兩罐"),
    ("stack4_scene0010", "view_el30_az195", "tuna_fish_can", "側視,兩罐應清楚可分"),
]
PROMPTS = {
    "bbq": "Describe visible object in front of you, paying close attention to its "
           "spatial dimensions and visual attributes.",
    "count_stack": "List every distinct physical object you can see on the table. "
                   "For each one, give its name. Then state clearly whether any object is "
                   "stacked on top of another object, and which is on top of which.",
}


def img_path(sc, vn):
    return CAP / f"multi_{sc.split('_')[0]}" / sc / f"{vn}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=256, dest="max_new")
    ap.add_argument("--out", default="RESULT_llava_failcases.md")
    ap.add_argument("--only", default=None, help="只跑某場景(除錯用)")
    a = ap.parse_args()
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_use_double_quant=True)
    print(f"載入 {MODEL}(4-bit NF4)...", flush=True)
    t0 = time.time()
    proc = LlavaNextProcessor.from_pretrained(MODEL)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=qcfg, device_map="cuda:0", dtype=torch.float16)
    model.eval()
    print(f"  載入完成 {time.time()-t0:.0f}s;VRAM {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)

    md = ["# LLaVA-1.6 對「管線漏找」失敗案例的描述\n",
          f"- 建檔 2026-09-25;程式 `srp/stage4_probe/llava_scene_desc.py`;模型 `{MODEL}`,**4-bit NF4**。",
          "- 目的:看 VLM 能不能指出被幾何管線漏掉的堆疊物(獨立於 hull 幾何的訊號)。",
          "- 案例來自 `RESULT_match_eval_*`(瘦中心drop / 瘦fp drop 的沒找到物體)。",
          "- **原始輸出未經編輯**;判讀另外標註。\n"]
    for sc, vn, missed, note in CASES:
        if a.only and a.only != sc:
            continue
        p = img_path(sc, vn)
        if not p.is_file():
            md.append(f"## {sc} / {vn}\n\n★ 找不到影像 {p}\n"); continue
        im = Image.open(p).convert("RGB")
        md.append(f"## {sc} / {vn}\n")
        md.append(f"- 管線漏掉的物體:**{missed}**;視角備註:{note}\n")
        for key, q in PROMPTS.items():
            conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
            prompt = proc.apply_chat_template(conv, add_generation_prompt=True)
            inputs = proc(images=im, text=prompt, return_tensors="pt").to("cuda:0")
            t1 = time.time()
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=a.max_new, do_sample=False)
            txt = proc.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            md.append(f"**prompt `{key}`** ({time.time()-t1:.0f}s)\n\n> {q}\n")
            md.append("```\n" + txt + "\n```\n")
            print(f"[{sc}/{vn}/{key}] {time.time()-t1:.0f}s", flush=True)
    outp = Path(__file__).resolve().parent / a.out
    outp.write_text("\n".join(md), encoding="utf-8")
    print(f"[存檔] {outp}")


if __name__ == "__main__":
    main()
