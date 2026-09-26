#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""plot_vlm_answers.py — 把 ask_vlm.py 的回答畫成圖:影像 + 問句 + 模型名 + 模型回答。

★ 新檔。跑在 webots_visual_hull(有 matplotlib 3.10.8 + Noto Sans CJK);
  推論在 llava 環境,兩邊分工,不動已設好的 llava 環境(沒有 matplotlib)。

輸入 CSV 需有欄位 image, model, prompt, answer(= ask_vlm.py --csv 的格式)。
若圖屬於 pair_crops*/,會自動從同目錄 index.json 補上 GT 資訊
(視角名、GT 是否 on、GT 上方物體),方便目視核對模型答得對不對。

用法:
  ./plot_vlm_answers.py --csv out.csv --out fig.png
  ./plot_vlm_answers.py --csv out.csv --out fig.png --cols 2      # 每列 2 張
通常不必自己跑:ask_vlm.py --fig fig.png 會自動呼叫本檔。
"""
import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = Path(__file__).resolve().parent
# 中文不要變豆腐框:CJK 字型要同時掛進 sans-serif 與 monospace
_CJK = next((n for n in ("Noto Sans CJK JP", "Droid Sans Fallback")
             if any(f.name == n for f in fm.fontManager.ttflist)), None)
if _CJK:
    plt.rcParams["font.sans-serif"] = [_CJK, "DejaVu Sans"]
    # ⚠ 不可用 family="monospace":matplotlib 的 monospace 不會 fallback 到 CJK,
    #    中文會變成豆腐框。Noto Sans CJK 本身含 Latin 字形,直接用它。
plt.rcParams["axes.unicode_minus"] = False

_IDX_CACHE = {}


def _abs(q):
    p = Path(q)
    return p if p.is_absolute() else HERE / p


def gt_of(img_path):
    """圖若來自 pair_crops*/,回 (視角名, GT是否on, GT上方物體);否則回 None。"""
    p = Path(img_path)
    idx = p.parent / "index.json"
    if not idx.is_file():
        return None
    if idx not in _IDX_CACHE:
        m = {}
        for r in json.loads(idx.read_text()):
            for fr in r.get("frames", []):
                m[fr["file"]] = (fr["view"], r.get("is_on"), r.get("upper"))
        _IDX_CACHE[idx] = m
    return _IDX_CACHE[idx].get(p.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=1, help="每列幾筆(預設 1,文字最寬好讀)")
    ap.add_argument("--wrap", type=int, default=0, help="回答每行字元數;0=依欄寬自動")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    if not rows:
        raise SystemExit(f"[錯誤] {a.csv} 沒有資料列")
    cols = max(1, a.cols)
    nrow = -(-len(rows) // cols)
    wrap = a.wrap or (92 // cols)

    def blk(r):                     # 先算文字,才知道每列要多高
        g = gt_of(_abs(r["image"]))
        L = [f"檔案: {Path(r['image']).name}", f"模型: {r['model']}", ""]
        L += ["問句:"] + textwrap.wrap(r["prompt"], wrap) + [""]
        L += ["回答:"] + textwrap.wrap(r["answer"], wrap)
        if g:
            L += ["", f"GT: 視角 {g[0]} / on={g[1]} / 上方={g[2]}"]
        return L

    blks = [blk(r) for r in rows]
    # 每列高度 = 該列最長文字塊的行數(每行約 0.17 吋),下限 2.6 吋讓影像不致太小
    hs = [max(2.6, 0.17 * max(len(blks[i]) for i in range(r0 * cols, min((r0 + 1) * cols, len(blks)))))
          for r0 in range(nrow)]
    fig, axes = plt.subplots(nrow, 2 * cols, squeeze=False,
                            figsize=(7.4 * cols, sum(hs)),
                            gridspec_kw={"width_ratios": [1, 1.7] * cols,
                                         "height_ratios": hs})
    for ax in axes.ravel():
        ax.axis("off")

    for i, (r, lines) in enumerate(zip(rows, blks)):
        ar, ac = divmod(i, cols)
        axi, axt = axes[ar][2 * ac], axes[ar][2 * ac + 1]
        axi.imshow(plt.imread(_abs(r["image"])))
        axt.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=8.2,
                 family=(_CJK or "DejaVu Sans"), linespacing=1.35, wrap=False)

    fig.tight_layout(h_pad=1.2)
    fig.savefig(a.out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"[存圖] {a.out}  {len(rows)} 筆")


if __name__ == "__main__":
    main()
