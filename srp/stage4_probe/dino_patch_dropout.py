#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""dino_patch_dropout.py — 量 gt_mask_feats.py 的 DINOv2 取樣有多少物體被 patch 網格降採樣丟掉。

背景:gt_mask_feats.py 的 dino_feats() 對【整張圖】跑一次 DINOv2 得 dense patch tokens,
再把每個遮罩用 cv2.resize(INTER_NEAREST) 降到 patch 網格(14px/patch)後取該區平均。
若降採樣後 mm.sum()==0(遮罩太小/太細,一個 patch 中心都沒蓋到)→ dino_feats 回 None
→ process() 行 124 把該 (物體,視角) 整筆丟掉。CLIP 那側不會丟(gt_modal 已濾掉全空遮罩)。

本腳本【不需要 DINOv2 模型】:行 73 的降採樣是純影像運算,可精確重現。

拿什麼:GT modal 遮罩 data/labels/<scene>/actual/annotations.json(同 gt_mask_feats.gt_modal:
        排除 ur5e、排除 m.sum()==0)。做什麼:重現 patch 網格降採樣,統計丟棄率與 patch 數分布。
分母   :60 場(stack3/stack4/stack5)所有 (物體×視角) 對,與現有 data/eval/gt_mask_feats/*.npz 同母體。
交叉驗證:與 npz 實際筆數比對(⚠ npz 那次跑會跳過缺 RGB 的視角,故 npz 筆數 ≤ 本統計分母)。
排除項 :GEX(skillet_lid/windex_bottle/colored_wood_blocks/dice)另列一欄,不預設濾除。
用法   : ./dino_patch_dropout.py [stack3 stack4 stack5]   輸出 RESULT_dino_patch_dropout.md
"""
import json
import sys
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
from pycocotools import mask as mask_utils

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "srp" / "io"))
from labels import LABELS  # noqa: E402

FEAT = REPO / "data" / "eval" / "gt_mask_feats"
GEX = {"skillet_lid", "windex_bottle", "colored_wood_blocks", "dice"}
PATCH = 14  # dinov2_vitb14


def gt_modal(scene):
    """與 gt_mask_feats.gt_modal 完全相同的讀法(排 ur5e、排全空遮罩)。"""
    ann = LABELS / scene / "actual" / "annotations.json"
    if not ann.is_file():
        return None
    d = json.loads(ann.read_text())
    cat = {c["id"]: c["name"] for c in d["categories"]}
    vof = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
    mo = {}
    for a in d["annotations"]:
        nm = cat[a["category_id"]]
        if nm == "ur5e":
            continue
        m = mask_utils.decode(a["segmentation"]).astype(bool)
        if m.sum() == 0:
            continue
        mo.setdefault(vof[a["image_id"]], {})[nm] = m
    return mo


def patch_count(seg):
    """重現 dino_feats 行 66/73:圖縮到 14 倍數 → 遮罩 NEAREST 降到 patch 網格,回 patch 數。"""
    H, W = seg.shape
    ph, pw = (H // PATCH * PATCH) // PATCH, (W // PATCH * PATCH) // PATCH
    mm = cv2.resize(seg.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST) > 0
    return int(mm.sum()), ph * pw


def main():
    targets = sys.argv[1:] or ["stack3", "stack4", "stack5"]
    scenes = []
    for a in targets:
        if "scene" in a:
            scenes.append(a)
        else:
            scenes += [Path(p).parent.parent.name
                       for p in glob.glob(str(LABELS / f"{a}_scene*/actual/annotations.json"))]
    scenes = sorted(set(scenes))

    rows = []            # (scene, view, obj, px, npatch)
    for sc in scenes:
        mo = gt_modal(sc)
        if not mo:
            continue
        for vn, objs in sorted(mo.items()):
            for o, seg in objs.items():
                npx = int(seg.sum())
                npatch, total_patch = patch_count(seg)
                rows.append((sc, vn, o, npx, npatch))
    if not rows:
        print("無資料"); return
    px = np.array([r[3] for r in rows]); npa = np.array([r[4] for r in rows])
    # labels 物體名帶 YCB 前綴(如 070-b_colored_wood_blocks),GEX 是裸名 → 去前綴再比
    isgex = np.array([r[2].split("_", 1)[-1] in GEX for r in rows])
    grid = patch_count(np.ones((720, 1280), bool))[1]

    def blk(sel, tag):
        n = int(sel.sum())
        if n == 0:
            return [f"| {tag} | 0 | – | – | – | – | – |"]
        d = npa[sel] == 0
        p = px[sel]
        return [f"| {tag} | {n} | {int(d.sum())} | {d.mean()*100:.2f}% | "
                f"{int(np.median(npa[sel]))} | {int((npa[sel]<=3).sum())} ({(npa[sel]<=3).mean()*100:.2f}%) | "
                f"{int(np.median(p))} |"]

    md = ["# DINOv2 patch 網格降採樣丟棄率(gt_mask_feats.py)\n",
          f"- 建檔 2026-09-23;程式 `srp/stage4_probe/dino_patch_dropout.py`;可復現。",
          f"- 分母 = {len(scenes)} 場({', '.join(targets)})所有 (物體×視角) 對,GT modal 非空遮罩。",
          f"- 重現 `dino_feats` 行 66/73:1280×720 → patch 網格 91×51 = **{grid} patches**(14px/patch),INTER_NEAREST。",
          "- 丟棄 = 降採樣後 patch 數 0 → `dino_feats` 回 None → `process()` 行 124 整筆丟掉(CLIP 側不丟)。\n",
          "## 統計\n",
          "| 母體 | 對數 | 丟棄數 | 丟棄率 | patch數中位 | ≤3 patch | 遮罩面積中位(px) |",
          "|---|---|---|---|---|---|---|"]
    md += blk(np.ones(len(rows), bool), "全部")
    md += blk(~isgex, "排除 GEX")
    md += blk(isgex, "只看 GEX")
    md.append("")

    # 被丟棄者與極小者的物體分布
    drop = defaultdict(int); tot = defaultdict(int); tiny = defaultdict(int)
    for (sc, vn, o, p, q) in rows:
        tot[o] += 1
        if q == 0:
            drop[o] += 1
        if q <= 3:
            tiny[o] += 1
    md.append("## 逐物體(只列有丟棄或 ≤3 patch 的)\n")
    md.append("| 物體 | 總對數 | 丟棄數 | 丟棄率 | ≤3 patch 數 | 是否 GEX |")
    md.append("|---|---|---|---|---|---|")
    for o in sorted(tot, key=lambda x: (-drop[x], -tiny[x])):
        if drop[o] == 0 and tiny[o] == 0:
            continue
        md.append(f"| {o} | {tot[o]} | {drop[o]} | {drop[o]/tot[o]*100:.1f}% | {tiny[o]} | "
                  f"{'是' if o.split(chr(95),1)[-1] in GEX else '否'} |")
    md.append("")

    # 與既有 npz 交叉驗證
    md.append("## 與既有 npz 交叉驗證\n")
    md.append("| 項目 | 數值 |")
    md.append("|---|---|")
    md.append(f"| 本統計分母(GT modal 對數) | {len(rows)} |")
    md.append(f"| 本統計預測保留數 | {int((npa>0).sum())} |")
    nz = sorted(FEAT.glob("*.npz"))
    if nz:
        actual = sum(len(np.load(f)["names"]) for f in nz)
        md.append(f"| 既有 npz 實際筆數({len(nz)} 場) | {actual} |")
        md.append(f"| 差值(預測保留 − 實際) | {int((npa>0).sum()) - actual} |")
        md.append("\n⚠ 差值 >0 屬正常:`process()` 行 118–119 會跳過缺 RGB 的視角"
                  "(該次跑的 `CAPTURES_ROOT` 預設是 `data/captures` 而非 `captures_fast`),"
                  "那些視角不在 npz 裡但算在本統計分母內。")
    out = HERE / "RESULT_dino_patch_dropout.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[存檔] {out}\n\n" + "\n".join(md))


if __name__ == "__main__":
    main()
