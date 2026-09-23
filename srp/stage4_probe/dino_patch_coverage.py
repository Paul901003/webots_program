#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""dino_patch_coverage.py — 體檢:gt_mask_feats.py 的 DINOv2 用 INTER_NEAREST 選 patch,比「真實覆蓋面積」虧多少。

問題:dino_feats 行 73 用 cv2.resize(mask,(91,51),INTER_NEAREST) 決定哪些 patch 屬於該遮罩,
      等於每個 14x14=196px 的 patch 只檢查【1 個取樣點】(全圖 4641 點 / 921600 px = 0.5% 取樣率),
      且選中的 patch 一律等權平均 → 覆蓋 15% 的邊界 patch 與覆蓋 100% 的核心 patch 貢獻相同。

本腳本【不需要 DINOv2 模型】:覆蓋率是純遮罩幾何。量四件事,決定值不值得改成覆蓋率加權:
  (1) NEAREST 選中的 patch,覆蓋率分布 → 半背景 patch 佔多少、佔多少特徵權重
  (2) NEAREST 選中集合 vs cov>=0.5 集合的差異 → 取樣點隨機性造成多少誤選/漏選
  (3) 被丟棄(NEAREST 選 0 個)的筆數,改用「有交集(cov>0)就算」能救回幾筆
  (4) 逐物體,看小物體受害程度

對齊:影像送模型前被 resize 成 H14xW14=(720//14)*14 x (1280//14)*14 = 714x1274,patch 網格 51x91=4641。
      覆蓋率在同一個 714x1274 上算(把每 14x14 block 的遮罩像素數 / 196),與模型輸入完全對齊。
      NEAREST 側則【原封不動重現】行 73(720x1280 -> 91x51),不做任何美化。

拿什麼:GT modal 遮罩 data/labels/<scene>/actual/annotations.json(同 gt_mask_feats.gt_modal)。
分母   :60 場(stack3/stack4/stack5)所有 (物體x視角) 對,與 data/eval/gt_mask_feats/*.npz 同母體。
排除項 :GEX(skillet_lid/windex_bottle/colored_wood_blocks/dice)另列,不預設濾除。
用法   : ./dino_patch_coverage.py [stack3 stack4 stack5]   輸出 RESULT_dino_patch_coverage.md
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

GEX = {"skillet_lid", "windex_bottle", "colored_wood_blocks", "dice"}
PATCH = 14


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


def analyse(seg):
    """回 (near 選中的 bool 網格, 覆蓋率網格 cov)。near 原封重現 dino_feats 行 73。"""
    H, W = seg.shape
    H14, W14 = (H // PATCH) * PATCH, (W // PATCH) * PATCH
    ph, pw = H14 // PATCH, W14 // PATCH
    u8 = seg.astype(np.uint8)
    near = cv2.resize(u8, (pw, ph), interpolation=cv2.INTER_NEAREST) > 0      # 行 73 原樣
    s14 = cv2.resize(u8, (W14, H14), interpolation=cv2.INTER_NEAREST)         # 對齊模型輸入
    cov = s14.reshape(ph, PATCH, pw, PATCH).sum(axis=(1, 3)) / float(PATCH * PATCH)
    return near, cov


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

    rows = []          # 每筆遮罩一列
    covsel_all = []    # NEAREST 選中 patch 的覆蓋率(攤平,跨全部遮罩)
    for sc in scenes:
        mo = gt_modal(sc)
        if not mo:
            continue
        for vn, objs in sorted(mo.items()):
            for o, seg in objs.items():
                near, cov = analyse(seg)
                c50 = cov >= 0.5
                n_near, n_c50, n_any = int(near.sum()), int(c50.sum()), int((cov > 0).sum())
                cs = cov[near]
                covsel_all.append(cs)
                rows.append(dict(
                    scene=sc, view=vn, obj=o, gex=o.split("_", 1)[-1] in GEX,
                    n_near=n_near, n_c50=n_c50, n_any=n_any,
                    # 誤選 = NEAREST 選了但覆蓋 <50%;漏選 = 覆蓋 >=50% 卻沒被選
                    n_false_in=int((near & ~c50).sum()), n_false_out=int((~near & c50).sum()),
                    # 特徵權重:等權平均下,來自覆蓋 <50% / <25% patch 的權重比例
                    w_lt50=float((cs < 0.5).mean()) if n_near else np.nan,
                    w_lt25=float((cs < 0.25).mean()) if n_near else np.nan,
                    cov_mean=float(cs.mean()) if n_near else np.nan,
                ))
    if not rows:
        print("無資料"); return
    A = {k: np.array([r[k] for r in rows]) for k in rows[0] if k not in ("scene", "view", "obj")}
    gex = A["gex"].astype(bool)
    allcov = np.concatenate([c for c in covsel_all if len(c)])

    def sec(sel, tag):
        n = int(sel.sum())
        if n == 0:
            return [f"| {tag} | 0 | – | – | – | – | – | – |"]
        ok = sel & (A["n_near"] > 0)
        mism = (A["n_false_in"][sel] + A["n_false_out"][sel])
        denom = np.maximum(A["n_c50"][sel] + A["n_false_in"][sel], 1)
        return [f"| {tag} | {n} | {np.nanmean(A['w_lt50'][ok])*100:.1f}% | "
                f"{np.nanmean(A['w_lt25'][ok])*100:.1f}% | {np.nanmean(A['cov_mean'][ok]):.3f} | "
                f"{A['n_false_in'][sel].mean():.1f} | {A['n_false_out'][sel].mean():.1f} | "
                f"{(mism/denom).mean()*100:.1f}% |"]

    md = ["# DINOv2 patch 選取體檢:INTER_NEAREST vs 真實覆蓋面積\n",
          "- 建檔 2026-09-23;程式 `srp/stage4_probe/dino_patch_coverage.py`;**不需 DINOv2 模型**(純遮罩幾何);可復現。",
          f"- 分母 = {len(scenes)} 場({', '.join(targets)})共 **{len(rows)}** 個 (物體×視角) 對,GT modal 非空遮罩。",
          "- patch 網格 51×91 = **4641**,每 patch 14×14 = **196 px**;覆蓋率 = 該 patch 落在遮罩內的像素數 / 196。",
          "- NEAREST 側原封重現 `dino_feats` 行 73(720×1280 → 91×51,每 patch 只檢查 1 個取樣點)。\n",
          "## (1)(2) 現行選法的品質\n",
          "| 母體 | 對數 | 特徵權重來自 cov<0.5 的 patch | 來自 cov<0.25 | 選中 patch 平均覆蓋率 | 誤選/筆 | 漏選/筆 | 與 cov≥0.5 選法的不一致率 |",
          "|---|---|---|---|---|---|---|---|"]
    md += sec(np.ones(len(rows), bool), "全部")
    md += sec(~gex, "排除 GEX")
    md += sec(gex, "只看 GEX")
    md += ["",
           "- **誤選** = NEAREST 選中但覆蓋 <50% 的 patch 數;**漏選** = 覆蓋 ≥50% 卻沒被選中的 patch 數。",
           "- **不一致率** = (誤選+漏選) / cov≥0.5 集合大小 → 取樣點隨機性造成的偏差幅度。\n",
           "## 選中 patch 的覆蓋率分布(全部遮罩攤平)\n",
           "| 覆蓋率區間 | patch 數 | 佔比 |", "|---|---|---|"]
    bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.01)]
    for lo, hi in bins:
        k = int(((allcov >= lo) & (allcov < hi)).sum())
        lab = "= 1.00(全覆蓋)" if lo == 1.0 else f"[{lo:.2f}, {hi:.2f})"
        md.append(f"| {lab} | {k} | {k/len(allcov)*100:.1f}% |")
    md += ["", f"合計 {len(allcov)} 個被選中的 patch;中位覆蓋率 **{np.median(allcov):.3f}**。\n",
           "## (3) 被丟棄的筆數能否救回\n", "| 母體 | NEAREST 選 0 個(被丟棄) | 其中 cov>0(有交集,可救回) | 可救回率 |", "|---|---|---|---|"]
    for tag, sel in (("全部", np.ones(len(rows), bool)), ("排除 GEX", ~gex), ("只看 GEX", gex)):
        d = sel & (A["n_near"] == 0)
        rec = int((d & (A["n_any"] > 0)).sum())
        md.append(f"| {tag} | {int(d.sum())} | {rec} | "
                  f"{rec/max(int(d.sum()),1)*100:.0f}% |" if d.sum() else f"| {tag} | 0 | 0 | – |")
    md += ["", "## (4) 逐物體(依「權重來自 cov<0.5」由高到低,取前 20)\n",
           "| 物體 | 對數 | 選中patch中位數 | 權重來自 cov<0.5 | 權重來自 cov<0.25 | 平均覆蓋率 | 是否 GEX |",
           "|---|---|---|---|---|---|---|"]
    byobj = defaultdict(list)
    for i, r in enumerate(rows):
        byobj[r["obj"]].append(i)
    stat = []
    for o, idx in byobj.items():
        idx = np.array(idx); ok = idx[A["n_near"][idx] > 0]
        if len(ok) == 0:
            continue
        stat.append((float(np.nanmean(A["w_lt50"][ok])), o, len(idx),
                     int(np.median(A["n_near"][ok])), float(np.nanmean(A["w_lt25"][ok])),
                     float(np.nanmean(A["cov_mean"][ok])), rows[idx[0]]["gex"]))
    for w50, o, n, med, w25, cm, g in sorted(stat, reverse=True)[:20]:
        md.append(f"| {o} | {n} | {med} | {w50*100:.1f}% | {w25*100:.1f}% | {cm:.3f} | {'是' if g else '否'} |")
    md.append("")
    out = HERE / "RESULT_dino_patch_coverage.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[存檔] {out}\n\n" + "\n".join(md))


if __name__ == "__main__":
    main()
