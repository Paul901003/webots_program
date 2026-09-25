#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""sam2d_sep_mv2.py — MobileSAMv2 遮罩有沒有把「堆疊上下物」在 2D 切開?

★ 新檔。既有 diag_stack_sep.py 寫死 SAM=data/eval/sam_only(舊遮罩),與現行管線用的
   mobilesamv2_fast 基準不一致(見 [[consistent-comparison-baseline]]),故另開新檔重測。
   既有結果(sam_only,60場/717視角):分開 84% / 併 1% / 漏 15%(REPORT_relation_separation.md §6.1)。

為什麼要問這個:若要用 VLM 場景描述偵測「漏找的堆疊物」,前提是【那個物體要有自己的 SAM 遮罩】
可以被指認。若 SAM 在 2D 就把上下物切成同一塊,描述再準也指不到遮罩,瓶頸就在分割而非關聯。

做什麼:對每個 on 對 (T=上物, B=下物),逐視角:
  用 GT modal 遮罩定位 T/B → 各自找 IoU 最大的 kept SAM 遮罩(與管線同一套 kept_object_masks 過濾)
  分開 = T/B 最佳匹配是【不同】遮罩且兩者 IoU>=THR
  併   = T/B 最佳匹配是【同一個】遮罩
  漏   = 至少一方沒有 IoU>=THR 的遮罩(通常是被遮住)
排除:GEX(skillet_lid/windex_bottle/colored_wood_blocks/dice)的 on 對整對不計。
用法: ./sam2d_sep_mv2.py [--thr 0.5]   輸出 RESULT_sam2d_sep_mv2.md + sam2d_sep_mv2_perpair.csv
env : SAM_ROOT(mobilesamv2_fast) HULL_ROOT(定場景清單)
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "srp" / "io")); sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import masks as MK            # noqa: E402
import viewpoints as VP       # noqa: E402
from labels import label_dir  # noqa: E402
from pycocotools import mask as RLE  # noqa: E402

SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "mobilesamv2_fast")))
HULL_ROOT = Path(os.environ.get("HULL_ROOT", str(REPO / "data" / "eval" / "srp_hull_mv2_v12_am1")))
GEX = {"skillet_lid", "windex_bottle", "colored_wood_blocks", "dice"}
# match_eval 判定「沒分開」的對(瘦fp drop / 瘦中心 drop),單獨標註
FLAG = {("stack4_scene0007", "sponge"), ("stack4_scene0010", "tuna_fish_can"),
        ("stack3_scene0005", "gelatin_box")}


# ★ 直接複用既有實作,不自寫(先前自寫用錯欄位名 subject/object,實際是 x/y,導致 0 對)
from stack_leak_nosep import on_pairs      # noqa: E402  relations.json 的 type=="on",欄位 x(上)/y(下)


def gt_modal(sc):
    f = label_dir(sc) / "actual" / "annotations.json"
    if not f.is_file():
        return {}
    d = json.loads(f.read_text())
    id2n = {c["id"]: c["name"].split("_", 1)[-1] for c in d["categories"] if c["name"] != "ur5e"}
    vof = {Path(im["file_name"]).stem: im["id"] for im in d["images"]}
    seg = {(a["image_id"], a["category_id"]): a["segmentation"] for a in d["annotations"]}
    per = {}
    for vn, iid in vof.items():
        for cid, nm in id2n.items():
            if (iid, cid) in seg:
                m = RLE.decode(seg[(iid, cid)]).astype(bool)
                if m.any():
                    per.setdefault(vn, {})[nm] = m
    return per


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()
    scenes = sorted(Path(p).parent.name for p in glob.glob(str(HULL_ROOT / "stack*_scene*/hull.npz")))
    views = sorted(VP.selected_view_names(12))
    rows = []; TOT = Counter()
    for sc in scenes:
        pairs = [(u, l) for (u, l) in on_pairs(sc) if u not in GEX and l not in GEX]
        if not pairs:
            continue
        per = gt_modal(sc)
        if not per:
            continue
        smask = {}
        for vn in views:
            vd = SAM_ROOT / sc / vn
            if (vd / "masks").is_dir():
                smask[vn] = [m for m, _ in MK.kept_object_masks(vd)]
        for (T, B) in pairs:
            c = Counter()
            for vn in views:
                g = per.get(vn) or {}; ms = smask.get(vn)
                if ms is None or T not in g or B not in g:
                    c["無GT"] += 1; continue

                def best(gt):
                    ious = [float((m & gt).sum()) / max(float((m | gt).sum()), 1.0) for m in ms]
                    if not ious:
                        return -1, 0.0
                    i = int(np.argmax(ious)); return i, ious[i]
                iT, vT = best(g[T]); iB, vB = best(g[B])
                if vT < a.thr or vB < a.thr:
                    c["漏"] += 1
                elif iT == iB:
                    c["併"] += 1
                else:
                    c["分開"] += 1
            rows.append((sc, T, B, c["分開"], c["併"], c["漏"], c["無GT"],
                         (sc, T) in FLAG or (sc, B) in FLAG))
            TOT.update(c)
    n = sum(TOT[k] for k in ("分開", "併", "漏"))
    md = [f"# MobileSAMv2 遮罩在 2D 有沒有把堆疊上下物切開(IoU≥{a.thr})\n",
          "- 建檔 2026-09-25;程式 `srp/stage4_probe/sam2d_sep_mv2.py`;可復現。",
          f"- SAM=`{SAM_ROOT.name}`(經 `kept_object_masks` 過濾,與管線一致);GT=labels/<場>/actual(modal);12 視角 A-3。",
          f"- 母體:{len(rows)} 個 on 對 × 12 視角,有效判定 **{n}** 個(視角,對);**排除 GEX**。",
          "- 分開=T/B 最佳匹配是不同遮罩且兩者 IoU≥thr;併=同一個遮罩;漏=至少一方無 IoU≥thr 的遮罩。",
          "- ⚠ 既有 `diag_stack_sep.py` 用舊 `sam_only` 遮罩,結果 84%/1%/15%(REPORT_relation_separation.md §6.1),基準不同。\n",
          "## 整體\n", "| 判定 | 次數 | 佔比 |", "|---|---|---|"]
    for k in ("分開", "併", "漏"):
        md.append(f"| {k} | {TOT[k]} | {TOT[k]/max(n,1)*100:.1f}% |")
    md += ["", f"(另有 {TOT['無GT']} 個(視角,對)因該視角缺 GT modal 遮罩而未判定)\n",
           "## match_eval 判定「沒分開」的對(重點)\n",
           "| 場景 | 上物 T | 下物 B | 分開 | 併 | 漏 | 無GT |", "|---|---|---|---|---|---|---|"]
    for (sc, T, B, s, m_, l, ng, fl) in rows:
        if fl:
            md.append(f"| {sc} | {T} | {B} | **{s}** | {m_} | {l} | {ng} |")
    md += ["", "## 逐對明細(依「分開」由少到多,前 15)\n",
           "| 場景 | 上物 T | 下物 B | 分開 | 併 | 漏 | 無GT |", "|---|---|---|---|---|---|---|"]
    for (sc, T, B, s, m_, l, ng, fl) in sorted(rows, key=lambda r: r[3])[:15]:
        md.append(f"| {sc} | {T} | {B} | {s} | {m_} | {l} | {ng} |")
    out = HERE / "RESULT_sam2d_sep_mv2.md"
    out.write_text("\n".join(md), encoding="utf-8")
    with open(HERE / "sam2d_sep_mv2_perpair.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene", "upper", "lower", "sep", "merged", "miss", "no_gt", "flagged"])
        for r in rows:
            w.writerow(list(r))
    print(f"[存檔] {out}\n\n" + "\n".join(md))


if __name__ == "__main__":
    main()
