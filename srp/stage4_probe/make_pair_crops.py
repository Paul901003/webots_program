#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""make_pair_crops.py — 照 Open3DSG 的做法,為【每個物體對】裁一張圖(box_i ∪ box_j)。

★ 新檔。依 Open3DSG(CVPR2024)Sec.3.2 + 補充材料 Sec.A:
   - 邊的視覺特徵取自「兩物體 bbox 聯集」的裁切:box_ij^k = box_ik ∪ box_jk
   - 視角需【兩物體同時可見】(論文用 vis(i,k)>t_vis ∨ A(box)>t_box,兩物都要滿足)
   - 關係 prompt:"Describe the relationship between [object1] and [object2]?"
   論文的視角選擇用深度判遮擋(w − d_k > t_occ);本專案不可用深度,改用 GT modal 遮罩面積
   (= 實際可見輪廓)判可見性 → ★上界模擬,清楚標註。

與先前失敗測試的差異(見 RESULT_llava_stack_batch.md,平衡準確率 52.8%):
   先前給【整張場景圖】問「列出所有物體、誰疊在誰上」→ 開放式全場清點,非論文做法。
   本版改為【成對裁切 + 成對提問 + 物體名當 context】,忠於 Open3DSG。

輸出: pair_crops/<scene>__<A>__<B>.png 與 index.json
      (含 scene/view/objA/objB/是否 on/誰在上/兩物面積/裁切尺寸)
用法: ./make_pair_crops.py [--max-nonon 60] [--min-area 300] [--margin 30]
"""
import argparse
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "srp" / "io")); sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import viewpoints as VP        # noqa: E402
from labels import label_dir   # noqa: E402
from stack_leak_nosep import on_pairs   # noqa: E402
from pycocotools import mask as RLE     # noqa: E402

CAP = REPO / "data" / "captures_fast"
HULL = REPO / "data" / "eval" / "srp_hull_mv2_v12_am1"
GEX = {"skillet_lid", "windex_bottle", "colored_wood_blocks", "dice"}
OUT = HERE / "pair_crops"


def modal(sc):
    """回 {view: {obj: (mask, area)}},只取 A-3 12 視角。"""
    d = json.loads((label_dir(sc) / "actual" / "annotations.json").read_text())
    id2n = {c["id"]: c["name"].split("_", 1)[-1] for c in d["categories"] if c["name"] != "ur5e"}
    vof = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
    sel = set(VP.selected_view_names(12))
    per = {}
    for a in d["annotations"]:
        nm = id2n.get(a["category_id"]); vn = vof[a["image_id"]]
        if nm is None or vn not in sel:
            continue
        m = RLE.decode(a["segmentation"]).astype(bool)
        ar = int(m.sum())
        if ar > 0:
            per.setdefault(vn, {})[nm] = (m, ar)
    return per


def bbox(m):
    ys, xs = np.nonzero(m)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-nonon", type=int, default=60, dest="max_nonon")
    ap.add_argument("--min-area", type=int, default=300, dest="min_area")
    ap.add_argument("--margin", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = sorted(Path(p).parent.name for p in glob.glob(str(HULL / "stack*_scene*/hull.npz")))
    on_recs, non_recs = [], []
    for sc in scenes:
        per = modal(sc)
        if not per:
            continue
        ons = {(u, l) for (u, l) in on_pairs(sc) if u not in GEX and l not in GEX}
        objs = sorted({o for d_ in per.values() for o in d_ if o not in GEX})
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                A, B = objs[i], objs[j]
                # ★ 視角選擇:兩物同時可見,取 min(面積) 最大者(論文口徑的無深度版)
                best = None
                for vn, dd in per.items():
                    if A in dd and B in dd and dd[A][1] >= a.min_area and dd[B][1] >= a.min_area:
                        s = min(dd[A][1], dd[B][1])
                        if best is None or s > best[0]:
                            best = (s, vn)
                if best is None:
                    continue
                vn = best[1]
                is_on = (A, B) in ons or (B, A) in ons
                upper = A if (A, B) in ons else (B if (B, A) in ons else None)
                rec = dict(scene=sc, view=vn, objA=A, objB=B, is_on=is_on, upper=upper,
                           areaA=per[vn][A][1], areaB=per[vn][B][1])
                (on_recs if is_on else non_recs).append(rec)
    random.shuffle(non_recs)
    recs = on_recs + non_recs[:a.max_nonon]
    made = []
    for r in recs:
        p = CAP / f"multi_{r['scene'].split('_')[0]}" / r["scene"] / f"{r['view']}.png"
        img = cv2.imread(str(p))
        if img is None:
            continue
        H, W = img.shape[:2]
        per = modal(r["scene"])[r["view"]]
        xa0, ya0, xa1, ya1 = bbox(per[r["objA"]][0]); xb0, yb0, xb1, yb1 = bbox(per[r["objB"]][0])
        x0 = max(min(xa0, xb0) - a.margin, 0); y0 = max(min(ya0, yb0) - a.margin, 0)
        x1 = min(max(xa1, xb1) + a.margin, W); y1 = min(max(ya1, yb1) + a.margin, H)
        fn = f"{r['scene']}__{r['objA']}__{r['objB']}.png"
        cv2.imwrite(str(OUT / fn), img[y0:y1, x0:x1])
        made.append({**r, "file": fn, "crop_wh": [x1 - x0, y1 - y0]})
    (OUT / "index.json").write_text(json.dumps(made, ensure_ascii=False, indent=1))
    n_on = sum(1 for r in made if r["is_on"])
    print(f"[輸出] {OUT}  共 {len(made)} 對;on {n_on}、非on {len(made)-n_on}")


if __name__ == "__main__":
    main()
