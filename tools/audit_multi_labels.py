#!/usr/bin/env python3
"""audit_multi_labels.py

稽核多物體標籤（n3/n4/n5），找出需要重拍的場景：
  - 完全缺失：某物體在所有 12 視角都無遮罩（area<=0）→ 列入「需重拍」
  - 影像不足：actual/planned images != 12
  - 位移/出界：actual vs planned > 5cm、或物體離工作中心 > 工作半徑
  - 可見度低：某物體 <4/12 視角有遮罩（僅警告，不一定要重拍）

用法: python tools/audit_multi_labels.py [3 4 5]
輸出: 終端報告 + data/eval/redo_scenes.txt（需重拍場景清單）
"""

import json, math, os, glob, sys
from collections import defaultdict

REPO   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = os.path.join(REPO, "data", "labels")
CAPS   = os.path.join(REPO, "data", "captures")
REDO_OUT = os.path.join(REPO, "data", "eval", "redo_scenes.txt")

CENTER = (0.35, 0.0)
WS_R   = 0.35
DISP_TH = 0.05
LOWVIS_TH = 4

groups = [int(a) for a in sys.argv[1:]] or [3, 4, 5]


def obj_view_counts(ann_path):
    """回傳 {物體名: 有遮罩的視角數}；找不到檔回傳 None。"""
    if not os.path.exists(ann_path):
        return None
    a = json.load(open(ann_path))
    objc = {c["id"]: c["name"] for c in a["categories"] if c["name"] != "ur5e"}
    v = defaultdict(int)
    for an in a["annotations"]:
        if an["category_id"] in objc and an.get("area", 0) > 0:
            v[an["category_id"]] += 1
    return {name: v.get(cid, 0) for cid, name in objc.items()}, len(a.get("images", []))


def main():
    redo = set()
    fully_missing, low_vis, img_bad, disp_rows, out_ws = [], [], [], [], []
    n_scenes = 0

    for g in groups:
        for d in sorted(glob.glob(f"{CAPS}/multi_n{g}/n{g}_scene*")):
            sc = os.path.basename(d)
            n_scenes += 1
            # 完整性 + 缺失（以 actual 為準）
            for mode in ("actual", "planned"):
                res = obj_view_counts(f"{LABELS}/{sc}/{mode}/annotations.json")
                if res is None:
                    img_bad.append(f"{sc}/{mode}: 無 annotations.json"); redo.add(sc); continue
                counts, n_img = res
                if n_img != 12:
                    img_bad.append(f"{sc}/{mode}: images={n_img}"); redo.add(sc)
                if mode == "actual":
                    for name, c in counts.items():
                        if c == 0:
                            fully_missing.append(f"{sc}: {name} (0/12)"); redo.add(sc)
                        elif c < LOWVIS_TH:
                            low_vis.append(f"{sc}: {name} ({c}/12)")
            # 位移
            man = f"{d}/scene_manifest.json"
            if os.path.exists(man):
                m = json.load(open(man))
                pl = {o["name"]: o["spawn_position_m"][:2] for o in m["planned"]["objects"]}
                ac = {o["name"]: o["position_m"][:2] for o in m["actual"]["viewpoints"][0]["objects"]}
                for name, p in pl.items():
                    a = ac.get(name)
                    if a is None:
                        continue
                    if math.dist(p, a) > DISP_TH:
                        disp_rows.append((math.dist(p, a), sc, name))
                    if math.dist(a, CENTER) > WS_R:
                        out_ws.append((math.dist(a, CENTER), sc, name))

    print(f"稽核場景數: {n_scenes}  (groups={groups})")
    print(f"\n[需重拍] 完全缺失(0/12)物體: {len(fully_missing)}")
    for x in fully_missing: print("  -", x)
    print(f"\n[需重拍] 影像/檔案異常: {len(img_bad)}")
    for x in img_bad: print("  -", x)
    print(f"\n[警告] 可見度低(<{LOWVIS_TH}/12): {len(low_vis)}")
    for x in low_vis[:30]: print("  -", x)
    print(f"\n[參考] 位移>{DISP_TH}m: {len(disp_rows)}；出工作空間: {len(out_ws)}")
    for dd, sc, name in sorted(disp_rows, reverse=True)[:15]:
        print(f"  {sc:16s} {name:24s} {dd:.3f} m")

    os.makedirs(os.path.dirname(REDO_OUT), exist_ok=True)
    with open(REDO_OUT, "w") as f:
        f.write("\n".join(sorted(redo)) + ("\n" if redo else ""))
    print(f"\n需重拍場景數: {len(redo)} → 已寫入 {REDO_OUT}")
    if redo:
        print("  " + "  ".join(sorted(redo)))


if __name__ == "__main__":
    main()
