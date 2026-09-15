#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""eval_mask_grouping — 物體級「遮罩分群還原」評估(完全相等 + 每物體 recall/precision 平均)。

問題:MobileSAMv2 遮罩分完群(+合併)後,各 GT 物體有沒有被還原成「它自己那組遮罩」。
單位=物體;不是群對、不是物理堆疊。無面積,一切以「遮罩個數」計 TP/FP/FN。

單一物體怎麼算一個數:
  由 GT modal 把每個 kept 遮罩指派給所屬物體 → M_gt(o)=物體 o 的全部 kept 遮罩(跨視角、含 MobileSAMv2 過切)。
  C*(o)=裝最多 o 遮罩的預測 instance。
    TP=C* 裡屬 o 的遮罩數;FN=o 但不在 C*(過切散掉);FP=C* 裡屬「其他非排除物體」的遮罩數(誤併)。
    Recall=TP/(TP+FN)、Precision=TP/(TP+FP);完全相等=(FN==0 且 FP==0)。
  指標一 完全相等成功率 = mean_o 1[完全相等];指標二 = mean_o Recall、mean_o Precision(皆 macro,每物體等權)。

排除(逐項,必列,對應輸出的「排除項回報」):
  ① ur5e/robot:GT modal 已排(手臂夾爪遮罩因 <0.5 覆蓋而未指派)。
  ② GLOBAL_EXCLUDE 4 類:不列入物體;其遮罩也不計入任何物體的 FP。
  ③ kept 遮罩「過半被單一物體覆蓋」<COV_THR(0.5) → 未指派,不進 M_gt。
  ④ 物體 12 視角內 0 可見遮罩(全遮擋/GT 門檻漏)→ recall 無定義,排除該物體。
  ⑤ C* 打平(正的最大交集並列 ≥2 instance)→ 記場景名、該物體不計。
  ⑥ 物體完全沒被任何 instance 收到(最大交集=0)→ 不算打平,recall=precision=0(計為失敗)。

輸入:
  GT modal:labels.label_dir/actual/annotations.json(RLE segmentation + categories,排 robot)。
  kept 遮罩:masks.kept_object_masks(data/eval/mobilesamv2_fast/<scene>/<view>) 的 (mask,filename)。
  預測分群:data/eval/<root>/<scene>/instances.json 的 instances[].masks(每 instance=一組 (view,name))。
  視角:viewpoints.selected_view_names(12)(A-3)。

用法: ./eval_mask_grouping.py <root1> [root2 ...]   (不給場景=全 303 多物場;n1 排除)
輸出:每 root 一行總表 + per-object CSV(eval_mask_grouping_<root>.csv);唯讀,不動任何現有檔。
"""
import sys
import json
import glob
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
import masks as MK          # kept_object_masks
import viewpoints as VP     # selected_view_names
from labels import LABELS, label_dir

EVAL = REPO / "data" / "eval"
MV2 = EVAL / "mobilesamv2_fast"
GEX = {"skillet_lid", "windex_bottle", "colored_wood_blocks", "dice"}
COV_THR = 0.5
GROUPS = ["n3", "n4", "n5", "occ3", "occ4", "occ5", "stack3", "stack4", "stack5"]  # 排除 n1 單物


def short(name):
    """026_sponge -> sponge;070-b_colored_wood_blocks -> colored_wood_blocks。"""
    return name.split("_", 1)[-1]


def modal_by_view(scene):
    """回 {view_stem: [(obj_short_name, bool_mask), ...]},排 robot。"""
    ann = label_dir(scene) / "actual" / "annotations.json"
    if not ann.exists():
        return {}
    d = json.load(open(ann))
    cat = {c["id"]: c for c in d["categories"]}
    vname = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
    out = defaultdict(list)
    for a in d["annotations"]:
        c = cat[a["category_id"]]
        if c.get("supercategory") == "robot":
            continue
        out[vname[a["image_id"]]].append((short(c["name"]), mask_utils.decode(a["segmentation"]).astype(bool)))
    return out


def assign_scene(scene, views):
    """回 mask_obj: {(view,name): obj_short},只含「過半被單一物體覆蓋 ≥COV_THR」者。
    另回 (n_kept, n_unassigned) 供排除③統計。"""
    modal = modal_by_view(scene)
    mask_obj = {}
    n_kept = 0
    n_unassigned = 0
    for vn in views:
        vdir = MV2 / scene / vn
        if not (vdir / "masks").is_dir():
            continue
        objs = modal.get(vn, [])
        km = MK.kept_object_masks(vdir)
        # 每視角建一張 GT label map(modal 遮罩對可見像素互斥;重疊時後者覆蓋,可忽略)
        glabel = None
        names = [None]  # idx 0 = 背景
        if objs:
            H, W = objs[0][1].shape
            glabel = np.zeros((H, W), np.int16)
            for oname, gt in objs:
                names.append(oname)
                glabel[gt] = len(names) - 1
        for b, fname in km:
            n_kept += 1
            area = int(b.sum())
            if area == 0 or glabel is None:
                n_unassigned += 1
                continue
            cnt = np.bincount(glabel[b], minlength=len(names))
            cnt[0] = 0  # 背景不算
            best_idx = int(cnt.argmax())
            if best_idx > 0 and cnt[best_idx] / area >= COV_THR:
                mask_obj[(vn, fname)] = names[best_idx]
            else:
                n_unassigned += 1
    return mask_obj, n_kept, n_unassigned


def pred_instances(root, scene, views):
    """回 [set((view,name))],每 set = 一個預測 instance,只保留 selected views。"""
    p = EVAL / root / scene / "instances.json"
    if not p.is_file():
        return None
    d = json.load(open(p))
    vset = set(views)
    out = []
    for it in d["instances"]:
        s = set()
        for vn, names in it["masks"].items():
            if vn in vset:
                for nm in names:
                    s.add((vn, nm))
        out.append(s)
    return out


def eval_root(root, scenes, views, assign_cache):
    """assign_cache: {scene: (mask_obj, n_kept, n_unassigned)}(一次性算好,4 root 共用)。"""
    rows = []          # per-object: (scene, obj, tp, fn, fp, recall, prec, exact)
    n_tie = 0
    tie_scenes = []
    n_obj_zeromask = 0     # 排除④
    n_scene_noann = 0
    n_scene_nopred = 0
    tot_kept = tot_unassigned = 0
    for scene in scenes:
        mask_obj, nk, nu = assign_cache[scene]
        tot_kept += nk
        tot_unassigned += nu
        if nk == 0:
            n_scene_noann += 1
            continue
        insts = pred_instances(root, scene, views)
        if insts is None:
            n_scene_nopred += 1
            continue
        # M_gt(o)
        Mgt = defaultdict(set)
        for k, o in mask_obj.items():
            Mgt[o].add(k)
        objs = [o for o in Mgt if o not in GEX]
        # 每物體
        for o in objs:
            mset = Mgt[o]
            if len(mset) == 0:          # 不會發生(defaultdict 只在有遮罩時建鍵)
                n_obj_zeromask += 1
                continue
            inters = [len(mset & C) for C in insts]
            mx = max(inters) if inters else 0
            if mx == 0:                  # 排除⑥:完全沒被收到 → 失敗(0/0)
                rows.append((scene, o, 0, len(mset), 0, 0.0, 0.0, False))
                continue
            winners = [i for i, v in enumerate(inters) if v == mx]
            if len(winners) > 1:         # 排除⑤:打平
                n_tie += 1
                tie_scenes.append(f"{scene}:{o}")
                continue
            C = insts[winners[0]]
            tp = mx
            fn = len(mset) - tp
            # FP：C* 裡屬「其他非排除物體」的遮罩
            fp = 0
            for k in C:
                oo = mask_obj.get(k)
                if oo is not None and oo != o and oo not in GEX:
                    fp += 1
            recall = tp / (tp + fn)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            exact = (fn == 0 and fp == 0)
            rows.append((scene, o, tp, fn, fp, recall, prec, exact))
    return rows, dict(n_tie=n_tie, tie_scenes=tie_scenes, n_scene_noann=n_scene_noann,
                      n_scene_nopred=n_scene_nopred, tot_kept=tot_kept, tot_unassigned=tot_unassigned,
                      n_obj_zeromask=n_obj_zeromask)


def main():
    import csv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    roots = [a for a in args if "/" not in a and not a.startswith("n") and not a.startswith("occ") and not a.startswith("stack")]
    scene_args = [a for a in args if a not in roots]
    if not roots:
        print("用法: ./eval_mask_grouping.py <root1> [root2 ...] [scenes...]"); return
    if scene_args:
        scenes = scene_args
    else:
        scenes = []
        for g in GROUPS:
            scenes += sorted(Path(p).name for p in glob.glob(str(MV2 / f"{g}_scene*")))
    views = sorted(VP.selected_view_names(12))
    print(f"場景數={len(scenes)}(排除 n1),視角數={len(views)},roots={roots}\n", flush=True)

    # 一次性算每場「遮罩→GT 物體」指派,4 root 共用(GT 解碼+載遮罩=最貴,只做一次)
    print("[指派] 算每場遮罩→GT 物體 ...", flush=True)
    assign_cache = {}
    for i, scene in enumerate(scenes):
        assign_cache[scene] = assign_scene(scene, views)
        if (i + 1) % 30 == 0:
            print(f"    {i+1}/{len(scenes)}", flush=True)
    print("[指派] 完成\n", flush=True)

    summ = []
    for root in roots:
        rows, ex = eval_root(root, scenes, views, assign_cache)
        with open(Path(__file__).parent / f"eval_mask_grouping_{root}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene", "obj", "tp", "fn", "fp", "recall", "precision", "exact"])
            w.writerows(rows)
        n = len(rows)
        exact = sum(1 for r in rows if r[7]) / n if n else float("nan")
        mrec = float(np.mean([r[5] for r in rows])) if n else float("nan")
        mpre = float(np.mean([r[6] for r in rows])) if n else float("nan")
        summ.append((root, n, exact, mrec, mpre, ex))
        print(f"[{root}] 物體數={n} 完全相等={exact:.3f} 平均Recall={mrec:.3f} 平均Precision={mpre:.3f}")
        print(f"    排除: 打平{ex['n_tie']} / 場無標註{ex['n_scene_noann']} / 場無預測{ex['n_scene_nopred']} / "
              f"未指派遮罩{ex['tot_unassigned']}/{ex['tot_kept']}({100*ex['tot_unassigned']/max(1,ex['tot_kept']):.0f}%)")

    print(f"\n{'root':<44}{'物體':>6}{'完全相等':>9}{'均Recall':>10}{'均Prec':>9}")
    for root, n, exact, mrec, mpre, ex in summ:
        print(f"{root:<44}{n:>6}{exact:>9.3f}{mrec:>10.3f}{mpre:>9.3f}")


if __name__ == "__main__":
    main()
