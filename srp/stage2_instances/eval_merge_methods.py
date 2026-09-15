#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""eval_merge_methods — 合併方案的物體級公正比較(同一份遮罩宇宙,只差分組)。

問題:CLIP 語意分群過切;各合併方案能否把過切收回又不砸 precision。
關鍵:遮罩宇宙**固定=baseline 的 instances[].masks**(已驗證對齊 GT 物體);
合併方案只提供「baseline 群 → 合併元件」的分組,套回 baseline 遮罩(成員群遮罩聯集,無損)。
避免各 root 自己寫的壞遮罩檔(merged 用碎片取代物體、veto 無 json)污染比較。

分組來源:
  baseline:不併(每個 baseline 群自成一群)。
  from-root(crossing/veto):讀該 root 的 instances.npz voxel labels,對 baseline labels 取多數
    → baseline 群 g 落在哪個合併元件;同元件的群併。(不信任該 root 的 json 遮罩)

指標:完全相等 / 每物體 recall / precision(macro),定義與排除同 eval_mask_grouping。
用法: ./eval_merge_methods.py [scenes...]   (不給=全 303 多物場)
輸出:總表 + eval_merge_methods.csv(per-object);唯讀。
"""
import sys
import json
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
sys.path.insert(0, str(REPO / "srp" / "io"))
import eval_mask_grouping as G   # assign_scene, GEX, GROUPS, MV2, EVAL, views
import viewpoints as VP

import os
BASE = os.environ.get("BASE_ROOT", "srp_hull_semcluster_surf_am1photo")
# 合併方案 → 其 instances.npz(voxel labels)所在 root;None=不併(baseline)
METHODS = {
    "baseline(不併)": None,
    "crossing":       "srp_hull_semcluster_surf_am1photo_merged",
    "veto":           "srp_hull_veto_merge",
}


def baseline_groups(scene, views):
    """回 {gid: set((view,name))},來自 baseline instances[].masks(固定遮罩宇宙)。"""
    p = G.EVAL / BASE / scene / "instances.json"
    d = json.loads(p.read_text())
    vset = set(views)
    out = {}
    for it in d["instances"]:
        s = set()
        for vn, names in it["masks"].items():
            if vn in vset:
                for nm in names:
                    s.add((vn, nm))
        out[it["instance"]] = s
    return out


def grouping_from_labels(scene, merge_root, gids):
    """回 {gid: comp_id}:baseline 群 g 的 voxel 在 merge_root labels 的多數元件。
    None → 每群自成一群(baseline / 無法還原時 fallback)。"""
    if merge_root is None:
        return {g: g for g in gids}
    bp = G.EVAL / BASE / scene / "instances.npz"
    mp = G.EVAL / merge_root / scene / "instances.npz"
    if not (bp.is_file() and mp.is_file()):
        return None
    bl = np.load(bp)["labels"]
    ml = np.load(mp)["labels"]
    if bl.shape != ml.shape:
        return None
    g2comp = {}
    for g in gids:
        vox = (bl == g)
        if not vox.any():
            g2comp[g] = ("solo", g)   # 無 voxel → 自成一群
            continue
        vals, cnts = np.unique(ml[vox], return_counts=True)
        g2comp[g] = int(vals[cnts.argmax()])
    return g2comp


def eval_partition(mask_obj, clusters):
    """clusters: list[set((view,name))]。回 per-object rows + tie 數。"""
    Mgt = defaultdict(set)
    for k, o in mask_obj.items():
        Mgt[o].add(k)
    objs = [o for o in Mgt if o not in G.GEX]
    rows = []
    n_tie = 0
    for o in objs:
        mset = Mgt[o]
        inters = [len(mset & C) for C in clusters]
        mx = max(inters) if inters else 0
        if mx == 0:
            rows.append((o, 0, len(mset), 0, 0.0, 0.0, False))
            continue
        winners = [i for i, v in enumerate(inters) if v == mx]
        if len(winners) > 1:   # 打平=遮罩被平均切開,不跳過,取第一個當 C*、照常算(必為失敗)
            n_tie += 1
        C = clusters[winners[0]]
        tp = mx
        fn = len(mset) - tp
        fp = sum(1 for k in C if (mask_obj.get(k) not in (None, o) and mask_obj.get(k) not in G.GEX))
        recall = tp / (tp + fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rows.append((o, tp, fn, fp, recall, prec, (fn == 0 and fp == 0)))
    return rows, n_tie


def build_clusters(bgroups, g2comp):
    """依 g2comp 把 baseline 群併成 clusters(成員群遮罩聯集,無損)。"""
    comp = defaultdict(set)
    for g, c in g2comp.items():
        comp[c] |= bgroups.get(g, set())
    return [s for s in comp.values() if s]


def main():
    import csv
    scenes = sys.argv[1:]
    if not scenes:
        scenes = []
        for grp in G.GROUPS:
            scenes += sorted(Path(p).name for p in glob.glob(str(G.MV2 / f"{grp}_scene*")))
    views = sorted(VP.selected_view_names(12))
    print(f"場景數={len(scenes)},視角數={len(views)},方案={list(METHODS)}\n", flush=True)

    # 一次性遮罩→GT 指派(所有方案共用)
    print("[指派] ...", flush=True)
    assign = {}
    for i, sc in enumerate(scenes):
        assign[sc] = G.assign_scene(sc, views)
        if (i + 1) % 60 == 0:
            print(f"    {i+1}/{len(scenes)}", flush=True)
    print("[指派] 完成\n", flush=True)

    agg = {m: {"rows": [], "tie": 0, "nofile": 0} for m in METHODS}
    for sc in scenes:
        mask_obj, nk, nu = assign[sc]
        if nk == 0:
            continue
        try:
            bg = baseline_groups(sc, views)
        except Exception:
            continue
        gids = list(bg)
        for m, root in METHODS.items():
            g2comp = grouping_from_labels(sc, root, gids)
            if g2comp is None:
                agg[m]["nofile"] += 1
                continue
            clusters = build_clusters(bg, g2comp)
            rows, tie = eval_partition(mask_obj, clusters)
            for r in rows:
                agg[m]["rows"].append((sc,) + r)
            agg[m]["tie"] += tie

    with open(Path(__file__).parent / "eval_merge_methods.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "scene", "obj", "tp", "fn", "fp", "recall", "precision", "exact"])
        for m in METHODS:
            for r in agg[m]["rows"]:
                w.writerow((m,) + r)

    print(f"{'方案':<16}{'物體':>6}{'完全相等':>9}{'均Recall':>10}{'均Prec':>9}{'打平':>6}{'缺檔場':>7}")
    for m in METHODS:
        rows = agg[m]["rows"]
        n = len(rows)
        if n == 0:
            print(f"{m:<16}{0:>6}"); continue
        exact = sum(1 for r in rows if r[7]) / n
        mrec = float(np.mean([r[5] for r in rows]))
        mpre = float(np.mean([r[6] for r in rows]))
        print(f"{m:<16}{n:>6}{exact:>9.3f}{mrec:>10.3f}{mpre:>9.3f}{agg[m]['tie']:>6}{agg[m]['nofile']:>7}")


if __name__ == "__main__":
    main()
