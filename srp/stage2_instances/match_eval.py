#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""match_eval.py — 依「每群 voxel ↔ GT 物體」的對應,量 找到/沒找到/過切/沒分開。

★ 新檔,不動任何既有腳本。讀 build_vis_index.py 產的 vis_index_12v.npz(hull 層級、與分群方法無關),
  評估只是純加總,不需要重算 z-buffer 或影像運算。

指標定義(使用者 2026-09-25 定):
  1. 每個 instance k 投影回各視角得【z-buffer 可見部分】遮罩 pm[k,v];與各 GT modal 遮罩算 2D IoU。
  2. obj(k) = argmax_o  SUM_v IoU(pm[k,v], gt[o,v])          ← 各視角【IoU 加總】最大者
  3. 依 obj 分組:S(o) = {k | obj(k)==o}
       |S(o)| == 0  → 物體 o【沒找到】
       |S(o)| >= 1  → 物體 o【找到】;代表群 = voxel 數最多者;其餘 |S(o)|-1 群 =【過切】
  4. 堆疊 on 對 (上,下) 兩者皆非 GEX:任一方【沒找到】→ 該對【沒分開】
       (沒找到 = 它被併進對方的群裡,沒有自己的群)
  ⚠ 不設 IoU 下限(使用者明確指示):只要是最大就算對應。跑完看代表群 IoU 分佈再決定是否需要。
  ⚠ 非一對一配對:多群可對應到同一 GT——這正是偵測過切的機制,不可改成 Hungarian。

精確性:IoU 由索引三量精確還原(z-buffer 下每像素只屬一個最前 voxel,逐 voxel 加總 == 逐像素計數)。
  已驗證:stack3_scene0019 的 48 組 IoU,索引加總 vs 直接算像素最大誤差 0.000e+00。

排除項:GEX = skillet_lid/windex_bottle/colored_wood_blocks/dice(方法不處理的物體)。
  GT 端排除;且 obj(k) 落在 GEX 的 instance 整個不計(不算成別人的過切)。
一致性:核對 vis_index 的 hull_root 與 n_surf 與實際 hull 相符,不符 raise(不靜默用錯索引)。

用法: ./match_eval.py --inst-root <root> [--hull-root srp_hull_mv2_v12_am1]
輸出: RESULT_match_eval_<root>.md + match_eval_<root>_perscene.csv
"""
import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "srp" / "io")); sys.path.insert(0, str(HERE))
from stack_leak_nosep import on_pairs      # noqa: E402  stage3 GT 的 on(支撐)關係

EVAL = REPO / "data" / "eval"
GEX = {"skillet_lid", "windex_bottle", "colored_wood_blocks", "dice"}


def scene_stats(inst_root, hull_root, sc):
    hp = EVAL / hull_root / sc / "hull.npz"
    ip = EVAL / inst_root / sc / "instances.npz"
    xp = EVAL / hull_root / sc / "vis_index_12v.npz"
    if not (hp.is_file() and ip.is_file() and xp.is_file()):
        return None
    Z = np.load(xp, allow_pickle=False)
    meta = json.loads(str(Z["build_meta"]))
    hz = np.load(hp); surf = hz["surface"]
    vox = np.array(np.nonzero(surf)).T; M = len(vox)
    if meta["hull_root"] != hull_root or int(Z["n_surf"]) != M:      # fail loud,不靜默用錯索引
        raise ValueError(f"{sc}: vis_index 與 hull 不符 "
                         f"(index hull={meta['hull_root']} n_surf={int(Z['n_surf'])} vs {hull_root} {M})")
    labels = np.load(ip)["labels"]
    if labels.shape != surf.shape:
        raise ValueError(f"{sc}: labels shape {labels.shape} != surface {surf.shape}")
    labv = labels[tuple(vox.T)]                                       # 每表面 voxel 的群 id
    ks = [int(k) for k in np.unique(labv) if k > 0]
    objs = [str(o) for o in Z["objs"]]
    nV, nO = len(Z["views"]), len(objs)
    if not ks or nO == 0:
        return None
    kpos = {k: i for i, k in enumerate(ks)}
    grp = np.full(M, -1, np.int32)                                    # voxel -> 群序號
    for k in ks:
        grp[labv == k] = kpos[k]
    # A[k,v] = |pm|;I[k,v,o] = |pm & gt|   —— 純加總
    A = np.zeros((len(ks), nV))
    tsel = grp[Z["tot_p"]] >= 0                                       # 只取屬於某群的 voxel(label 0 排除)
    np.add.at(A, (grp[Z["tot_p"]][tsel], Z["tot_v"][tsel]), Z["tot_c"][tsel])
    I = np.zeros((len(ks), nV, nO))
    gsel = grp[Z["gtc_p"]] >= 0
    np.add.at(I, (grp[Z["gtc_p"]][gsel], Z["gtc_v"][gsel], Z["gtc_o"][gsel]), Z["gtc_c"][gsel])
    G = Z["gtarea"].astype(np.float64)[None, :, :]                    # (1,V,O)
    U = A[:, :, None] + G - I
    IOU = np.where(U > 0, I / np.maximum(U, 1e-9), 0.0)               # (K,V,O)
    obj_of = IOU.sum(axis=1).argmax(axis=1)                           # ★ 各視角 IoU 加總最大
    nvox = np.array([int((labv == k).sum()) for k in ks])
    # 分組
    S = defaultdict(list)
    for ki in range(len(ks)):
        S[objs[obj_of[ki]]].append(ki)
    nong = [o for o in objs if o not in GEX]
    found, miss, extra, rep_iou = [], [], 0, []
    for o in nong:
        lst = S.get(o, [])
        if not lst:
            miss.append(o); continue
        found.append(o); extra += len(lst) - 1
        rep = lst[int(np.argmax(nvox[lst]))]                          # 代表群 = voxel 最多
        v_has = Z["gtarea"][:, objs.index(o)] > 0
        rep_iou.append(float(IOU[rep, v_has, objs.index(o)].mean()) if v_has.any() else 0.0)
    # 堆疊沒分開
    pairs = [(u, l) for (u, l) in on_pairs(sc) if u not in GEX and l not in GEX
             and u in objs and l in objs]
    nosep = sum(1 for (u, l) in pairs if (u in miss or l in miss))
    return dict(n_obj=len(nong), n_found=len(found), n_miss=len(miss), n_extra=extra,
                n_inst=len(ks), rep_iou=float(np.mean(rep_iou)) if rep_iou else 0.0,
                n_pair=len(pairs), n_nosep=nosep,
                rep_iou_list=rep_iou, miss_list=miss)


def grp_of(sc):
    return "stack" if sc.startswith("stack") else ("occ" if sc.startswith("occ") else "n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst-root", required=True)
    ap.add_argument("--hull-root", default="srp_hull_mv2_v12_am1")
    a = ap.parse_args()
    scenes = sorted(Path(p).parent.name for p in
                    glob.glob(str(EVAL / a.inst_root / "*_scene*" / "instances.npz")))
    scenes = [s for s in scenes if not s.startswith("n1_")]
    rows = []; allrep = []
    G = {g: defaultdict(list) for g in ("n", "occ", "stack", "all")}
    for sc in scenes:
        r = scene_stats(a.inst_root, a.hull_root, sc)
        if r is None:
            continue
        rows.append((sc, r)); allrep += r["rep_iou_list"]
        for g in (grp_of(sc), "all"):
            G[g]["obj"].append(r["n_obj"]); G[g]["found"].append(r["n_found"])
            G[g]["miss"].append(r["n_miss"]); G[g]["extra"].append(r["n_extra"])
            G[g]["inst"].append(r["n_inst"]); G[g]["rep"].append(r["rep_iou"])
            G[g]["pair"].append(r["n_pair"]); G[g]["nosep"].append(r["n_nosep"])
    md = [f"# 對應式評估(找到/沒找到/過切/沒分開):{a.inst_root}\n",
          "- 建檔 2026-09-25;程式 `match_eval.py`;索引 `build_vis_index.py`;可復現。",
          f"- hull=`{a.hull_root}`;GT=labels/<場>/actual(**modal**,z-buffer 可見部分);12 視角 A-3 selected。",
          "- 對應:每群取【各視角 IoU 加總】最大的 GT 物體;多群可對應同一物體(非一對一)。",
          "- 同一物體被多群對應 → voxel 最多者為代表,其餘計為**過切**;無群對應 → **沒找到**。",
          "- **沒分開** = 堆疊 on 對中任一方沒找到(它被併進對方群裡);僅 stack 組有 on 對。",
          "- **不設 IoU 下限**(依指示);代表群 IoU 分佈見文末。",
          "- **排除 GEX**(skillet_lid/windex_bottle/colored_wood_blocks/dice):GT 端排除,"
          "且對應到 GEX 的 instance 整個不計。\n",
          "## 分組彙總\n",
          "| 組 | 場數 | GT物體 | 找到 | 沒找到 | 找到率 | 過切群數 | 過切/物體 | inst數 | 代表群mIoU | on對 | 沒分開 | 沒分開率 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in ("n", "occ", "stack", "all"):
        d = G[g]
        if not d["obj"]:
            continue
        o, f, m, e = sum(d["obj"]), sum(d["found"]), sum(d["miss"]), sum(d["extra"])
        pr, ns = sum(d["pair"]), sum(d["nosep"])
        md.append(f"| {g} | {len(d['obj'])} | {o} | {f} | {m} | {f/max(o,1)*100:.1f}% | {e} | "
                  f"{e/max(o,1):.3f} | {np.mean(d['inst']):.2f} | {np.mean(d['rep']):.3f} | "
                  f"{pr} | {ns} | {ns/pr*100:.1f}% |" if pr else
                  f"| {g} | {len(d['obj'])} | {o} | {f} | {m} | {f/max(o,1)*100:.1f}% | {e} | "
                  f"{e/max(o,1):.3f} | {np.mean(d['inst']):.2f} | {np.mean(d['rep']):.3f} | 0 | 0 | – |")
    if allrep:
        q = np.percentile(allrep, [0, 5, 10, 25, 50])
        md += ["", "## 代表群 IoU 分佈(判斷是否需要 IoU 下限)\n",
               "| min | p5 | p10 | p25 | 中位 | <0.1 的比例 | <0.05 的比例 |", "|---|---|---|---|---|---|---|",
               f"| {q[0]:.3f} | {q[1]:.3f} | {q[2]:.3f} | {q[3]:.3f} | {q[4]:.3f} | "
               f"{np.mean(np.array(allrep)<0.1)*100:.2f}% | {np.mean(np.array(allrep)<0.05)*100:.2f}% |"]
    out = HERE / f"RESULT_match_eval_{a.inst_root}.md"
    out.write_text("\n".join(md), encoding="utf-8")
    with open(HERE / f"match_eval_{a.inst_root}_perscene.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["scene", "n_obj", "n_found", "n_miss", "n_extra",
                                        "n_inst", "rep_iou", "n_pair", "n_nosep", "miss"])
        for sc, r in rows:
            w.writerow([sc, r["n_obj"], r["n_found"], r["n_miss"], r["n_extra"], r["n_inst"],
                        f"{r['rep_iou']:.4f}", r["n_pair"], r["n_nosep"], "|".join(r["miss_list"])])
    print(f"[存檔] {out}  場數={len(rows)}\n\n" + "\n".join(md))


if __name__ == "__main__":
    main()
