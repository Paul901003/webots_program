#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""threshold_stats.py — 純特徵層級:用 GT 標定的「同物/異物 cos 分佈」比較三特徵、找最佳 sem_thr。
不跑切群、不跑 3D pipeline。門檻直接從同/異物分佈找(見 optimal_thr)。

流程(去偏後特徵):同一場景兩兩遮罩 cos → 依 GT 物體標籤分 intra(同物)/inter(異物):
  - AUC = P(同物 cos > 異物 cos):特徵可分性(1=完全分得開,0.5=分不開)。
  - 最佳 sem_thr = 讓「cos≥cut 判同物」的平衡準確度最高的切點(cut),回傳距離 1-cut 與該準確度。
  - 另報分佈 mean/p10/p90/max(重點看異物 max = 最難分的異物對)。

來源(--source):
  diag(預設,快):讀抽樣 48 場的合併特徵 _diag/siglip_probe/{clip_b32,siglip_b32,siglip_b16}.npy
  full(全 367 場):即時對每場遮罩用 modal GT(IoU>0.5)標物體、讀 per-view 特徵
輸出 threshold_stats_<source>.csv。
用法: ./threshold_stats.py [--source diag|full]   (webots_visual_hull env)
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[3]
DIAG = REPO / "data" / "eval" / "_diag" / "siglip_probe"
SAM_ROOT = REPO / "data" / "eval" / "mobilesamv2_fast"
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
sys.path.insert(0, str(REPO / "srp" / "io"))
# tag → (顯示名, per-view 特徵檔, bg 檔;bg=None 用 CLIP MC.F_BG)
GROUPS = [("clip_b32", "CLIP-B32", "clip_mean_feats.npy", None),
          ("siglip_b32", "SigLIP2-B32", "siglip_b32_feats.npy", "siglip_b32_bg.npy"),
          ("siglip_b16", "SigLIP2-B16", "siglip_b16_feats.npy", "siglip_b16_bg.npy")]
GROUPS_SCAN = ["n1", "n3", "n4", "n5", "occ3", "occ4", "occ5", "stack3", "stack4", "stack5"]
IOU_MIN = 0.5
MIN_AREA = 200


def get_bg(bg_file):
    if bg_file is None:
        import mask_clip_cluster as MC   # CLIP F_BG(載 CLIP model 一次)
        return MC.F_BG.astype(np.float64)
    p = DIAG / bg_file
    if not p.is_file():
        return "MISSING"
    v = np.load(p).astype(np.float64)
    return v / (np.linalg.norm(v) + 1e-9)


def debias(F, bg):
    F = F.astype(np.float64)
    F = F - (F @ bg)[:, None] * bg[None, :]
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


def pair_cos(feats, scenes, names):
    """同一場景內兩兩 cos，依 GT 名分 intra/inter。scenes/names 與 feats 同序，nan 特徵略過。"""
    by_scene = defaultdict(list)
    for i in range(len(feats)):
        if np.isfinite(feats[i, 0]):
            by_scene[scenes[i]].append(i)
    intra, inter = [], []
    for ids in by_scene.values():
        ids = np.array(ids)
        if len(ids) < 2:
            continue
        S = feats[ids] @ feats[ids].T
        nm = np.array([names[i] for i in ids])
        iu = np.triu_indices(len(ids), k=1)
        same = nm[iu[0]] == nm[iu[1]]
        intra.append(S[iu][same]); inter.append(S[iu][~same])
    return np.concatenate(intra), np.concatenate(inter)


def auc(pos, neg):
    n1, n2 = len(pos), len(neg)
    if n1 == 0 or n2 == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def optimal_thr(intra, inter):
    """找 cosine 切點 cut：cos≥cut 判同物。回傳最大平衡準確度的 (sem_thr=1-cut, bacc, cut)。"""
    cuts = np.linspace(0.0, 1.0, 401)
    intra = np.sort(intra); inter = np.sort(inter)
    tpr = 1 - np.searchsorted(intra, cuts) / len(intra)   # P(同物 cos ≥ cut)
    tnr = np.searchsorted(inter, cuts) / len(inter)         # P(異物 cos < cut)
    bacc = 0.5 * (tpr + tnr)
    k = int(np.argmax(bacc))
    return round(1 - cuts[k], 4), round(float(bacc[k]), 4), round(float(cuts[k]), 4)


def load_labels_full():
    """全 367 場:回 (scene_list, name_list, per-view-dir list, mask-filename list)(標到 GT 的遮罩)。"""
    import cv2
    from pycocotools import mask as mask_utils
    import viewpoints as VP
    from labels import LABELS
    sel = set(VP.selected_view_names(12))
    rows = []   # (scene, view, vdir, mask_fn, gt_name)
    scenes = []
    for g in GROUPS_SCAN:
        scenes += sorted(p.name for p in SAM_ROOT.glob(f"{g}_scene*") if p.is_dir())
    for sc in scenes:
        annf = LABELS / sc / "actual" / "annotations.json"
        if not annf.is_file():
            continue
        d = json.loads(annf.read_text())
        cat = {c["id"]: c["name"] for c in d["categories"]}
        vof = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
        gt = {}   # view → {name: mask}
        for a in d["annotations"]:
            nm = cat[a["category_id"]]
            if nm == "ur5e":
                continue
            m = mask_utils.decode(a["segmentation"]).astype(bool)
            gt.setdefault(vof[a["image_id"]], {})[nm] = m
        for vd in sorted((SAM_ROOT / sc).glob("view_*")):
            if vd.name not in sel or vd.name not in gt:
                continue
            gts = gt[vd.name]
            for mp in sorted((vd / "masks").glob("mask_*.png")):
                seg = cv2.imread(str(mp), 0) > 127
                if int(seg.sum()) < MIN_AREA:
                    continue
                best_g, best_i = None, 0.0
                for nm, gm in gts.items():
                    u = int((seg | gm).sum()); i = int((seg & gm).sum()) / u if u else 0
                    if i > best_i:
                        best_i, best_g = i, nm
                if best_g is not None and best_i > IOU_MIN:
                    rows.append((sc, vd.name, vd, mp.name, best_g))
    return rows


def feats_for(rows, feat_file):
    """依 rows 順序取 per-view 特徵(用檔名查),回 (feats, scenes, names)。無效=nan。"""
    import masks as MK
    dim = None
    cache = {}
    vecs = []
    for sc, view, vd, mfn, gt in rows:
        key = str(vd)
        if key not in cache:
            cache[key] = MK.mask_feats(vd, feat_file)
        f = cache[key].get(mfn)
        vecs.append(f)
        if f is not None and dim is None:
            dim = len(f)
    dim = dim or 512
    arr = np.stack([v if v is not None else np.full(dim, np.nan, np.float32) for v in vecs])
    return arr, [r[0] for r in rows], [r[4] for r in rows]


def run(source):
    if source == "diag":
        man = json.loads((DIAG / "manifest.json").read_text())
        items = man["items"]
        scenes = [it["scene"] for it in items]; names = [it["gt_name"] for it in items]
        loader = lambda ff: (np.load(DIAG / {"clip_mean_feats.npy": "clip_b32.npy",
                             "siglip_b32_feats.npy": "siglip_b32.npy",
                             "siglip_b16_feats.npy": "siglip_b16.npy"}[ff]), scenes, names)
        print(f"[diag] 抽樣 {len(set(scenes))} 場 / {len(items)} 遮罩")
    else:
        rows = load_labels_full()
        print(f"[full] {len(set(r[0] for r in rows))} 場 / {len(rows)} 有主遮罩")
        loader = lambda ff: feats_for(rows, ff)

    out_rows = []
    for tag, label, feat_file, bg_file in GROUPS:
        bg = get_bg(bg_file)
        if isinstance(bg, str):
            print(f"[skip] {label}: bg 未產"); continue
        feats, sc, nm = loader(feat_file)
        feats = debias(feats, bg)
        intra, inter = pair_cos(feats, sc, nm)
        a = auc(intra, inter)
        thr, bacc, cut = optimal_thr(intra, inter)
        out_rows.append({"group": label, "n_intra": len(intra), "n_inter": len(inter),
                         "intra_mean": round(float(intra.mean()), 4),
                         "inter_mean": round(float(inter.mean()), 4),
                         "inter_p90": round(float(np.percentile(inter, 90)), 4),
                         "inter_max": round(float(inter.max()), 4),
                         "AUC": round(a, 4), "最佳sem_thr": thr, "cut(cos)": cut,
                         "平衡準確度@最佳": bacc})
    if not out_rows:
        print("無結果"); return
    print(f"\n{'組':<13}{'同物µ':>7}{'異物µ':>7}{'異物max':>8}{'AUC':>7}{'最佳thr':>8}{'準確度':>8}")
    for r in out_rows:
        print(f"{r['group']:<13}{r['intra_mean']:>7}{r['inter_mean']:>7}{r['inter_max']:>8}"
              f"{r['AUC']:>7}{r['最佳sem_thr']:>8}{r['平衡準確度@最佳']:>8}")
    print("\nAUC 越高=特徵越能分同/異物;最佳thr=cosine 距離切點(給 pipeline 分群用,若要跑下游)。")
    op = DIAG / f"threshold_stats_{source}.csv"
    with open(op, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
    print(f"→ {op}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["diag", "full"], default="diag")
    args = ap.parse_args()
    run(args.source)


if __name__ == "__main__":
    main()
