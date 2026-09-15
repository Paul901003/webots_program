#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""甜甜圈(父)遮罩 vs 其包含的(子)遮罩 — 語意 cosine 分佈,分同物體/不同物體。

怎麼生 / 設置:
- 資料:mobilesamv2_fast 遮罩;GT=labels/<scene>/actual 的 modal 遮罩(只貼標不進計算);A-3 12 視角;captures_fast RGB。
- 分群前先去手臂+夾爪:遮罩 ≥0.5 落在 srp_arm_masks(FK手臂+夾爪剪影)→ 丟(對齊修正後管線 DROP_ARM)。
- 父子對:原始遮罩中 j 有 ≥0.8 面積落在更大的 i 內(= donut_masks 觸發條件)。
- 父特徵=甜甜圈(挖洞後)i 的 CLIP;子特徵=挖洞後 j 的 CLIP。皆現算(MC.clip_feats mean)+去偏(減 F_BG)。cosine=父·子。
- 貼標(決策1a/τ=0.7/決策3a):對「甜甜圈父」與「子」各算 max_g |m∩modal_g|/|m|;<0.7 → 整對排除。父GT==子GT→同物體,否則不同物體。
用法: env CAPTURES_ROOT=captures_fast SAM_ROOT=mobilesamv2_fast; ./analyze.py [scene|group|(空=全部)] [--sanity]
輸出:pairs.csv(每對) + hist.png + 印同/異物體 min/25/50/75/90%+AUC。RESULT.md 另記設置/數據/結果。
"""
import json, sys, argparse, csv
import numpy as np, cv2
from collections import defaultdict
from pathlib import Path
from pycocotools import mask as cocomask
from scipy.stats import rankdata
sys.path.insert(0, "srp/io"); sys.path.insert(0, "srp/stage2_instances")
import masks as MK, viewpoints as VP, labels as L
import mask_clip_cluster as MC
from voxel_sem_cluster_donut import donut_masks
MV2 = Path("data/eval/mobilesamv2_fast"); CAP = Path("data/captures_fast"); ARM = Path("data/eval/srp_arm_masks")
FBG = MC.F_BG.astype(np.float64); TAU = 0.7; ARM_THR = 0.5; NEST = 0.8

def debias(f):
    f = np.asarray(f, np.float64); f = f - (f @ FBG) * FBG
    return f / (np.linalg.norm(f) + 1e-9)

def modal(sc):
    ann = json.loads((L.label_dir(sc) / "actual" / "annotations.json").read_text())
    cat = {c["id"]: c["name"] for c in ann["categories"]}; id2v = {im["id"]: Path(im["file_name"]).stem for im in ann["images"]}
    md = defaultdict(dict)
    for a in ann["annotations"]:
        if a["category_id"] == 1: continue
        s = a["segmentation"]; c = s["counts"].encode() if isinstance(s["counts"], str) else s["counts"]
        md[id2v[a["image_id"]]][cat[a["category_id"]]] = cocomask.decode({"size": s["size"], "counts": c}).astype(bool)
    return md

def gtlabel(m, gmv):
    a = int(m.sum())
    if a == 0: return None
    best = None; bc = TAU
    for on, om in gmv.items():
        cov = (m & om).sum() / a
        if cov >= bc: bc = cov; best = on
    return best

def scene_pairs(sc, sanity=False):
    md = modal(sc); rows = []
    sdir = CAP / f"multi_{sc.split('_')[0]}" / sc
    for vn in sorted(VP.selected_view_names(12)):
        vd = MV2 / sc / vn;
        if not vd.is_dir(): continue
        km = MK.kept_object_masks(vd); ms0 = [m for m, _ in km]; names = [n for _, n in km]
        if not ms0: continue
        ap = ARM / sc / f"{vn}_arm.png"; arm = (cv2.imread(str(ap), 0) > 127) if ap.is_file() else None
        if arm is not None:                        # 分群前去手臂+夾爪
            keep = [k for k, m in enumerate(ms0) if (m & arm).sum() / max(int(m.sum()), 1) < ARM_THR]
            ms0 = [ms0[k] for k in keep]; names = [names[k] for k in keep]
        if len(ms0) < 2: continue
        don = donut_masks(ms0, thr=NEST)           # 甜甜圈(父挖洞)
        areas = [int(m.sum()) for m in ms0]
        gmv = md.get(vn, {})
        rgb = cv2.cvtColor(cv2.imread(str(sdir / f"{vn}.png")), cv2.COLOR_BGR2RGB)
        for i in range(len(ms0)):
            for j in range(len(ms0)):
                if i == j or areas[j] == 0: continue
                if areas[i] > areas[j] and (ms0[i] & ms0[j]).sum() / areas[j] >= NEST:  # j 含於 i
                    gi = gtlabel(don[i], gmv); gj = gtlabel(don[j], gmv)
                    if gi is None or gj is None: continue      # 決策3a:標不清就排除
                    fi, fj = MC.clip_feats(rgb, [don[i], don[j]], "mean")
                    if fi is None or fj is None: continue
                    cos = float(debias(fi) @ debias(fj))
                    same = (gi == gj)
                    rows.append((sc, vn, names[i], names[j], gi, gj, same, cos))
                    if sanity:
                        print(f"  {vn} 父{names[i]}({gi}) ⊃ 子{names[j]}({gj}) | {'同' if same else '異'}物體 cos={cos:.3f}")
    return rows

def auc(s, t):
    s, t = np.array(s), np.array(t)
    if len(s) == 0 or len(t) == 0: return float("nan")
    r = rankdata(np.concatenate([s, t])); U = r[:len(s)].sum() - len(s) * (len(s) + 1) / 2
    return U / (len(s) * len(t))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("targets", nargs="*"); ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()
    if not args.targets: scenes = sorted(p.name for p in MV2.glob("*_scene*") if (L.label_dir(p.name) / "actual" / "annotations.json").is_file())
    else:
        scenes = []
        for a in args.targets:
            scenes += [a] if "scene" in a else sorted(p.name for p in MV2.glob(f"{a}_scene*"))
    allrows = []
    for i, sc in enumerate(scenes):
        try:
            r = scene_pairs(sc, args.sanity); allrows += r
            if not args.sanity and (i + 1) % 20 == 0: print(f"  {i+1}/{len(scenes)} ({len(allrows)} 對)", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}", flush=True)
    HERE = Path("srp/stage2_instances/experiments/donut_parent_child_sem_20260828")
    with open(HERE / "pairs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene", "view", "parent", "child", "parent_gt", "child_gt", "same", "cos"]); w.writerows(allrows)
    same = [r[7] for r in allrows if r[6]]; diff = [r[7] for r in allrows if not r[6]]
    print(f"\n===== 甜甜圈父-子 語意 cosine ({len(scenes)}場, 同物體{len(same)}對 / 不同物體{len(diff)}對) =====")
    for nm, x in [("同物體", same), ("不同物體", diff)]:
        if x: p = np.percentile(x, [5, 25, 50, 75, 95]); print(f"  {nm}: 5/25/50/75/95% = {p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f}/{p[3]:.3f}/{p[4]:.3f}  mean={np.mean(x):.3f}")
    print(f"  AUC(同>異) = {auc(same, diff):.3f}")
    if same and diff:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        b = np.linspace(-0.2, 1, 37); fig, a = plt.subplots(figsize=(9, 5))
        a.hist(same, bins=b, density=True, alpha=0.55, color="#2c7fb8", label=f"same object (n={len(same)})")
        a.hist(diff, bins=b, density=True, alpha=0.55, color="#d95f0e", label=f"different object (n={len(diff)})")
        a.axvline(np.median(same), color="#2c7fb8", ls="--"); a.axvline(np.median(diff), color="#d95f0e", ls="--")
        a.set_xlabel("debiased CLIP cosine (donut parent vs contained child)"); a.set_ylabel("density"); a.legend()
        a.set_title("Donut parent vs contained child — semantic similarity")
        fig.tight_layout(); fig.savefig(HERE / "hist.png", dpi=120); print("saved hist.png")

if __name__ == "__main__": main()
