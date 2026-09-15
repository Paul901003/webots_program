#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""sweep.py — 三組特徵 × 去偏開關 × sem_thr 掃描,測試組跑 pipeline 比 recall(基準一致)。

對每個 (feat, debias, sem_thr):除特徵/門檻外全相同(同 mv2_bf hull、同遮罩、同視角、同 eval),
  voxel_sem_cluster → build_hull_gt → eval_hull_gt,收分組 recall(分 stack/非 stack)。
公平:CLIP 也一起掃門檻(不拿固定 0.3 比 SigLIP2 最佳)。
輸出 sweep_result.csv(每 config 的 n3/stack3/全體 recall+precision)。
用法: ./sweep.py [--scenes n3 stack3] [--thrs 0.30 0.40 0.50] [--feats clip b32 b16]
                 [--debias 1 0] [--hit-iou 0.7]
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EVAL = REPO / "data" / "eval"
DIAG = EVAL / "_diag" / "siglip_probe"
S2 = REPO / "srp" / "stage2_instances"
PY = sys.executable   # webots_visual_hull
FEAT_FILES = {"clip": "clip_mean_feats.npy", "b32": "siglip_b32_feats.npy", "b16": "siglip_b16_feats.npy"}
BG_FILES = {"clip": "", "b32": str(DIAG / "siglip_b32_bg.npy"), "b16": str(DIAG / "siglip_b16_bg.npy")}
HULL = str(EVAL / "srp_hull_mobilesamv2_bf")
SAM = str(EVAL / "mobilesamv2_fast")
CAPS = str(REPO / "data" / "captures_fast")


def run(cmd, env):
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print("  [ERR]", " ".join(cmd[-4:]), r.stderr[-400:], flush=True)
    return r.returncode == 0


def read_recall(root, hit_iou):
    """讀 eval_hull_gt 的分組 summary,回 {group: (recall, precision)}。"""
    p = EVAL / root / f"hull_gt_summary_iou{hit_iou:g}.csv"
    out = {}
    if p.is_file():
        for row in csv.DictReader(open(p, encoding="utf-8")):
            out[row["group"]] = (row["recall"], row["precision"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["n3", "stack3"])
    ap.add_argument("--thrs", nargs="+", type=float, default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55])
    ap.add_argument("--feats", nargs="+", default=["clip", "b32", "b16"])
    ap.add_argument("--debias", nargs="+", default=["1", "0"])
    ap.add_argument("--hit-iou", type=float, default=0.7, dest="hit_iou")
    args = ap.parse_args()

    rows = []
    for feat in args.feats:
        for deb in args.debias:
            for thr in args.thrs:
                root = f"_sweep_{feat}_d{deb}_t{thr:g}"
                env = dict(os.environ, HULL_ROOT=HULL, SAM_ROOT=SAM, CAPTURES_ROOT=CAPS,
                           OUT_ROOT=root, FEAT_FILE=FEAT_FILES[feat], DEBIAS=deb, BG_FILE=BG_FILES[feat])
                tag = f"{feat} debias={deb} thr={thr:g}"
                ok = run([PY, str(S2 / "voxel_sem_cluster.py"), *args.scenes], env)
                if ok:
                    run([PY, str(S2 / "build_hull_gt.py"), "--root", root, *args.scenes], env)
                    run([PY, str(S2 / "eval_hull_gt.py"), "--root", root,
                         "--hit-iou", str(args.hit_iou), *args.scenes], env)
                rec = read_recall(root, args.hit_iou)
                row = {"feat": feat, "debias": deb, "sem_thr": thr}
                for g in args.scenes + ["全體"]:
                    r, p = rec.get(g, ("", ""))
                    row[f"{g}_recall"] = r; row[f"{g}_prec"] = p
                rows.append(row)
                print(f"[{tag}] " + " ".join(f"{g}:R={rec.get(g,('?',))[0]}" for g in args.scenes + ["全體"]),
                      flush=True)

    outp = DIAG / "sweep_result.csv"
    with open(outp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n→ {outp}")
    # 每 (feat,debias) 取最佳 sem_thr(依全體 recall)
    print("\n== 各組最佳門檻(依全體 recall) ==")
    best = {}
    for r in rows:
        k = (r["feat"], r["debias"])
        v = float(r["全體_recall"]) if r["全體_recall"] else -1
        if k not in best or v > best[k][0]:
            best[k] = (v, r)
    for (feat, deb), (v, r) in sorted(best.items()):
        print(f"  {feat:<5} debias={deb} 最佳thr={r['sem_thr']:<5} "
              + " ".join(f"{g}:R={r[g+'_recall']}" for g in args.scenes + ["全體"]))


if __name__ == "__main__":
    main()
