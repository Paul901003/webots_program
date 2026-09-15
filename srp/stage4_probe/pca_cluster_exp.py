#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""pca_cluster_exp.py — 實驗:debiased CLIP 特徵先全域 PCA 再 cos 分群,vs baseline。

公平比法:對每場,baseline(cos average-linkage 切 SEM_THR=0.40)產生 N 群;
PCA 版在【相同群數 N】下(maxclust)重分群 → 隔離「PCA 這一步」的效果,不受門檻校準干擾。
重用 cluster_purity_eval.cluster_scene(同一份 mobilesamv2 clip_mean_feats + debias + GT 標籤)。

指標(只算有 GT 標籤的遮罩,sklearn):
  homogeneity  群純度(不混別物;越高越好)
  completeness 同物聚一起(不被拆散;越高越好)
  overseg      每物平均拆幾群(越低越好)
用法: ./pca_cluster_exp.py [--k 16 32 64 128] [group...]
"""
import argparse, sys, glob
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_score, completeness_score
from collections import Counter, defaultdict

import os
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_purity_eval import cluster_scene, EVAL, ROOT, GT_OUT  # noqa


def overseg(y, c):
    """每物平均拆到幾群。"""
    d = defaultdict(set)
    for yi, ci in zip(y, c):
        d[yi].add(ci)
    return np.mean([len(s) for s in d.values()]) if d else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("groups", nargs="*")
    args = ap.parse_args()

    scenes = sorted(os.path.basename(p) for p in glob.glob(str(GT_OUT / "*_scene*")))  # 從 GT 列(semcluster 目錄已刪)
    if args.groups:
        scenes = [s for s in scenes if s.split("_")[0] in args.groups]

    # 1) 收集每場:debiased 特徵 F、GT 物標籤、baseline 群
    data = []          # (F, y_lbl, c_base_lbl, N_all)
    allF = []
    for sc in scenes:
        try:
            ref, clof, gtof, F, gj, modal = cluster_scene(sc)
        except Exception:
            continue
        if F is None or len(F) < 2:
            continue
        idx = [i for i, r in enumerate(ref) if gtof[r] is not None]   # 有 GT 標籤的遮罩
        if len(idx) < 2:
            continue
        y = [gtof[ref[i]] for i in idx]
        cb = [clof[ref[i]] for i in idx]
        Nall = len(set(clof[r] for r in ref))                        # baseline 全群數(含未標)
        data.append((np.asarray(F, float), np.array(idx), y, cb, Nall))
        allF.append(np.asarray(F, float))
    allF = np.vstack(allF)
    print(f"場數 {len(data)}  總遮罩 {len(allF)}  特徵維 {allF.shape[1]}")

    # baseline 指標
    b_h = [homogeneity_score(y, cb) for _, _, y, cb, _ in data]
    b_c = [completeness_score(y, cb) for _, _, y, cb, _ in data]
    b_o = [overseg(y, cb) for _, _, y, cb, _ in data]
    print(f"\n{'版本':<14}{'homogeneity':>13}{'completeness':>14}{'overseg':>10}")
    print(f"{'baseline':<14}{np.mean(b_h):>13.3f}{np.mean(b_c):>14.3f}{np.mean(b_o):>10.3f}")

    # 2) 各 K:全域 PCA → 每場相同群數(maxclust=Nall)重分群
    for k in args.k:
        kk = min(k, allF.shape[1], allF.shape[0])
        pca = PCA(n_components=kk).fit(allF)
        ev = pca.explained_variance_ratio_.sum()
        H, C, O = [], [], []
        for F, idx, y, cb, Nall in data:
            Fp = pca.transform(F)
            if len(Fp) <= Nall:
                cp_all = np.arange(len(Fp))            # 群數≥遮罩數 → 各自一群
            else:
                cp_all = fcluster(linkage(pdist(Fp, "cosine"), "average"), t=Nall, criterion="maxclust")
            cp = [cp_all[i] for i in idx]
            H.append(homogeneity_score(y, cp)); C.append(completeness_score(y, cp)); O.append(overseg(y, cp))
        print(f"{'PCA-'+str(kk):<14}{np.mean(H):>13.3f}{np.mean(C):>14.3f}{np.mean(O):>10.3f}"
              f"   (保留變異 {ev*100:.0f}%)")


if __name__ == "__main__":
    main()
