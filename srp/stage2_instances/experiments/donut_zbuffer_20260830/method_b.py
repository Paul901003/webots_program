#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""method_b — 截面積突降(肩台)訊號,分「相疊on vs 同物過切」。

對每對相鄰群(讀 clean_pairs.csv 的分類):取 A、B voxel;下群 L(z 中心低)、上群 U。
接觸處:area_L=L 頂幾層截面積中位、area_U=U 底幾層截面積中位。
shoulder = 1 - min/max(area_L,area_U)（0=同寬平滑、1=大肩台）。
不同寬相疊→肩台大;同物過切→平滑。輸出各類分佈 + AUC(相疊on > 同物過切)。
用法: ./method_b.py   (讀 clean_pairs.csv + 各場 instances.npz)
"""
import sys, csv
import numpy as np
from collections import defaultdict
from pathlib import Path
from scipy.stats import rankdata
REPO = Path(__file__).resolve().parents[4]
EVAL = REPO/"data"/"eval"; INST = "srp_hull_semcluster_surf_am1photo"; HERE = Path(__file__).resolve().parent
BAND = 3   # 接觸處取幾層


def auc(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    r = rankdata(np.concatenate([pos, neg])); U = r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2
    return U/(len(pos)*len(neg))


def area_profile(vox):
    """回 dict k->area(voxel數)。"""
    a = defaultdict(int)
    for k in vox[:, 2]: a[int(k)] += 1
    return a


def shoulder(labels, i, j):
    A = np.argwhere(labels == i); B = np.argwhere(labels == j)
    if len(A) < 10 or len(B) < 10: return None
    # 下群 L / 上群 U
    if A[:, 2].mean() <= B[:, 2].mean(): L, U = A, B
    else: L, U = B, A
    pa = area_profile(L); pu = area_profile(U)
    kLtop = L[:, 2].max(); kUbot = U[:, 2].min()
    # L 頂幾層、U 底幾層 的截面積中位(用各自 profile,避免混到對方)
    aL = np.median([pa[k] for k in range(kLtop-BAND, kLtop+1) if k in pa]) if any(k in pa for k in range(kLtop-BAND, kLtop+1)) else pa[kLtop]
    aU = np.median([pu[k] for k in range(kUbot, kUbot+BAND+1) if k in pu]) if any(k in pu for k in range(kUbot, kUbot+BAND+1)) else pu[kUbot]
    if max(aL, aU) < 1: return None
    return 1.0 - min(aL, aU)/max(aL, aU)


def main():
    rows = list(csv.DictReader(open(HERE/"clean_pairs.csv")))
    # 依場分組,少讀 npz
    by_scene = defaultdict(list)
    for r in rows: by_scene[r["scene"]].append(r)
    out = []
    for sc, rs in by_scene.items():
        p = EVAL/INST/sc/"instances.npz"
        if not p.is_file(): continue
        labels = np.load(p, allow_pickle=True)["labels"]
        for r in rs:
            s = shoulder(labels, int(r["i"]), int(r["j"]))
            if s is None: continue
            out.append((r["class"], s, float(r["plane_med"]), float(r["zov"]), r["objA"], r["objB"], sc))
    cls = defaultdict(list)
    for r in out: cls[r[0]].append(r)
    print(f"n={len(out)}")
    print(f"{'類型':<10}{'n':>5}{'肩台shoulder中位':>16}")
    for c in ["同物過切", "相疊on", "並排相觸", "並排有縫"]:
        rs = cls.get(c, [])
        if rs: print(f"{c:<10}{len(rs):>5}{np.median([r[1] for r in rs]):>16.2f}")
        else: print(f"{c:<10}{0:>5}")
    same = cls.get("同物過切", []); onp = cls.get("相疊on", [])
    if same and onp:
        aB = auc([r[1] for r in onp], [r[1] for r in same])
        aP = auc([r[2] for r in onp], [r[2] for r in same])
        aZ = auc([r[3] for r in same], [r[3] for r in onp])  # z重疊:同物>相疊,方向反
        print(f"\n★ AUC(相疊on vs 同物過切):")
        print(f"    方法B 肩台      = {aB:.3f}")
        print(f"    平面階梯 median = {aP:.3f}  (對照)")
        print(f"    z重疊(低=相疊) = {aZ:.3f}  (對照)")
        # 組合:B 或 z重疊
        combo = []
        for r in onp: combo.append(max(r[1], 1-r[3]))  # 肩台大 或 z重疊低(1-zov大)
        combo_s = []
        for r in same: combo_s.append(max(r[1], 1-r[3]))
        print(f"    B 或 z重疊(取大)= {auc(combo, combo_s):.3f}  (組合)")
        # 同物誤判來源(shoulder 大的同物)
        from collections import Counter
        bad = [r for r in same if r[1] > 0.4]
        print(f"\n同物過切卻肩台>0.4(B 誤拆)n={len(bad)}/{len(same)}:")
        for o, c in Counter([r[4] for r in bad]).most_common(8): print(f"    {o}: {c}")
    # 存
    with open(HERE/"method_b_pairs.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["class", "shoulder", "plane_med", "zov", "objA", "objB", "scene"]); w.writerows(out)


if __name__ == "__main__":
    main()
