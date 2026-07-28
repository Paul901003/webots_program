#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""diag_geo_fail.py — 診斷純幾何配對(geo_match)為何失敗:是 ncut 切錯,還是 SAM 遮罩本身分不開。

唯讀(輸出 data/eval/_diag/geo/)。對每個融合 stack 場景,建與 geo_match 相同的「(視角,遮罩)節點 + 命中 voxel」:
  - **遮罩純度**:每節點命中的 voxel 多少屬上物/下物 → 純度=多數占比。高=遮罩乾淨切到單一物。
  - **oracle 分群**:用 GT 把每節點完美歸到上/下物 → voxel 投票 → on 恢復(=class-agnostic SAM + 完美配對的上界)。
  - **ncut 分群**:geo_match 的實際結果。
判讀:
  - oracle 高、ncut 低 → 遮罩夠乾淨,是 **ncut 切錯**(演算法可修,走 a)。
  - oracle 也低 / 純度低 → **SAM 遮罩本身分不開**(走 b 學習 / c 語意)。
用法: ./srp/stage4_probe/diag_geo_fail.py [scenes...]
"""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
sys.path.insert(0, str(REPO / "srp" / "stage4_probe"))
import eval_mesh as EM          # noqa: E402
import geo_match as G           # noqa: E402  (visible_labels, ncut2, iou3, MIN_VOX)

HULL = REPO / "data" / "eval" / "srp_hull"
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "srp" / "io")); from labels import LABELS  # data/labels 分層(類別/數量/場景)
OUT = REPO / "data" / "eval" / "_diag" / "geo"
IOU_OK = 0.25


def build_nodes(L, fmask):
    """回傳 nodes, vox_nodes(融合 voxel 全域索引 -> 命中節點清單)。"""
    nk, V = L.shape
    Lf = L[fmask]
    nodes = []
    for vi in range(V):
        ids, cnt = np.unique(Lf[:, vi], return_counts=True)
        for mid, c in zip(ids, cnt):
            if mid > 0 and c >= G.MIN_VOX:
                nodes.append((vi, int(mid)))
    if len(nodes) < 2:
        return None, None
    nidx = {nd: i for i, nd in enumerate(nodes)}
    fidx = np.where(fmask)[0]
    vox_nodes = []
    for gv in fidx:
        vox_nodes.append([nidx[(vi, int(L[gv, vi]))] for vi in range(V)
                          if (vi, int(L[gv, vi])) in nidx])
    return nodes, (fidx, vox_nodes, nidx)


def vote(pred_group, fidx, vox_nodes, nk):
    pred = -np.ones(nk, np.int8)
    for gv, nds in zip(fidx, vox_nodes):
        if not nds:
            continue
        g = pred_group[nds]
        pred[gv] = 0 if (g == 0).sum() >= (g == 1).sum() else 1
    return pred


def on_recover(pred, fmask, true, gt, top, base, labels, fused, shape, gi, gj, gk):
    sel = fmask & (true > 0); yp = pred[sel]; yt = true[sel]; dec = yp >= 0
    if not dec.any():
        return 0.0, 0
    a1 = (np.where(yp == 0, 1, 2)[dec] == yt[dec]).mean()
    a2 = (np.where(yp == 0, 2, 1)[dec] == yt[dec]).mean()
    acc = max(a1, a2); top_is0 = a1 >= a2
    gtop = 0 if top_is0 else 1
    def og(m):
        g = np.zeros(shape, bool); g[gi[m], gj[m], gk[m]] = True; return g
    iouT = G.iou3(og((pred == gtop) & fmask), gt[top] & (labels == fused))
    iouB = G.iou3(og((pred == (1 - gtop)) & fmask), gt[base] & (labels == fused))
    return float(acc), int(iouT >= IOU_OK and iouB >= IOU_OK)


def process(scene):
    hp = HULL / scene / "hull.npz"; ip = HULL / scene / "instances.npz"
    rp = LABELS / scene / "relations.json"
    if not (hp.is_file() and ip.is_file() and rp.is_file()):
        return None
    ons = [(r["x"], r["y"]) for r in json.loads(rp.read_text())["relations"] if r["type"] == "on"]
    if not ons:
        return None
    top, base = ons[0]
    z = np.load(hp); occ = z["occupancy"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
    shape = occ.shape; labels = np.load(ip)["labels"]
    gt = EM.solid_mesh_occ(scene, gm, vs, shape)
    if not gt or top not in gt or base not in gt:
        return None
    gi, gj, gk = np.nonzero(occ); nk = len(gi)
    P = gm + (np.stack([gi, gj, gk], 1) + 0.5) * vs
    true = np.zeros(nk, np.int8)
    true[gt[top][gi, gj, gk]] = 1; true[gt[base][gi, gj, gk]] = 2
    lab = labels[gi, gj, gk]
    fused, best = None, 0
    for k in np.unique(lab[lab > 0]):
        mk = lab == k; n = min((mk & (true == 1)).sum(), (mk & (true == 2)).sum())
        if n > best:
            best, fused = n, k
    if fused is None:
        return None
    fmask = lab == fused
    L = G.visible_labels(scene, P)
    if L is None:
        return None
    nodes, packed = build_nodes(L, fmask)
    if nodes is None:
        return None
    fidx, vox_nodes, nidx = packed
    M = len(nodes)
    # 每節點 GT 純度 + 多數物
    node_gt = np.zeros(M, np.int8); node_pur = np.zeros(M)
    for nd, i in nidx.items():
        vi, mid = nd
        hit = (L[:, vi] == mid) & fmask
        tt = true[hit]
        nt = (tt == 1).sum(); nb = (tt == 2).sum()
        if nt + nb == 0:
            node_pur[i] = 0; continue
        node_gt[i] = 0 if nt >= nb else 1     # 0=上物群,1=下物群
        node_pur[i] = max(nt, nb) / (nt + nb)
    # oracle 分群(用 node_gt)
    pred_or = vote(node_gt, fidx, vox_nodes, nk)
    acc_or, on_or = on_recover(pred_or, fmask, true, gt, top, base, labels, fused, shape, gi, gj, gk)
    # ncut 分群
    Waff = np.zeros((M, M))
    for nds in vox_nodes:
        for i in range(len(nds)):
            for j in range(i + 1, len(nds)):
                Waff[nds[i], nds[j]] += 1; Waff[nds[j], nds[i]] += 1
    grp = G.ncut2(Waff)
    pred_nc = vote(grp, fidx, vox_nodes, nk)
    acc_nc, on_nc = on_recover(pred_nc, fmask, true, gt, top, base, labels, fused, shape, gi, gj, gk)
    return {"scene": scene, "n_nodes": M, "mean_purity": round(float(node_pur.mean()), 3),
            "oracle_acc": round(acc_or, 3), "oracle_on": on_or,
            "ncut_acc": round(acc_nc, 3), "ncut_on": on_nc}


def main():
    scenes = sys.argv[1:] or sorted(Path(p).name for p in glob.glob(str(HULL / "stack*")))
    rows = [r for r in (process(s) for s in scenes) if r]
    if not rows:
        print("無資料"); return
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "geo_fail_diag.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    m = lambda k: float(np.mean([r[k] for r in rows]))
    # ncut 失敗(on=0)中,oracle 成功(on=1)的比例 = 「ncut 切錯」佔比
    ncut_fail = [r for r in rows if r["ncut_on"] == 0]
    fixable = [r for r in ncut_fail if r["oracle_on"] == 1]
    print(f"全 {len(rows)} 融合場:平均 遮罩節點數 {m('n_nodes'):.1f} | 遮罩純度 {m('mean_purity'):.3f}")
    print(f"  oracle(完美配對) voxel {m('oracle_acc'):.3f} on恢復 {m('oracle_on'):.0%}")
    print(f"  ncut(實際)      voxel {m('ncut_acc'):.3f} on恢復 {m('ncut_on'):.0%}")
    print(f"\nncut 失敗 {len(ncut_fail)} 場中,oracle 本可成功 = {len(fixable)} 場 "
          f"({len(fixable)/max(1,len(ncut_fail)):.0%}) → 這些是『ncut 切錯』(可修)")
    print(f"其餘 {len(ncut_fail)-len(fixable)} 場 oracle 也失敗 → 『SAM 遮罩本身分不開』(需學習/語意)")
    print(f"\n判讀:遮罩純度高 + oracle on 高 → 訊號在,瓶頸是 ncut 演算法(走 a 修幾何);")
    print(f"     純度低 / oracle on 也低 → SAM 遮罩不夠(走 b/c)。")
    print(f"→ {OUT/'geo_fail_diag.csv'}")


if __name__ == "__main__":
    main()
