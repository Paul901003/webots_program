#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""geo_match.py — 純幾何(無語意、無學習)分離融合 instance:遮罩共享-voxel 圖 → normalized-cut 切 2 群。

用「現有 class-agnostic SAM 遮罩」(sam_only,不需 sam_clip/grounded_sam)分開堆疊上下物:
  ① 每 voxel 在「可見(hull z-buffer 最前)」的視角落進哪塊 SAM 遮罩(id 跨視角亂沒關係)。
  ② 把每個 (視角, 遮罩) 當節點,節點間關聯 = 共享多少 voxel(同物強、跨物只共享接觸面少數→弱)。
  ③ Fiedler / normalized-cut 切成 2 群(從最弱接縫切,勝過 associate 的 union-find 焊接)。
  ④ 每 voxel 看可見視角多數落進哪群 → 歸該物體。
評估同 sep_probe(唯讀,輸出 data/eval/_diag/geo/):voxel 準確、on 恢復,對比 86% 的 GT 遮罩上界。
用法: ./srp/stage4_probe/geo_match.py [scenes...]   預設全 stack。
"""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import camera as cam            # noqa: E402
import eval_mesh as EM          # noqa: E402
import associate as A           # noqa: E402  (view_label_image)

HULL = REPO / "data" / "eval" / "srp_hull"
LABELS = REPO / "data" / "labels"
CAPTURES = REPO / "data" / "captures"
SAMR = REPO / "data" / "eval" / "sam_only"
OUT = REPO / "data" / "eval" / "_diag" / "geo"
EPS = 1e-4
MIN_VOX = 20    # 遮罩節點需被 ≥ 此 voxel 命中才納入
IOU_OK = 0.25


def iou3(a, b):
    i = np.logical_and(a, b).sum(); u = np.logical_or(a, b).sum()
    return i / u if u else 0.0


def visible_labels(scene, P):
    """回傳 L (nk, V) int32:每 voxel 在每視角「可見時」的 SAM 遮罩 id(被遮=0)。"""
    grp = scene.split("_")[0]; sdir = CAPTURES / f"multi_{grp}" / scene
    nk = P.shape[0]; cols = []
    for vdir in sorted((SAMR / scene).glob("view_*")):
        pf = sdir / f"{vdir.name}_pose.json"
        if not pf.is_file():
            continue
        img, _ = A.view_label_image(vdir, "large")
        if img is None:
            continue
        H, W = img.shape
        C, Rb = cam.load_pose(pf); Rw2c, t = cam.pose_to_w2c(C, Rb)
        K = cam.intrinsics(W, H)
        X = P @ Rw2c.T + t; z = X[:, 2]; ok = z > 1e-9; zz = np.where(ok, z, 1.0)
        u = np.round(K[0, 0] * X[:, 0] / zz + K[0, 2]).astype(int)
        vv = np.round(K[1, 1] * X[:, 1] / zz + K[1, 2]).astype(int)
        inb = ok & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
        pix = np.where(inb, vv * W + u, 0)
        depth = np.full(H * W, np.inf); np.minimum.at(depth, pix[inb], z[inb])
        vis = inb & (z <= depth[pix] + EPS)
        lab = np.zeros(nk, np.int32); lab[vis] = img.reshape(-1)[pix[vis]]
        cols.append(lab)
    return np.stack(cols, 1) if cols else None


def ncut2(W):
    """normalized-cut 2-way:回傳每節點 0/1 群標。"""
    d = W.sum(1) + 1e-9
    L = np.diag(d) - W
    vals, vecs = eigh(L, np.diag(d))      # 廣義特徵
    f = vecs[:, 1]                          # Fiedler 向量
    return (f > np.median(f)).astype(int)


def separate(scene, P, fused_mask):
    """對融合 instance 的 voxel(fused_mask)做幾何配對 → 回傳每 voxel 群標(0/1,-1=無票)。"""
    L = visible_labels(scene, P)
    if L is None:
        return None
    nk, V = L.shape
    # 收集遮罩節點 (view, mask_id):被 ≥MIN_VOX 個融合 voxel 命中
    nodes = []
    Lf = L[fused_mask]
    for vi in range(V):
        ids, cnt = np.unique(Lf[:, vi], return_counts=True)
        for mid, c in zip(ids, cnt):
            if mid > 0 and c >= MIN_VOX:
                nodes.append((vi, int(mid)))
    if len(nodes) < 2:
        return None
    node_idx = {nd: i for i, nd in enumerate(nodes)}
    M = len(nodes)
    # 節點關聯 = 共享 voxel 數(跨視角)
    Waff = np.zeros((M, M))
    # 每 voxel 的命中節點清單
    fidx = np.where(fused_mask)[0]
    vox_nodes = [[] for _ in range(len(fidx))]
    for a, gv in enumerate(fidx):
        for vi in range(V):
            nd = (vi, int(L[gv, vi]))
            if nd in node_idx:
                vox_nodes[a].append(node_idx[nd])
    for nds in vox_nodes:
        for i in range(len(nds)):
            for j in range(i + 1, len(nds)):
                Waff[nds[i], nds[j]] += 1; Waff[nds[j], nds[i]] += 1
    grp_node = ncut2(Waff)
    # voxel 投票:可見視角中,命中節點屬哪群多
    pred = -np.ones(nk, np.int8)
    for a, gv in enumerate(fidx):
        if not vox_nodes[a]:
            continue
        g = grp_node[vox_nodes[a]]
        pred[gv] = 0 if (g == 0).sum() >= (g == 1).sum() else 1
    return pred


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
    shape = occ.shape
    labels = np.load(ip)["labels"]
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
        mk = lab == k
        n = min((mk & (true == 1)).sum(), (mk & (true == 2)).sum())
        if n > best:
            best, fused = n, k
    if fused is None:
        return {"scene": scene, "fused": 0}
    fmask = (lab == fused)
    pred = separate(scene, P, fmask)
    if pred is None:
        return None
    sel = fmask & (true > 0)
    yp = pred[sel]; yt = true[sel]
    dec = yp >= 0
    # 群 0/1 對 top/base:用準確率高的對應
    if dec.any():
        # 群0=top 或 群1=top,取高者
        a1 = (np.where(yp == 0, 1, 2)[dec] == yt[dec]).mean()
        a2 = (np.where(yp == 0, 2, 1)[dec] == yt[dec]).mean()
        acc = max(a1, a2); top_is0 = a1 >= a2
    else:
        acc, top_is0 = 0.0, True
    novote = float((~dec).mean())
    g_top = 0 if top_is0 else 1
    predTop = (pred == g_top) & fmask; predBase = (pred == (1 - g_top)) & fmask
    def og(m):
        g = np.zeros(shape, bool); g[gi[m], gj[m], gk[m]] = True; return g
    gT = gt[top] & (labels == fused)
    gB = gt[base] & (labels == fused)
    iouT = iou3(og(predTop), gT); iouB = iou3(og(predBase), gB)
    on_ok = int(iouT >= IOU_OK and iouB >= IOU_OK)
    return {"scene": scene, "fused": 1, "pair": f"{top}|{base}", "n_vox": int(sel.sum()),
            "voxel_acc": round(float(acc), 3), "no_vote": round(novote, 3),
            "iouT": round(iouT, 3), "iouB": round(iouB, 3), "on_recover": on_ok}


def main():
    scenes = sys.argv[1:] or sorted(Path(p).name for p in glob.glob(str(HULL / "stack*")))
    rows = [r for r in (process(s) for s in scenes) if r and r.get("fused")]
    if not rows:
        print("無融合場景"); return
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["scene", "pair", "n_vox", "voxel_acc", "no_vote", "iouT", "iouB", "on_recover"]
    with open(OUT / "geo_match.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    m = lambda k: float(np.mean([r[k] for r in rows]))
    print(f"{'scene':<18}{'voxel準確':>9}{'無票':>6}{'IoU上':>7}{'IoU下':>7}{'on恢復':>7}")
    for r in sorted(rows, key=lambda x: x["voxel_acc"])[:12]:
        print(f"{r['scene']:<18}{r['voxel_acc']:>9}{r['no_vote']:>6}{r['iouT']:>7}{r['iouB']:>7}{r['on_recover']:>7}")
    print("-" * 64)
    print(f"全 {len(rows)} 融合場(純幾何配對):voxel準確 {m('voxel_acc'):.3f} | 無票 {m('no_vote'):.3f} | "
          f"on恢復率 {m('on_recover'):.0%}")
    print(f"對比 GT 遮罩上界:voxel 0.886 / on恢復 86%")
    print(f"→ {OUT/'geo_match.csv'}")


if __name__ == "__main__":
    main()
