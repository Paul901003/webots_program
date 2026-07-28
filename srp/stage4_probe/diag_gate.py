#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""diag_gate.py — 驗證:可見性閘控(hull 自身 z-buffer)能否提供「分開堆疊上下物」的依據。

唯讀讀取既有 pipeline(hull/instances/SAM/GT),不動 associate.py、不覆寫任何結果;輸出到新目錄
data/eval/_diag/gate/。對每個 stack 場景的「融合 instance」(同時涵蓋 on 的上物+底物):
  - 每佔據 voxel 建兩種跨視角 SAM 標籤向量:
      (a) 無閘控 = associate 現行(取投影像素標籤,不管被不被遮);
      (b) z-buffer 閘控 = 只有該視線「最前面」的 voxel 取標籤,被遮者該視角不取(視為未觀測)。
  - 用 GT 把 voxel 分真上物/真底物;比接觸面「跨上下」相鄰對的一致度 same/den:
      (a) 應偏高(>0.5,解釋為何被併)、(b) 若 < 0.5 → 閘控提供了分離依據。
  - 量死角:閘控後完全無可見視角的 voxel 比例(=依據不足之處)。
用法: ./srp/stage4_probe/diag_gate.py [scenes...]   預設幾個代表性 stack 場景。
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import camera as cam            # noqa: E402
import eval_mesh as EM          # noqa: E402
import associate as A           # noqa: E402  (view_label_image,唯讀使用)

HULL = REPO / "data" / "eval" / "srp_hull"
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "srp" / "io")); from labels import LABELS  # data/labels 分層(類別/數量/場景)
CAPTURES = REPO / "data" / "captures"
OUT = REPO / "data" / "eval" / "_diag" / "gate"
EPS = 1e-4   # z-buffer 同深容差(m)


def label_vectors(scene, P):
    """回傳 La(無閘控), Lb(z-buffer 閘控) 各 (nk, V) int16。"""
    grp = scene.split("_")[0]
    sdir = CAPTURES / f"multi_{grp}" / scene
    nk = P.shape[0]
    cols_a, cols_b = [], []
    for vdir in sorted((REPO / "data" / "eval" / "sam_only" / scene).glob("view_*")):
        pf = sdir / f"{vdir.name}_pose.json"
        if not pf.is_file():
            continue
        img, _ = A.view_label_image(vdir, "large")
        if img is None:
            continue
        H, W = img.shape
        C, Rb = cam.load_pose(pf); Rw2c, t = cam.pose_to_w2c(C, Rb)
        K = cam.intrinsics(W, H)
        X = P @ Rw2c.T + t
        z = X[:, 2]; ok = z > 1e-9
        zz = np.where(ok, z, 1.0)
        u = np.round(K[0, 0] * X[:, 0] / zz + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * X[:, 1] / zz + K[1, 2]).astype(int)
        inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        pix = np.where(inb, v * W + u, 0)
        imgf = img.reshape(-1)
        la = np.zeros(nk, np.int16); la[inb] = imgf[pix[inb]]
        # z-buffer:每像素最小深度,voxel 為最前才取標籤
        depth = np.full(H * W, np.inf)
        np.minimum.at(depth, pix[inb], z[inb])
        visible = inb & (z <= depth[pix] + EPS)
        lb = np.zeros(nk, np.int16); lb[visible] = imgf[pix[visible]]
        cols_a.append(la); cols_b.append(lb)
    return np.stack(cols_a, 1), np.stack(cols_b, 1)


def agree(L, ea, eb):
    La_, Lb_ = L[ea], L[eb]
    both = (La_ > 0) & (Lb_ > 0)
    den = both.sum(1)
    same = ((La_ == Lb_) & both).sum(1)
    m = den > 0
    return (float((same[m] / den[m]).mean()) if m.any() else float("nan")), int(m.sum())


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
    true[gt[top][gi, gj, gk]] = 1
    true[gt[base][gi, gj, gk]] = 2
    lab_inst = labels[gi, gj, gk]
    # 融合 instance = 同時涵蓋真上物與真底物最多的 instance
    fused, best = None, 0
    for k in np.unique(lab_inst[lab_inst > 0]):
        mk = lab_inst == k
        n = min((mk & (true == 1)).sum(), (mk & (true == 2)).sum())
        if n > best:
            best, fused = n, k
    if fused is None:
        return {"scene": scene, "fused": 0}   # 沒融合(已分開)

    La, Lb = label_vectors(scene, P)
    # 6-鄰接邊(只在融合 instance 內)
    idx3 = -np.ones(shape, np.int64); idx3[gi, gj, gk] = np.arange(nk)
    ea, eb = [], []
    for ax in range(3):
        sa = [slice(None)] * 3; sb = [slice(None)] * 3
        sa[ax] = slice(0, -1); sb[ax] = slice(1, None)
        ia = idx3[tuple(sa)].ravel(); ib = idx3[tuple(sb)].ravel()
        m = (ia >= 0) & (ib >= 0); ea.append(ia[m]); eb.append(ib[m])
    a = np.concatenate(ea); b = np.concatenate(eb)
    inf = (lab_inst[a] == fused) & (lab_inst[b] == fused)
    a, b = a[inf], b[inf]
    cross = ((true[a] == 1) & (true[b] == 2)) | ((true[a] == 2) & (true[b] == 1))
    wtop = (true[a] == 1) & (true[b] == 1)
    wbase = (true[a] == 2) & (true[b] == 2)

    ca_a, nc = agree(La, a[cross], b[cross])
    ca_b, _ = agree(Lb, a[cross], b[cross])
    wt_b, _ = agree(Lb, a[wtop], b[wtop])
    wb_b, _ = agree(Lb, a[wbase], b[wbase])
    # 死角:融合 instance 內真上/底 voxel 閘控後 0 可見視角比例
    vis_b = (Lb > 0).sum(1)
    fin = lab_inst == fused
    mt = fin & (true == 1); mb = fin & (true == 2)
    dz_t = float((vis_b[mt] == 0).mean()) if mt.any() else float("nan")
    dz_b = float((vis_b[mb] == 0).mean()) if mb.any() else float("nan")
    return {"scene": scene, "fused": 1, "pair": f"{top}|{base}",
            "n_top": int(mt.sum()), "n_base": int(mb.sum()), "n_cross_edge": nc,
            "cross_noGate": round(ca_a, 3), "cross_gated": round(ca_b, 3),
            "within_top_gated": round(wt_b, 3), "within_base_gated": round(wb_b, 3),
            "deadzone_top": round(dz_t, 3), "deadzone_base": round(dz_b, 3)}


def main():
    scenes = sys.argv[1:] or ["stack3_scene0001", "stack3_scene0007", "stack3_scene0013",
                              "stack5_scene0001", "stack3_scene0016"]
    rows = [r for r in (process(s) for s in scenes) if r]
    OUT.mkdir(parents=True, exist_ok=True)
    fused = [r for r in rows if r.get("fused")]
    if fused:
        cols = ["scene", "pair", "n_top", "n_base", "n_cross_edge",
                "cross_noGate", "cross_gated", "within_top_gated", "within_base_gated",
                "deadzone_top", "deadzone_base"]
        with open(OUT / "gate_diag.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            w.writerows([{k: r.get(k, "") for k in cols} for r in fused])
        print(f"{'scene':<18}{'跨界無閘':>9}{'跨界閘控':>9}{'內上閘':>8}{'內底閘':>8}{'死角上':>7}{'死角底':>7}")
        for r in fused:
            print(f"{r['scene']:<18}{r['cross_noGate']:>9}{r['cross_gated']:>9}"
                  f"{r['within_top_gated']:>8}{r['within_base_gated']:>8}"
                  f"{r['deadzone_top']:>7}{r['deadzone_base']:>7}")
        print(f"\n判讀:跨界無閘>0.5(解釋被併);跨界閘控<0.5 且 內上/內底閘控仍高 → 閘控提供分離依據。")
        print(f"     死角高 → 該物大量 voxel 閘控後無可見視角 → 依據不足。")
        print(f"→ {OUT / 'gate_diag.csv'}")
    print(f"\n(無融合場景: {[r['scene'] for r in rows if not r.get('fused')]})")


if __name__ == "__main__":
    main()
