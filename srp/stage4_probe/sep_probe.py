#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""sep_probe.py — 逐 voxel 分離測試(最簡規則版):成對語意遮罩 + 可見性閘控 多數決。

問題:堆疊上下物相觸 → hull 融成一個 instance。要切回兩物,只能逐 voxel 判「屬上物/下物」。
本版用最簡單的規則先探底:
  對融合 instance 的每個 voxel,在「它可見(hull z-buffer 最前)」的視角裡,看它投影落進
  上物遮罩還是下物遮罩 → 多數決歸屬。成對語意遮罩先用 GT modal(上界:給乾淨遮罩能不能分)。
評估(唯讀,不動 pipeline,輸出 data/eval/_diag/sep/):
  - per-voxel 準確率(vs GT 實心 mesh 的上/下物);
  - on 恢復:預測切兩塊 → 各自 vs top/base GT 3D IoU 都≥門檻才算成功;
  - 對照:z-cut oracle(最佳水平切高,平面切上界)。
用法: ./srp/stage4_probe/sep_probe.py [scenes...]   預設全 stack。
"""
import csv
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
sys.path.insert(0, str(REPO / "controllers" / "ycb_supervisor"))
import camera as cam            # noqa: E402
import eval_mesh as EM          # noqa: E402
import eval_reproj2d as RP      # noqa: E402  (load_modal_by_view)
from config import PROMPT_TABLE  # noqa: E402

HULL = REPO / "data" / "eval" / "srp_hull"
LABELS = REPO / "data" / "labels"
CAPTURES = REPO / "data" / "captures"
OUT = REPO / "data" / "eval" / "_diag" / "sep"
EPS = 1e-4
IOU_OK = 0.25   # on 恢復:每塊 vs GT 物體 3D IoU 門檻
SRC = "gt"      # gt=GT modal 遮罩(上界);gsam=grounded_sam 預測遮罩


def _sanitize(value):
    out = []
    for ch in value.strip().lower():
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
    return "".join(out)


def _gsam_class(name):
    if name in PROMPT_TABLE:
        return _sanitize(PROMPT_TABLE[name])
    parts = name.split("_"); start = 1 if parts[0].isdigit() else 0
    return _sanitize(" ".join(parts[start:]))


def load_gsam_by_view(scene, top, base):
    """grounded_sam 預測遮罩 → {view: {top:maskT, base:maskB}}(鍵用 YCB 物名)。"""
    grp = scene.split("_")[0]
    d = None
    for gd in sorted(glob.glob(str(REPO / "data" / "eval" / "grounded_sam_*"))):
        cand = Path(gd) / f"multi_{grp}" / scene
        if cand.is_dir():
            d = cand; break
    if d is None:
        return {}
    out = {}
    for ycb in (top, base):
        cls = _gsam_class(ycb)
        for f in d.glob(f"view_*_mask_{cls}.png"):
            v = f.name.split("_mask_")[0]
            m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                out.setdefault(v, {})[ycb] = m > 127
    return out


def get_masks(scene, top, base):
    if SRC == "gsam":
        return load_gsam_by_view(scene, top, base)
    modal, _ = RP.load_modal_by_view(scene)
    return modal or {}


def iou3(a, b):
    i = np.logical_and(a, b).sum(); u = np.logical_or(a, b).sum()
    return i / u if u else 0.0


def vote_separate(scene, P, gi, gj, gk, top, base):
    """回傳 pred(每 voxel: 1=上物,2=下物,0=無票),用成對 GT modal 遮罩 + 可見性閘控多數決。"""
    grp = scene.split("_")[0]
    sdir = CAPTURES / f"multi_{grp}" / scene
    modal = get_masks(scene, top, base)
    if not modal:
        return None
    nk = P.shape[0]
    vt = np.zeros(nk, np.int32); vb = np.zeros(nk, np.int32)
    for v, objs in modal.items():
        if top not in objs or base not in objs:
            continue
        pf = sdir / f"{v}_pose.json"
        if not pf.is_file():
            continue
        mT, mB = objs[top], objs[base]; H, W = mT.shape
        C, Rb = cam.load_pose(pf); Rw2c, t = cam.pose_to_w2c(C, Rb)
        K = cam.intrinsics(W, H)
        X = P @ Rw2c.T + t; z = X[:, 2]; ok = z > 1e-9
        zz = np.where(ok, z, 1.0)
        u = np.round(K[0, 0] * X[:, 0] / zz + K[0, 2]).astype(int)
        vv = np.round(K[1, 1] * X[:, 1] / zz + K[1, 2]).astype(int)
        inb = ok & (u >= 0) & (u < W) & (vv >= 0) & (vv < H)
        pix = np.where(inb, vv * W + u, 0)
        # 可見性:hull z-buffer,只有最前 voxel 投票
        depth = np.full(H * W, np.inf); np.minimum.at(depth, pix[inb], z[inb])
        vis = inb & (z <= depth[pix] + EPS)
        inT = mT.reshape(-1)[pix]; inB = mB.reshape(-1)[pix]
        vt += (vis & inT).astype(np.int32)
        vb += (vis & inB).astype(np.int32)
    pred = np.where(vt == vb, 0, np.where(vt > vb, 1, 2)).astype(np.int8)
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
    true[gt[top][gi, gj, gk]] = 1
    true[gt[base][gi, gj, gk]] = 2
    lab = labels[gi, gj, gk]
    fused, best = None, 0
    for k in np.unique(lab[lab > 0]):
        mk = lab == k
        n = min((mk & (true == 1)).sum(), (mk & (true == 2)).sum())
        if n > best:
            best, fused = n, k
    if fused is None:
        return {"scene": scene, "fused": 0}

    fin = (lab == fused) & (true > 0)        # 融合塊內、有 GT 歸屬的 voxel
    pred = vote_separate(scene, P, gi, gj, gk, top, base)
    if pred is None:
        return None
    sel = fin
    yt = true[sel]; yp = pred[sel]
    decided = yp > 0
    acc = float((yp[decided] == yt[decided]).mean()) if decided.any() else 0.0
    novote = float((~decided).mean())
    # on 恢復:預測上物塊 vs GT top、預測下物塊 vs GT base(在融合塊內)
    occ_idx = np.zeros(nk, bool); occ_idx[sel] = True
    predTop = (pred == 1) & occ_idx; predBase = (pred == 2) & occ_idx
    def occ_of(maskvox):
        g = np.zeros(shape, bool); g[gi[maskvox], gj[maskvox], gk[maskvox]] = True; return g
    gT = gt[top] & (labels == fused); gB = gt[base] & (labels == fused)
    iouT = iou3(occ_of(predTop), gT); iouB = iou3(occ_of(predBase), gB)
    on_ok = int(iouT >= IOU_OK and iouB >= IOU_OK)
    # z-cut oracle(最佳水平切高,平面切上界)
    zc = P[sel, 2]
    best_zacc = 0.0
    for thr in np.percentile(zc, np.arange(10, 91, 5)):
        p = np.where(zc >= thr, 1, 2)
        best_zacc = max(best_zacc, (p == yt).mean(), ((3 - p) == yt).mean())
    return {"scene": scene, "fused": 1, "pair": f"{top}|{base}", "n_vox": int(sel.sum()),
            "voxel_acc": round(acc, 3), "no_vote": round(novote, 3),
            "iouT": round(iouT, 3), "iouB": round(iouB, 3), "on_recover": on_ok,
            "zcut_oracle_acc": round(float(best_zacc), 3)}


def main():
    global SRC
    args = sys.argv[1:]
    if "--src" in args:
        i = args.index("--src"); SRC = args[i + 1]; del args[i:i + 2]
    scenes = args or sorted(Path(p).name for p in glob.glob(str(HULL / "stack*")))
    print(f"[遮罩來源 SRC={SRC}]")
    rows = [r for r in (process(s) for s in scenes) if r and r.get("fused")]
    if not rows:
        print("無融合場景"); return
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["scene", "pair", "n_vox", "voxel_acc", "no_vote", "iouT", "iouB",
            "on_recover", "zcut_oracle_acc"]
    with open(OUT / f"sep_probe_{SRC}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    m = lambda k: float(np.mean([r[k] for r in rows]))
    print(f"{'scene':<18}{'voxel準確':>9}{'無票':>6}{'IoU上':>7}{'IoU下':>7}{'on恢復':>7}{'z切上界':>8}")
    for r in sorted(rows, key=lambda x: x["voxel_acc"])[:12]:
        print(f"{r['scene']:<18}{r['voxel_acc']:>9}{r['no_vote']:>6}{r['iouT']:>7}{r['iouB']:>7}"
              f"{r['on_recover']:>7}{r['zcut_oracle_acc']:>8}")
    print("-" * 70)
    print(f"全 {len(rows)} 融合場:voxel準確 {m('voxel_acc'):.3f} | 無票 {m('no_vote'):.3f} | "
          f"on恢復率 {m('on_recover'):.0%} | z切上界 {m('zcut_oracle_acc'):.3f}")
    print(f"\n判讀:voxel準確高+on恢復率高 → 成對遮罩+可見性(規則)就能分,不需學習。")
    print(f"     若規則低、但 z切上界高 → 形狀/高度有訊號,值得上小模型;兩者都低 → 接觸面真無訊號。")
    print(f"→ {OUT/'sep_probe.csv'}")


if __name__ == "__main__":
    main()
