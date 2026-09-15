#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""merge_bridge_frag.py — 步驟4:對仍 div<0.5 的碎片群,若之間隔著「無標籤表面 voxel 區塊」相連 → 併(碎片對碎片),
並把該無標籤區吸收進併後群;迭代到無可併。完整群(div≥THC)不參與 → 堆疊不受影響。
用法: BASE_ROOT=<inst-root> OUT_SUFFIX=_bridged HULL_ROOT_NAME=srp_hull_mv2_v12_am1 ./merge_bridge_frag.py [scenes...]
"""
import os, sys, json, glob, datetime as _dt
import numpy as np
from scipy import ndimage
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "data" / "eval"
HULL = os.environ.get("HULL_ROOT_NAME", "srp_hull_mv2_v12_am1")
BASE = os.environ.get("BASE_ROOT", "srp_hull_divB_t50_surf_guard")
OUTSUF = os.environ.get("OUT_SUFFIX", "_bridged")
THC = float(os.environ.get("DIV_THETA", "0.5"))
ST = ndimage.generate_binary_structure(3, 3)


def div_of(grad, m):
    g = np.stack([grad[0][m], grad[1][m], grad[2][m]], 1)
    mag = np.linalg.norm(g, axis=1); keep = mag > 1e-6
    if keep.sum() < 20: return 0.0
    return float(1 - np.linalg.norm((g[keep] / mag[keep, None]).mean(0)))


def process(sc):
    ip = EVAL / BASE / sc / "instances.npz"; hp = EVAL / HULL / sc / "hull.npz"
    if not (ip.is_file() and hp.is_file()):
        return
    z = np.load(ip); lab = z["labels"].copy(); gm = z["grid_min"]; vs = z["voxel_size"]
    hz = np.load(hp); occ = hz["occupancy"]; surf = hz["surface"].astype(bool)
    grad = np.gradient(occ.astype(float))
    n_merge = 0
    for _ in range(20):
        ids = [int(i) for i in np.unique(lab) if i > 0]
        div = {i: div_of(grad, lab == i) for i in ids}
        frag = {i for i in ids if div[i] < THC}
        if len(frag) < 2:
            break
        U = surf & (lab == 0)
        Ul, nu = ndimage.label(U, ST)
        par = {i: i for i in ids}
        def find(x):
            while par[x] != x: par[x] = par[par[x]]; x = par[x]
            return x
        bridged_regions = {}   # region id -> base fragment it fed into
        merged = False
        for r in range(1, nu + 1):
            reg = (Ul == r)
            nb = ndimage.binary_dilation(reg, ST) & (lab > 0)
            adj = set(int(x) for x in np.unique(lab[nb]) if x > 0)
            fadj = sorted(i for i in adj if i in frag)     # 相鄰的碎片(碎片對碎片)
            if len(fadj) >= 2:
                base = fadj[0]
                for j in fadj[1:]:
                    if find(base) != find(j):
                        par[find(j)] = find(base); merged = True; n_merge += 1
                bridged_regions[r] = base
        if not merged:
            break
        newmap = {i: find(i) for i in ids}
        for i in ids:
            if newmap[i] != i:
                lab[lab == i] = newmap[i]
        for r, base in bridged_regions.items():       # 吸收橋接的無標籤區
            lab[(Ul == r)] = find(base)
    # 重編號連續
    out = np.zeros_like(lab); rel = {}
    for i in [int(x) for x in np.unique(lab) if x > 0]:
        rel[i] = len(rel) + 1; out[lab == i] = rel[i]
    meta = {"script": "merge_bridge_frag.py", "step": 4, "base": BASE, "hull": HULL,
            "div_theta": THC, "n_merge": n_merge, "built": _dt.datetime.now().isoformat(timespec="seconds")}
    d = EVAL / (BASE + OUTSUF) / sc; d.mkdir(parents=True, exist_ok=True)
    save = {"labels": out, "grid_min": gm, "voxel_size": vs, "build_meta": json.dumps(meta, ensure_ascii=False)}
    if "occupancy" in z.files: save["occupancy"] = z["occupancy"]
    np.savez_compressed(d / "instances.npz", **save)
    print(f"[{sc}] 步驟4 併 {n_merge} 對碎片 → {int(out.max())} 實例", flush=True)


def main():
    scenes = sys.argv[1:]
    for sc in scenes:
        try: process(sc)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err]{sc}:{e}")


if __name__ == "__main__":
    main()
