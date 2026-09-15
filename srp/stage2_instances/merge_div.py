#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""merge_div — 用「每群法向多樣性(完整度)」做合併,產可視化/評估用 root。

每群完整度 div = 1 - ‖mean(unit normal)‖,法向取自 np.gradient(occupancy),在 labels==k 上算(已驗證 cosine 0.89)。
高 div = 完整物體(別併),低 div = 過切碎片(該併)。3D 連通閘門套在所有候選上。

兩種合併(各掃 θ):
  A 純完整度閘門:3D 連通相鄰對 (X,Y),若「兩群都完整(div≥θ)」→ 不併;否則(至少一個碎片)→ 併。
  B span3d + 完整度否決:先取 span3d 邊(span_frac>0.6 且 3D 連通),再否決「兩群都完整」的邊。
分組套回 baseline 遮罩/voxel(無損重編 labels),寫 root(含 build_meta)。
env:SAM_ROOT/CAPTURES 預設 fast。 用法: ./merge_div.py [scenes...]   (空=全 303 多物場)
輸出 root:srp_hull_divA_t{40,50,60}、srp_hull_divB_t{40,50,60}。
"""
import sys
import os
import json
import glob
from pathlib import Path

import numpy as np

SUFFIX = os.environ.get("OUT_SUFFIX", "")   # 區分不同 base(如 _mv20)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
sys.path.insert(0, str(REPO / "srp" / "io"))
import eval_mask_grouping as G
import eval_merge_methods as M
import eval_merge_rules as RU
import viewpoints as VP

THETAS = [float(x) for x in os.environ.get("THETAS", "0.4,0.5,0.6").split(",")]
SPAN_THR = 0.6


def group_div(occ, labels, gids):
    g0, g1, g2 = np.gradient(occ.astype(float))
    out = {}
    for k in gids:
        m = (labels == k)
        grad = np.stack([g0[m], g1[m], g2[m]], 1)
        mag = np.linalg.norm(grad, axis=1); keep = mag > 1e-6
        if keep.sum() < 20:
            out[k] = 0.0; continue
        u = grad[keep] / mag[keep, None]
        out[k] = float(1 - np.linalg.norm(u.mean(0)))
    return out


def write_root(scene, bg, g2comp, out_root, meta):
    bp = G.EVAL / M.BASE / scene / "instances.npz"
    z = np.load(bp); labels = z["labels"].astype(np.int32); new = np.zeros_like(labels)
    reps = {}; g2c = {}
    for g in bg:
        r = g2comp[g]
        if r not in reps:
            reps[r] = len(reps) + 1
        g2c[g] = reps[r]
    for g, c in g2c.items():
        new[labels == g] = c
    save = {"labels": new, "grid_min": z["grid_min"], "voxel_size": z["voxel_size"],
            "build_meta": json.dumps(meta, ensure_ascii=False)}
    if "occupancy" in z.files:
        save["occupancy"] = z["occupancy"]
    out = G.EVAL / out_root / scene; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "instances.npz", **save)
    # instances.json(遮罩無損聯集,供其他工具)
    from collections import defaultdict
    cm = defaultdict(lambda: defaultdict(set))
    for g, c in g2c.items():
        for (vn, nm) in bg[g]:
            cm[c][vn].add(nm)
    insts = [{"instance": c, "n_vox": int((new == c).sum()),
              "masks": {vn: sorted(s) for vn, s in sorted(cm[c].items())}} for c in sorted(set(g2c.values()))]
    (out / "instances.json").write_text(json.dumps(
        {"scene": scene, "n_instances": len(insts), "meta": meta, "instances": insts}, indent=2, ensure_ascii=False))


def main():
    scenes = sys.argv[1:]
    if not scenes:
        scenes = []
        for grp in G.GROUPS:
            scenes += sorted(Path(p).name for p in glob.glob(str(G.MV2 / f"{grp}_scene*")))
    views = sorted(VP.selected_view_names(12))
    Rd = 3; offs = [(dx, dy) for dx in range(-Rd, Rd + 1) for dy in range(-Rd, Rd + 1) if dx * dx + dy * dy <= Rd * Rd]
    ker = np.ones((RU.KB, RU.KB), np.uint8)
    print(f"場景={len(scenes)},θ={THETAS}", flush=True)
    for i, sc in enumerate(scenes):
        try:
            bg = M.baseline_groups(sc, views)
            gids = list(bg)
            occ = np.load(G.EVAL / RU.HULL / sc / "hull.npz")["occupancy"]
            labels = np.load(G.EVAL / M.BASE / sc / "instances.npz")["labels"]
            div = group_div(occ, labels, gids)
            adj, pm, sf, c3 = RU.signals(sc, views, bg, offs, ker)
            span_edges = [k for k in c3 if sf.get(k, 0) > SPAN_THR]
            for th in THETAS:
                tag = f"t{int(th*100)}"
                both_complete = lambda e: (div.get(e[0], 0) >= th and div.get(e[1], 0) >= th)
                # A:3D 連通對,至少一個碎片才併
                eA = [e for e in c3 if not both_complete(e)]
                gA = RU.union_find(gids, eA)
                write_root(sc, bg, gA, f"srp_hull_divA_{tag}{SUFFIX}",
                           {"script": "merge_div.py", "method": "A_gate", "theta": th, "src": M.BASE})
                # B:span 邊,否決兩群都完整
                eB = [e for e in span_edges if not both_complete(e)]
                gB = RU.union_find(gids, eB)
                write_root(sc, bg, gB, f"srp_hull_divB_{tag}{SUFFIX}",
                           {"script": "merge_div.py", "method": "B_span_veto", "theta": th, "span_thr": SPAN_THR, "src": M.BASE})
        except Exception as e:
            print(f"[fail] {sc}: {e}")
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(scenes)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
