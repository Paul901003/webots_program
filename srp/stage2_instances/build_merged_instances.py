#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""build_merged_instances.py — 用群對交叉投影比例合併過切群,產「合併後 instances」新 root。

怎麼生 / 設置:
- 讀 <inst-root> 的 instances.{npz,json} + reproj_labels.npz(proj_group)。
- 群對交叉 max(A→B,B→A) > τ → 連邊;連通元件=合併後物體(用全部群,不靠 GT)。
- 重編 labels(voxel 的群→元件 id)、mask_clusters(群→元件)、instances 列;存到 <out-root>。
- meta 記 tau/來源 inst_root/script。
用法: ./build_merged_instances.py [scene|group|(空=舊367)] --inst-root srp_hull_semcluster_surf_am1photo \
        --out-root srp_hull_semcluster_surf_am1photo_merged --tau 0.05
"""
import argparse, sys, json
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; EVAL = REPO / "data" / "eval"

def _zov(a0, a1, b0, b1):
    inter = max(0, min(a1, b1) - max(a0, b0)); short = min(a1 - a0, b1 - b0)
    return inter / short if short > 0 else (1.0 if inter > 0 else 0.0)

def crossing_sig(rl):
    """回 groups, sig[(A,B)]=交叉比例, zovr[(A,B)]=z重疊(高度區間)。"""
    og = rl["own_group"].astype(int); vis = rl["vis"]; pg = rl["proj_group"].astype(int); coords = rl["coords"].astype(int)
    groups = [int(g) for g in np.unique(og) if g > 0]
    idx = {g: np.where(og == g)[0] for g in groups}; vv = {g: int(vis[idx[g]].sum()) for g in groups}
    zr = {g: (coords[idx[g], 2].min(), coords[idx[g], 2].max()) for g in groups}
    sig = {}; zovr = {}
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            A, B = groups[a], groups[b]
            if vv[A] == 0 or vv[B] == 0: continue
            fab = int(((pg[idx[A]] == B) & vis[idx[A]]).sum()) / vv[A]
            fba = int(((pg[idx[B]] == A) & vis[idx[B]]).sum()) / vv[B]
            sig[(A, B)] = max(fab, fba); zovr[(A, B)] = _zov(*zr[A], *zr[B])
    return groups, sig, zovr

def union_find(groups, edges):
    par = {g: g for g in groups}
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in edges: par[find(a)] = find(b)
    comp = defaultdict(list)
    for g in groups: comp[find(g)].append(g)
    return list(comp.values())

def build(sc, inst_root, out_root, tau, tz):
    d = EVAL / inst_root / sc
    rlp = d / "reproj_labels.npz"; ip = d / "instances.npz"; jp = d / "instances.json"
    if not (rlp.is_file() and ip.is_file()): return None
    rl = np.load(rlp); groups, sig, zovr = crossing_sig(rl)
    if not groups: return None
    edges = [(a, b) for (a, b), s in sig.items() if s > tau and (tz is None or zovr.get((a, b), 0) > tz)]
    comps = union_find(groups, edges)
    g2new = {}
    for ci, c in enumerate(sorted(comps, key=lambda c: -len(c)), 1):
        for g in c: g2new[g] = ci
    z = np.load(ip); labels = z["labels"].astype(np.int32); gm = z["grid_min"]; vs = float(z["voxel_size"])
    new = np.zeros_like(labels)
    for g, ng in g2new.items(): new[labels == g] = ng
    out = EVAL / out_root / sc; out.mkdir(parents=True, exist_ok=True)
    meta = {"script": "build_merged_instances.py", "src_inst": inst_root,
            "merge": "group_cross+zoverlap" if tz is not None else "group_cross", "tau": tau, "tz": tz,
            "n_groups_before": len(groups), "n_after": int(new.max())}
    save = {"labels": new, "grid_min": gm, "voxel_size": vs, "build_meta": json.dumps(meta, ensure_ascii=False)}
    if "occupancy" in z.files: save["occupancy"] = z["occupancy"]
    np.savez_compressed(out / "instances.npz", **save)
    # instances.json:重編 mask_clusters + instances 列
    if jp.is_file():
        j = json.loads(jp.read_text()); mc = j.get("mask_clusters", {})
        mc2 = {vn: {nm: g2new.get(int(g), 0) for nm, g in cl.items() if int(g) in g2new} for vn, cl in mc.items()}
        insts = []
        for k in range(1, int(new.max()) + 1):
            m = defaultdict(set)
            for vn, cl in mc2.items():
                for nm, gg in cl.items():
                    if gg == k: m[vn].add(nm)
            insts.append({"instance": k, "n_vox": int((new == k).sum()),
                          "masks": {vn: sorted(s) for vn, s in sorted(m.items())}})
        (out / "instances.json").write_text(json.dumps(
            {"scene": sc, "voxel": vs, "n_instances": len(insts), "meta": meta,
             "mask_clusters": mc2, "instances": insts}, indent=2, ensure_ascii=False))
    return int(new.max())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("targets", nargs="*")
    ap.add_argument("--inst-root", default="srp_hull_semcluster_surf_am1photo")
    ap.add_argument("--out-root", default="srp_hull_semcluster_surf_am1photo_merged")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--tz", type=float, default=None, help="z重疊否決門檻;不給=只用交叉")
    args = ap.parse_args(); root = EVAL / args.inst_root
    if not args.targets:
        groups = ("n1", "n3", "n4", "n5", "occ3", "occ4", "occ5", "stack3", "stack4", "stack5")
        scenes = sorted(p.name for g in groups for p in root.glob(f"{g}_scene*"))
    else:
        scenes = []
        for a in args.targets: scenes += [a] if "scene" in a else sorted(p.name for p in root.glob(f"{a}_scene*"))
    done = 0
    for i, sc in enumerate(scenes):
        try:
            r = build(sc, args.inst_root, args.out_root, args.tau, args.tz)
            if isinstance(r, int): done += 1
            if (i + 1) % 60 == 0: print(f"  {i+1}/{len(scenes)}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}", flush=True)
    print(f"完成:產 {done} 場合併 instances 於 data/eval/{args.out_root}/ (tau={args.tau}, tz={args.tz})")

if __name__ == "__main__": main()
