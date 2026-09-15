#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""materialize_merge — 把 span+3D連通閘門 的合併分組寫成可視化 root(供 hull_viz 目視)。

分組 = eval_merge_rules 的 span 規則(view-frac > S)且過 3D voxel 連通閘門,套回 baseline 群。
輸出:data/eval/<out-root>/<scene>/instances.npz(重編 labels + grid_min/voxel_size/occupancy + build_meta)。
遮罩宇宙、訊號、閘門與 eval_merge_rules 完全一致(唯讀輸入,只多寫 npz 供可視化)。
用法: ./materialize_merge.py <scene...> [--thr 0.6] [--out-root srp_hull_span3d]
之後看:SRP_VIZ_ARGS="<scene> 1 srp_hull_span3d" webots worlds/hull_viz.wbt
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
sys.path.insert(0, str(REPO / "srp" / "io"))
import eval_mask_grouping as G
import eval_merge_methods as M
import eval_merge_rules as RU
import viewpoints as VP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--thr", type=float, default=0.6)
    ap.add_argument("--out-root", default="srp_hull_span3d")
    args = ap.parse_args()

    views = sorted(VP.selected_view_names(12))
    Rd = 3
    offs = [(dx, dy) for dx in range(-Rd, Rd + 1) for dy in range(-Rd, Rd + 1) if dx * dx + dy * dy <= Rd * Rd]
    ker = np.ones((RU.KB, RU.KB), np.uint8)

    for sc in args.scenes:
        bg = M.baseline_groups(sc, views)
        gids = list(bg)
        adj, pm, sf, c3 = RU.signals(sc, views, bg, offs, ker)
        edges = [k for k in c3 if sf.get(k, 0) > args.thr]
        grouping = RU.union_find(gids, edges)          # g -> representative
        reps = {}
        g2comp = {}
        for g in gids:
            r = grouping[g]
            if r not in reps:
                reps[r] = len(reps) + 1
            g2comp[g] = reps[r]

        bp = G.EVAL / M.BASE / sc / "instances.npz"
        z = np.load(bp)
        labels = z["labels"].astype(np.int32)
        new = np.zeros_like(labels)
        for g, c in g2comp.items():
            new[labels == g] = c
        meta = {"script": "materialize_merge.py", "rule": "span+3d_gate", "thr": args.thr,
                "src_inst": M.BASE, "n_groups_before": len(gids), "n_after": int(new.max())}
        save = {"labels": new, "grid_min": z["grid_min"], "voxel_size": z["voxel_size"],
                "build_meta": json.dumps(meta, ensure_ascii=False)}
        if "occupancy" in z.files:
            save["occupancy"] = z["occupancy"]
        out = G.EVAL / args.out_root / sc
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "instances.npz", **save)
        # instances.json:每 comp 的遮罩(baseline 群聯集,供對照)
        from collections import defaultdict
        comp_masks = defaultdict(lambda: defaultdict(set))
        for g, c in g2comp.items():
            for (vn, nm) in bg[g]:
                comp_masks[c][vn].add(nm)
        insts = [{"instance": c, "n_vox": int((new == c).sum()),
                  "masks": {vn: sorted(s) for vn, s in sorted(comp_masks[c].items())}}
                 for c in sorted(comp_masks)]
        (out / "instances.json").write_text(json.dumps(
            {"scene": sc, "n_instances": len(insts), "meta": meta, "instances": insts}, indent=2, ensure_ascii=False))
        print(f"{sc}: {len(gids)} 群 → {int(new.max())} 物體  寫入 {out}")


if __name__ == "__main__":
    main()
