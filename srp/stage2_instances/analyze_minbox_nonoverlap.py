#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""analyze_minbox_nonoverlap.py — 最少不重疊 bbox 分割:每群→AABB,反覆併「重疊≥m」的盒,直到互不重疊。
最終盒=物體。評估真疊物(relations "on")/分開物的 clean/lost/mixed/split。掃 m。讀存好標籤,不重投影。
用法: ./analyze_minbox_nonoverlap.py --groups stack
"""
import argparse, sys, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; EVAL = REPO / "data" / "eval"
sys.path.insert(0, str(REPO / "srp" / "io")); from labels import label_dir

def ov_extent(a, b):   # 每軸重疊 voxel 數(<=0 表沒重疊)
    lo = np.maximum(a[0], b[0]); hi = np.minimum(a[1], b[1]); return hi - lo + 1

def merge_boxes(boxes, m):
    """boxes: list of [lo(3,), hi(3,), set(groups)]。反覆併重疊≥m 的盒。"""
    boxes = [[np.array(lo), np.array(hi), set(g)] for lo, hi, g in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if (ov_extent(boxes[i], boxes[j]) >= m).all():   # 三軸都重疊≥m
                    lo = np.minimum(boxes[i][0], boxes[j][0]); hi = np.maximum(boxes[i][1], boxes[j][1])
                    boxes[i] = [lo, hi, boxes[i][2] | boxes[j][2]]; boxes.pop(j); changed = True; break
            if changed: break
    return boxes

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inst-root", default="srp_hull_semcluster_surf_am1photo")
    ap.add_argument("--groups", default="stack"); args = ap.parse_args(); root = EVAL / args.inst_root
    files = [f for f in sorted(root.glob("*_scene*/reproj_labels.npz")) if f.parent.name.startswith(tuple(args.groups.split(",")))]
    for m in [1, 2, 4, 8]:
        agg = {"stacked": Counter(), "separate": Counter()}; nbox = 0; ngt = 0; nsc = 0
        for f in files:
            sc = f.parent.name; z = np.load(f, allow_pickle=True)
            coords = z["coords"].astype(int); og = z["own_group"].astype(int); gt = z["gt_obj"].astype(int)
            onames = json.loads(str(z["build_meta"]))["onames"]
            rel = json.loads((label_dir(sc) / "relations.json").read_text())["relations"]
            stacked_objs = set()
            for r in rel:
                if r["type"] == "on":
                    for nm in (r["x"], r["y"]):
                        if nm in onames: stacked_objs.add(onames.index(nm))
            boxes0 = []; gobj = {}
            for g in np.unique(og):
                if g <= 0: continue
                ii = np.where(og == g)[0]
                if len(ii) < 5: continue
                c = coords[ii]; o = gt[(og == g) & (gt >= 0)]
                gobj[g] = Counter(o.tolist()).most_common(1)[0][0] if len(o) else None
                if gobj[g] is not None: boxes0.append([c.min(0), c.max(0), {int(g)}])
            if not boxes0: continue
            boxes = merge_boxes(boxes0, m)
            # 每盒物體=成員群多數 gobj
            comp_maj = [Counter([gobj[g] for g in b[2]]).most_common(1)[0][0] for b in boxes]
            comp_objs = [[gobj[g] for g in b[2]] for b in boxes]
            objs = set(gobj[g] for b in boxes for g in b[2])
            obj2comp = defaultdict(set); majcnt = Counter(comp_maj)
            for ci, b in enumerate(boxes):
                for g in b[2]: obj2comp[gobj[g]].add(ci)
            nbox += len(boxes); ngt += len(objs); nsc += 1
            for o in objs:
                cs = [ci for ci, cm in enumerate(comp_maj) if cm == o]
                if majcnt.get(o, 0) == 0: cat = "lost"
                elif len(cs) == 1 and len(set(comp_objs[cs[0]])) == 1 and len(obj2comp[o]) == 1: cat = "clean"
                elif len(obj2comp[o]) > 1: cat = "split"
                else: cat = "mixed"
                agg["stacked" if o in stacked_objs else "separate"][cat] += 1
        print(f"\n===== m={m}(重疊≥m voxel 才併) | 盒/場 {nbox/nsc:.2f} vs GT/場 {ngt/nsc:.2f} =====")
        for grp in ("stacked", "separate"):
            d = agg[grp]; tot = sum(d.values()); lab = "真疊物" if grp == "stacked" else "分開物"
            print(f"  [{lab} {tot}] clean {d['clean']/tot*100:.0f}% | lost {d['lost']/tot*100:.0f}% | mixed {d['mixed']/tot*100:.0f}% | split {d['split']/tot*100:.0f}%")

if __name__ == "__main__": main()
