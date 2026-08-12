#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""mixed_instance_cause.py — 把每個 mixed instance(來源遮罩跨 ≥2 GT)拆成成因:
  · 分群失敗:主物與某達標混入物的遮罩「同群」(語意把兩物併了)
  · 幾何:主物與混入物在不同群,混入物遮罩是被投影邊界滲漏「記」進來的

複用 cluster_purity_eval.cluster_scene(和 srp_hull_semcluster_clip 建置完全一致)。
輸出 data/eval/_diag/cluster_purity/mixed_instance_cause.csv + 終端彙總。
用法: ./mixed_instance_cause.py [group...]
"""
import sys, json, glob, csv
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster_purity_eval import cluster_scene, EVAL, ROOT   # noqa

OUT = EVAL / "_diag" / "cluster_purity" / "mixed_instance_cause.csv"
P_CNT, P_FRAC = 2, 0.25


def short(o):
    return o.split("_", 1)[-1]


def main():
    groups = set(sys.argv[1:])
    scenes = sorted(p.split("/")[-2] for p in glob.glob(str(EVAL / ROOT / "*_scene*/instances.json")))
    if groups:
        scenes = [s for s in scenes if s.split("_")[0] in groups]
    rows = []
    cnt = Counter(); by_grp = defaultdict(Counter)
    for sc in scenes:
        try:
            r = cluster_scene(sc)
        except Exception:
            continue
        if not r:
            continue
        ref, clof, gtof, F, gj, modal = r
        gv = {x: gtof[x] for x in ref}; clv = {x: clof[x] for x in ref}
        try:
            d = json.loads((EVAL / ROOT / sc / "instances.json").read_text())
        except Exception:
            continue
        grp = sc.split("_")[0]
        for it in d["instances"]:
            pairs = []
            for v, names in it["masks"].items():
                for nm in names:
                    x = (v, nm)
                    if x in gv and gv[x]:
                        pairs.append((gv[x], clv[x]))
            objc = Counter(o for o, _ in pairs)
            if len(objc) < 2:
                continue
            vals = sorted(objc.values())
            if not (vals[-2] >= P_CNT and vals[-2] / sum(vals) >= P_FRAC):
                continue
            dom = objc.most_common(1)[0][0]
            obj_cl = defaultdict(set)
            for o, c in pairs:
                obj_cl[o].add(c)
            # 達標混入物 + 是否與主物同群
            minors = []; shared_all = set()
            for o, v2 in objc.items():
                if o == dom:
                    continue
                meets = v2 >= P_CNT and v2 / sum(objc.values()) >= P_FRAC
                if not meets:
                    continue
                sh = obj_cl[dom] & obj_cl[o]
                shared_all |= sh
                minors.append((o, v2, sh))
            if not minors:
                continue
            kind = "分群失敗" if shared_all else "幾何"
            cnt[kind] += 1; by_grp[grp][kind] += 1
            rows.append({
                "scene": sc, "group": grp, "instance": it["instance"], "kind": kind,
                "dominant": short(dom), "dom_masks": objc[dom], "dom_clusters": sorted(obj_cl[dom]),
                "minorities": "; ".join(f"{short(o)}×{v2}(群{sorted(sh) or '不同'})" for o, v2, sh in minors),
                "shared_clusters": sorted(shared_all) or "",
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    keys = ["scene", "group", "instance", "kind", "dominant", "dom_masks",
            "dom_clusters", "minorities", "shared_clusters"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)
    tot = sum(cnt.values())
    print(f"→ {OUT}")
    print(f"全部 mixed instance: {tot}")
    for k in ("分群失敗", "幾何"):
        print(f"  {k}: {cnt[k]} ({cnt[k]/tot*100:.1f}%)" if tot else f"  {k}: 0")
    print("分組:")
    for g in sorted(by_grp):
        c = by_grp[g]; t = sum(c.values())
        print(f"  {g:<7} 共{t:<3} 分群失敗={c['分群失敗']:<3} 幾何={c['幾何']}")


if __name__ == "__main__":
    main()
