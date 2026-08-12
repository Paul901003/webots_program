#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""am_compare.py — 比 allow_miss 0/1/2/3 對 semcluster recall 的影響(整體 + 逐物漏檢率)。
只讀各 am 的 hull_gt.json + gt.json(不重跑分群),用 decide_hits 定命中。
輸出每 am 的 missed_by_object CSV + 終端小物對照。"""
import sys, json, glob, csv
from pathlib import Path
from collections import Counter, defaultdict

REPO = Path("/home/cho/webots_program")
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
from eval_hull_gt import decide_hits   # noqa
EVAL = REPO / "data" / "eval"
GT = EVAL / "gt_reproj"
AMS = [0, 1, 2, 3]
HIT = 0.6
SMALL = ["062_dice", "031_spoon", "022_windex_bottle", "028_skillet_lid", "030_fork",
         "070-a_colored_wood_blocks", "070-b_colored_wood_blocks", "061_foam_brick"]


def per_am(am):
    root = EVAL / f"srp_hull_semcluster_clip_am{am}"
    appear = Counter(); missed = Counter()
    for hp in glob.glob(str(root / "*_scene*/hull_gt.json")):
        sc = Path(hp).parent.name
        gj = json.loads((GT / sc / "gt.json").read_text())
        unocc = gj["unoccluded_views"]
        gt_eval = list(gj["gt_objects"])   # 全放置物體(unocc=0 也算,auto-miss)
        hulls = json.loads(Path(hp).read_text())["hulls"]
        hit_d, red_d = decide_hits(hulls, gt_eval, HIT)
        found = {hit_d[k] for k in hulls if hit_d[k] and k not in red_d and hit_d[k] in gt_eval}
        for g in gt_eval:
            appear[g] += 1
            if g not in found:
                missed[g] += 1
    return appear, missed


def main():
    data = {am: per_am(am) for am in AMS}
    # 每 am 存 missed_by_object CSV
    for am in AMS:
        appear, missed = data[am]
        rows = sorted(((g, appear[g], missed[g], round(missed[g] / appear[g], 3))
                       for g in appear if missed[g] > 0), key=lambda x: -x[2])
        out = EVAL / f"srp_hull_semcluster_clip_am{am}" / f"missed_by_object_iou{HIT:g}.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["object", "eval_appear", "missed", "miss_rate"]); w.writerows(rows)
    # 整體 recall
    print(f"=== 整體 recall @{HIT} ===")
    for am in AMS:
        appear, missed = data[am]
        tot = sum(appear.values()); mis = sum(missed.values())
        print(f"  am{am}: recall={1-mis/tot:.3f} ({tot-mis}/{tot})  漏 {mis}")
    # 小/薄物逐 am 漏檢率
    print(f"\n=== 小/薄物漏檢率 @{HIT}(出現次數 | am0 / am1 / am2 / am3)===")
    print(f"  {'物體':<28}{'出現':>4}   {'am0':>6}{'am1':>7}{'am2':>7}{'am3':>7}")
    for g in SMALL:
        ap = data[0][0].get(g, 0)
        if ap == 0:
            continue
        rates = []
        for am in AMS:
            a, m = data[am]; rates.append(m.get(g, 0) / a.get(g, 1) if a.get(g, 0) else 0)
        star = "  ← am1改善" if rates[1] < rates[0] - 0.03 else ("  ← am1變差" if rates[1] > rates[0] + 0.03 else "")
        print(f"  {g:<28}{ap:>4}   " + "".join(f"{r:>7.2f}" for r in rates) + star)


if __name__ == "__main__":
    main()
