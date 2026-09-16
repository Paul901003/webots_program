#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""stack_leak_nosep.py — GT 表面標籤上的乾淨分離評估。

★ 為什麼是這支(2026-09-16 重建,舊 job-tmp perobj_leak.py 已失,落地存檔避免再消失):
  found@/mIoU/堆疊s@(見 RESULT_hull_vote_reassign_eval.md,已作廢)是聚合指標,看不出「相異 mesh 被歸同一 instance」的過併。
  使用者定案指標 = 把 pipeline instance「重投影回 GT 遮罩」判「堆疊有沒有分開 / 混太嚴重」,分 n/occ/stack/all。

定義(全部在同一顆前景 hull 的表面 voxel;GT 物體歸屬用 gtlabel=重投影比 GT 遮罩的 3D 版):
  - GT 物體 per voxel: srp_hull_gtlabel_<base>/<sc>/instances.npz labels(1..N=物體,0=過估;build_meta.labelmap 給名)。
  - pipeline instance per voxel: <inst-root>/<sc>/instances.npz labels(>0=實例)。兩者同 hull 同網格。
  - 主導物體 dom(i) = instance i 內 gtlab>0 voxel 最多的 GT 物體。
  - correct(o) = 物體 o 的 voxel 落在 dom(i)=o 的預測 instance 的比例。
  - 混/洩漏 leak(o→q) = 物體 o 的 voxel 落在 dom(i)=q (q≠o) instance 的比例。
  - unassigned(o) = 物體 o 的 voxel 預測為 label 0 的比例。這是漏標，不可被當成低 leak。
  - fragment(o) = correct voxel 落在最大正確 instance 以外的比例，量同物體過切。
  - 主 instance main(o) = 持有 o 最多 voxel 的 instance;沒分開(u,l)= main(u)==main(l)>0(on 對)。
  - instance 純度 purity(i)= dom(i) 佔 i 內 gtlab>0 voxel 的比例。
  - 分組 n/occ/stack/all;stack 的洩漏平均只算「有參與 on 關係」的物體;排 GEX。

GEX(方法不處理,排除):skillet_lid, windex_bottle, colored_wood_blocks, dice。

★ 驗證錨點(--validate;必須重現才算「跟之前一樣」;瘦=am1 hull+gtlabel_am1):
  stack4_scene0002 soup→tuna:fp leak(soup)≈28%、tuna 主 instance 純度≈65%;中心 leak≈0.5%、純度≈99%。
  stack3_scene0005 gelatin→foam:fp≈3%、中心≈76%。 stack4_scene0014 sponge→sugar:fp≈4%、中心≈73%。
  stack4_scene0006 gelatin→soup:fp≈2%、中心≈77%。 沒分開率:n=0%、occ≤0.7%、stacked 7~17%。

用法:
  ./stack_leak_nosep.py --validate                         # 只跑 4 錨點場,印 leak+純度,對錨點
  ./stack_leak_nosep.py --inst-root <root> --gt-root srp_hull_gtlabel_am1 [scenes...]   # 分組報表(空=全 303 多物場)
"""
import argparse
import sys
import json
import glob
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
from labels import label_dir  # noqa: E402

EVAL = REPO / "data" / "eval"
GEX = {"skillet_lid", "windex_bottle", "colored_wood_blocks", "dice"}


def on_pairs(sc):
    try:
        rel = json.loads((label_dir(sc) / "relations.json").read_text())
    except Exception:
        return []
    out = []
    for r in rel.get("relations", []):
        if r.get("type") == "on":
            x = r.get("x", "").split("_", 1)[-1]
            y = r.get("y", "").split("_", 1)[-1]
            if x and y:
                out.append((x, y))
    return out


def load_gt(gt_root, sc):
    p = EVAL / gt_root / sc / "instances.npz"
    if not p.is_file():
        return None, None
    z = np.load(p, allow_pickle=True)
    lab = z["labels"]
    meta = json.loads(str(z["build_meta"])) if "build_meta" in z.files else {}
    idx2name = {int(k): v for k, v in meta.get("labelmap", {}).items()}
    return lab, idx2name


def label_stats(gtlab, plab, idx2name=None):
    """Compare GT and prediction labels on the same voxel grid."""
    if plab.shape != gtlab.shape:
        raise ValueError(f"label shape mismatch: gt={gtlab.shape}, pred={plab.shape}")
    idx2name = idx2name or {}
    objs = [int(o) for o in np.unique(gtlab) if o > 0]
    name2idx = {idx2name.get(o, str(o)): o for o in objs}
    dom = {}
    for i in np.unique(plab):
        if i <= 0:
            continue
        m = (plab == i) & (gtlab > 0)
        if not m.any():
            continue
        vals, cnts = np.unique(gtlab[m], return_counts=True)
        dom[int(i)] = int(vals[cnts.argmax()])
    per = {}
    for o in objs:
        voxel_of_o = gtlab == o
        total = int(voxel_of_o.sum())
        assigned = plab[voxel_of_o]
        positive = assigned[assigned > 0]
        main_i = 0
        if len(positive):
            values, counts = np.unique(positive, return_counts=True)
            main_i = int(values[counts.argmax()])
        correct_by_inst = {}
        leak_by_obj = {}
        for i in np.unique(positive):
            count = int((positive == i).sum())
            owner = dom.get(int(i))
            if owner == o:
                correct_by_inst[int(i)] = count
            elif owner is not None:
                leak_by_obj[owner] = leak_by_obj.get(owner, 0) + count
        correct = sum(correct_by_inst.values())
        leak = sum(leak_by_obj.values())
        unassigned = int((assigned == 0).sum())
        largest_correct = max(correct_by_inst.values(), default=0)
        per[o] = {
            "name": idx2name.get(o, str(o)),
            "tot": total,
            "main_i": main_i,
            "correct": correct,
            "correct_frac": correct / max(total, 1),
            "unassigned": unassigned,
            "unassigned_frac": unassigned / max(total, 1),
            "leak": leak,
            "leak_frac": leak / max(total, 1),
            "leak_by_obj": leak_by_obj,
            "leak_by_obj_frac": {q: count / max(total, 1) for q, count in leak_by_obj.items()},
            "correct_instance_count": len(correct_by_inst),
            "largest_correct": largest_correct,
            "largest_correct_frac": largest_correct / max(total, 1),
            "fragment": correct - largest_correct,
            "fragment_frac": (correct - largest_correct) / max(total, 1),
        }

    def purity(i):
        m = (plab == i) & (gtlab > 0)
        count = int(m.sum())
        if count == 0:
            return 0.0
        _, counts = np.unique(gtlab[m], return_counts=True)
        return int(counts.max()) / count
    return {"objs": objs, "idx2name": idx2name, "name2idx": name2idx,
            "per": per, "dom": dom, "purity": purity}


def scene_stats(inst_root, gt_root, sc):
    gtlab, idx2name = load_gt(gt_root, sc)
    if gtlab is None:
        return None
    path = EVAL / inst_root / sc / "instances.npz"
    if not path.is_file():
        return None
    return label_stats(gtlab, np.load(path)["labels"], idx2name)


def grp_of(sc):
    if sc.startswith(("stack", "stkb")):
        return "stack"
    if sc.startswith("occ"):
        return "occ"
    if sc.startswith("n"):
        return "n"
    return "?"


def validate(gt_root_thin="srp_hull_gtlabel_am1"):
    anchors = [
        ("stack4_scene0002", "tomato_soup_can", "tuna_fish_can"),
        ("stack3_scene0005", "gelatin_box", "foam_brick"),
        ("stack4_scene0014", "sponge", "sugar_box"),
        ("stack4_scene0006", "gelatin_box", "tomato_soup_can"),
    ]
    roots = {"fp": "srp_hull_semcluster_reNNfpSd_am1", "中心": "srp_hull_semcluster_reNNcSd_am1",
             "fp_div": "srp_hull_divB_t50_reNNfpSd_am1", "中心_div": "srp_hull_divB_t50_reNNcSd_am1"}
    print(f"=== 錨點驗證(gt={gt_root_thin})===")
    for sc, a, b in anchors:
        print(f"\n[{sc}] on 對: {a}(洩漏測這個)→ {b}(純度測其主 instance)")
        for tag, root in roots.items():
            st = scene_stats(root, gt_root_thin, sc)
            if st is None:
                print(f"  {tag:8s}{root}: 缺資料")
                continue
            ai = st["name2idx"].get(a)
            bi = st["name2idx"].get(b)
            if ai is None or bi is None:
                print(f"  {tag:8s}: 場內找不到物體 {a}/{b}(有:{list(st['name2idx'])})")
                continue
            leak_a = st["per"][ai]["leak_frac"] * 100
            main_b = st["per"][bi]["main_i"]
            pur_b = st["purity"](main_b) * 100 if main_b > 0 else 0
            same = st["per"][ai]["main_i"] == st["per"][bi]["main_i"] and main_b > 0
            print(f"  {tag:8s}leak({a})={leak_a:5.1f}%  {b}主inst純度={pur_b:5.1f}%  沒分開={'是' if same else '否'}")


def report(inst_root, gt_root, scenes):
    metric_names = ("correct_frac", "leak_frac", "unassigned_frac", "fragment_frac")
    groups = {g: {name: [] for name in metric_names} for g in ("n", "occ", "stack", "all")}
    nosep = {g: [0, 0] for g in ("n", "occ", "stack", "all")}  # [沒分開, on對數]
    for sc in scenes:
        st = scene_stats(inst_root, gt_root, sc)
        if st is None:
            continue
        g = grp_of(sc)
        if g not in groups:
            continue
        pairs = on_pairs(sc)
        onobjs = set()
        for upper, lower in pairs:
            if upper in GEX or lower in GEX:
                continue
            iu = st["name2idx"].get(upper)
            il = st["name2idx"].get(lower)
            if iu is None or il is None:
                continue
            onobjs.update((iu, il))
            same = st["per"][iu]["main_i"] == st["per"][il]["main_i"] and st["per"][iu]["main_i"] > 0
            for key in (g, "all"):
                nosep[key][0] += int(same)
                nosep[key][1] += 1
        # n/occ take all objects; stack takes objects involved in an on relation.
        for obj in st["objs"]:
            info = st["per"][obj]
            if info["name"] in GEX or (g == "stack" and obj not in onobjs):
                continue
            for key in (g, "all"):
                for name in metric_names:
                    groups[key][name].append(info[name])
    print(f"\n=== {inst_root}  vs  {gt_root} ===")
    print(f"{'組':>6}{'物體':>6}{'correct%':>10}{'leak%':>9}{'未指派%':>10}{'過切%':>9}{'on對':>6}{'主群同%':>10}")
    for group in ("n", "occ", "stack", "all"):
        values = groups[group]
        mean = {name: float(np.mean(values[name])) * 100 if values[name] else 0.0 for name in metric_names}
        same, total = nosep[group]
        same_pct = same / total * 100 if total else 0.0
        print(f"{group:>6}{len(values['leak_frac']):>6}{mean['correct_frac']:>9.2f}%"
              f"{mean['leak_frac']:>8.2f}%{mean['unassigned_frac']:>9.2f}%"
              f"{mean['fragment_frac']:>8.2f}%{total:>6}{same_pct:>9.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--inst-root")
    ap.add_argument("--gt-root", default="srp_hull_gtlabel_am1")
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    if a.validate:
        validate()
        return
    if not a.inst_root:
        sys.exit("需 --inst-root(或 --validate)")
    scenes = a.scenes
    if not scenes:
        scenes = sorted(Path(p).parent.name for p in
                        glob.glob(str(EVAL / a.inst_root / "*_scene*" / "instances.npz")))
        scenes = [s for s in scenes if not s.startswith("n1_")]
    report(a.inst_root, a.gt_root, scenes)


if __name__ == "__main__":
    main()
