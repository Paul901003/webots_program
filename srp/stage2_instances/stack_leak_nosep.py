#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""stack_leak_nosep.py — 定案指標:重投影對 GT 遮罩(3D gtlabel 版)算「混/洩漏% + 沒分開率」,分 n/occ/stack/all。無門檻。

★ 為什麼是這支(2026-09-16 重建,舊 job-tmp perobj_leak.py 已失,落地存檔避免再消失):
  found@/mIoU/堆疊s@(見 RESULT_hull_vote_reassign_eval.md,已作廢)是聚合指標,看不出「相異 mesh 被歸同一 instance」的過併。
  使用者定案指標 = 把 pipeline instance「重投影回 GT 遮罩」判「堆疊有沒有分開 / 混太嚴重」,分 n/occ/stack/all。

定義(全部在同一顆前景 hull 的表面 voxel;GT 物體歸屬用 gtlabel=重投影比 GT 遮罩的 3D 版):
  - GT 物體 per voxel: srp_hull_gtlabel_<base>/<sc>/instances.npz labels(1..N=物體,0=過估;build_meta.labelmap 給名)。
  - pipeline instance per voxel: <inst-root>/<sc>/instances.npz labels(>0=實例)。兩者同 hull 同網格。
  - 主導物體 dom(i) = instance i 內 gtlab>0 voxel 最多的 GT 物體。
  - 混/洩漏 leak(o) = 物體 o 的 voxel 落在「dom(i)≠o」instance 的比例(連續、無門檻;plab==0 不算洩漏,算漏標另計)。
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


def scene_stats(inst_root, gt_root, sc):
    gtlab, idx2name = load_gt(gt_root, sc)
    if gtlab is None:
        return None
    ip = EVAL / inst_root / sc / "instances.npz"
    if not ip.is_file():
        return None
    plab = np.load(ip)["labels"]
    if plab.shape != gtlab.shape:
        return None
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
        Vo = (gtlab == o)
        tot = int(Vo.sum())
        pos = plab[Vo]
        pos = pos[pos > 0]
        main_i = 0
        if len(pos):
            iv, ic = np.unique(pos, return_counts=True)
            main_i = int(iv[ic.argmax()])
        leak = 0
        for i in np.unique(pos):
            if dom.get(int(i), -1) != o:
                leak += int((pos == i).sum())
        per[o] = {"name": idx2name.get(o, str(o)), "tot": tot,
                  "main_i": main_i, "leak": leak, "leak_frac": leak / max(tot, 1)}

    def purity(i):
        m = (plab == i) & (gtlab > 0)
        n = int(m.sum())
        if n == 0:
            return 0.0
        vals, cnts = np.unique(gtlab[m], return_counts=True)
        return int(cnts.max()) / n
    return {"objs": objs, "idx2name": idx2name, "name2idx": name2idx,
            "per": per, "dom": dom, "purity": purity}


def grp_of(sc):
    if sc.startswith("stack"):
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
    groups = {"n": [], "occ": [], "stack": [], "all": []}
    nosep = {"n": [0, 0], "occ": [0, 0], "stack": [0, 0], "all": [0, 0]}  # [沒分開, on對數]
    for sc in scenes:
        st = scene_stats(inst_root, gt_root, sc)
        if st is None:
            continue
        g = grp_of(sc)
        pairs = on_pairs(sc)
        onobjs = set()
        for (u, l) in pairs:
            if u in GEX or l in GEX:
                continue
            iu = st["name2idx"].get(u)
            il = st["name2idx"].get(l)
            if iu is None or il is None:
                continue
            onobjs.add(iu)
            onobjs.add(il)
            same = st["per"][iu]["main_i"] == st["per"][il]["main_i"] and st["per"][iu]["main_i"] > 0
            for k in (g, "all"):
                nosep[k][0] += int(same)
                nosep[k][1] += 1
        # 洩漏:n/occ 用全物體;stack 用有參與 on 的物體
        for o in st["objs"]:
            if st["per"][o]["name"] in GEX:
                continue
            if g == "stack" and o not in onobjs:
                continue
            lf = st["per"][o]["leak_frac"]
            groups[g].append(lf)
            groups["all"].append(lf)
    print(f"\n=== {inst_root}  vs  {gt_root} ===")
    print(f"{'組':>6}{'物體數':>7}{'平均洩漏%':>10}{'on對':>6}{'沒分開%':>9}")
    for g in ("n", "occ", "stack", "all"):
        lk = groups[g]
        avg = float(np.mean(lk)) * 100 if lk else 0.0
        ns, tot = nosep[g]
        nsp = ns / tot * 100 if tot else 0.0
        print(f"{g:>6}{len(lk):>7}{avg:>9.2f}%{tot:>6}{nsp:>8.1f}%")


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
