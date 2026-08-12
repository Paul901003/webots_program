#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""cluster_purity_eval.py — 直接評估 semcluster「分群這一步」的品質,並關聯到下游 hull。

基準與 srp_hull_semcluster_clip 建置完全一致:mobilesamv2_fast 遮罩、clip_mean_feats、
DEBIAS(CLIP F_BG)、sem_thr=0.40、A-3 12 視角。重跑確定性分群拿 mask→cluster(cl),
用 GT(modal IoU>0.5)標每張遮罩,算:

Part1 分群純度(每場 + 全域):
  · weighted_purity  每群主物佔比、以遮罩加權
  · mixed_rate       含 ≥2 GT 物體的群佔比(群數 & 遮罩加權兩版)
  · contamination    落在「主物≠自己」群的遮罩比例
  · ARI/NMI/homogeneity/completeness/vmeasure(vs GT 物體標籤,sklearn)
  · obj_overseg      每物 mask 被拆到幾個群(mask 級過切)
Part2 下游關聯(回答「混群是否傷 hull」):
  · 每物 contaminated? → 對照 found(recall@0.6)、over-cover(hull過估倍率)
  · 每場 mixed instance 率(instances.json 的來源遮罩跨 ≥2 GT → precision 傷害)

用法: ./cluster_purity_eval.py [group...]     (空=全部 367 場)
env: SAM_ROOT(預設 mobilesamv2_fast) ROOT(預設 srp_hull_semcluster_clip) SEM_THR(0.40) HIT_IOU(0.6)
"""
import os, sys, json, csv
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

REPO = Path("/home/cho/webots_program")
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import masks as MK, viewpoints as VP   # noqa  (mask_clip_cluster 延遲載入:它一被 import 就載 CLIP,純讀 dump 用不到)
from eval_hull_gt import decide_hits                            # noqa
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
try:
    from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                                 homogeneity_completeness_v_measure)
    HAVE_SK = True
except Exception:
    HAVE_SK = False

EVAL = REPO / "data" / "eval"
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(EVAL / "mobilesamv2_fast")))
ROOT = os.environ.get("ROOT", "srp_hull_semcluster_clip")
GT_OUT = EVAL / "gt_reproj"
SEM_THR = float(os.environ.get("SEM_THR", "0.40"))
HIT_IOU = float(os.environ.get("HIT_IOU", "0.6"))
GT_IOU = 0.5                       # SAM 遮罩歸屬某 GT 物體的 modal IoU 門檻
P_CNT, P_FRAC = 2, 0.25            # 混群/污染判定:夥伴 ≥2 遮罩且 ≥25%
_BG = None   # 延遲載入:只有真的要重算特徵(debias)時才 import mask_clip_cluster → 才載 CLIP


def _bg():
    global _BG
    if _BG is None:
        import mask_clip_cluster as MC   # ← 這行才會載入 CLIP 模型
        _BG = MC.F_BG.astype(np.float64)
    return _BG
TMP = EVAL / "_diag" / "cluster_purity"; TMP.mkdir(parents=True, exist_ok=True)


def debias(F):
    bg = _bg()   # 首次呼叫才載 CLIP;純讀 dump 不會走到這
    F = F.astype(np.float64)
    F = F - (F @ bg)[:, None] * bg[None, :]
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


def iou(a, b):
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else 0.0


def load_modal(scene):
    z = np.load(GT_OUT / scene / "gt.npz")
    return {(k[len("modal_"):].rsplit("__", 1)[0], k[len("modal_"):].rsplit("__", 1)[1]): z[k].astype(bool)
            for k in z.files if k.startswith("modal_")}


def _params():
    """決定 cl/gt/F 的所有參數;存進 dump、讀取時比對,不一致就重算(不藏基準不一致)。"""
    return {"sam_root": SAM_ROOT.name, "feat": "clip_mean_feats.npy",
            "thr": SEM_THR, "nv": 12, "gt_iou": GT_IOU, "debias": 1}


def cluster_scene(scene):
    """重跑確定性分群 → 回 ref, clof, gtof, F(去偏), gj, modal。ref=(view,fname)。
    貴的部分(讀遮罩+分群+GT標記)存 MASK_LABELS/<scene>.npz;參數一致就直接讀(秒開),
    否則重算並覆存。GT(gj/modal)仍每次讀新的。MASK_LABELS_FORCE=1 強制重算重存。"""
    gj = json.loads((GT_OUT / scene / "gt.json").read_text())
    modal = load_modal(scene)
    dp = EVAL / ROOT / scene / "mask_labels.npz"   # 跟該 root 的 instances/hull_gt 放一起,不跨 root 衝突
    if os.environ.get("MASK_LABELS_FORCE", "") != "1" and dp.is_file():
        try:
            z = np.load(dp, allow_pickle=True)
            if json.loads(str(z["params"])) == _params():
                ref = [(str(v), str(m)) for v, m in zip(z["views"], z["masks"])]
                clof = {r: int(c) for r, c in zip(ref, z["cl"])}
                gtof = {r: (str(g) if str(g) else None) for r, g in zip(ref, z["gt"])}
                return ref, clof, gtof, z["F"], gj, modal
        except Exception:
            pass   # dump 壞了 → 重算重存
    sel = sorted(set(VP.selected_view_names(12)))
    allf, ref, mask_by = [], [], {}
    for vd in sorted((SAM_ROOT / scene).glob("view_*")):
        if vd.name not in sel:
            continue
        km = MK.kept_object_masks(vd); names = [nm for _, nm in km]
        for (m, nm), f in zip(km, MK.feats_list(vd, names, feat_file="clip_mean_feats.npy")):
            if f is not None:
                allf.append(f); ref.append((vd.name, nm)); mask_by[(vd.name, nm)] = m
    if len(allf) < 2:
        return None
    F = debias(np.array(allf))
    cl = fcluster(linkage(pdist(F, "cosine"), "average"), t=SEM_THR, criterion="distance")
    clof = {r: int(c) for r, c in zip(ref, cl)}
    gtof = {}
    for r, m in mask_by.items():
        vn = r[0]; best, bo = 0.0, None
        for obj in gj["gt_objects"]:
            mm = modal.get((obj, vn))
            if mm is not None:
                i = iou(m, mm)
                if i > best:
                    best, bo = i, obj
        gtof[r] = bo if best >= GT_IOU else None
    dp.parent.mkdir(parents=True, exist_ok=True)   # 存下:參數 key + 遮罩→群/GT/特徵
    np.savez_compressed(dp, params=json.dumps(_params()),
                        views=np.array([r[0] for r in ref]), masks=np.array([r[1] for r in ref]),
                        cl=np.array([clof[r] for r in ref], np.int32),
                        gt=np.array([gtof[r] or "" for r in ref]), F=F)
    return ref, clof, gtof, F, gj, modal


def purity_metrics(ref, clof, gtof):
    """Part1 純度指標(只看有 GT 標的遮罩)。"""
    lab = [r for r in ref if gtof[r] is not None]     # labeled masks
    n_all, n_lab = len(ref), len(lab)
    if n_lab == 0:
        return None
    # 每群 GT 物體組成
    by_cl = defaultdict(Counter)
    for r in lab:
        by_cl[clof[r]][gtof[r]] += 1
    n_cl = len(by_cl)
    wpur_num = sum(max(c.values()) for c in by_cl.values())
    weighted_purity = wpur_num / n_lab
    mixed_cl = [cid for cid, c in by_cl.items()
                if len(c) >= 2 and sorted(c.values())[-2] >= P_CNT
                and sorted(c.values())[-2] / sum(c.values()) >= P_FRAC]
    mixed_rate_cl = len(mixed_cl) / n_cl
    masks_in_mixed = sum(sum(by_cl[cid].values()) for cid in mixed_cl) / n_lab
    # 污染:遮罩的物體 ≠ 其群主物
    dom = {cid: c.most_common(1)[0][0] for cid, c in by_cl.items()}
    contamination = sum(1 for r in lab if gtof[r] != dom[clof[r]]) / n_lab
    # 每物過切:mask 被拆到幾個群
    obj_cl = defaultdict(set)
    for r in lab:
        obj_cl[gtof[r]].add(clof[r])
    obj_overseg = np.mean([len(s) for s in obj_cl.values()])
    out = {"n_masks": n_all, "n_labeled": n_lab, "n_bg": n_all - n_lab, "n_clusters": n_cl,
           "weighted_purity": round(weighted_purity, 4), "mixed_rate_cl": round(mixed_rate_cl, 4),
           "masks_in_mixed_frac": round(masks_in_mixed, 4), "contamination": round(contamination, 4),
           "obj_overseg": round(float(obj_overseg), 3)}
    if HAVE_SK:
        yt = [gtof[r] for r in lab]; yp = [clof[r] for r in lab]
        h, c, v = homogeneity_completeness_v_measure(yt, yp)
        out.update({"ARI": round(adjusted_rand_score(yt, yp), 4),
                    "NMI": round(normalized_mutual_info_score(yt, yp), 4),
                    "homogeneity": round(h, 4), "completeness": round(c, 4), "vmeasure": round(v, 4)})
    return out, by_cl, dom


def per_object(scene, ref, clof, gtof, F, gj, modal, by_cl):
    """Part2:每物 contaminated? + found + over-cover。"""
    unocc = gj["unoccluded_views"]
    gt_eval = list(gj["gt_objects"])   # 全放置物體(unocc=0 也算,auto-miss)
    # found(recall@HIT_IOU)
    hp = EVAL / ROOT / scene / "hull_gt.json"
    found = set()
    hulls = {}
    if hp.is_file():
        hj = json.loads(hp.read_text()); hulls = hj["hulls"]
        hit_d, red_d = decide_hits(hulls, gt_eval, HIT_IOU)
        for k in hulls:
            if hit_d[k] and k not in red_d and hit_d[k] in gt_eval:
                found.add(hit_d[k])
    hz = None
    hzp = EVAL / ROOT / scene / "hull_gt.npz"
    if hzp.is_file():
        hz = np.load(hzp)
    idx = {r: i for i, r in enumerate(ref)}
    rows = []
    for g in gt_eval:
        g_refs = [r for r in ref if gtof[r] == g]
        if not g_refs:
            rows.append({"scene": scene, "object": g, "n_masks": 0, "contaminated": "",
                         "found": int(g in found), "note": "no-mask"}); continue
        home = Counter(clof[r] for r in g_refs).most_common(1)[0][0]
        comp = by_cl.get(home, Counter())
        others = [(o, c) for o, c in comp.items() if o != g]
        partner, pc = (max(others, key=lambda x: x[1]) if others else (None, 0))
        labeled = sum(comp.values())
        contaminated = int(partner is not None and pc >= P_CNT and (pc / labeled if labeled else 0) >= P_FRAC)
        cos_gp = None
        if partner:
            Fg = F[[idx[r] for r in ref if gtof[r] == g]]
            Fp = F[[idx[r] for r in ref if gtof[r] == partner]]
            cos_gp = round(float((Fg @ Fp.T).mean()), 3)
        # over-cover:best hull 的重投影面積 / modal 面積
        over, best_iou = None, 0.0; best_hull = None
        for k, info in hulls.items():
            a = info["per_gt"].get(g, {}).get("avg", 0)
            if a > best_iou:
                best_iou, best_hull = a, k
        if hz is not None and best_hull is not None:
            rr = []
            for vn in unocc.get(g, []):
                rk = f"reproj_{best_hull}__{vn}"
                if rk in hz.files and (g, vn) in modal:
                    ma = int(modal[(g, vn)].sum())
                    if ma > 0:
                        rr.append(int(hz[rk].sum()) / ma)
            if rr:
                over = round(float(np.mean(rr)), 2)
        rows.append({"scene": scene, "object": g, "n_masks": len(g_refs), "home": home,
                     "contaminated": contaminated, "partner": partner, "partner_cnt": pc,
                     "cos_partner": cos_gp, "found": int(g in found), "best_iou": round(best_iou, 3),
                     "overcover": over, "unocc": len(unocc.get(g, []))})
    return rows


def mixed_instances(scene, gtof_by_view):
    """每場 mixed instance 率:instances.json 來源遮罩跨 ≥2 GT。"""
    ij = EVAL / ROOT / scene / "instances.json"
    if not ij.is_file():
        return None
    insts = json.loads(ij.read_text()).get("instances", [])
    n_mixed = 0
    for it in insts:
        comp = Counter()
        for v, names in it.get("masks", {}).items():
            for nm in names:
                g = gtof_by_view.get((v, nm))
                if g:
                    comp[g] += 1
        if len(comp) >= 2 and sorted(comp.values())[-2] >= P_CNT and \
           sorted(comp.values())[-2] / sum(comp.values()) >= P_FRAC:
            n_mixed += 1
    return {"scene": scene, "n_inst": len(insts), "n_mixed_inst": n_mixed,
            "mixed_inst_rate": round(n_mixed / len(insts), 4) if insts else 0.0}


def resolve(groups):
    base = EVAL / ROOT
    scenes = sorted(p.parent.name for p in base.glob("*_scene*/hull_gt.json"))
    if groups:
        scenes = [s for s in scenes if s.split("_")[0] in set(groups)]
    return scenes


def main():
    groups = sys.argv[1:]
    scenes = resolve(groups)
    per_scene, per_obj, per_inst = [], [], []
    for i, sc in enumerate(scenes):
        try:
            r = cluster_scene(sc)
            if r is None:
                continue
            ref, clof, gtof, F, gj, modal = r
            pm = purity_metrics(ref, clof, gtof)
            if pm is None:
                continue
            metrics, by_cl, dom = pm
            metrics = {"scene": sc, "group": sc.split("_")[0], **metrics}
            per_scene.append(metrics)
            per_obj += per_object(sc, ref, clof, gtof, F, gj, modal, by_cl)
            gv = {r_: gtof[r_] for r_ in ref}   # (view,fname)->obj
            mi = mixed_instances(sc, gv)
            if mi:
                per_inst.append(mi)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(scenes)}", flush=True)

    # 存 CSV
    def dump(rows, name):
        if not rows:
            return
        keys = list({k for r in rows for k in r})
        pref = ["scene", "group", "object"]
        keys = [k for k in pref if k in keys] + [k for k in keys if k not in pref]
        with open(TMP / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader()
            w.writerows(rows)
        print(f"  → {TMP/name}")
    dump(per_scene, "cluster_purity_perscene.csv")
    dump(per_obj, "cluster_purity_perobject.csv")
    dump(per_inst, "cluster_purity_perinst.csv")

    # 全域 + 分組彙總
    def agg(rows, key):
        v = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(float(np.mean(v)), 4) if v else None
    print("\n" + "=" * 66)
    print(f"分群純度評估  root={ROOT}  thr={SEM_THR}  {len(per_scene)} 場  (sklearn={HAVE_SK})")
    print("=" * 66)
    cols = ["weighted_purity", "mixed_rate_cl", "masks_in_mixed_frac", "contamination",
            "obj_overseg"] + (["ARI", "NMI", "homogeneity", "completeness", "vmeasure"] if HAVE_SK else [])
    print("\n【Part1 全域】")
    for c in cols:
        print(f"  {c:<20} {agg(per_scene, c)}")
    print("\n【Part1 分組】weighted_purity / mixed_rate_cl / contamination / ARI")
    for grp in sorted(set(r["group"] for r in per_scene)):
        gr = [r for r in per_scene if r["group"] == grp]
        print(f"  {grp:<7} pur={agg(gr,'weighted_purity')}  mix={agg(gr,'mixed_rate_cl')}  "
              f"cont={agg(gr,'contamination')}  ARI={agg(gr,'ARI')}")

    # Part2 關聯:contaminated vs clean → found-rate / over-cover
    print("\n【Part2 關聯:混群是否傷 hull】")
    valid = [r for r in per_obj if r.get("contaminated") in (0, 1)]
    for flag, name in [(1, "污染物(混群)"), (0, "乾淨物")]:
        g = [r for r in valid if r["contaminated"] == flag]
        fr = np.mean([r["found"] for r in g]) if g else 0
        ov = [r["overcover"] for r in g if isinstance(r.get("overcover"), (int, float))]
        print(f"  {name:<14} n={len(g):<4} found率={fr:.3f}  over-cover均={np.mean(ov):.2f}" if g else f"  {name}: 無")
    mir = agg(per_inst, "mixed_inst_rate")
    print(f"\n  mixed instance 率(precision 傷害,全域均): {mir}")
    print(f"  總 instance {sum(r['n_inst'] for r in per_inst)},其中 mixed {sum(r['n_mixed_inst'] for r in per_inst)}")


if __name__ == "__main__":
    main()
