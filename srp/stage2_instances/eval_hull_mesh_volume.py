#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""eval_hull_mesh_volume.py — 整顆前景 hull(class-agnostic 佔據)vs 全部 GT mesh 聯集 的體積級 FN/FP。

不是逐 instance(那是 eval_mesh.py)。這裡把「一個場景的前景 hull」當一整塊,和「該場所有 GT 物體
的實心 mesh 聯集(amodal solid,eval_mesh.solid_mesh_occ 填實)」直接做體積比對:
  未檢測 FN = mesh_union & ~hull   (GT 有、hull 沒雕到的體積)
  過檢測 FP = hull & ~mesh_union   (hull 有、落在所有 GT mesh 外的體積 = ghost/過估)
  交集 inter、IoU = inter/union。
體積 = voxel 數 × voxel_size³(以毫升 mL 報,1 mL=1e-6 m³)。

★ 母體說明(audit):hull 是 class-agnostic 全前景聯集,涵蓋場上所有物體,故 mesh_union 也**含全部
  放置物體(不套 GLOBAL_EXCLUDE)**——排除某些物體會讓 hull 在其上的體積誤算成 FP。若要看排除版,
  另加旗標。FN/FP 是**幾何體積**,非「找到率」。

用法: ./eval_hull_mesh_volume.py <scenes|group|(空=舊367)> --root srp_hull_mv2_v12_am0 [--tag ""] [--csv out.csv]
env:mesh 快取走 eval_mesh 的 MESH_CACHE。需 webots_visual_hull。
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
sys.path.insert(0, str(REPO / "srp" / "io"))
import eval_mesh as EM   # noqa: E402  (solid_mesh_occ / gt_objects)

EVAL = REPO / "data" / "eval"
OLD_GROUPS = ["n1", "n3", "n4", "n5", "occ3", "occ4", "occ5", "stack3", "stack4", "stack5"]


def scene_group(sc):
    return sc.split("_")[0]


def expand_scenes(args_scenes, root):
    scenes = []
    for a in args_scenes:
        if "_scene" in a:
            scenes.append(a)
        else:  # group 名
            scenes += sorted(Path(p).name for p in (EVAL / root).glob(f"{a}_scene*"))
    return scenes


def process(scene, root, tag):
    hp = EVAL / root / scene / f"hull{('_'+tag) if tag else ''}.npz"
    if not hp.is_file():
        return None
    z = np.load(hp)
    hull = z["occupancy"].astype(bool)
    gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = hull.shape
    gt = EM.solid_mesh_occ(scene, gm, vs, shape)
    if not gt:
        return None
    mesh = np.zeros(shape, bool)
    for name, occ in gt.items():
        mesh |= occ.astype(bool)
    inter = int((hull & mesh).sum())
    fn = int((mesh & ~hull).sum())          # 未檢測:GT 有 hull 沒有
    fp = int((hull & ~mesh).sum())          # 過檢測:hull 有 GT 沒有
    hv = int(hull.sum()); mv = int(mesh.sum())
    union = hv + mv - inter
    ml = vs ** 3 * 1e6                        # voxel → 毫升
    return {"scene": scene, "n_obj": len(gt),
            "hull_mL": hv * ml, "mesh_mL": mv * ml,
            "FN_mL": fn * ml, "FP_mL": fp * ml, "inter_mL": inter * ml,
            "FN_frac": fn / mv if mv else 0.0,      # 未檢測 / GT 體積
            "FP_frac": fp / hv if hv else 0.0,      # 過檢測 / hull 體積
            "IoU": inter / union if union else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--root", default="srp_hull_mv2_v12_am0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    scenes = expand_scenes(args.scenes, args.root) if args.scenes else \
        [Path(p).name for g in OLD_GROUPS for p in sorted((EVAL / args.root).glob(f"{g}_scene*"))]
    rows = []
    for i, sc in enumerate(scenes):
        try:
            r = process(sc, args.root, args.tag)
            if r:
                rows.append(r)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(scenes)}", flush=True)
    if not rows:
        print("無資料"); return
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"→ {args.csv}")

    def agg(rs, label):
        hv = sum(r["hull_mL"] for r in rs); mv = sum(r["mesh_mL"] for r in rs)
        fn = sum(r["FN_mL"] for r in rs); fp = sum(r["FP_mL"] for r in rs)
        it = sum(r["inter_mL"] for r in rs); un = hv + mv - it
        # 兩種平均:體積加權(總量比)與逐場平均(macro)
        fn_w = fn / mv if mv else 0; fp_w = fp / hv if hv else 0
        fn_m = float(np.mean([r["FN_frac"] for r in rs])); fp_m = float(np.mean([r["FP_frac"] for r in rs]))
        iou = it / un if un else 0
        print(f"{label:<8} 場{len(rs):>3} | mesh {mv/len(rs):6.1f}mL hull {hv/len(rs):6.1f}mL/場 | "
              f"未檢測 {fn_w*100:5.1f}%(加權) {fn_m*100:5.1f}%(逐場) | "
              f"過檢測 {fp_w*100:5.1f}%(加權) {fp_m*100:5.1f}%(逐場) | IoU {iou:.3f}")

    print(f"\n== {args.root} (tag={args.tag or 'hull'}) 前景hull vs 全GT mesh聯集 ==")
    print("  未檢測FN=mesh−hull / mesh體積 ; 過檢測FP=hull−mesh / hull體積 (含全部物體,不套GLOBAL_EXCLUDE)")
    by = defaultdict(list)
    for r in rows:
        by[scene_group(r["scene"])].append(r)
    for g in OLD_GROUPS:
        if by.get(g):
            agg(by[g], g)
    agg(rows, "ALL")


if __name__ == "__main__":
    main()
