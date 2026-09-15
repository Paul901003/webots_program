#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""cg_associate_faithful.py — ConceptGraphs 忠實版(免深度,用 surface voxel 當點雲)。

對照官方源碼(concept-graphs/conceptgraph/slam)逐項對齊,與舊 cg_associate.py 的差異:
  關聯 spatial : 舊=voxel 集合交集 → 本檔=【3D 軸對齊 bbox IoU】(compute_iou_batch)
  關聯門檻     : 舊 sim_threshold=1.2(錯) → 本檔=【0】(config 實值);agg=(1+bias)*spatial+(1-bias)*visual
  幀級關聯     : 每幀先算完所有 detection×object 相似度,再一起指派(同 merge_detections_to_objects)
  detection 去噪: 舊 largest_cc(6-連通) → 本檔=【DBSCAN(eps,min_pts) 取最大 cluster】(pcd_denoise_dbscan)
  merge_overlap: 舊 voxel 交集/min 0.5/0.8 → 本檔=【點-最近鄰重疊比】(在 downsample 距離內) 0.7/0.7
  filter       : 舊 MIN_OBJ_VOX(voxel 數) → 本檔=【num_detections ≥ 3】(obj_min_detections)
  特徵融合     : 兩版相同(偵測數加權平均 + L2),與 CG 一致
depth→免深度替代: 點雲用 hull 表面 voxel(z-buffer 可見);距離參數(eps/downsample)沿用 CG 公尺值(場景尺度相近)。

輸出 data/eval/<OUT_ROOT|srp_hull_cgf>/<scene>/instances.{npz,json}(labels 只含表面 voxel)。
用法: ./cg_associate_faithful.py [scene|group|(空=全部)] [--n-views 12]
env: HULL_ROOT SAM_ROOT CAPTURES_ROOT ARM_MASK_ROOT OUT_ROOT
     + 可覆寫 CG 參數: SIM_THRESHOLD(0) PHYS_BIAS(0) MERGE_OVERLAP_THR(0.7) MERGE_VISUAL_SIM(0.7)
       DBSCAN_EPS(0.05) DBSCAN_MIN(10) DOWNSAMPLE(0.025) OBJ_MIN_DET(3) MIN_VOX_DET(16)
"""
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np
from collections import Counter
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import camera as cam                 # noqa: E402
import masks as MK                   # noqa: E402
import viewpoints as VP              # noqa: E402
import cg_associate as CGA           # noqa: E402  重用 detection 端 helper(z-buffer/mask_subtract/arm)

HULL_ROOT = Path(os.environ.get("HULL_ROOT", str(REPO / "data" / "eval" / "srp_hull_v12")))
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "sam_only_fast")))
CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures_fast")))

# ── CG 參數(預設 = 官方 config 實值;公尺制,場景尺度與 CG 相近故直接沿用) ──
PHYS_BIAS = float(os.environ.get("PHYS_BIAS", "0.0"))
SIM_THRESHOLD = float(os.environ.get("SIM_THRESHOLD", "0.0"))       # config: sim_threshold=0
MERGE_OVERLAP_THR = float(os.environ.get("MERGE_OVERLAP_THR", "0.7"))
MERGE_VISUAL_SIM = float(os.environ.get("MERGE_VISUAL_SIM", "0.7"))
DBSCAN_EPS = float(os.environ.get("DBSCAN_EPS", "0.05"))            # config: dbscan_eps=0.05 m
DBSCAN_MIN = int(os.environ.get("DBSCAN_MIN", "10"))               # config: dbscan_min_points=10
DOWNSAMPLE = float(os.environ.get("DOWNSAMPLE", "0.025"))          # config: downsample_voxel_size=0.025 m
OBJ_MIN_DET = int(os.environ.get("OBJ_MIN_DET", "3"))              # config: obj_min_detections=3
MIN_VOX_DET = int(os.environ.get("MIN_VOX_DET", "16"))             # config: min_points_threshold=16
# 關聯 spatial 型別(CG spatial_sim_type):iou=軸對齊 bbox 3D IoU(CG 預設);overlap=點-最近鄰重疊比(CG 原生選項,對堆疊較敏感)
SPATIAL = os.environ.get("SPATIAL", "iou")
# 配對法(CG match_method):sim_sum=agg 過單一門檻;sep_thresh=空間、語意【各自】過門檻(CG config 預設,官方 batch code 未實作)
MATCH_METHOD = os.environ.get("MATCH_METHOD", "sim_sum")
SEMANTIC_THRESHOLD = float(os.environ.get("SEMANTIC_THRESHOLD", "0.5"))   # config: semantic_threshold=0.5(raw CLIP)
PHYSICAL_THRESHOLD = float(os.environ.get("PHYSICAL_THRESHOLD", "0.5"))   # config: physical_threshold=0.5


def bbox_of(P, li):
    pts = P[li]
    return pts.min(0), pts.max(0)


def iou_3d(ba, bb):
    """軸對齊 3D bbox IoU(同 compute_iou_batch:inter/union 體積)。"""
    lo = np.maximum(ba[0], bb[0]); hi = np.minimum(ba[1], bb[1])
    inter = np.prod(np.clip(hi - lo, 0, None))
    va = np.prod(ba[1] - ba[0]); vb = np.prod(bb[1] - bb[0])
    return inter / (va + vb - inter + 1e-10)


def dbscan_largest(li, P):
    """CG pcd_denoise_dbscan 的 voxel 版:先 downsample 到 DOWNSAMPLE 格,DBSCAN(eps,min)取最大 cluster,
    再回填「落在最大 cluster 格內」的原生 voxel。回傳保留的 local index 陣列。"""
    li = np.asarray(list(li), np.int64)
    if len(li) < DBSCAN_MIN:
        return li                                   # 太少不去噪(同 CG:cluster<5 就退回原樣)
    pts = P[li]
    cell = np.floor(pts / DOWNSAMPLE).astype(np.int64)     # downsample:每格一代表點
    _, cidx, inv = np.unique(cell, axis=0, return_index=True, return_inverse=True)   # 回傳序固定(unique,index,inverse)
    inv = inv.ravel()                              # numpy2.0 return_inverse 可能是 (N,1)
    cpts = pts[cidx]                                # 代表點(每格一個)
    if len(cpts) < DBSCAN_MIN:
        return li
    lab = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN).fit_predict(cpts)
    cnt = Counter(lab[lab >= 0])
    if not cnt:
        return li                                   # 全 noise → 退回原樣(同 CG)
    big = cnt.most_common(1)[0][0]
    keep_cell = set(np.where(lab == big)[0].tolist())
    mask = np.array([inv[k] in keep_cell for k in range(len(li))])
    kept = li[mask]
    return kept if len(kept) >= 5 else li           # 同 CG:最大 cluster < 5 點就退回原樣


def downsample_pts(li, P):
    """把 voxel 世界座標 downsample 到 DOWNSAMPLE 格,每格一代表點(同 CG voxel_down_sample)。"""
    pts = P[np.asarray(li, np.int64)]
    cell = np.floor(pts / DOWNSAMPLE).astype(np.int64)
    _, idx = np.unique(cell, axis=0, return_index=True)
    return pts[idx]


def voxel_nn_overlap(li_i, li_j, P):
    """CG compute_overlap_matrix:先 downsample 到 0.025m,再算 i 的點有多少比例落在 j 的 DOWNSAMPLE 內(最近鄰)。"""
    if len(li_i) == 0 or len(li_j) == 0:
        return 0.0
    pi = downsample_pts(li_i, P); pj = downsample_pts(li_j, P)     # 先 downsample(同 CG,點變稀疏)
    tree = cKDTree(pj)
    d, _ = tree.query(pi, k=1)
    return float((d < DOWNSAMPLE).sum()) / len(pi)


def process(scene, n_views, out_root):
    hp = HULL_ROOT / scene / "hull.npz"
    if not hp.is_file():
        print(f"[skip] {scene}: 無 hull"); return None
    z = np.load(hp)
    shape = z["occupancy"].shape; gm = z["grid_min"]; vs = float(z["voxel_size"])
    if "surface" not in z.files:
        print(f"[skip] {scene}: 無 surface(先跑 add_surface_mask)"); return None
    surf = z["surface"].astype(bool)
    sflat = np.flatnonzero(surf.ravel()); ns = len(sflat)
    if ns == 0:
        print(f"[skip] {scene}: 空 surface"); return None
    gi, gj, gk = np.unravel_index(sflat, shape)
    P = gm + (np.stack([gi, gj, gk], 1) + 0.5) * vs
    # 全域 local index 表:grid→local(給 z-buffer 的 vox_at 用同一套 local 編號)
    group = scene.split("_")[0]
    sdir = CAPTURES / f"multi_{group}" / scene
    views = sorted(VP.selected_view_names(n_views))

    objects = []          # {vox:set(local), ft:np(512), n:int, bbox:(min,max), masks:{view:set}}
    n_used = 0
    for vn in views:
        vd = SAM_ROOT / scene / vn
        pose = sdir / f"{vn}_pose.json"
        cf = vd / "clip_mean_feats.npy"
        if not (vd.is_dir() and pose.is_file() and cf.is_file()):
            continue
        km = MK.kept_object_masks(vd)
        if not km:
            continue
        featmap = MK.mask_feats(vd)
        if not featmap:
            continue
        H, W = km[0][0].shape
        C, Rb = cam.load_pose(pose)
        vox_at = CGA.zbuffer_visible(P, C, Rb, W, H, vs)      # 每像素最近可見表面 voxel 的 local idx(-1)
        arm = CGA.load_arm(scene, vn, (H, W))
        masks_sub = CGA.mask_subtract_contained([m for m, _ in km])
        names = [nm for _, nm in km]
        n_used += 1
        # ── 1) 本幀所有 detection(voxel set + 特徵),各自 DBSCAN 去噪 ──
        dets = []
        for mi, m in enumerate(masks_sub):
            ft = featmap.get(names[mi])
            if ft is None:
                continue
            mm = m.ravel()
            if arm is not None:
                mm = mm & (~arm.ravel())
            vis = vox_at[mm]; vis = vis[vis >= 0]
            if len(vis) == 0:
                continue
            vset = dbscan_largest(np.unique(vis), P)          # CG:detection pcd 也 DBSCAN 去噪取最大坨
            if len(vset) < MIN_VOX_DET:
                continue
            ftn = ft / (np.linalg.norm(ft) + 1e-9)
            dets.append({"vox": set(vset.tolist()), "ft": ftn, "bbox": bbox_of(P, vset),
                         "name": names[mi], "view": vn})
        if not dets:
            continue
        # ── 2) 幀級關聯:先算完 detection×object 的 agg,再一起指派(同 CG merge_detections_to_objects) ──
        assign = []
        for det in dets:
            dli = np.fromiter(det["vox"], np.int64)
            best_j, best_score = -1, -np.inf
            for j, ob in enumerate(objects):
                if SPATIAL == "overlap":                      # CG compute_overlap_matrix_2set:bbox IoU>0 才算點-nn 重疊
                    sp = (voxel_nn_overlap(dli, np.fromiter(ob["vox"], np.int64), P)
                          if iou_3d(det["bbox"], ob["bbox"]) >= 1e-6 else 0.0)
                else:
                    sp = iou_3d(det["bbox"], ob["bbox"])      # spatial = bbox 3D IoU
                vs_sim = float(det["ft"] @ ob["ft"])          # visual = CLIP cos
                if MATCH_METHOD == "sep_thresh":              # CG config 預設:空間、語意各自過門檻才算候選
                    if sp <= PHYSICAL_THRESHOLD or vs_sim <= SEMANTIC_THRESHOLD:
                        continue
                    score = (1.0 + PHYS_BIAS) * sp + (1.0 - PHYS_BIAS) * vs_sim   # 候選中取 agg 最高
                else:                                         # sim_sum:agg 過單一門檻
                    score = (1.0 + PHYS_BIAS) * sp + (1.0 - PHYS_BIAS) * vs_sim
                if score > best_score:
                    best_score, best_j = score, j
            if MATCH_METHOD == "sep_thresh":
                assign.append(best_j)                         # 無候選 → best_j=-1 → 新建
            else:
                assign.append(best_j if best_score > SIM_THRESHOLD else -1)
        # ── 3) 依指派融合(新物體本幀不參與同幀後續 detection 的比對,同 CG) ──
        for det, j in zip(dets, assign):
            if j >= 0:
                ob = objects[j]
                ob["vox"] |= det["vox"]
                ob["bbox"] = bbox_of(P, np.fromiter(ob["vox"], np.int64))
                ft2 = ob["ft"] * ob["n"] + det["ft"] * 1      # 偵測數加權(det n=1)
                ob["ft"] = ft2 / (np.linalg.norm(ft2) + 1e-9)
                ob["n"] += 1
                ob["masks"].setdefault(det["view"], set()).add(det["name"])
            else:
                objects.append({"vox": set(det["vox"]), "ft": det["ft"].copy(), "n": 1,
                                "bbox": det["bbox"], "masks": {det["view"]: {det["name"]}}})

    # ── 4) 收尾:denoise(DBSCAN) → filter(num_det≥3) → merge_overlap(點-nn 0.7 + visual 0.7) ──
    for ob in objects:                                        # denoise_objects
        kept = dbscan_largest(np.fromiter(ob["vox"], np.int64), P)
        ob["vox"] = set(kept.tolist())
        if ob["vox"]:
            ob["bbox"] = bbox_of(P, np.fromiter(ob["vox"], np.int64))
    objects = [o for o in objects if o["n"] >= OBJ_MIN_DET and o["vox"]]   # filter_objects

    changed = True                                            # merge_overlap_objects(迭代到穩定)
    while changed and len(objects) > 1:
        changed = False
        pairs = []
        for i in range(len(objects)):
            for j in range(len(objects)):
                if i == j:
                    continue
                if iou_3d(objects[i]["bbox"], objects[j]["bbox"]) < 1e-6:
                    continue                                  # bbox 不重疊直接跳(同 CG)
                li_i = np.fromiter(objects[i]["vox"], np.int64)
                li_j = np.fromiter(objects[j]["vox"], np.int64)
                ov = voxel_nn_overlap(li_i, li_j, P)          # i 落在 j 附近的比例
                if ov > MERGE_OVERLAP_THR:
                    pairs.append((ov, i, j))
        pairs.sort(reverse=True)                              # 重疊大者先併
        for ov, i, j in pairs:
            if i >= len(objects) or j >= len(objects):
                continue
            if float(objects[i]["ft"] @ objects[j]["ft"]) <= MERGE_VISUAL_SIM:
                continue                                      # 視覺不像 → 不併(text 條件無 caption 故略)
            a, b = objects[i], objects[j]                     # 把 i 併進 j
            b["vox"] |= a["vox"]
            b["bbox"] = bbox_of(P, np.fromiter(b["vox"], np.int64))
            ft = b["ft"] * b["n"] + a["ft"] * a["n"]
            b["ft"] = ft / (np.linalg.norm(ft) + 1e-9); b["n"] += a["n"]
            for vw, fs in a["masks"].items():
                b["masks"].setdefault(vw, set()).update(fs)
            objects.pop(i)
            changed = True
            break

    # ── 5) 輸出 labels(只表面 voxel)+ json ──
    labels = np.zeros(shape, np.int32)
    order = sorted(range(len(objects)), key=lambda k: -len(objects[k]["vox"]))
    inst_list = []
    for newid, k in enumerate(order, 1):
        li = np.fromiter(objects[k]["vox"], np.int64)
        labels[gi[li], gj[li], gk[li]] = newid
        inst_list.append({"instance": newid, "n_vox": len(objects[k]["vox"]),
                          "n_det": objects[k]["n"],
                          "masks": {vw: sorted(fs) for vw, fs in sorted(objects[k]["masks"].items())}})
    out = REPO / "data" / "eval" / out_root / scene
    out.mkdir(parents=True, exist_ok=True)
    meta = {"script": "cg_associate_faithful.py", "n_views": n_used, "hull_root": HULL_ROOT.name,
            "sam_root": SAM_ROOT.name, "spatial": SPATIAL, "match_method": MATCH_METHOD,
            "semantic_threshold": SEMANTIC_THRESHOLD, "physical_threshold": PHYSICAL_THRESHOLD, "sim_threshold": SIM_THRESHOLD,
            "phys_bias": PHYS_BIAS, "merge_overlap_thr": MERGE_OVERLAP_THR, "merge_visual_sim": MERGE_VISUAL_SIM,
            "dbscan_eps": DBSCAN_EPS, "dbscan_min": DBSCAN_MIN, "downsample": DOWNSAMPLE, "obj_min_det": OBJ_MIN_DET}
    np.savez_compressed(out / "instances.npz", labels=labels, grid_min=gm, voxel_size=vs,
                        build_meta=json.dumps(meta, ensure_ascii=False))
    (out / "instances.json").write_text(json.dumps(
        {"scene": scene, "voxel": vs, "n_instances": len(inst_list), "meta": meta,
         "instances": inst_list}, indent=2, ensure_ascii=False))
    print(f"[{scene}] {n_used} 視角 → {len(inst_list)} 物體 "
          f"(voxel/物體={[len(objects[k]['vox']) for k in order]})", flush=True)
    return len(inst_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--n-views", type=int, default=12)
    args = ap.parse_args()
    out_root = os.environ.get("OUT_ROOT", "srp_hull_cgf")
    if not args.targets:
        scenes = sorted(Path(p).parent.name for p in glob.glob(str(HULL_ROOT / "*_scene*/hull.npz")))
    else:
        scenes = []
        for a in args.targets:
            if "scene" in a:
                scenes.append(a)
            else:
                scenes += [Path(p).parent.name for p in glob.glob(str(HULL_ROOT / f"{a}_scene*/hull.npz"))]
        scenes = sorted(set(scenes))
    for sc in scenes:
        try:
            process(sc, args.n_views, out_root)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")


if __name__ == "__main__":
    main()
