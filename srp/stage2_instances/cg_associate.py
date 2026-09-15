#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""cg_associate.py — ConceptGraphs 式實例關聯(免 depth,用 surface voxel 當點雲)。

每視角:表面 voxel 投影 + z-buffer 取「可見」表面 voxel;每個 SAM mask(先 mask_subtract_contained
掏掉合體)→ 落在 mask 內的可見 voxel = 一個 detection(帶該 mask CLIP 特徵)。跨視角**增量關聯**:
空間(voxel 集合交集)+ 視覺(CLIP cos),超門檻併入、否則新建。物體從關聯**湧現**,不預先切 hull。
輸出 data/eval/srp_hull_cg/<scene>/instances.{npz,json}(labels 只含表面 voxel)。

可視化: SRP_VIZ_ARGS="<scene> 1 srp_hull_cg surface" webots worlds/hull_viz.wbt
用法: ./cg_associate.py [scene|group|(空=全部)] [--n-views 12]
env: HULL_ROOT(srp_hull_v12) SAM_ROOT(sam_only_fast) CAPTURES_ROOT(captures_fast) ARM_MASK_ROOT
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
_ST3 = ndimage.generate_binary_structure(3, 1)
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import camera as cam          # noqa: E402
import masks as MK            # noqa: E402
import viewpoints as VP       # noqa: E402

HULL_ROOT = Path(os.environ.get("HULL_ROOT", str(REPO / "data" / "eval" / "srp_hull_v12")))
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "sam_only_fast")))
CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures_fast")))
ARM_ROOT = os.environ.get("ARM_MASK_ROOT")

# ── 參數 ──
# 關聯 = spatial(voxel 重疊) + visual(CLIP),兩者一起進 agg 過門檻(見 SIM_THRESHOLD 處)。
# sim_threshold=1.2 使「純視覺相似但空間不重疊」(如兩個 cups,sp≈0、vis≈0.9<1.2)不會被錯併。
MIN_VOX_DET = 15      # detection 最少可見 voxel(濾雜訊)
MERGE_THR = 0.5       # 物體間合併:空間重疊(交集/較小者)> 此值(= CG merge_overlap_thresh)
MIN_OBJ_VOX = 40      # 最終物體最少 voxel:低於此的小碎片 object 丟掉(設 label 0)
# CG merge_overlap_objects:空間重疊 > MERGE_THR「且」視覺相似 > 此值 才合併(text 條件我無 per-object
# caption 特徵故略)。CG README 值 0.8;<0 = 關閉(退回純空間合併)。
MERGE_VISUAL_SIM = float(os.environ.get("MERGE_VISUAL_SIM", "0.8"))
# 完全照 ConceptGraphs:agg = (1+PHYS_BIAS)*spatial + (1-PHYS_BIAS)*visual(sim_sum),
# 門檻加在「加總後」:agg < SIM_THRESHOLD 視為不配對(對應 CG 的 agg<sim_threshold→-inf)。
# spatial=voxel 重疊比、visual=CLIP 餘弦;PHYS_BIAS 同 CG 預設 0.0。兩者皆可 env 覆寫方便掃描。
PHYS_BIAS = float(os.environ.get("PHYS_BIAS", "0.0"))
SIM_THRESHOLD = float(os.environ.get("SIM_THRESHOLD", "1.2"))   # CG README 實際值:sim_threshold=1.2


def mask_subtract_contained(masks, th1=0.8, th2=0.7):
    """大 mask 減掉被它包含的小 mask(bbox 判定,同 ConceptGraphs)。masks: list bool(H,W)。"""
    n = len(masks)
    if n == 0:
        return masks
    boxes = np.zeros((n, 4))
    for i, m in enumerate(masks):
        ys, xs = np.where(m)
        boxes[i] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] if len(xs) else [0, 0, 0, 0]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    lt = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    rb = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    inter = (rb - lt).clip(min=0)
    ia = inter[:, :, 0] * inter[:, :, 1]
    iob1 = ia / np.maximum(areas[:, None], 1)      # 交集 / box_i
    iob2 = iob1.T                                   # 交集 / box_j
    contained = (iob1 < th2) & (iob2 > th1)         # box_j 幾乎全在 box_i 內、box_i 明顯更大
    out = [m.copy() for m in masks]
    ci, cj = contained.nonzero()
    for i, j in zip(ci, cj):
        out[i] = out[i] & (~masks[j])
    return out


def zbuffer_visible(P, C, Rb, W, H, vs, rmax=8):
    """voxel 立方體 footprint 投影 + 像素級 z-buffer → vox_at(H*W,):每像素最近 voxel 的 local idx(-1=無)。

    不再拿 voxel「中心點」當一個像素(那會讓中心被擋的部分可見 voxel 整個被丟)。
    改成:每個 voxel 依深度投影成一小塊像素(半徑 ≈ fx·(vs/2)/深度 = 立方體投影大小),
    在像素層級比深度取最近;voxel 只要在它的 footprint 內贏得「任一」像素就算可見。
    完全被擋在後面的仍會排除。"""
    Rwc, t = cam.pose_to_w2c(C, Rb)
    K = cam.intrinsics(W, H)
    X = P @ Rwc.T + t
    zc = X[:, 2]
    ok = zc > 1e-9
    zz = np.where(ok, zc, 1.0)
    fx = K[0, 0]; fy = K[1, 1]
    u = np.round(fx * X[:, 0] / zz + K[0, 2]).astype(np.int64)
    v = np.round(fy * X[:, 1] / zz + K[1, 2]).astype(np.int64)
    r = np.zeros(len(P), np.int64)                  # 每 voxel 投影半徑(像素)
    r[ok] = np.clip(np.round(fx * (vs * 0.5) / zc[ok]).astype(np.int64), 0, rmax)
    R = int(r.max()) if len(r) else 0
    base = np.arange(len(P))
    pix_l, vox_l, z_l = [], [], []
    for dy in range(-R, R + 1):                     # 對 footprint 內每個 offset,以 per-voxel r 遮罩
        for dx in range(-R, R + 1):
            sel = ok & (np.abs(dx) <= r) & (np.abs(dy) <= r)
            if not sel.any():
                continue
            uu = u[sel] + dx; vv = v[sel] + dy
            inb = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            if not inb.any():
                continue
            idx = base[sel][inb]
            pix_l.append(vv[inb] * W + uu[inb]); vox_l.append(idx); z_l.append(zc[idx])
    vox_at = np.full(H * W, -1, np.int64)
    if not pix_l:
        return vox_at
    pix = np.concatenate(pix_l); vox = np.concatenate(vox_l); zz2 = np.concatenate(z_l)
    order = np.argsort(zz2)                          # 近→遠
    uniq, first = np.unique(pix[order], return_index=True)   # 每像素第一個(最近)
    vox_at[uniq] = vox[order][first]
    return vox_at


def largest_cc(vset, gi, gj, gk, shape):
    """detection 的 voxel 取最大 3D 連通坨(同 ConceptGraphs 的 process_pcd 取最大 cluster):
    合體 mask 框到跨物體的 voxel 會分成多坨,只留最大那坨,濾掉橋接雜點。"""
    li = np.fromiter(vset, np.int64)
    if len(li) == 0:
        return vset
    g = np.zeros(shape, bool)
    g[gi[li], gj[li], gk[li]] = True
    lab, n = ndimage.label(g, _ST3)
    if n <= 1:
        return vset
    comp = lab[gi[li], gj[li], gk[li]]
    big = np.bincount(comp)[1:].argmax() + 1
    return set(li[comp == big].tolist())


def load_arm(scene, view, shape):
    if not ARM_ROOT:
        return None
    import cv2
    p = Path(ARM_ROOT) / scene / f"{view}.png"
    if not p.is_file():
        return None
    a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return (a > 127) if (a is not None and a.shape == tuple(shape)) else None


def merge_objects(objects):
    """物體間去重:vox 交集/較小者 > MERGE_THR 就合併,迭代到穩定。"""
    changed = True
    while changed and len(objects) > 1:
        changed = False
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                a, b = objects[i]["vox"], objects[j]["vox"]
                if not a or not b:
                    continue
                inter = len(a & b)
                if inter / min(len(a), len(b)) > MERGE_THR:
                    if float(objects[i]["ft"] @ objects[j]["ft"]) <= MERGE_VISUAL_SIM:
                        continue         # CG: 空間重疊但外觀不像 → 不併(避免併掉真的相鄰不同物體)
                    na, nb = objects[i]["n"], objects[j]["n"]
                    objects[i]["vox"] = a | b
                    ft = objects[i]["ft"] * na + objects[j]["ft"] * nb
                    objects[i]["ft"] = ft / (np.linalg.norm(ft) + 1e-9)
                    objects[i]["n"] = na + nb
                    for vw, fs in objects[j]["masks"].items():         # 合併來源遮罩
                        objects[i]["masks"].setdefault(vw, set()).update(fs)
                    objects.pop(j)
                    changed = True
                    break
            if changed:
                break
    return objects


def process(scene, n_views, out_root="srp_hull_cg"):
    hp = HULL_ROOT / scene / "hull.npz"
    if not hp.is_file():
        print(f"[skip] {scene}: 無 hull"); return None
    z = np.load(hp)
    shape = z["occupancy"].shape; gm = z["grid_min"]; vs = float(z["voxel_size"])
    if "surface" not in z.files:
        print(f"[skip] {scene}: 無 surface mask(先跑 add_surface_mask)"); return None
    surf = z["surface"].astype(bool)
    sflat = np.flatnonzero(surf.ravel()); ns = len(sflat)
    if ns == 0:
        print(f"[skip] {scene}: 空 surface"); return None
    gi, gj, gk = np.unravel_index(sflat, shape)
    P = gm + (np.stack([gi, gj, gk], 1) + 0.5) * vs      # (ns,3) 表面 voxel 世界座標

    group = scene.split("_")[0]
    sdir = CAPTURES / f"multi_{group}" / scene
    views = sorted(VP.selected_view_names(n_views))

    objects = []                                          # 湧現物體:{vox:set(local), ft:512, n:int}
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
        featmap = MK.mask_feats(vd)           # {遮罩檔名: 特徵},用檔名查(與過濾解耦)
        if not featmap:
            continue
        H, W = km[0][0].shape
        C, Rb = cam.load_pose(pose)
        vox_at = zbuffer_visible(P, C, Rb, W, H, vs)
        arm = load_arm(scene, vn, (H, W))
        masks_sub = mask_subtract_contained([m for m, _ in km])
        names = [nm for _, nm in km]          # 各遮罩檔名(與 masks_sub 同序)
        n_used += 1
        for mi, m in enumerate(masks_sub):
            ft = featmap.get(names[mi])       # 以檔名查特徵
            if ft is None:
                continue
            mm = m.ravel()
            if arm is not None:
                mm = mm & (~arm.ravel())
            vis = vox_at[mm]
            vis = vis[vis >= 0]
            if len(vis) == 0:
                continue
            vset = set(np.unique(vis).tolist())
            vset = largest_cc(vset, gi, gj, gk, shape)   # 只留最大連通坨,濾合體mask的跨物體voxel
            if len(vset) < MIN_VOX_DET:
                continue
            ftn = ft / (np.linalg.norm(ft) + 1e-9)
            # 完全照 ConceptGraphs:對每個既有物體算 agg=(1+bias)*spatial+(1-bias)*visual,
            # 門檻加在「加總後」——agg < SIM_THRESHOLD 視為不配對(=CG 的 -inf)。
            # spatial 與 visual 一起過門檻:visual 高可補 spatial 低、visual 低可拖垮 spatial 高。
            best_j, best_agg = -1, -np.inf
            for j, ob in enumerate(objects):
                sp = len(vset & ob["vox"]) / len(vset)
                vis = float(ftn @ ob["ft"])
                agg = (1.0 + PHYS_BIAS) * sp + (1.0 - PHYS_BIAS) * vis
                if agg < SIM_THRESHOLD:                    # 加總後低於門檻 → 不配對(-inf)
                    continue
                if agg > best_agg:
                    best_agg, best_j = agg, j
            if best_j >= 0:
                ob = objects[best_j]
                ob["vox"] |= vset
                ft2 = ob["ft"] * ob["n"] + ftn            # CLIP 特徵累積成物體平均外觀
                ob["ft"] = ft2 / (np.linalg.norm(ft2) + 1e-9)
                ob["n"] += 1
                ob["masks"].setdefault(vn, set()).add(names[mi])   # 記此 detection 來源遮罩
            else:
                objects.append({"vox": set(vset), "ft": ftn.copy(), "n": 1,
                                "masks": {vn: {names[mi]}}})

    objects = merge_objects(objects)
    objects = [o for o in objects if len(o["vox"]) >= MIN_OBJ_VOX]   # 丟小碎片
    # labels(只表面 voxel);大物體先給小 id
    labels = np.zeros(shape, np.int32)
    order = sorted(range(len(objects)), key=lambda k: -len(objects[k]["vox"]))
    inst_list = []
    for newid, k in enumerate(order, 1):
        li = np.fromiter(objects[k]["vox"], dtype=np.int64)
        labels[gi[li], gj[li], gk[li]] = newid
        inst_list.append({"instance": newid, "n_vox": len(objects[k]["vox"]),
                          "masks": {vw: sorted(fs) for vw, fs in sorted(objects[k]["masks"].items())}})

    out = REPO / "data" / "eval" / out_root / scene
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "instances.npz", labels=labels, grid_min=gm, voxel_size=vs)
    (out / "instances.json").write_text(json.dumps({
        "scene": scene, "n_objects": len(objects), "n_views_used": n_used,
        "voxels_per_obj": sorted((len(o["vox"]) for o in objects), reverse=True),
        "params": {"SIM_THRESHOLD": SIM_THRESHOLD, "PHYS_BIAS": PHYS_BIAS,
                   "MIN_VOX_DET": MIN_VOX_DET, "MERGE_THR": MERGE_THR,
                   "MERGE_VISUAL_SIM": MERGE_VISUAL_SIM},
        "instances": inst_list,          # 每 instance 用到的來源遮罩 {view: [mask檔]}
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{scene}] {n_used} 視角 → {len(objects)} 物體, voxel/物體={sorted((len(o['vox']) for o in objects), reverse=True)}")
    return len(objects)


def resolve(t):
    if not t:
        return sorted(Path(p).parent.name for p in glob.glob(str(HULL_ROOT / "*_scene*/hull.npz")))
    out = []
    for a in t:
        if "scene" in a:
            out.append(a)
        else:
            out += [Path(p).parent.name for p in glob.glob(str(HULL_ROOT / f"{a}_scene*/hull.npz"))]
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--n-views", type=int, default=12)
    args = ap.parse_args()
    scenes = resolve(args.targets)
    out_root = os.environ.get("OUT_ROOT", "srp_hull_cg")   # env 覆寫輸出根(如 mobilesam 用 srp_hull_cg_mv2)
    for i, sc in enumerate(scenes):
        try:
            process(sc, args.n_views, out_root)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")
        if (i + 1) % 30 == 0:
            print(f"...{i+1}/{len(scenes)}", flush=True)


if __name__ == "__main__":
    main()
