#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""inst_mask_attrib.py — 最終 instance 的「遮罩歸屬」對 GT 物體的一致性。

問「每個 GT 物體(mesh)該有的那組 SAM 遮罩,有沒有正確落到對應的那個 hull instance」:
  gt 標籤 : 每張 SAM 遮罩 → modal IoU>GT_IOU 的 GT 物體(gt_reproj 的 modal_<obj>__<view>)
  pred標籤: 每張 SAM 遮罩 → instances.json 裡屬哪個 instance(最終 hull,含投票+3D連通後結果)
  只取「有 GT 標籤且有 instance 標籤」的遮罩,算:
    completeness(≈分群 recall,同物遮罩聚同一 instance)
    homogeneity  (≈分群 precision,一個 instance 不混不同物的遮罩)
用法: ROOT=<method> ./inst_mask_attrib.py [group...]   (空=全部舊367 prefix)
env: SAM_ROOT(mobilesamv2_fast) ROOT(instances 根) GT_IOU(0.5)
"""
import os, sys, json, glob
from pathlib import Path
import numpy as np
from sklearn.metrics import homogeneity_score, completeness_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
import masks as MK  # noqa: E402
EVAL = REPO / "data" / "eval"
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(EVAL / "mobilesamv2_fast")))
ROOT = os.environ.get("ROOT", "srp_hull_semcluster_base")
GT_OUT = EVAL / "gt_reproj"
GT_IOU = float(os.environ.get("GT_IOU", "0.5"))


def iou(a, b):
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else 0.0


def load_modal(scene):
    z = np.load(GT_OUT / scene / "gt.npz")
    d = {}
    for k in z.files:
        if k.startswith("modal_"):
            obj, view = k[len("modal_"):].rsplit("__", 1)
            d.setdefault(view, {})[obj] = z[k].astype(bool)
    return d


def scene_labels(scene):
    """回傳 (gt_lab, inst_lab):兩個等長 list,對應同一批「有 GT 且有 instance」的遮罩。"""
    ij = EVAL / ROOT / scene / "instances.json"
    if not ij.is_file():
        return None
    insts = json.loads(ij.read_text()).get("instances", [])
    # (view,fname) → instance id
    inst_of = {}
    for it in insts:
        iid = it["instance"]
        for vw, files in it.get("masks", {}).items():
            for f in files:
                inst_of[(vw, f)] = iid
    if not inst_of:
        return None
    modal = load_modal(scene)                       # view → {obj: mask}
    gt_lab, inst_lab = [], []
    for vd in sorted((SAM_ROOT / scene).glob("view_*")):
        vw = vd.name
        gm = modal.get(vw, {})
        if not gm:
            continue
        for m, fname in MK.kept_object_masks(vd):
            key = (vw, fname)
            if key not in inst_of:
                continue                            # 這遮罩沒被任何 instance 用到
            best, bi = 0.0, None
            for obj, gmask in gm.items():
                if gmask.shape != m.shape:
                    continue
                v = iou(m, gmask)
                if v > best:
                    best, bi = v, obj
            if bi is not None and best >= GT_IOU:    # 有 GT 歸屬
                gt_lab.append(bi); inst_lab.append(inst_of[key])
    return (gt_lab, inst_lab) if len(gt_lab) >= 2 else None


def main():
    groups = sys.argv[1:]
    scenes = sorted(Path(p).parent.name for p in glob.glob(str(GT_OUT / "*_scene*/gt.json")))
    if groups:
        scenes = [s for s in scenes if s.split("_")[0] in groups]
    H, C, nmask = [], [], 0
    for sc in scenes:
        try:
            r = scene_labels(sc)
        except Exception:
            r = None
        if r is None:
            continue
        gt, inst = r
        H.append(homogeneity_score(gt, inst)); C.append(completeness_score(gt, inst)); nmask += len(gt)
    if H:
        print(f"[{ROOT}] {len(H)} 場 / {nmask} 遮罩 | "
              f"completeness(分群recall) {np.mean(C):.3f}  homogeneity(分群precision) {np.mean(H):.3f}")
    else:
        print(f"[{ROOT}] 無可算場景(instances.json 可能未存 masks)")


if __name__ == "__main__":
    main()
