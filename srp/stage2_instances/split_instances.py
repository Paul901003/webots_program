#!/home/cho/.pyenv/versions/grounded_sam/bin/python3
"""split_instances.py — 用「大遮罩切割 + CLIP 語意」把黏連的 hull instance 直接分裂。

對每個已雕好的 instance(associate 產的 labels==k):
  步驟1 大遮罩 A = associate 記錄的 instances.json masks[inst][view](**直接讀,不重算投影**)。
  步驟2 「塊」= 剩餘 R=A−∪B_i / 各內含小遮罩 B_i(sam_only 該視角遮罩中 |A∩B|/|B|>CONTAIN);
        每塊 square_mean_crop → CLIP → 去偏 → 512d。
  步驟3 該 instance 所有塊凝聚聚類(average,dist=1−cos,門檻 SEM_THR)→ g 群 = g 物體。
  步驟4 每 voxel 跨視角投影落哪塊 → 投該塊的群;投票 → voxel 歸群。instance 拆成 g 個。
不碰 associate、不改遮罩。輸出新 instances.npz/json(可餵 eval.py)。需 grounded_sam(clip)。
用法: ./srp/stage2_instances/split_instances.py stack3_scene0001 [--sem-thr 0.35] [--diag]
env: CAPTURES_ROOT SAM_ROOT(sam_only_fast) HULL_ROOT(srp_hull_fast) ARM_MASK_ROOT
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import camera as cam                                  # noqa: E402
import masks as MK                                    # noqa: E402
import refine_masks as RM                             # noqa: E402  (encode_debias/get_f_bg/MIN_AREA)

CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures")))
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "sam_only")))
HULL_ROOT = Path(os.environ.get("HULL_ROOT", str(REPO / "data" / "eval" / "srp_hull")))
ARM_MASK_ROOT = Path(os.environ.get("ARM_MASK_ROOT", str(REPO / "data" / "eval" / "srp_arm_masks")))
CONTAIN = 0.3     # 小遮罩落在大遮罩內的比例門檻(>此算內含)


def load_arm(scene, view):
    p = ARM_MASK_ROOT / scene / f"{view}_arm.png"
    if not p.is_file():
        return None
    a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return (a > 127) if a is not None else None


def process(scene, model, prep, f_bg, device, sem_thr, diag=False):
    hp = HULL_ROOT / scene / "hull.npz"
    ip = HULL_ROOT / scene / "instances.npz"
    ij = HULL_ROOT / scene / "instances.json"
    if not (hp.is_file() and ip.is_file() and ij.is_file()):
        print(f"[skip] {scene}: 缺 hull/instances(.npz/.json)"); return None
    z = np.load(hp); occ = z["occupancy"]; grid_min = z["grid_min"]; vs = float(z["voxel_size"])
    shape = occ.shape
    labels = np.load(ip)["labels"]
    inst_masks = {it["instance"]: it["masks"] for it in json.loads(ij.read_text())["instances"]}

    group = scene.split("_")[0]
    sdir = CAPTURES / f"multi_{group}" / scene
    # 預載每視角:{檔名:mask}(含巢狀)、pose、arm、原圖
    vinfo = {}
    for vdir in sorted((SAM_ROOT / scene).glob("view_*")):
        pose = sdir / f"{vdir.name}_pose.json"
        if not pose.is_file():
            continue
        km = MK.kept_object_masks(vdir)
        if not km:
            continue
        C, Rb = cam.load_pose(pose); Rwc, t = cam.pose_to_w2c(C, Rb)
        img = cv2.cvtColor(cv2.imread(str(sdir / f"{vdir.name}.png")), cv2.COLOR_BGR2RGB)
        vinfo[vdir.name] = {"n2m": {name: m for m, name in km}, "Rwc": Rwc, "t": t,
                            "arm": load_arm(scene, vdir.name), "rgb": img,
                            "hw": km[0][0].shape}

    new_labels = np.zeros(shape, np.int32)
    next_id = 0
    for k in range(1, int(labels.max()) + 1):
        vox = np.array(np.nonzero(labels == k)).T
        nk = len(vox)
        if nk == 0:
            continue
        P = grid_min + (vox + 0.5) * vs
        vm = inst_masks.get(k, {})               # {view: [大遮罩檔名]}(associate 記錄)
        block_feats = []                          # 全局塊 feat
        vox_blocks = [[] for _ in range(nk)]      # voxel -> [全局塊 id]
        for vn, anames in vm.items():
            V = vinfo.get(vn)
            if V is None:
                continue
            n2m = V["n2m"]
            avail = [n2m[a] for a in anames if a in n2m]
            if not avail:
                continue
            A = max(avail, key=lambda m: int(m.sum()))    # 大遮罩(associate 記的;多個取最大)
            aA = int(A.sum())
            # 內含小遮罩:該視角其他遮罩 >CONTAIN 落在 A 內
            subs = [m for nm, m in n2m.items()
                    if m is not A and 0 < int(m.sum()) < aA and int((A & m).sum()) / int(m.sum()) > CONTAIN]
            R = (A & ~np.logical_or.reduce(subs)) if subs else A
            blocks = list(subs) + ([R] if int(R.sum()) >= RM.MIN_AREA else [])
            if len(blocks) < 2:                    # 沒內含小遮罩 → 這視角無從比較,跳過
                continue
            feats = RM.encode_debias(V["rgb"], blocks, model, prep, f_bg, device)
            base = len(block_feats)
            block_feats.extend(feats)
            # voxel 投影落哪塊
            H, W = V["hw"]; K = cam.intrinsics(W, H)
            X = P @ V["Rwc"].T + V["t"]; zc = X[:, 2]; ok = zc > 1e-9; zz = np.where(ok, zc, 1.0)
            u = np.round(K[0, 0] * X[:, 0] / zz + K[0, 2]).astype(int)
            v = np.round(K[1, 1] * X[:, 1] / zz + K[1, 2]).astype(int)
            inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            idx = np.where(inb)[0]
            for p in idx:
                if V["arm"] is not None and V["arm"][v[p], u[p]]:
                    continue
                for bi, bm in enumerate(blocks):
                    if bm[v[p], u[p]]:
                        vox_blocks[p].append(base + bi)
                        break

        if len(block_feats) < 2:                   # 整個 instance 沒有可比的塊 → 不分
            new_labels[tuple(vox.T)] = next_id + 1; next_id += 1
            if diag:
                print(f"  inst{k}: 塊<2,不分 (nk={nk})")
            continue
        F = np.array(block_feats)
        Z = linkage(pdist(F, metric="cosine"), method="average")
        cl = fcluster(Z, t=sem_thr, criterion="distance")
        g = int(cl.max())
        if diag:
            print(f"  inst{k}: nk={nk} 塊={len(block_feats)} → {g} 群  群大小={np.bincount(cl)[1:].tolist()}")
        if g == 1:
            new_labels[tuple(vox.T)] = next_id + 1; next_id += 1
            continue
        # voxel 投票歸群
        assign = np.zeros(nk, int)
        for p in range(nk):
            if vox_blocks[p]:
                assign[p] = int(np.bincount([cl[b] for b in vox_blocks[p]], minlength=g + 1).argmax())
        if (assign == 0).any() and (assign > 0).any():
            maj = int(np.bincount(assign[assign > 0], minlength=g + 1).argmax())
            assign[assign == 0] = maj
        for grp in range(1, g + 1):
            sel = assign == grp
            if sel.sum():
                next_id += 1
                new_labels[tuple(vox[sel].T)] = next_id

    return new_labels, grid_min, vs, shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--sem-thr", type=float, default=0.35, dest="sem_thr")
    ap.add_argument("--root", default="srp_hull_split")
    ap.add_argument("--diag", action="store_true", help="只印診斷(每 instance 分幾群),不寫檔")
    args = ap.parse_args()
    import clip, torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, prep = clip.load("ViT-B/32", device=device); model.eval()
    f_bg = RM.get_f_bg(model, prep, device)
    out_root = REPO / "data" / "eval" / args.root
    for sc in args.scenes:
        print(f"[{sc}]")
        r = process(sc, model, prep, f_bg, device, args.sem_thr, diag=args.diag)
        if r is None or args.diag:
            continue
        new_labels, grid_min, vs, shape = r
        od = out_root / sc; od.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(od / "instances.npz", labels=new_labels, grid_min=grid_min, voxel_size=vs)
        insts = [{"instance": i, "n_vox": int((new_labels == i).sum())}
                 for i in range(1, int(new_labels.max()) + 1) if (new_labels == i).any()]
        (od / "instances.json").write_text(json.dumps(
            {"scene": sc, "voxel": vs, "n_instances": len(insts), "instances": insts},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  → {len(insts)} instance → {od}")


if __name__ == "__main__":
    main()
