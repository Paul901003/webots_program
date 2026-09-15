#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""split_puremask.py — 純遮罩 CLIP 語意分群 → voxel 切分黏連 instance。

對每個 instance:各視角「遮罩內有 inst voxel 投影點」的 SAM 遮罩全納入 → CLIP 特徵
(square_mean_crop 填正確 CLIP mean → 224 → CLIP normalize → encode_image → 去偏 f_bg)
→ 凝聚聚類(average, 1−cos, 門檻 SEM_THR)→ 每 voxel 各視角落的遮罩都投其群、取最高票 → 切成 g 塊。
單物體過分裂先不處理。輸出 instances.npz/json(可餵 eval.py)。需 open_clip。
用法: ./srp/stage2_instances/split_puremask.py stack3_scene0001 [stack3 stack4 stack5] [--sem-thr 0.3] [--root srp_hull_puremask]
env: CAPTURES_ROOT SAM_ROOT HULL_ROOT ARM_MASK_ROOT
"""
import argparse, os, sys, json, glob
import numpy as np, cv2, torch, open_clip
from pathlib import Path
from PIL import Image
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "srp" / "io"))
import camera as cam, masks as MK

REPO = Path(__file__).resolve().parents[2]
CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures")))
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "sam_only")))
HULL_ROOT = Path(os.environ.get("HULL_ROOT", str(REPO / "data" / "eval" / "srp_hull")))
ARM = Path(os.environ.get("ARM_MASK_ROOT", str(REPO / "data" / "eval" / "srp_arm_masks")))
dev = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model = clip_model.to(dev).eval()
CMEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CSTD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
CLIP_FILL = np.array([123, 117, 104], np.uint8)   # round(CLIP mean×255):填後 CLIP normalize≈0
MIN_VOX = 20


def sqcrop(rgb, seg):
    ys, xs = np.nonzero(seg)
    if xs.size == 0: return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    c = rgb[y0:y1, x0:x1].copy(); c[~seg[y0:y1, x0:x1]] = CLIP_FILL
    h, w = c.shape[:2]; s = max(h, w); cv = np.empty((s, s, 3), np.uint8); cv[:] = CLIP_FILL
    oy, ox = (s - h) // 2, (s - w) // 2; cv[oy:oy + h, ox:ox + w] = c
    return cv


@torch.no_grad()
def clip_feats(rgb, segs):
    ims, val = [], []
    for i, s in enumerate(segs):
        c = sqcrop(rgb, s)
        if c is not None:
            im = cv2.resize(c, (224, 224))
            ims.append((torch.from_numpy(im).permute(2, 0, 1).float() / 255 - CMEAN) / CSTD); val.append(i)
    out = [None] * len(segs)
    if ims:
        f = clip_model.encode_image(torch.stack(ims).to(dev)).float()
        f = (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()
        for k, i in enumerate(val): out[i] = f[k]
    return out


@torch.no_grad()
def _fbg():
    im = np.empty((100, 100, 3), np.uint8); im[:] = CLIP_FILL
    x = (torch.from_numpy(cv2.resize(im, (224, 224))).permute(2, 0, 1).float() / 255 - CMEAN) / CSTD
    f = clip_model.encode_image(x[None].to(dev)).float()[0]
    return (f / f.norm()).cpu().numpy()
F_BG = _fbg()


def debias(F):
    F2 = F - (F @ F_BG)[:, None] * F_BG[None, :]
    return F2 / (np.linalg.norm(F2, axis=1, keepdims=True) + 1e-9)


def load_arm(scene, view):
    p = ARM / scene / f"{view}_arm.png"
    a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.is_file() else None
    return (a > 127) if a is not None else None


def process(scene, sem_thr, out_root):
    hp = HULL_ROOT / scene / "hull.npz"; ip = HULL_ROOT / scene / "instances.npz"
    if not (hp.is_file() and ip.is_file()):
        print(f"[skip] {scene}: 缺 hull/instances"); return None
    z = np.load(hp); occ = z["occupancy"]; gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = occ.shape
    labels = np.load(ip)["labels"]
    group = scene.split("_")[0]; sdir = CAPTURES / f"multi_{group}" / scene
    # 預載每視角:遮罩、pose、arm、原圖
    views = []
    for vd in sorted((SAM_ROOT / scene).glob("view_*")):
        pf = sdir / f"{vd.name}_pose.json"
        if not pf.is_file(): continue
        km = MK.kept_object_masks(vd)
        if not km: continue
        C, Rb = cam.load_pose(pf); Rwc, t = cam.pose_to_w2c(C, Rb)
        rgb = cv2.cvtColor(cv2.imread(str(sdir / f"{vd.name}.png")), cv2.COLOR_BGR2RGB)
        views.append({"ms": [m for m, _ in km], "Rwc": Rwc, "t": t,
                      "arm": load_arm(scene, vd.name), "rgb": rgb, "hw": km[0][0].shape})
    new_labels = np.zeros(shape, np.int32); nid = 0
    for k in range(1, int(labels.max()) + 1):
        vox = np.array(np.nonzero(labels == k)).T; nk = len(vox)
        if nk == 0: continue
        P = gm + (vox + 0.5) * vs
        feats, mask_pix = [], []   # 每覆蓋遮罩的 CLIP 特徵 + (view_idx, mask)
        for vi, V in enumerate(views):
            H, W = V["hw"]; K = cam.intrinsics(W, H)
            X = P @ V["Rwc"].T + V["t"]; zc = X[:, 2]; ok = zc > 1e-9; zz = np.where(ok, zc, 1.0)
            u = np.round(K[0, 0] * X[:, 0] / zz + K[0, 2]).astype(int); v = np.round(K[1, 1] * X[:, 1] / zz + K[1, 2]).astype(int)
            inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H); idx = np.where(inb)[0]
            if len(idx) == 0: continue
            proj = np.zeros((H, W), bool); proj[v[idx], u[idx]] = True
            cover = [m for m in V["ms"] if int((m & proj).sum()) > 0]
            if not cover: continue
            fe = clip_feats(V["rgb"], cover)
            for m, f in zip(cover, fe):
                if f is not None:
                    feats.append(f); mask_pix.append((vi, m, u, v, idx))
        if len(feats) < 2:
            new_labels[tuple(vox.T)] = nid + 1; nid += 1; continue
        cl = fcluster(linkage(pdist(debias(np.array(feats)), "cosine"), "average"), t=sem_thr, criterion="distance")
        g = int(cl.max())
        print(f"  inst{k}: 遮罩{len(feats)} → {g}群 大小{np.bincount(cl)[1:].tolist()}", flush=True)
        if g == 1:
            new_labels[tuple(vox.T)] = nid + 1; nid += 1; continue
        # voxel 都投:落哪個遮罩就投該遮罩的群,取最高票
        votes = np.zeros((nk, g + 1), int)
        for (vi, m, u, v, idx), c in zip(mask_pix, cl):
            arm = views[vi]["arm"]
            for p in idx:
                if arm is not None and arm[v[p], u[p]]: continue
                if m[v[p], u[p]]: votes[p, c] += 1
        assign = votes.argmax(1)
        if (assign == 0).any() and (assign > 0).any():
            assign[assign == 0] = int(np.bincount(assign[assign > 0], minlength=g + 1).argmax())
        for grp in range(1, g + 1):
            sel = assign == grp
            if sel.sum() >= 1:
                nid += 1; new_labels[tuple(vox[sel].T)] = nid
    out_dir = out_root / scene; out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "instances.npz", labels=new_labels, grid_min=gm, voxel_size=vs)
    insts = [{"instance": i, "n_vox": int((new_labels == i).sum())} for i in range(1, int(new_labels.max()) + 1) if (new_labels == i).any()]
    (out_dir / "instances.json").write_text(json.dumps({"scene": scene, "voxel": vs, "n_instances": len(insts), "instances": insts}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{scene}] 原 instance {int(labels.max())} → split {len(insts)}", flush=True)
    return len(insts)


def resolve(targets):
    out = []
    for a in targets:
        if "scene" in a: out.append(a)
        else: out += [Path(p).parent.name for p in glob.glob(str(HULL_ROOT / f"{a}_scene*/instances.npz"))]
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+")
    ap.add_argument("--sem-thr", type=float, default=0.3, dest="sem_thr")
    ap.add_argument("--root", default="srp_hull_puremask")
    args = ap.parse_args()
    out_root = REPO / "data" / "eval" / args.root
    for sc in resolve(args.targets):
        try: process(sc, args.sem_thr, out_root)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")


if __name__ == "__main__":
    main()
