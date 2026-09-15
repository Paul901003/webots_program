#!/home/cho/.pyenv/versions/grounded_sam/bin/python3
"""refine_masks.py — 用 CLIP 語意拆分 SAM 巢狀大遮罩(切開黏連 hull 的前處理)。

SAM(AMG)常把多物體圈成一個大遮罩、內含小遮罩(巢狀)。本程式:
  每視角找「大遮罩 A 內含子遮罩 B_i」→ 比較 各子遮罩 B_i vs 剩餘 R=A−∪B_i 的 CLIP 語意
  (去偏:扣掉灰底共性向量 f_bg)→ 若彼此語意差異大(max cos 距離 > SEM_THR)判 A 為多物體混合
  → 拆成 {R, B_i} 各自獨立取代 A。輸出細分遮罩集(結構同 sam_only)供下游 associate 切開。

輸出 SAM_OUT_ROOT/<scene>/<view>/masks/mask_*.png(+meta.txt)。需 grounded_sam 環境(clip/torch/cv2)。
用法: ./srp/stage2_instances/refine_masks.py stack3_scene0001 [stack3] [--sem-thr 0.35]
env: CAPTURES_ROOT SAM_ROOT(輸入遮罩) SAM_OUT_ROOT(輸出,預設 sam_only_sem) FORCE
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import clip

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "instance_hull"))
import masks as MK                                   # noqa: E402
from precompute_clip import square_mean_crop, CLIP_MEAN  # noqa: E402  (crop 邏輯複用)

CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures")))
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "sam_only")))
OUT_ROOT = Path(os.environ.get("SAM_OUT_ROOT", str(REPO / "data" / "eval" / "sam_only_sem")))
BG_FEAT = REPO / "data" / "eval" / "clip_bg_feat.npy"
FORCE = os.environ.get("FORCE") == "1"
CONTAIN = 0.8      # 子遮罩 >此比例落在大遮罩內 = 被包含
MIN_AREA = 500     # 剩餘/子遮罩最小面積(丟碎片)


def resolve_scenes(targets):
    out = []
    for a in targets:
        if "scene" in a:
            out.append(a)
        else:
            g = f"n{a}" if a.isdigit() else a
            out += [d.name for d in sorted((CAPTURES / f"multi_{g}").glob(f"{g}_scene*"))]
    return out


def get_f_bg(model, prep, device):
    if BG_FEAT.is_file():
        return np.load(BG_FEAT)
    bg = np.empty((100, 100, 3), np.uint8); bg[:] = CLIP_MEAN
    from PIL import Image
    with torch.no_grad():
        f = model.encode_image(prep(Image.fromarray(bg)).unsqueeze(0).to(device)).float().cpu().numpy()[0]
    f = f / np.linalg.norm(f)
    BG_FEAT.parent.mkdir(parents=True, exist_ok=True); np.save(BG_FEAT, f)
    return f


@torch.no_grad()
def encode_debias(rgb, segs, model, prep, f_bg, device):
    """各 seg → square_mean_crop → CLIP → 去偏(扣 f_bg) → L2 norm。回 (n,512)。"""
    crops = []
    for s in segs:
        c = square_mean_crop(rgb, s)
        crops.append(prep(c) if c is not None else None)
    feats = np.zeros((len(segs), 512), np.float32)
    valid = [i for i, c in enumerate(crops) if c is not None]
    if valid:
        b = torch.stack([crops[i] for i in valid]).to(device)
        f = model.encode_image(b).float().cpu().numpy()
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
        f = f - (f @ f_bg)[:, None] * f_bg[None, :]          # 去偏(正交投影)
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
        for k, i in enumerate(valid):
            feats[i] = f[k]
    return feats


def refine_view(rgb, kept, model, prep, f_bg, sem_thr, device):
    """kept: [(bool_mask, name)]。回傳細分後的遮罩 list[bool]。"""
    ms = [m for m, _ in kept]
    n = len(ms)
    areas = [int(m.sum()) for m in ms]
    # 找巢狀:children[i] = 落在 ms[i] 內的較小遮罩 index
    children = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i == j or areas[j] >= areas[i] or areas[j] == 0:
                continue
            inter = int((ms[i] & ms[j]).sum())
            if inter and inter / areas[j] > CONTAIN:
                children[i].append(j)

    drop = set()      # 被判多物體、要移除的大遮罩 A
    extra = []        # 新增的剩餘遮罩 R
    for i, ch in children.items():
        if not ch:
            continue
        subs = [ms[j] for j in ch]
        union = np.logical_or.reduce(subs)
        R = ms[i] & ~union
        blocks, tags = [], []
        for s in subs:
            blocks.append(s); tags.append("sub")
        if int(R.sum()) >= MIN_AREA:
            blocks.append(R); tags.append("R")
        if len(blocks) < 2:
            continue
        feats = encode_debias(rgb, blocks, model, prep, f_bg, device)
        # 兩兩 cos 距離最大值
        mx = 0.0
        for a in range(len(blocks)):
            for b in range(a + 1, len(blocks)):
                if feats[a].any() and feats[b].any():
                    mx = max(mx, 1.0 - float(feats[a] @ feats[b]))
        if mx > sem_thr:                # 多物體混合 → 拆
            drop.add(i)
            if int(R.sum()) >= MIN_AREA:
                extra.append(R)

    out = [ms[k] for k in range(n) if k not in drop]   # 原遮罩保留(除非被拆);不對原遮罩做面積過濾
    out += extra                                        # 加入剩餘 R(已在上面過 MIN_AREA)
    return out, len(drop)


def process_scene(scene, model, prep, f_bg, sem_thr, device):
    g = scene.split("_")[0]
    sdir = CAPTURES / f"multi_{g}" / scene
    sam = SAM_ROOT / scene
    if not sam.is_dir():
        print(f"[skip] {scene}: 無 SAM"); return
    n_split = 0
    for vdir in sorted(sam.glob("view_*")):
        out_dir = OUT_ROOT / scene / vdir.name
        if (out_dir / "masks").is_dir() and not FORCE:
            continue
        img = cv2.imread(str(sdir / f"{vdir.name}.png"))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        kept = MK.kept_object_masks(vdir)
        refined, ndrop = refine_view(rgb, kept, model, prep, f_bg, sem_thr, device) if kept else ([], 0)
        refined.sort(key=lambda m: -int(m.sum()))
        md = out_dir / "masks"; md.mkdir(parents=True, exist_ok=True)
        meta = []
        for i, m in enumerate(refined):
            cv2.imwrite(str(md / f"mask_{i:03d}.png"), (m.astype(np.uint8) * 255))
            meta.append(f"mask_{i:03d}  area={int(m.sum()):8d}")
        (out_dir / "meta.txt").write_text("\n".join(meta), encoding="utf-8")
        n_split += ndrop
    print(f"[{scene}] 完成(拆開 {n_split} 個混合大遮罩)→ {OUT_ROOT / scene}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+")
    ap.add_argument("--sem-thr", type=float, default=0.35, dest="sem_thr")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, prep = clip.load("ViT-B/32", device=device); model.eval()
    f_bg = get_f_bg(model, prep, device)
    scenes = resolve_scenes(args.targets)
    print(f"refine_masks: {len(scenes)} 場景  sem_thr={args.sem_thr}  → {OUT_ROOT}")
    for sc in scenes:
        try:
            process_scene(sc, model, prep, f_bg, args.sem_thr, device)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")


if __name__ == "__main__":
    main()
