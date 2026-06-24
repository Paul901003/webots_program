#!/home/cho/.pyenv/versions/grounded_sam/bin/python3
"""filter_candidates_clip.py — 程式②:用 CLIP 外觀特徵驗證/拆分候選 hull。

讀程式① voxel_candidates.py 的 candidates.json(每個 hull 各視角對應的 mask),
對每塊 mask 用 CLIP(ViT-B/32)抽特徵(摳出遮罩區域→encode→L2 normalize)。
每個 hull 內,把所有 mask 依「餘弦相似度 >= --sim」單連結分群:
  - 全部夠相似 → 一群 → 一個物體(幾何幻影橋若把外觀不同的兩物焊在一起,這裡會被拆)。
  - 分成多群 → 各群=各自物體(中心用該群 mask 質心射線最小平方交會重算)。

輸出: data/eval/voxel_clip/<scene>/instances.json(schema 同其他關聯法,可直接餵 eval/carve)。
需在 grounded_sam 環境(clip/torch/cv2)。重用 associate.py 的射線幾何。
用法: ./instance_hull/filter_candidates_clip.py n5_scene0031 [--sim 0.85]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import clip

sys.path.insert(0, str(Path(__file__).resolve().parent))
import associate as ag   # noqa: E402  (pixel_to_ray / load_pose / intrinsics)

REPO = Path(__file__).resolve().parents[1]
CAPTURES = REPO / "data" / "captures"
SAM_ROOT = REPO / "data" / "eval" / "sam_only"
CAND_ROOT = REPO / "data" / "eval" / "voxel_candidates"
OUT_ROOT = REPO / "data" / "eval" / "voxel_clip"


def masked_crop(rgb, seg):
    ys, xs = np.nonzero(seg)
    if xs.size == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = rgb[y0:y1 + 1, x0:x1 + 1].copy()
    m = seg[y0:y1 + 1, x0:x1 + 1]
    crop[~m] = 0                                  # 背景塗黑,聚焦物體
    return Image.fromarray(crop)


def ls_ray_intersect(origins, dirs):
    """多條射線最小平方交會點。origins,dirs: (k,3)。"""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d in zip(origins, dirs):
        d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ o
    return np.linalg.solve(A + 1e-9 * np.eye(3), b)


def uf_components(n, edges):
    p = list(range(n))
    def f(x):
        while p[x] != x:
            p[x] = p[p[x]]; x = p[x]
        return x
    for a, b in edges:
        p[f(a)] = f(b)
    comp = {}
    for i in range(n):
        comp.setdefault(f(i), []).append(i)
    return list(comp.values())


def process_scene(scene, model, preprocess, device, args):
    group = scene.split("_")[0]
    scene_dir = CAPTURES / f"multi_{group}" / scene
    cand_path = CAND_ROOT / scene / "candidates.json"
    if not cand_path.is_file():
        print(f"[skip] {scene}: 找不到 {cand_path}(先跑 voxel_candidates.py)")
        return
    cands = json.loads(cand_path.read_text())["candidates"]
    out_dir = OUT_ROOT / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    # 快取每視角 RGB、pose、內參
    rgb_c, pose_c, K_c = {}, {}, {}
    def get_view(vn):
        if vn not in rgb_c:
            img = cv2.imread(str(scene_dir / f"{vn}.png"))
            rgb_c[vn] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
            pp = scene_dir / f"{vn}_pose.json"
            C, R = ag.load_pose(pp); pose_c[vn] = (C, R)
            H, W = (rgb_c[vn].shape[:2] if rgb_c[vn] is not None else (720, 1280))
            K_c[vn] = ag.intrinsics(W, H)
        return rgb_c[vn], pose_c[vn], K_c[vn]

    instances = []
    report = [f"scene: {scene}  (CLIP filter, sim={args.sim})", f"候選 hull: {len(cands)}", ""]
    for hull in cands:
        items = []   # (view, file, feat, centroid_uv)
        feats = []
        for vn, files in hull["masks"].items():
            rgb, _, _ = get_view(vn)
            if rgb is None:
                continue
            for f in files:
                seg = cv2.imread(str(SAM_ROOT / scene / vn / "masks" / f), cv2.IMREAD_GRAYSCALE)
                if seg is None:
                    continue
                seg = seg > 127
                crop = masked_crop(rgb, seg)
                if crop is None:
                    continue
                ys, xs = np.nonzero(seg)
                with torch.no_grad():
                    t = preprocess(crop).unsqueeze(0).to(device)
                    fe = model.encode_image(t).float()
                    fe = (fe / fe.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
                items.append((vn, f, (float(xs.mean()), float(ys.mean()))))
                feats.append(fe)
        if not feats:
            continue
        feats = np.stack(feats)
        sim = feats @ feats.T                      # 餘弦相似度矩陣(已normalize)
        # 單連結分群:相似度 >= sim 連邊
        edges = [(i, j) for i in range(len(feats)) for j in range(i + 1, len(feats))
                 if sim[i, j] >= args.sim]
        comps = uf_components(len(feats), edges)
        report.append(f"hull_{hull['id']:02d}: {len(feats)} masks → {len(comps)} 物體 "
                      f"(內部平均相似度 {sim[np.triu_indices(len(feats),1)].mean():.3f})"
                      if len(feats) > 1 else f"hull_{hull['id']:02d}: 1 mask")
        for comp in comps:
            if len(comp) < args.min_masks:
                continue
            per_view = {}
            origins, dirs = [], []
            for idx in comp:
                vn, f, (u, v) = items[idx]
                per_view.setdefault(vn, []).append(f)
                C, R = pose_c[vn]
                origins.append(C); dirs.append(ag.pixel_to_ray(u, v, C, R, K_c[vn]))
            if len(per_view) < 1:
                continue
            center = ls_ray_intersect(np.array(origins), np.array(dirs)) if len(origins) >= 2 \
                else np.array(hull["center"])
            instances.append({"center": [round(float(x), 4) for x in center],
                              "support": len(per_view), "masks": per_view})

    instances.sort(key=lambda a: -a["support"])
    # GT 參考
    gt = []
    mani = scene_dir / "scene_manifest.json"
    if mani.is_file():
        for o in json.loads(mani.read_text())["actual"]["viewpoints"][0]["objects"]:
            gt.append((o["name"], np.array(o["position_m"])))
    report += ["", f"最終 instances: {len(instances)}  (GT 物體數: {len(gt)})"]
    for k, inst in enumerate(instances):
        c = np.array(inst["center"]); line = f"inst_{k:02d}: support={inst['support']}"
        if gt:
            nm, dmin = min(((n, float(np.linalg.norm(c - p))) for n, p in gt), key=lambda a: a[1])
            line += f"  最近GT={nm} ({dmin*100:.1f}cm)"
        report.append(line)
    txt = "\n".join(report); print(txt)
    (out_dir / "instances.json").write_text(json.dumps(
        {"scene": scene, "method": "voxel_clip",
         "centers": [i["center"] for i in instances], "instances": instances},
        indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "filter_report.txt").write_text(txt + "\n", encoding="utf-8")
    print(f"→ {out_dir}/instances.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["n5_scene0031"])
    ap.add_argument("--sim", type=float, default=0.85, help="餘弦相似度門檻(>=此值算同物體)")
    ap.add_argument("--min-masks", type=int, default=1, dest="min_masks", help="一個物體最少 mask 數")
    args = ap.parse_args()
    scenes = ag.resolve_scenes(args.scenes or ["n5_scene0031"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    if not scenes:
        sys.exit("沒有場景")
    for i, scene in enumerate(scenes, 1):
        print(f"\n===== [{i}/{len(scenes)}] {scene} =====")
        try:
            process_scene(scene, model, preprocess, device, args)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[error] {scene}: {e}")


if __name__ == "__main__":
    main()
