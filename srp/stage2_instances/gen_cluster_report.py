#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""gen_cluster_report.py — 連通前語意群版報告:每個 mask_cluster 群的遮罩縮圖+影像+群內CLIP相似度熱圖。

讀 instances.json 的 mask_clusters(遮罩→群 id,連通前),按【語意群】分組顯示,和 gen_hull_report
(instance 版,連通後)對照——看「同一群的遮罩是不是同一個物體」。
輸出 data/eval/<root>/<scene>/cluster_report.html。
用法: SAM_ROOT=$PWD/data/eval/mobilesamv2_fast ./gen_cluster_report.py <scene> --root <root>
"""
import argparse, os, sys, json
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import masks as MK              # noqa: E402
import mask_clip_cluster as MC  # noqa: E402  F_BG(去偏,與 voxel_sem_cluster 同式)
import gen_hull_report as GHR   # noqa: E402  重用 thumb_b64 / heatmap_b64

_BG = MC.F_BG.astype(np.float64)


def debias(F):
    F = F.astype(np.float64)
    F = F - (F @ _BG)[:, None] * _BG[None, :]
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


def build(scene, root):
    ij = REPO / "data" / "eval" / root / scene / "instances.json"
    if not ij.is_file():
        print(f"[skip] {scene}: 無 {ij}"); return
    mc = json.loads(ij.read_text()).get("mask_clusters", {})
    if not mc:
        print(f"[skip] {scene}: 無 mask_clusters"); return
    sam_root = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "mobilesamv2_fast")))
    group = scene.split("_")[0]; cap = REPO / "data" / "captures_fast" / f"multi_{group}" / scene
    clu = defaultdict(list)                       # 群 id → [(view, name)]
    for view, mm in mc.items():
        for name, cid in mm.items():
            clu[int(cid)].append((view, name))
    vcache = {}; rgbcache = {}
    def getv(view):
        if view not in vcache:
            vd = sam_root / scene / view
            km = MK.kept_object_masks(vd); names = [n for _, n in km]
            feats = list(MK.feats_list(vd, names, feat_file="clip_mean_feats.npy"))
            vcache[view] = ({n: m for m, n in km},
                            {names[i]: feats[i] for i in range(len(names))})
        if view not in rgbcache:
            p = cap / f"{view}.png"
            rgbcache[view] = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) if p.is_file() else None
        return vcache[view], rgbcache[view]

    parts = [f"<h2>{scene} — {root}｜連通前語意群 共 {len(clu)} 群</h2>",
             "<p>每一區=一個語意群(mask_cluster);縮圖從原圖裁遮罩 bbox。看同群是不是同一物體。</p>"]
    for cid in sorted(clu, key=lambda c: -len(clu[c])):
        items = sorted(clu[cid])
        thumbs = []; feats = []; labels = []
        for view, name in items:
            (mdict, fdict), rgb = getv(view)
            m = mdict.get(name); f = fdict.get(name)
            if m is None or rgb is None:
                continue
            th = GHR.thumb_b64(rgb, m)
            if th:
                thumbs.append((th, view, name))
            if f is not None:
                feats.append(np.asarray(f, np.float64)); labels.append(view.replace("view_", ""))
        parts.append(f"<h3>群 {cid} — {len(items)} 遮罩</h3>")
        parts.append("<div style='display:flex;flex-wrap:wrap;gap:4px;align-items:flex-end'>")
        rptdir = REPO / "data" / "eval" / root / scene
        for th, view, name in thumbs:
            mpng = sam_root / scene / view / "masks" / name
            rel = os.path.relpath(mpng, rptdir) if mpng.is_file() else ""
            relrgb = os.path.relpath(cap / f"{view}.png", rptdir)
            vs = view.replace("view_", "")
            oi = f"<a href='{rel}' target='_blank'>" if rel else ""; ci = "</a>" if rel else ""
            parts.append(f"<div style='text-align:center;font-size:9px;color:#666'>"
                         f"{oi}<img src='data:image/png;base64,{th}'>{ci}<br>{vs} "
                         f"<a href='{relrgb}' target='_blank' style='color:#888'>原圖</a></div>")
        parts.append("</div>")
        if len(feats) >= 2:
            heat = GHR.heatmap_b64(debias(np.array(feats)), labels)
            parts.append(f"<img src='data:image/png;base64,{heat}' style='max-width:560px;margin:6px 0'>")
    out = REPO / "data" / "eval" / root / scene / "cluster_report.html"
    out.write_text("<html><meta charset='utf-8'><body style='font-family:sans-serif'>"
                   + "\n".join(parts) + "</body></html>", encoding="utf-8")
    print(f"[{scene}] 連通前 {len(clu)} 群 → {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene"); ap.add_argument("--root", required=True)
    a = ap.parse_args(); build(a.scene, a.root)


if __name__ == "__main__":
    main()
