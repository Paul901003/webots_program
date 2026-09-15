#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""gen_donut_report.py — donut(去大遮罩)版連通前語意群報告。

同 gen_cluster_report.py,但:縮圖顯示【挖洞後(甜甜圈)遮罩】、熱圖用【重算的甜甜圈 CLIP 去偏特徵】,
與 voxel_sem_cluster_donut.py 的分群一致 → 看「去除大遮罩後同群是不是同一物體、大遮罩是否被挖空」。
輸出 data/eval/<root>/<scene>/donut_cluster_report.html。
用法: SAM_ROOT=$PWD/data/eval/mobilesamv2_fast ./gen_donut_report.py <scene> --root <root>
"""
import argparse, os, sys, json
from collections import defaultdict
from pathlib import Path
import numpy as np, cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import masks as MK               # noqa: E402
import mask_clip_cluster as MC   # noqa: E402
import gen_hull_report as GHR    # noqa: E402  重用 thumb_b64 / heatmap_b64
from voxel_sem_cluster_donut import donut_masks, donut_feats   # noqa: E402  同一挖洞+特徵快取

_BG = MC.F_BG.astype(np.float64)


def debias(F):
    F = F.astype(np.float64)
    F = F - (F @ _BG)[:, None] * _BG[None, :]
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


def build(scene, root):
    ij = REPO / "data" / "eval" / root / scene / "instances.json"
    if not ij.is_file():
        print(f"[skip] {scene}: 無 {ij}"); return
    html = REPO / "data" / "eval" / root / scene / "donut_cluster_report.html"
    if html.is_file() and os.environ.get("FORCE", "") != "1":
        return   # 續跑保護:已產報告就跳過(並行 worker + 序列步驟2 不重複;FORCE=1 強制重做)
    mc = json.loads(ij.read_text()).get("mask_clusters", {})
    if not mc:
        print(f"[skip] {scene}: 無 mask_clusters"); return
    sam_root = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "mobilesamv2_fast")))
    group = scene.split("_")[0]; cap = REPO / "data" / "captures_fast" / f"multi_{group}" / scene
    clu = defaultdict(list)                       # 群 id → [(view, name)]
    for view, mm in mc.items():
        for name, cid in mm.items():
            clu[int(cid)].append((view, name))
    rptdir = REPO / "data" / "eval" / root / scene
    donutdir = rptdir / "donut_masks"; donutdir.mkdir(parents=True, exist_ok=True)  # 存挖洞後遮罩 PNG 供超連結
    vcache = {}; rgbcache = {}
    def getv(view):
        """回 ({name: 挖洞後遮罩}, {name: 重算去偏特徵}), rgb;順便把挖洞遮罩存 PNG(donut_masks/<view>_<name>)。"""
        if view not in vcache:
            vd = sam_root / scene / view
            km = MK.kept_object_masks(vd); names = [n for _, n in km]
            ms0 = [m for m, _ in km]
            ms = donut_masks(ms0)                 # ★挖洞後遮罩(與分群一致)
            p = cap / f"{view}.png"
            rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) if p.is_file() else None
            fmap = donut_feats(vd, rgb, ms, names) if rgb is not None else {}  # ★讀快取(分群已存),不重算
            for i in range(len(names)):           # 存挖洞後遮罩 PNG(超連結用,看得到甜甜圈)
                cv2.imwrite(str(donutdir / f"{view}_{names[i]}"), (ms[i].astype(np.uint8) * 255))
            vcache[view] = ({names[i]: ms[i] for i in range(len(names))},
                            {names[i]: fmap.get(names[i]) for i in range(len(names))})
            rgbcache[view] = rgb
        return vcache[view], rgbcache[view]

    parts = [f"<h2>{scene} — {root}｜去大遮罩(甜甜圈)後 連通前語意群 共 {len(clu)} 群</h2>",
             "<p>縮圖=挖洞後遮罩(大遮罩已減掉被包含的小遮罩)。看同群是不是同一物體、大遮罩是否被挖空。</p>"]
    for cid in sorted(clu, key=lambda c: -len(clu[c])):
        items = sorted(clu[cid])
        thumbs = []; feats = []; labels = []
        for view, name in items:
            (mdict, fdict), rgb = getv(view)
            m = mdict.get(name); f = fdict.get(name)
            if m is None or rgb is None or int(m.sum()) == 0:
                continue
            th = GHR.thumb_b64(rgb, m)
            if th:
                thumbs.append((th, view, name))
            if f is not None:
                feats.append(np.asarray(f, np.float64)); labels.append(view.replace("view_", ""))
        parts.append(f"<h3>群 {cid} — {len(items)} 遮罩</h3>")
        parts.append("<div style='display:flex;flex-wrap:wrap;gap:4px;align-items:flex-end'>")
        for th, view, name in thumbs:
            reldonut = os.path.relpath(donutdir / f"{view}_{name}", rptdir)  # 挖洞後遮罩 PNG
            relrgb = os.path.relpath(cap / f"{view}.png", rptdir)
            vs = view.replace("view_", "")
            parts.append(f"<div style='text-align:center;font-size:9px;color:#666'>"
                         f"<a href='{reldonut}' target='_blank'><img src='data:image/png;base64,{th}'></a><br>{vs} "
                         f"<a href='{relrgb}' target='_blank' style='color:#888'>原圖</a></div>")
        parts.append("</div>")
        if len(feats) >= 2:
            heat = GHR.heatmap_b64(debias(np.array(feats)), labels)
            parts.append(f"<img src='data:image/png;base64,{heat}' style='max-width:560px;margin:6px 0'>")
    out = REPO / "data" / "eval" / root / scene / "donut_cluster_report.html"
    out.write_text("<html><meta charset='utf-8'><body style='font-family:sans-serif'>"
                   + "\n".join(parts) + "</body></html>", encoding="utf-8")
    print(f"[{scene}] 去大遮罩 連通前 {len(clu)} 群 → {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="+")   # 場景名/群名(如 stack3)/多個;一個進程跑完,CLIP 模型只載一次
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    base = REPO / "data" / "eval" / a.root
    scenes = []
    for tg in a.targets:
        if "scene" in tg:
            scenes.append(tg)
        else:                                # 群名 → 展開該組所有已產 instances 的場
            scenes += [Path(p).parent.name for p in
                       __import__("glob").glob(str(base / f"{tg}_scene*/instances.json"))]
    for sc in sorted(set(scenes)):
        try:
            build(sc, a.root)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")


if __name__ == "__main__":
    main()
