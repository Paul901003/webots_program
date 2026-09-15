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


def group_voxel_overlap(scene, mc):
    """算每個語意群的 voxel 集合(多數決歸屬)+ 群間重疊矩陣。回 (cids, Mcount, Mjac, gsize)。"""
    import camera as cam, viewpoints as VP  # noqa
    import cg_associate as CG               # noqa
    from voxel_sem_cluster_donut import donut_masks
    from collections import Counter
    HB = REPO / "data" / "eval" / "srp_hull_mv2_v12_am1"
    hp = HB / scene / "hull.npz"
    if not hp.is_file():
        return None
    z = np.load(hp); surf = z["surface"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
    voxarr = np.array(np.nonzero(surf)).T; P = gm + (voxarr + 0.5) * vs
    sam_root = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "mobilesamv2_fast")))
    group = scene.split("_")[0]; sdir = REPO / "data" / "captures_fast" / f"multi_{group}" / scene
    gvox = defaultdict(set)
    for vn in sorted(VP.selected_view_names(12)):
        vd = sam_root / scene / vn; pf = sdir / f"{vn}_pose.json"
        if not (vd.is_dir() and pf.is_file()): continue
        km = MK.kept_object_masks(vd); names = [n for _, n in km]; ms0 = [m for m, _ in km]
        if not ms0: continue
        C, Rb = cam.load_pose(pf); H, W = ms0[0].shape
        vox_at = CG.zbuffer_visible(P, C, Rb, W, H, vs).reshape(H, W)
        vmc = defaultdict(Counter)
        ms = ms0  # ★不挖空(surf版)
        for mi, m in enumerate(ms):
            ys, xs = np.where(m); vv = vox_at[ys, xs]
            for v in vv[vv >= 0]: vmc[int(v)][mi] += 1
        mv = defaultdict(set)
        for v, cnt in vmc.items(): mv[cnt.most_common(1)[0][0]].add(v)
        cl = mc.get(vn, {})
        for mi, nm in enumerate(names):
            cid = cl.get(nm)
            if cid is not None: gvox[cid] |= mv.get(mi, set())
    cids = sorted([c for c in gvox if gvox[c]], key=lambda c: -len(gvox[c]))
    n = len(cids); Mcount = np.zeros((n, n), int); Mjac = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: continue
            a, b = gvox[cids[i]], gvox[cids[j]]
            inter = len(a & b); Mcount[i, j] = inter; Mjac[i, j] = inter / max(len(a | b), 1)
    return cids, Mcount, Mjac, {c: len(gvox[c]) for c in cids}


def matrix_html(scene, mc):
    r = group_voxel_overlap(scene, mc)
    if r is None: return ""
    cids, Mcount, Mjac, gsize = r
    out = ["<h3>語意群間 voxel 重疊矩陣(Jaccard 比例)</h3>",
           "<p>格子=兩群 voxel 重疊比例 Jaccard(交集/聯集,多數決歸屬);滑鼠移上看共享 voxel 數。"
           "紅底=Jaccard>0.05(=合併門檻,重疊高候選同物體)。群號與下方各群縮圖對應。</p>",
           "<table border=1 style='border-collapse:collapse;font-size:11px;text-align:center'>"]
    out.append("<tr><th>群\\群</th>" + "".join(f"<th>群{c}<br>({gsize[c]}vox)</th>" for c in cids) + "</tr>")
    for i, ci in enumerate(cids):
        row = [f"<th>群{ci}</th>"]
        for j in range(len(cids)):
            if i == j: row.append("<td style='background:#eee'>-</td>")
            else:
                bg = "#f88" if Mjac[i, j] > 0.05 else "#fff"
                row.append(f"<td style='background:{bg}' title='共享{Mcount[i,j]}vox'>{Mjac[i,j]:.3f}</td>")
        out.append("<tr>" + "".join(row) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def build(scene, root):
    ij = REPO / "data" / "eval" / root / scene / "instances.json"
    if not ij.is_file():
        print(f"[skip] {scene}: 無 {ij}"); return
    html = REPO / "data" / "eval" / root / scene / "surf_report_matrix.html"
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
            ms = ms0                               # ★不挖空(surf版)
            p = cap / f"{view}.png"
            rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) if p.is_file() else None
            fmap = {names[i]: (lambda a: None if a is None else a)(MK.mask_feats(vd).get(names[i])) for i in range(len(names))}  # ★預存clip_mean(不挖空)
            for i in range(len(names)):           # 存挖洞後遮罩 PNG(超連結用,看得到甜甜圈)
                cv2.imwrite(str(donutdir / f"{view}_{names[i]}"), (ms[i].astype(np.uint8) * 255))
            vcache[view] = ({names[i]: ms[i] for i in range(len(names))},
                            {names[i]: fmap.get(names[i]) for i in range(len(names))})
            rgbcache[view] = rgb
        return vcache[view], rgbcache[view]

    parts = [f"<h2>{scene} — {root}｜連通前語意群(不挖空) 共 {len(clu)} 群</h2>",
             "<p>縮圖=原始SAM遮罩(不挖空)。看同群是不是同一物體、大遮罩是否被挖空。</p>",
             matrix_html(scene, mc)]   # ★群間 voxel 重疊矩陣(群號與下方縮圖對應)
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
    out = REPO / "data" / "eval" / root / scene / "surf_report_matrix.html"
    out.write_text("<html><meta charset='utf-8'><body style='font-family:sans-serif'>"
                   + "\n".join(parts) + "</body></html>", encoding="utf-8")
    print(f"[{scene}] 連通前(不挖空) {len(clu)} 群 → {out}", flush=True)


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
