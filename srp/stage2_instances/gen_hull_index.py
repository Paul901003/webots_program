#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""gen_hull_index.py — 為某 instances root 產總索引 index.html(依組別排,每格=reproj縮圖+連report.html+鬼影統計)。

讀 data/eval/<root>/<scene>/{instances.npz, reproj_instances.png, report.html}。
每場統計:實例數 / GT物體數 / 鬼影數(與所有 GT mesh 3D 重疊=0 的實例)。
輸出 data/eval/<root>/index.html(縮圖用相對路徑,本機瀏覽)。
用法: ./gen_hull_index.py --root srp_hull_semcluster_surf_am1photo
"""
import argparse, sys, re
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/"srp/io")); sys.path.insert(0, str(REPO/"srp/stage2_instances"))
import eval_mesh as EM
EVAL = REPO/"data"/"eval"
GORDER = ["n1", "n3", "n4", "n5", "occ3", "occ4", "occ5", "stack3", "stack4", "stack5"]


def grp(sc):
    m = re.match(r"([a-z]+\d)", sc); return m.group(1) if m else "?"


def ghost_stats(root, sc):
    """回 (實例數, GT物體數, 鬼影數)。鬼影=與所有 GT mesh 重疊 0 的實例。"""
    p = EVAL/root/sc/"instances.npz"
    if not p.is_file(): return (0, 0, 0)
    try:
        z = np.load(p, allow_pickle=True); labels = z["labels"]
        gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = labels.shape
        labs = [int(x) for x in np.unique(labels) if x > 0]
        gtocc = EM.solid_mesh_occ(sc, gm, vs, shape)
        ng = 0
        for L in labs:
            m = (labels == L)
            if max((int((m & oc).sum()) for oc in gtocc.values()), default=0) == 0:
                ng += 1
        return (len(labs), len(gtocc), ng)
    except Exception:
        return (0, 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    base = EVAL/a.root
    scenes = sorted(p.parent.name for p in base.glob("*_scene*/instances.npz"))
    bygrp = {}
    for sc in scenes:
        bygrp.setdefault(grp(sc), []).append(sc)
    total_ghost = 0; total_over = 0
    cells = {}
    for g in GORDER:
        if g not in bygrp: continue
        items = []
        for sc in sorted(bygrp[g]):
            ni, ngt, ngh = ghost_stats(a.root, sc)
            total_ghost += ngh; total_over += max(ni-ngt, 0)
            has_png = (base/sc/"reproj_instances.png").is_file()
            has_rep = (base/sc/"report.html").is_file()
            thumb = f'<img src="{sc}/reproj_instances.png" loading="lazy">' if has_png else '<div class="noimg">no reproj</div>'
            link = f'{sc}/report.html' if has_rep else '#'
            flag = f'<span class="gh">鬼影{ngh}</span>' if ngh else ''
            over = f'<span class="ov">過切+{ni-ngt}</span>' if ni > ngt else ''
            items.append(f'<a class="cell" href="{link}"><div class="th">{thumb}</div>'
                         f'<div class="cap">{sc}<br>{ni}inst / {ngt}GT {over}{flag}</div></a>')
        cells[g] = items
    html = ['<meta charset="utf-8"><title>hull gallery</title>', '''<style>
body{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#ddd}
h2{border-bottom:1px solid #444;padding-bottom:4px;margin-top:28px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px}
.cell{display:block;background:#1c1c1c;border:1px solid #333;border-radius:6px;overflow:hidden;text-decoration:none;color:#ddd}
.cell:hover{border-color:#888}
.th img{width:100%;display:block}.noimg{padding:40px;text-align:center;color:#666}
.cap{padding:6px 8px;font-size:12px;line-height:1.4}
.gh{color:#f66;font-weight:bold;margin-left:6px}.ov{color:#fb0;margin-left:6px}
.summary{background:#1c1c1c;padding:10px;border-radius:6px;margin-bottom:10px}
</style>''']
    html.append(f'<div class="summary"><b>{a.root}</b> — {len(scenes)} 場 / 全體鬼影實例 {total_ghost} / 過切超出 GT {total_over}。'
                f'點縮圖進該場 report.html(hull+來源遮罩+CLIP)。紅=有鬼影,黃=過切。</div>')
    for g in GORDER:
        if g not in cells: continue
        html.append(f'<h2>{g} ({len(cells[g])} 場)</h2><div class="grid">')
        html.extend(cells[g]); html.append('</div>')
    out = base/"index.html"; out.write_text("\n".join(html), encoding="utf-8")
    print(f"index → {out}  ({len(scenes)} 場, 鬼影 {total_ghost})")


if __name__ == "__main__":
    main()
