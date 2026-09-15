#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""gen_hull_report.py — 每場景每方法產一份自帶 HTML 報告(hull 表 + 每 hull 遮罩相似度熱度圖)。

讀 data/eval/<root>/<scene>/instances.json(每 hull 的來源遮罩 {view:[mask]})、
SAM_ROOT 的遮罩+clip_mean_feats、captures 的 RGB、clip_text_feats.npz(64 名詞)。
每個 hull:
  · 顏色格(可視化 PALETTE 的 RGB 直接塗滿)
  · 來源遮罩:小縮圖(從原圖裁 bbox)→ 點了開該 mask PNG;附該遮罩 top1
  · hull 平均 CLIP 特徵 → 對 64 名詞 top1 / top5 phrase
  · 來源遮罩兩兩 cos 相似度熱度圖(base64 內嵌)
輸出 data/eval/<root>/<scene>/report.html。

SAM_ROOT 預設依 root 自動選(root 含 _mv2 → mobilesamv2_fast,否則 sam_only_fast),可用 env 覆寫。
用法: ./gen_hull_report.py [scene|group|(空=全部)] --root srp_hull_cg_mv2
env: SAM_ROOT CAPTURES_ROOT
"""
import argparse
import base64
import glob
import io
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
import masks as MK   # noqa: E402

CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures_fast")))

PALETTE = [[0.12, 0.47, 0.71], [1.0, 0.50, 0.05], [0.17, 0.63, 0.17], [0.84, 0.15, 0.16],
           [0.58, 0.40, 0.74], [0.55, 0.34, 0.29], [0.89, 0.47, 0.76], [0.74, 0.74, 0.13],
           [0.09, 0.75, 0.81], [0.5, 0.5, 0.5]]

_Z = np.load(REPO / "data" / "eval" / "clip_text_feats.npz", allow_pickle=True)
TPH = [str(x) for x in _Z["phrases"]]
TF = _Z["feats"].astype(np.float32); TF = TF / (np.linalg.norm(TF, axis=1, keepdims=True) + 1e-9)


def sam_root_for(root):
    env = os.environ.get("SAM_ROOT")
    if env:
        return Path(env)
    name = "mobilesamv2_fast" if "_mv2" in root else "sam_only_fast"
    return REPO / "data" / "eval" / name


def view_index(scene, view, sam_root):
    """回 {mask檔名: (feature or None, mask_bool)}。特徵取 clip_mean_feats(對齊 kept_object_masks)。"""
    vd = sam_root / scene / view
    km = MK.kept_object_masks(vd)            # [(mask_bool, name)]
    cmf = vd / "clip_mean_feats.npy"
    feats = np.load(cmf) if cmf.is_file() else None
    out = {}
    for i, (mask, name) in enumerate(km):
        f = None
        if feats is not None and i < len(feats) and not np.isnan(feats[i]).any():
            f = feats[i].astype(np.float32)
        out[name] = (f, mask)
    return out


def clean_phrase(p):
    """去掉 CLIP prompt 模板前綴,只留物體名。"""
    for pre in ("a photo of an ", "a photo of a ", "a photo of "):
        if p.startswith(pre):
            return p[len(pre):]
    return p


def topk_phrases(mean_feat, k=5):
    """平均特徵 → top1 物體名 與 top-k 去重物體名清單(cos 值一併回)。"""
    if mean_feat is None:
        return "—", []
    v = mean_feat / (np.linalg.norm(mean_feat) + 1e-9)
    sim = TF @ v
    order = np.argsort(-sim)
    phrases = []
    for j in order:
        p = clean_phrase(TPH[int(j)])
        if p not in [q for q, _ in phrases]:
            phrases.append((p, float(sim[int(j)])))
        if len(phrases) >= k:
            break
    top1 = phrases[0][0]
    return top1, phrases


def thumb_b64(rgb, mask, box=64):
    """從原圖裁遮罩 bbox → 縮到 box px → base64 PNG(黑底只留遮罩內)。"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return ""
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = rgb[y0:y1, x0:x1].copy()
    mcrop = mask[y0:y1, x0:x1]
    crop[~mcrop] = crop[~mcrop] // 4          # 遮罩外壓暗,突顯物體
    h, w = crop.shape[:2]
    s = box / max(h, w)
    crop = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode() if ok else ""


def heatmap_b64(feats, labels):
    """feats:(k,512) 正規化 → k×k cos 熱度圖 → base64 PNG。一律標數值,大小隨遮罩數成長不設上限。"""
    k = len(feats)
    F = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    S = F @ F.T
    cell = 0.5                                   # 每格英吋:格子固定大小 → 越多遮罩圖越大
    side = max(3.0, k * cell + 1.8)
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(S, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(k):                           # 一律標數值
        for j in range(k):
            ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=100); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build(scene, root):
    sdir_json = REPO / "data" / "eval" / root / scene / "instances.json"
    if not sdir_json.is_file():
        print(f"[skip] {scene}: 無 {root}/instances.json"); return None
    d = json.loads(sdir_json.read_text())
    insts = d.get("instances", [])
    sam_root = sam_root_for(root)
    group = scene.split("_")[0]
    cap_dir = CAPTURES / f"multi_{group}" / scene
    report_dir = REPO / "data" / "eval" / root / scene
    report = report_dir / "report.html"

    # 各視角一次性建索引(特徵+遮罩)與 RGB 快取
    vcache = {}; rgbcache = {}
    def get_view(view):
        if view not in vcache:
            vcache[view] = view_index(scene, view, sam_root)
        if view not in rgbcache:
            p = cap_dir / f"{view}.png"
            rgbcache[view] = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) if p.is_file() else None
        return vcache[view], rgbcache[view]

    rows = []; heat_sections = []
    for idx, it in enumerate(insts):
        hid = it["instance"]; col = PALETTE[(hid - 1) % len(PALETTE)]
        rgb_css = f"rgb({int(col[0]*255)},{int(col[1]*255)},{int(col[2]*255)})"
        mk = it.get("masks", {})
        # 蒐集該 hull 所有來源遮罩:縮圖 + 特徵 + 每遮罩 top1
        thumbs = []; feats = []; labels = []
        for view in sorted(mk):
            vidx, rgb = get_view(view)
            vshort = view.replace("view_", "")
            for name in mk[view]:
                f, mask = vidx.get(name, (None, None))
                # 影像連結(相對路徑)
                mpng = sam_root / scene / view / "masks" / name
                rel = os.path.relpath(mpng, report_dir)
                relrgb = os.path.relpath(cap_dir / f"{view}.png", report_dir)
                tb = thumb_b64(rgb, mask) if (rgb is not None and mask is not None) else ""
                cap = f"{vshort}/{name.replace('mask_','m').replace('.png','')}"
                if f is not None:                        # 該遮罩自己的 top1 / top5
                    mt1, mt5 = topk_phrases(f, 5)
                    top5_str = " · ".join(f"{p} ({c:.2f})" for p, c in mt5)
                else:
                    mt1, top5_str = "—", ""
                title = f"{cap} · top5: {top5_str}" if top5_str else cap
                img = (f'<img src="data:image/png;base64,{tb}" '
                       f'style="width:52px;height:52px;object-fit:cover;border-radius:3px">') if tb else esc(cap)
                thumbs.append(
                    f'<span style="display:inline-block;text-align:center;vertical-align:top;'
                    f'margin:3px;width:66px" title="{esc(title)}">'
                    f'<a href="{esc(rel)}" target="_blank">{img}</a>'
                    f'<span style="display:block;font-size:9px;line-height:1.15">{esc(mt1)}</span>'
                    f'<a href="{esc(relrgb)}" target="_blank" style="font-size:9px;color:#888">原圖</a>'
                    f'</span>')
                if f is not None:
                    feats.append(f); labels.append(cap)
        # hull 平均特徵 → top1/top5
        if feats:
            mean_f = np.mean(np.stack(feats), 0)
            top1, top5 = topk_phrases(mean_f, 5)
            heat = heatmap_b64(np.stack(feats), labels)
            heat_sections.append(
                f'<h3 style="margin:18px 0 4px">Hull {hid} '
                f'<span style="display:inline-block;width:14px;height:14px;background:{rgb_css};'
                f'vertical-align:middle;border:1px solid #999"></span> — 來源遮罩 cos 相似度 '
                f'({len(feats)} 張)</h3>'
                f'<div style="overflow:auto"><img src="data:image/png;base64,{heat}"></div>')
        else:
            top1, top5 = "—", []
        n_views = len(mk); n_masks = sum(len(v) for v in mk.values())
        summary = (f'<div style="font-size:12px;color:#555;margin-bottom:4px">'
                   f'共 {n_views} 視角 / {n_masks} 張遮罩(每張下方=該遮罩 top1,滑鼠移上顯示 top5)</div>')
        thumbs_html = summary + "".join(thumbs)
        rows.append(
            f'<tr>'
            f'<td style="text-align:center;font-weight:bold">{hid}</td>'
            f'<td style="background:{rgb_css};min-width:60px" title="{rgb_css}"></td>'
            f'<td style="font-weight:bold">{esc(top1)}</td>'
            f'<td style="font-size:12px">{esc(" · ".join(f"{p} ({c:.2f})" for p, c in top5))}</td>'
            f'<td style="text-align:center">{it.get("n_vox","")}</td>'
            f'<td style="line-height:1.6">{thumbs_html}</td>'
            f'</tr>')

    html = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<title>{esc(scene)} · {esc(root)}</title>
<style>
body{{font-family:system-ui,'Noto Sans TC',sans-serif;margin:20px;color:#222}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ccc;padding:5px;vertical-align:top}}
th{{background:#f0f0f0;position:sticky;top:0}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:24px}}
</style></head><body>
<h1>{esc(scene)} — {esc(root)}</h1>
<p style="color:#666">hull 數 {len(insts)}｜遮罩來源 = {esc(sam_root.name)}｜top1/top5 = 該 hull 來源遮罩 CLIP 平均特徵對 64 名詞</p>
<h2>Hull 一覽</h2>
<table>
<tr><th>hull</th><th>顏色</th><th>top1</th><th>top5</th><th>voxel</th><th>來源遮罩(點縮圖開 mask,原圖連結在下)</th></tr>
{"".join(rows)}
</table>
<h2>各 Hull 來源遮罩相似度熱度圖</h2>
{"".join(heat_sections) if heat_sections else "<p>(無有效 CLIP 特徵)</p>"}
</body></html>"""
    report.write_text(html, encoding="utf-8")
    print(f"[{scene}] {root}: {len(insts)} hull → {report}", flush=True)
    return report


def resolve(targets, root):
    base = REPO / "data" / "eval" / root
    if not targets:
        return sorted(Path(p).parent.name for p in glob.glob(str(base / "*_scene*/instances.json")))
    out = []
    for a in targets:
        if "scene" in a:
            out.append(a)
        else:
            out += [Path(p).parent.name for p in glob.glob(str(base / f"{a}_scene*/instances.json"))]
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    for sc in resolve(args.targets, args.root):
        try:
            build(sc, args.root)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")


if __name__ == "__main__":
    main()
