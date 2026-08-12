#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""gen_hull_gt_report.py — 以 hull 為單位、逐 hull 畫「全部 A-3 視角」的重投影疊圖 + recall。

每個 hull:對其命中(或最相關)GT,在**每個 A-3 視角**即時重投影,疊 modal GT 遮罩
  (紅=hull過估、綠=modal漏、青=交集);每視角標:view名 | 2D IoU | 是否計入。
遮擋判定:modal面積/amodal面積 >= OCC_THRESH(0.9) 才「計入」recall;否則標
  「遮擋>10%不計入(可見X%)」——**仍畫出重投影**(不參與平均而已)。
重投影/modal 全視角即時算(不依賴 build_hull_gt 只存無遮擋視角),故不用重跑 build。
recall/命中 判定沿用 eval_hull_gt.decide_hits(可調 --hit-iou)。
用法: ./gen_hull_gt_report.py --root <method> [--hit-iou 0.7] [scene|group|(空=全部)]
"""
import argparse
import base64
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "data" / "eval"
GT_OUT = EVAL / "gt_reproj"
CAP = REPO / "data" / "captures_fast"
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import camera as cam                     # noqa: E402
import masks as MK                       # noqa: E402  遮罩+特徵讀取
import mask_clip_cluster as MC           # noqa: E402  F_BG(去偏)
from build_hull_gt import reproject      # noqa: E402  即時重投影
from eval_hull_gt import decide_hits     # noqa: E402  命中判定(可調門檻)
import viewpoints as VP                  # noqa: E402
from labels import LABELS                # noqa: E402

PALETTE = [(230, 60, 60), (60, 160, 230), (60, 200, 90), (230, 160, 40), (170, 90, 220),
           (40, 200, 200), (230, 100, 170), (150, 150, 60), (100, 100, 230), (60, 230, 150)]
OCC_THRESH = 0.9
# 遮罩來源=建置該 root 用的 SAM(srp_hull_semcluster_clip 用 mobilesamv2_fast);可用 SAM_ROOT 覆寫
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(EVAL / "mobilesamv2_fast")))
THUMB_CAP = 400         # 縮圖上限(幾乎不裁,只防病態超大 hull)
HEAT_CAP = 120          # 熱度圖遮罩上限(超過才等距抽樣;一律標數值)
_BG = MC.F_BG.astype(np.float64)
_TZ = np.load(EVAL / "clip_text_feats.npz", allow_pickle=True)
TPH = [str(x) for x in _TZ["phrases"]]
_TF = _TZ["feats"].astype(np.float32); TF = _TF / (np.linalg.norm(_TF, axis=1, keepdims=True) + 1e-9)


def clean_phrase(p):
    for pre in ("a photo of an ", "a photo of a ", "a photo of "):
        if p.startswith(pre):
            return p[len(pre):]
    return p


def topk_phrases(mean_feat, k=5):
    """raw 平均特徵 → top1 + 去重 top-k 名詞(cos)。名詞比對用 raw(CLIP 文字對齊需 raw 空間)。"""
    if mean_feat is None:
        return "—", []
    v = mean_feat / (np.linalg.norm(mean_feat) + 1e-9)
    sim = TF @ v.astype(np.float32); order = np.argsort(-sim); ph = []
    for j in order:
        p = clean_phrase(TPH[int(j)])
        if p not in [q for q, _ in ph]:
            ph.append((p, float(sim[int(j)])))
        if len(ph) >= k:
            break
    return ph[0][0], ph


def debias(F):
    """去偏(投影掉 bg)+L2 — 與 voxel_sem_cluster 分群完全同式,熱度圖顯示的就是分群依據。"""
    F = F.astype(np.float64)
    F = F - (F @ _BG)[:, None] * _BG[None, :]
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


def view_masks_feats(scene, view):
    """{mask檔名: (bool_mask, raw_feat or None)};遮罩來源=SAM_ROOT(建置一致)。"""
    vd = SAM_ROOT / scene / view
    fm = MK.mask_feats(vd, "clip_mean_feats.npy")   # {name: raw feat or None}(對齊全部遮罩)
    out = {}
    for mp in sorted((vd / "masks").glob("mask_*.png")):
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            out[mp.name] = (m > 127, fm.get(mp.name))
    return out


def crop_b64(rgb, mask, box=64, darken=True, q=80):
    """遮罩 bbox 裁切縮圖 → base64 JPG(內嵌縮圖用)。"""
    c = crop_arr(rgb, mask, box, darken)
    if c is None:
        return ""
    ok, buf = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf).decode() if ok else ""


def heatmap_b64(F_deb, labels):
    """去偏特徵 (k,D) → k×k cos 熱度圖(base64)。超過 HEAT_CAP 等距抽樣。"""
    k = len(F_deb)
    S = F_deb @ F_deb.T
    cell = 0.42; side = max(3.0, k * cell + 1.8)
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(S, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    fs = 5 if k <= 30 else (4 if k <= 60 else 3)   # 一律標數值,字級隨遮罩數縮小
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=fs, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); fig.tight_layout()
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=95); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def iou2(a, b):
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else 0.0


def load_obj_masks(scene, kind):
    """{物體名: {view: mask(bool)}},kind=actual(modal)/amodal;含所有視角。"""
    d = json.loads((LABELS / scene / kind / "annotations.json").read_text())
    catn = {c["id"]: c["name"] for c in d["categories"]}
    vof = {im["id"]: Path(im["file_name"]).stem for im in d["images"]}
    out = {}
    for a in d["annotations"]:
        m = mask_utils.decode(a["segmentation"]).astype(bool)
        out.setdefault(catn[a["category_id"]], {})[vof[a["image_id"]]] = m
    return out


def make_overlay(rgb, reproj, modal):
    """全解析度疊圖:紅=hull過估、綠=modal漏、青=交集(無 modal 時只畫重投影紅)。"""
    ov = rgb.copy()
    if modal is not None:
        ov[reproj & ~modal] = (0, 0, 255)
        ov[modal & ~reproj] = (0, 255, 0)
        ov[reproj & modal] = (200, 200, 0)
    else:
        ov[reproj] = (0, 0, 255)
    return cv2.addWeighted(rgb, 0.45, ov, 0.55, 0)


def overlay_b64(out, width=230):
    """全解析疊圖 → 縮到 width → base64(內嵌縮圖用)。"""
    h, w = out.shape[:2]
    small = cv2.resize(out, (width, int(h * width / w)))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return base64.b64encode(buf).decode() if ok else ""


def crop_arr(rgb, mask, box, darken):
    """遮罩 bbox 裁切 → 縮到 box px。darken=True 打暗遮罩外。回 BGR 陣列 or None。"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = rgb[y0:y1, x0:x1].copy()
    if darken:
        mcrop = mask[y0:y1, x0:x1]; crop[~mcrop] = crop[~mcrop] // 4
    h, w = crop.shape[:2]; s = box / max(h, w)
    return cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def process(scene, root, hit_iou):
    hp = EVAL / root / scene / "hull_gt.json"
    if not hp.is_file():
        return None
    hj = json.loads(hp.read_text())
    hz = np.load(EVAL / root / scene / "hull_gt.npz")
    gm = hz["grid_min"]; vs = float(hz["voxel_size"])
    gj = json.loads((GT_OUT / scene / "gt.json").read_text())
    unocc = gj["unoccluded_views"]
    placed = list(gj["gt_objects"])                                  # 分母=全放置物體
    gt_occluded = [g for g in placed if len(unocc.get(g, [])) == 0]  # 無≥90%可見視角:一樣計為漏,只標數量
    hulls = hj["hulls"]
    hit_d, red_d = decide_hits(hulls, placed, hit_iou)
    found = set(); net = 0
    for k in hulls:
        if hit_d[k] and k not in red_d and hit_d[k] in placed:
            found.add(hit_d[k]); net += 1
    recall = len(found) / len(placed) if placed else 0
    precision = net / len(hulls) if hulls else 0

    modal = load_obj_masks(scene, "actual")
    amodal = load_obj_masks(scene, "amodal")
    group = scene.split("_")[0]; sdir = CAP / f"multi_{group}" / scene
    rdir = EVAL / root / scene            # 報告所在,算相對連結
    assets = rdir / "report_assets"; assets.mkdir(parents=True, exist_ok=True)  # 點開用大圖(疊圖/crop)
    sel = sorted(VP.selected_view_names(12))
    # 重投影由 build_hull_gt 直接存於 hull_gt.npz(全 A-3 視角);報告只「讀」,不算、不快取
    reproj = {}
    for key in hz.files:
        if key.startswith("reproj_"):
            kk, vv = key[len("reproj_"):].split("__", 1)
            reproj[(kk, vv)] = hz[key]

    # hull → 來源遮罩 {view:[mask檔]}(instances.json);hull_id 與 instance 同號
    # mask_clusters: view→{遮罩檔:群id(cl)},用來標「切它的主群 vs 幾何蓋到的跨群零星」
    ij = EVAL / root / scene / "instances.json"
    hull_masks = {}; mask_cl = {}
    if ij.is_file():
        ijd = json.loads(ij.read_text())
        mask_cl = ijd.get("mask_clusters", {})
        for it in ijd.get("instances", []):
            hull_masks[str(it["instance"])] = it.get("masks", {})
    def cl_of(v, nm):
        return mask_cl.get(v, {}).get(nm)
    CLPAL = ["#e63946", "#457b9d", "#2a9d8f", "#e9a227", "#8e6cc0", "#1aa5a5",
             "#e0669e", "#8a8a3c", "#5a5ad0", "#3aae6a"]   # 群色(依群id取模)
    _vmf, _rgbc = {}, {}   # 視角快取:遮罩+特徵 / RGB
    def vmf(v):
        if v not in _vmf:
            _vmf[v] = view_masks_feats(scene, v)
        return _vmf[v]
    def vrgb(v):
        if v not in _rgbc:
            p = sdir / f"{v}.png"
            _rgbc[v] = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) if p.is_file() else None
        return _rgbc[v]

    P = "<style>body{font-family:sans-serif;margin:14px}.hull{margin:12px 0;padding:8px;border:1px solid #ddd;border-radius:6px}" \
        ".v{display:inline-block;margin:2px;text-align:center;font-size:11px;vertical-align:top}" \
        ".v img{display:block;border:1px solid #aaa}.no{opacity:.5}.ok{color:#188}.oc{color:#c60}" \
        ".sem{margin-top:8px;padding-top:6px;border-top:1px dashed #ccc;font-size:12px}" \
        ".th{display:inline-block;text-align:center;vertical-align:top;margin:2px;font-size:9px;line-height:1.2;width:54px}" \
        ".th img{display:block;margin:0 auto;border-radius:3px;border:1px solid #bbb}" \
        ".orig{font-size:8px;color:#88a}.heat{margin-top:6px;overflow:auto}" \
        ".clchip{display:inline-block;font-size:8px;color:#fff;padding:0 3px;border-radius:2px;line-height:1.5}" \
        ".th{width:54px}.th .clchip{margin:1px 0}" \
        ".sw{display:inline-block;width:12px;height:12px;vertical-align:middle;margin-right:3px;border:1px solid #333}</style>"
    P += f"<h2>{scene} — {root} (hit-IoU={hit_iou:g})</h2>"
    P += (f"<p><b>recall</b>={recall:.3f} ({len(found)}/{len(placed)}) &nbsp; "
          f"<b>precision</b>={precision:.3f} ({net}/{len(hulls)})<br>"
          f"找到: {sorted(found) or '—'}<br>漏掉: {sorted(set(placed)-found) or '—'}<br>"
          f"其中無≥90%可見視角: {gt_occluded or '—'}</p>")
    P += ("<p>疊圖: <span class=sw style='background:rgb(255,0,0)'></span>hull過估 "
          "<span class=sw style='background:rgb(0,255,0)'></span>modal漏 "
          "<span class=sw style='background:rgb(0,200,200)'></span>交集 &nbsp;|&nbsp; "
          "<span class=ok>計入</span> / <span class=oc>遮擋>10%不計入</span></p>")

    for k, info in hulls.items():
        col = PALETTE[(int(k) - 1) % len(PALETTE)]
        tg = hit_d[k] or (max(info["per_gt"], key=lambda g: info["per_gt"][g].get("avg", 0))
                          if info["per_gt"] else None)
        avg = info["per_gt"].get(tg, {}).get("avg", 0) if tg else 0
        tag = f"命中 <b>{tg}</b>" if hit_d[k] else f"<i>未命中</i>(最相關 {tg})"
        if k in red_d:
            tag += " <span style='color:#c60'>[多餘/重複]</span>"
        P += f"<div class=hull><span class=sw style='background:rgb{col}'></span>" \
             f"<b>hull {k}</b> — {tag} &nbsp; 計入視角平均IoU={avg}<br>"
        for v in sel:
            rj = reproj.get((k, v))
            if rj is None:
                continue
            rgbp = sdir / f"{v}.png"
            if not rgbp.is_file():
                continue
            rgb = cv2.imread(str(rgbp))
            mm = modal.get(tg, {}).get(v) if tg else None
            am = amodal.get(tg, {}).get(v) if tg else None
            occ = (mm.sum() / am.sum()) if (mm is not None and am is not None and am.sum() > 0) else None
            counted = occ is not None and occ >= OCC_THRESH
            iou = iou2(rj, mm) if mm is not None else None
            full = make_overlay(rgb, rj, mm)          # 全解析疊圖(紅/綠/青)
            b64 = overlay_b64(full)                   # 內嵌縮圖
            of = assets / f"ov_h{k}_{v}.jpg"          # 點開=帶疊圖的大圖(實體檔,分頁有檔名)
            cv2.imwrite(str(of), full, [cv2.IMWRITE_JPEG_QUALITY, 88])
            iou_s = f"IoU={iou:.2f}" if iou is not None else "無modal"
            if mm is None:
                lab = "<span class=oc>此視角無此物</span>"
            elif counted:
                lab = f"<span class=ok>計入 · {iou_s}</span>"
            else:
                lab = f"<span class=oc>遮擋不計入(可見{occ*100:.0f}%) · {iou_s}</span>"
            cls = "v" if (mm is not None and counted) else "v no"
            rel_ov = os.path.relpath(of, rdir)
            P += f"<span class='{cls}'>{v.replace('view_','')}<br>" \
                 f"<a href='{rel_ov}' target=_blank><img src='data:image/jpeg;base64,{b64}'></a><br>{lab}</span>"

        # ── 來源遮罩 + 每遮罩群標籤(cl):標「切它的主群 ★」vs「幾何蓋到的跨群零星 ⚠」──
        mk = hull_masks.get(k, {})
        items = []; cl_count = Counter()   # item=(f, cap, cl, img, orig_link, tip, rel_m)
        for v in sorted(mk):
            vi = vmf(v); rgb = vrgb(v); vs = v.replace("view_", "")
            for nm in mk[v]:
                mask, f = vi.get(nm, (None, None))
                has = rgb is not None and mask is not None
                tb = crop_b64(rgb, mask, 50, darken=True) if has else ""
                stem = nm.replace(".png", ""); orig_link = ""
                if has:
                    cf = assets / f"crop_{v}_{stem}.jpg"
                    ca = crop_arr(rgb, mask, 384, darken=False)
                    if ca is not None:
                        cv2.imwrite(str(cf), ca, [cv2.IMWRITE_JPEG_QUALITY, 88])
                        orig_link = f"<a href='{os.path.relpath(cf, rdir)}' target=_blank class=orig>原圖crop</a>"
                mt1, mt5 = topk_phrases(f, 5) if f is not None else ("—", [])
                top5s = " · ".join(f"{p} {c:.2f}" for p, c in mt5)
                cap = f"{vs}/{nm.replace('mask_', 'm').replace('.png', '')}"
                cl = cl_of(v, nm)
                if cl is not None:
                    cl_count[cl] += 1
                rel_m = os.path.relpath(SAM_ROOT / scene / v / "masks" / nm, rdir)
                tip = f"{cap} · 群{cl} · top1={mt1} · top5: {top5s}"
                img = (f"<img src='data:image/jpeg;base64,{tb}' style='width:50px;height:50px;object-fit:cover'>"
                       if tb else cap)
                items.append((f, cap, cl, img, orig_link, tip, rel_m))
        dom = cl_count.most_common(1)[0][0] if cl_count else None      # 主群 = 切它的群
        items.sort(key=lambda x: (0 if x[2] == dom else 1, x[2] if x[2] is not None else 999))
        thumbs = []
        for f, cap, cl, img, orig_link, tip, rel_m in items[:THUMB_CAP]:
            col = CLPAL[(cl - 1) % len(CLPAL)] if cl is not None else "#999"
            chip = f"<span class=clchip style='background:{col}'>群{cl}{'★' if cl == dom else '⚠'}</span>"
            thumbs.append(f"<span class=th title='{tip}'><a href='{rel_m}' target=_blank>{img}</a>"
                          f"<br>{chip}<br>{orig_link}</span>")
        feats = [x[0] for x in items if x[0] is not None]
        labs = [f"g{x[2]}:{x[1]}" for x in items if x[0] is not None]   # 依群排序,標籤帶群
        n_v = len(mk); n_m = sum(len(x) for x in mk.values())
        strag = " · ".join(f"群{c}×{n}" for c, n in cl_count.most_common() if c != dom)
        P += (f"<div class=sem><b>來源遮罩</b> — {n_v}視角/{n_m}遮罩 &nbsp; "
              f"切它的主群=<b>群{dom}×{cl_count[dom] if dom else 0}</b>"
              + (f" &nbsp;<span class=oc>幾何蓋到的跨群零星: {strag}</span>" if strag else "")
              + (f" &nbsp;(縮圖前{THUMB_CAP}/{n_m})" if len(items) > THUMB_CAP else ""))
        P += "<br>" + "".join(thumbs) if thumbs else "<br><i>無來源遮罩</i>"
        if len(feats) >= 2:
            F = feats; ll = labs
            if len(F) > HEAT_CAP:
                idx = np.linspace(0, len(F) - 1, HEAT_CAP).round().astype(int)
                F = [F[i] for i in idx]; ll = [ll[i] for i in idx]
            heat = heatmap_b64(debias(np.stack(F)), ll)
            note = f"(顯示 {len(F)}/{len(feats)})" if len(feats) > HEAT_CAP else ""
            P += f"<div class=heat><div style='font-size:11px;color:#666'>遮罩兩兩 cos(去偏;依群排序,對角亮塊=同群){note}</div>" \
                 f"<img src='data:image/png;base64,{heat}'></div>"
        P += "</div>"
        P += "</div>"

    out = EVAL / root / scene / f"hull_gt_report_iou{hit_iou:g}.html"
    out.write_text(P, encoding="utf-8")
    print(f"[{scene}] {root} iou{hit_iou:g}: recall={recall:.2f} → {out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--root", required=True)
    ap.add_argument("--hit-iou", type=float, default=0.7, dest="hit_iou")
    args = ap.parse_args()
    base = EVAL / args.root
    if not args.targets:
        scenes = sorted(p.parent.name for p in base.glob("*_scene*/hull_gt.json"))
    else:
        scenes = []
        for a in args.targets:
            scenes.append(a) if "scene" in a else scenes.extend(
                p.parent.name for p in base.glob(f"{a}_scene*/hull_gt.json"))
        scenes = sorted(set(scenes))
    for sc in scenes:
        try:
            process(sc, args.root, args.hit_iou)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")


if __name__ == "__main__":
    main()
