#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""mask_clip_cluster.py — 對各 hull 的「來源遮罩片」做 CLIP 語意分群,判斷該切幾群。

讀 nested_masks.json(每 hull 每視角 {來源遮罩 A: [內含小遮罩 B_i]})。
每個來源遮罩取「片」:剩餘片 R = A − ∪B_i、各內含小遮罩 B_i(面積 < min_area 的片丟掉)。
每片 square_mean_crop(填 CLIP mean)→ 224 → CLIP normalize → encode_image → 去偏(扣 f_bg)→ L2。
整個 hull 的所有片一起凝聚分群(average, 1−cos, 門檻 SEM_THR)→ g 群 = 語意物體數。
g==1 不切;g≥2 判定該切。★只輸出分群,不碰 voxel。

原理:同物體內含(cups⊃cups)→ R 與 B 語意近 → 併 1 群 → 不切;
      異物體內含(sponge⊃blocks)→ R 與 B 語意遠 → 分 2 群 → 該切。

輸出 HULL_ROOT/<scene>/clusters.json:
  {"scene":..., "sem_thr":.., "min_area":..,
   "instances":{"<inst>":{"n_groups":g, "n_pieces":N, "group_sizes":[...],
                          "pieces":[{"view":..,"kind":"R"|"B","mask":..,"group":c}, ...]}}}
用法:
  ./srp/stage2_instances/mask_clip_cluster.py                  # 全部場景
  ./srp/stage2_instances/mask_clip_cluster.py stack3_scene0001 # 單場景
  ./srp/stage2_instances/mask_clip_cluster.py stack3 [--sem-thr 0.3] [--min-area 200]
env: SAM_ROOT HULL_ROOT CAPTURES_ROOT
"""
import argparse, os, sys, json, glob
import numpy as np, cv2, torch, open_clip
from pathlib import Path
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "srp" / "io"))
import masks as MK

REPO = Path(__file__).resolve().parents[2]
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "sam_only")))
HULL_ROOT = Path(os.environ.get("HULL_ROOT", str(REPO / "data" / "eval" / "srp_hull")))
CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures")))
dev = "cuda" if torch.cuda.is_available() else "cpu"
# ★lazy-load:import 本模組不載 CLIP 模型;只有真的要算特徵(clip_feats)才載。
# 只讀預存特徵/F_BG 的用途(如報告)完全不碰模型/GPU。行為不變(首次算特徵時載,結果一樣)。
_clip_model = None
def _clip():
    global _clip_model
    if _clip_model is None:
        m, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        _clip_model = m.to(dev).eval()
    return _clip_model
CMEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CSTD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
CLIP_FILL = np.array([123, 117, 104], np.uint8)   # round(CLIP mean×255):填後 normalize≈0


BLUR_KSIZE = 51   # 背景高斯模糊 kernel(強烈;越大越模糊)


def square_crop_blur(rgb, seg, ksize=BLUR_KSIZE):
    """bbox 為中心擴成正方形(取原圖含上下文)→ 遮罩外強烈高斯模糊、遮罩內清晰。
    保留上下文、讓 CLIP 注意力集中在清晰目標(vs 填均值把背景抹掉)。"""
    ys, xs = np.nonzero(seg)
    if xs.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    s = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    Y0, X0 = cy - s // 2, cx - s // 2
    Y1, X1 = Y0 + s, X0 + s
    pt, pl = max(0, -Y0), max(0, -X0)
    pb, pr = max(0, Y1 - rgb.shape[0]), max(0, X1 - rgb.shape[1])
    rp = cv2.copyMakeBorder(rgb, pt, pb, pl, pr, cv2.BORDER_REFLECT)
    sp = cv2.copyMakeBorder(seg.astype(np.uint8), pt, pb, pl, pr, cv2.BORDER_CONSTANT, None, 0).astype(bool)
    Y0 += pt; Y1 += pt; X0 += pl; X1 += pl
    sq = rp[Y0:Y1, X0:X1]; ss = sp[Y0:Y1, X0:X1]
    k = ksize | 1   # 保證奇數
    out = cv2.GaussianBlur(sq, (k, k), 0)
    out[ss] = sq[ss]   # 遮罩內貼回清晰原圖
    return out


def sqcrop(rgb, seg, bg_mode="mean"):
    if bg_mode == "blur":
        return square_crop_blur(rgb, seg)
    ys, xs = np.nonzero(seg)
    if xs.size == 0: return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    c = rgb[y0:y1, x0:x1].copy(); c[~seg[y0:y1, x0:x1]] = CLIP_FILL
    h, w = c.shape[:2]; s = max(h, w); cv = np.empty((s, s, 3), np.uint8); cv[:] = CLIP_FILL
    oy, ox = (s - h) // 2, (s - w) // 2; cv[oy:oy + h, ox:ox + w] = c
    return cv


@torch.no_grad()
def clip_feats(rgb, segs, bg_mode="mean"):
    ims, val = [], []
    for i, s in enumerate(segs):
        c = sqcrop(rgb, s, bg_mode)
        if c is not None:
            im = cv2.resize(c, (224, 224))
            ims.append((torch.from_numpy(im).permute(2, 0, 1).float() / 255 - CMEAN) / CSTD); val.append(i)
    out = [None] * len(segs)
    if ims:
        f = _clip().encode_image(torch.stack(ims).to(dev)).float()
        f = (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()
        for k, i in enumerate(val): out[i] = f[k]
    return out


_FBG_CACHE = Path(__file__).resolve().parent / "clip_fbg_vitb32.npy"   # F_BG 是固定常數,存檔;讀檔者(報告)免載模型
_F_BG = None
@torch.no_grad()
def _fbg():
    """CLIP 純填充色特徵(去偏用)。固定值→存檔;首次算後存,之後(含報告)讀檔不載模型。"""
    global _F_BG
    if _F_BG is not None:
        return _F_BG
    if _FBG_CACHE.is_file():
        _F_BG = np.load(_FBG_CACHE); return _F_BG
    im = np.empty((100, 100, 3), np.uint8); im[:] = CLIP_FILL
    x = (torch.from_numpy(cv2.resize(im, (224, 224))).permute(2, 0, 1).float() / 255 - CMEAN) / CSTD
    f = _clip().encode_image(x[None].to(dev)).float()[0]
    _F_BG = (f / f.norm()).cpu().numpy()
    np.save(_FBG_CACHE, _F_BG)
    return _F_BG


def __getattr__(name):   # PEP 562 模組級 lazy:MC.F_BG / MC.clip_model 首次存取才觸發(import 時不載)
    if name == "F_BG":
        return _fbg()
    if name == "clip_model":
        return _clip()
    raise AttributeError(f"module 'mask_clip_cluster' has no attribute {name!r}")


def cached_feats(view_dir, rgb, ms):
    """優先讀預存 clip_mean_feats.npy(填均值);沒有才即時算。三方法共用,不重算 CLIP。"""
    p = Path(view_dir) / "clip_mean_feats.npy"
    if p.is_file():
        arr = np.load(p)
        if len(arr) == len(ms):
            return [None if np.isnan(a).any() else a for a in arr]
    return clip_feats(rgb, ms, "mean")


def debias(F):
    bg = _fbg()
    F2 = F - (F @ bg)[:, None] * bg[None, :]
    return F2 / (np.linalg.norm(F2, axis=1, keepdims=True) + 1e-9)


def process_scene(scene, sem_thr, min_area):
    nmp = HULL_ROOT / scene / "nested_masks.json"
    if not nmp.is_file():
        print(f"[skip] {scene}: 缺 nested_masks.json"); return None
    nm = json.load(open(nmp))
    group = scene.split("_")[0]; sdir = CAPTURES / f"multi_{group}" / scene
    view_cache = {}

    def load_view(view):
        if view not in view_cache:
            n2m = {n: m for m, n in MK.kept_object_masks(SAM_ROOT / scene / view)}
            rgb = cv2.cvtColor(cv2.imread(str(sdir / f"{view}.png")), cv2.COLOR_BGR2RGB)
            view_cache[view] = (n2m, rgb)
        return view_cache[view]

    out_inst = {}
    for K, views in nm["instances"].items():
        pieces = []   # (seg, view, kind, name)
        for view, srcs in views.items():
            n2m, _ = load_view(view)
            for a, subs in srcs.items():
                if a not in n2m: continue
                R = n2m[a].copy(); valid = []
                for b in subs:
                    if b in n2m:
                        R = R & ~n2m[b]; valid.append(b)
                if int(R.sum()) >= min_area:
                    pieces.append((R, view, "R", a))
                for b in valid:
                    if int(n2m[b].sum()) >= min_area:
                        pieces.append((n2m[b], view, "B", b))
        # CLIP 特徵(依 view 分批)
        feats = [None] * len(pieces); by_view = {}
        for i, (seg, view, kind, name) in enumerate(pieces):
            by_view.setdefault(view, []).append(i)
        for view, idxs in by_view.items():
            _, rgb = load_view(view)
            fs = clip_feats(rgb, [pieces[i][0] for i in idxs])
            for i, f in zip(idxs, fs): feats[i] = f
        keep = [i for i in range(len(pieces)) if feats[i] is not None]
        plist = []
        if len(keep) < 2:
            g = 1 if keep else 0
            for i in keep:
                seg, view, kind, name = pieces[i]
                plist.append({"view": view, "kind": kind, "mask": name, "group": 1})
            out_inst[K] = {"n_groups": g, "n_pieces": len(keep),
                           "group_sizes": [len(keep)] if keep else [], "pieces": plist}
            continue
        F = debias(np.array([feats[i] for i in keep]))
        cl = fcluster(linkage(pdist(F, "cosine"), "average"), t=sem_thr, criterion="distance")
        g = int(cl.max())
        for j, i in enumerate(keep):
            seg, view, kind, name = pieces[i]
            plist.append({"view": view, "kind": kind, "mask": name, "group": int(cl[j])})
        out_inst[K] = {"n_groups": g, "n_pieces": len(keep),
                       "group_sizes": np.bincount(cl)[1:].tolist(), "pieces": plist}
    out = {"scene": scene, "sem_thr": sem_thr, "min_area": min_area, "instances": out_inst}
    (nmp.parent / "clusters.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_inst


def resolve(targets):
    if not targets:
        return sorted(p.parent.name for p in HULL_ROOT.glob("*/nested_masks.json"))
    out = []
    for a in targets:
        if "scene" in a:
            if (HULL_ROOT / a / "nested_masks.json").is_file():
                out.append(a)
        else:
            out += [Path(p).parent.name for p in glob.glob(str(HULL_ROOT / f"{a}_scene*/nested_masks.json"))]
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--sem-thr", type=float, default=0.3, dest="sem_thr")
    ap.add_argument("--min-area", type=int, default=200, dest="min_area")
    args = ap.parse_args()
    scenes = resolve(args.targets)
    if not scenes:
        print("找不到符合的場景(需 HULL_ROOT/<scene>/nested_masks.json)"); return
    n_split = 0
    for sc in scenes:
        r = process_scene(sc, args.sem_thr, args.min_area)
        if r:
            n_split += sum(1 for v in r.values() if v["n_groups"] >= 2)
    print(f"場景 {len(scenes)}; 判定該切(g≥2)的 hull 數 {n_split} → 各場景 clusters.json "
          f"(sem_thr={args.sem_thr}, min_area={args.min_area})")


if __name__ == "__main__":
    main()
