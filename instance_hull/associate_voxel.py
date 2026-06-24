#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""associate_voxel.py — mask-native 關聯(實驗,結果獨立存放)。

不再把遮罩壓成質心點做射線交會(那會丟掉形狀→細長物裂開、任意配對→幻影點)。
改成直接用整塊遮罩:把工作空間切體素,每個體素投影回各 view,看落在「第幾號遮罩」內。
  ① space carving:體素落在前景遮罩內的 view 數 ≥ keep_frac×N → 保留(visual hull)。
  ② 遮罩歸屬連通:相鄰體素「跨 view 的遮罩號大致一致」才連通(union-find)。
     同一物體的體素在各 view 落在同一塊遮罩→歸屬一致;相鄰/重疊物體與幻影橋的
     體素歸屬對不起來→自動斷開。解掉方法A(前景union)切不開相鄰物的死穴。
  ③ 每個連通元件=一個 instance;其各 view 遮罩號→遮罩檔名,即跨視角關聯。

無質心、無射線配對、無幻影點;用 SAM 的逐物體分割補 visual hull 切不開相鄰物的弱點。
完全不用深度/標籤。

輸出: data/eval/instance_hull_voxel/<scene>/instances.json(+ assoc_report.txt)
       —— 與 instance_hull/、_hdbscan/、_dbscan/ 完全分開。
需在 webots_visual_hull 環境(numpy/cv2)。

用法: ./instance_hull/associate_voxel.py n3_scene0001  (或組號 3 / 多組 1 3 4 5)
       [--voxel 0.015] [--keep-frac 0.6] [--agree-frac 0.5] [--min-vox 8]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
CAPTURES = REPO / "data" / "captures"
SAM_ROOT = REPO / "data" / "eval" / "sam_only"
OUT_ROOT = REPO / "data" / "eval" / "instance_hull_voxel"   # ← 獨立目錄

HFOV_RAD = 1.4746
MAX_AREA_FRAC = 0.30
BORDER = 2
# 工作空間體素範圍(物體所在)
WS_X = (-0.05, 0.75)
WS_Y = (-0.45, 0.45)
WS_Z = (-0.02, 0.40)


def rpy_to_R(roll, pitch, yaw):
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


BODY_TO_OPENCV = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)


def load_pose(pose_path):
    meta = json.loads(pose_path.read_text(encoding="utf-8"))
    if "position_m" not in meta and isinstance(meta.get("camera"), dict):
        meta = meta["camera"]
    p = meta["position_m"]
    C = np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
    r = meta["rotation_rpy_rad"]
    return C, rpy_to_R(r["roll"], r["pitch"], r["yaw"])


def intrinsics(W, H):
    fx = W / (2.0 * math.tan(HFOV_RAD / 2.0))
    return fx, W / 2.0, H / 2.0


def touches_border(seg):
    return bool(seg[:BORDER].any() or seg[-BORDER:].any()
                or seg[:, :BORDER].any() or seg[:, -BORDER:].any())


def load_label_image(view_dir):
    """回傳 (label_img, bitmask_img, filenames)。
    label_img: int, 0=bg, n=第n塊遮罩,小面積後畫覆蓋→較具體者勝(單標籤用)。
    bitmask_img: int64, 第k塊遮罩覆蓋處 set bit (k-1)(多標籤用,一像素可屬多遮罩)。
    filenames[k-1] = 第k塊遮罩檔名。"""
    mask_dir = view_dir / "masks"
    items = []
    for mp in sorted(mask_dir.glob("mask_*.png")):
        seg = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if seg is None:
            continue
        b = seg > 127
        H, W = b.shape
        area = int(b.sum())
        if area == 0 or area > MAX_AREA_FRAC * H * W or touches_border(b):
            continue
        items.append((area, b, mp.name))
    if not items:
        return None, None, []
    items.sort(key=lambda x: -x[0])          # 大面積先畫,小面積後畫覆蓋
    H, W = items[0][1].shape
    label = np.zeros((H, W), dtype=np.int32)
    bitmask = np.zeros((H, W), dtype=np.int64)
    files = []
    for idx, (_, b, name) in enumerate(items, start=1):
        label[b] = idx
        if idx - 1 < 63:                     # int64 bit 上限
            bitmask[b] |= np.int64(1) << np.int64(idx - 1)
        files.append(name)
    return label, bitmask, files


def resolve_scenes(targets):
    scenes = []
    for a in targets:
        if "scene" in a:
            scenes.append(a)
        else:
            scenes += [d.name for d in sorted((CAPTURES / f"multi_n{a}").glob(f"n{a}_scene*"))]
    return scenes


def sig_merge(instances, overlap_th, min_common=2):
    """簽章合併:兩元件在「共用同一遮罩(檔名)」的 view 比例 >= overlap_th → 同物體,合併。
    用遮罩身分(離散)判斷,不看距離 → 細體素切碎的同物體碎片黏回,不誤併不同物體。
    每元件:{center(np), n_vox, masks{view:set(檔名)}}。迭代併最相似的一對到無可併。"""
    insts = [{"center": np.array(i["center"], float), "n_vox": i["n_vox"],
              "masks": {v: set(f) for v, f in i["masks"].items()}} for i in instances]
    while len(insts) > 1:
        best = None  # (overlap, a, b)
        for a in range(len(insts)):
            for b in range(a + 1, len(insts)):
                ma, mb = insts[a]["masks"], insts[b]["masks"]
                common = [v for v in ma if v in mb]
                if len(common) < min_common:
                    continue
                shared = sum(1 for v in common if ma[v] & mb[v])
                ov = shared / len(common)
                if ov >= overlap_th and (best is None or ov > best[0]):
                    best = (ov, a, b)
        if best is None:
            break
        _, a, b = best
        A, B = insts[a], insts[b]
        for v, fs in B["masks"].items():
            A["masks"].setdefault(v, set()).update(fs)
        tot = A["n_vox"] + B["n_vox"]
        A["center"] = (A["center"] * A["n_vox"] + B["center"] * B["n_vox"]) / tot
        A["n_vox"] = tot
        insts.pop(b)
    out = [{"center": [round(float(x), 4) for x in i["center"]],
            "support": len(i["masks"]), "n_vox": i["n_vox"],
            "masks": {v: sorted(fs) for v, fs in i["masks"].items()}} for i in insts]
    out.sort(key=lambda a: -a["support"])
    return out


def build_adjacency(label_img, maxlabel, gap):
    """回傳 (maxlabel+1)² bool:遮罩 a,b 在影像上相接(或間距<=gap px)→ True;含對角(自己)。
    背景(0)與任何都不相鄰。用來『容許一物體多遮罩』:相鄰遮罩視為同物體。"""
    A = np.eye(maxlabel + 1, dtype=bool)
    A[0, :] = False; A[:, 0] = False
    for d in range(1, gap + 1):
        for a, b in ((label_img[:, :-d], label_img[:, d:]),
                     (label_img[:-d, :], label_img[d:, :])):
            m = (a > 0) & (b > 0) & (a != b)
            la, lb = a[m], b[m]
            A[la, lb] = True; A[lb, la] = True
    return A


def split_component_by_mask(Lc, ADJ, agree_th, min_common, min_sub):
    """把一個元件的體素,依『跨 view 對應到的遮罩(含相鄰遮罩視為同物體)』全局分群(治幻影橋)。
    Lc: (k,n) 各體素單標籤簽章(0=背景/外);ADJ[v]: 該 view 的遮罩相鄰矩陣。
    貪婪:支持最多的體素當種子,收所有「與種子在多數共同 view 對應到『同一或相鄰』遮罩」者為一群。
    碗緣/碗身遮罩相鄰→同群(不切碎);orange/foam 遮罩有空隙不相鄰→不同群(切開)。
    回傳:list of bool mask。"""
    k, n = Lc.shape
    assigned = np.full(k, -1)
    inside = (Lc > 0).sum(1)
    nclust = 0
    for seed in np.argsort(-inside):
        if assigned[seed] != -1:
            continue
        sig = Lc[seed]
        both = (sig > 0) & (Lc > 0)
        comp = np.zeros_like(both)
        for v in range(n):
            sv = int(sig[v])
            if sv > 0:
                comp[:, v] = ADJ[v][Lc[:, v], sv]      # 同一或相鄰遮罩
        comp &= both
        den = both.sum(1); eq = comp.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(den > 0, eq / np.maximum(den, 1), 0.0)
        grp = (assigned == -1) & (den >= min_common) & (frac >= agree_th)
        grp[seed] = True
        assigned[grp] = nclust
        nclust += 1
    return [assigned == c for c in range(nclust) if (assigned == c).sum() >= min_sub]


def _build_instance(views, comp_L, comp_B, center, n_vox, thresh, args, instances):
    """由一個(子)元件的體素簽章建一個 instance,append 到 instances。各 view 取支持夠的遮罩號→檔名。"""
    per_view = {}
    for vi, v in enumerate(views):
        if args.multi_label:
            bm = comp_B[:, vi]
            nbits = len(v["files"])
            chosen = [k + 1 for k in range(min(nbits, 63))
                      if int((((bm >> np.int64(k)) & 1) == 1).sum()) >= thresh]
        else:
            labs = comp_L[:, vi]; labs = labs[labs > 0]
            if labs.size == 0:
                continue
            vals, cnts = np.unique(labs, return_counts=True)
            chosen = [int(x) for x in vals[cnts >= thresh]]
        files = [v["files"][int(l) - 1] for l in chosen if 0 < l <= len(v["files"])]
        if files:
            per_view[v["name"]] = files
    if len(per_view) < 2:
        return
    instances.append({"center": [round(float(x), 4) for x in center],
                      "support": len(per_view), "n_vox": int(n_vox),
                      "masks": per_view})


class UF:
    def __init__(self, n):
        self.p = np.arange(n)
    def find(self, x):
        p = self.p
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def process_scene(scene, args):
    group = scene.split("_")[0]
    scene_dir = CAPTURES / f"multi_{group}" / scene
    sam_dir = SAM_ROOT / scene
    if not sam_dir.is_dir():
        print(f"[skip] {scene}: 找不到 SAM 遮罩 {sam_dir}(先跑 sam_only.py)")
        return
    tag = "instance_hull_voxel"
    if abs(args.voxel - 0.015) > 1e-6:
        tag += f"_v{round(args.voxel * 1000)}"      # 非預設體素 → 標於目錄名,不蓋既有
    if abs(args.keep_frac - 0.6) > 1e-6:
        tag += f"_k{round(args.keep_frac * 100)}"
    if args.min_vox != 8:
        tag += f"_mv{args.min_vox}"
    if args.multi_label:
        tag += "_ml"
    if args.mask_split:
        tag += "_ms"
    if args.max_seethrough is not None:
        tag += f"_st{args.max_seethrough}"
    out_dir = REPO / "data" / "eval" / tag / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 讀各 view 姿態 + label/bitmask 影像
    views = []
    for vdir in sorted(sam_dir.glob("view_*")):
        name = vdir.name
        pose_path = scene_dir / f"{name}_pose.json"
        if not pose_path.is_file():
            continue
        label, bitmask, files = load_label_image(vdir)
        if label is None:
            continue
        C, R = load_pose(pose_path)
        H, W = label.shape
        fx, cx, cy = intrinsics(W, H)
        vd_ = {"name": name, "C": C, "R": R, "fx": fx, "cx": cx, "cy": cy,
               "W": W, "H": H, "label": label, "bitmask": bitmask, "files": files}
        if args.mask_split:
            vd_["adj"] = build_adjacency(label, len(files), args.split_gap)
        views.append(vd_)
    n = len(views)
    if n < 2:
        print(f"[skip] {scene}: 有效 view < 2")
        return

    # 2) 體素格
    vx = args.voxel
    xs = np.arange(WS_X[0], WS_X[1], vx)
    ys = np.arange(WS_Y[0], WS_Y[1], vx)
    zs = np.arange(WS_Z[0], WS_Z[1], vx)
    nx, ny, nz = len(xs), len(ys), len(zs)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)   # (M,3)
    M = P.shape[0]

    # 3) 每 view:投影所有體素 → 單標籤 L(最小者勝)+ 多標籤 B(bitmask,可屬多遮罩)
    L = np.zeros((M, n), dtype=np.int32)
    B = np.zeros((M, n), dtype=np.int64) if args.multi_label else None
    ST = np.zeros(M, dtype=np.int32)        # see-through:投影到影像內但落在背景的 view 數
    for vi, v in enumerate(views):
        X = (P - v["C"]) @ v["R"] @ BODY_TO_OPENCV.T        # world→optical
        z = X[:, 2]
        valid = z > 1e-6
        u = np.full(M, -1.0); vv = np.full(M, -1.0)
        u[valid] = v["fx"] * X[valid, 0] / z[valid] + v["cx"]
        vv[valid] = v["fx"] * X[valid, 1] / z[valid] + v["cy"]
        ui = np.round(u).astype(np.int64); vi_ = np.round(vv).astype(np.int64)
        inb = valid & (ui >= 0) & (ui < v["W"]) & (vi_ >= 0) & (vi_ < v["H"])
        lab = np.zeros(M, dtype=np.int32)
        lab[inb] = v["label"][vi_[inb], ui[inb]]
        L[:, vi] = lab
        ST += (inb & (lab == 0)).astype(np.int32)   # 影像內但背景 = 該相機看穿成空地
        if args.multi_label:
            bm = np.zeros(M, dtype=np.int64)
            bm[inb] = v["bitmask"][vi_[inb], ui[inb]]
            B[:, vi] = bm

    # 4) space carving:落在前景遮罩內的 view 數 >= keep_frac×n
    inside_cnt = (L > 0).sum(axis=1)
    keep_min = max(2, int(math.ceil(args.keep_frac * n)))
    keep = inside_cnt >= keep_min
    if args.max_seethrough is not None:
        # see-through 過濾:被 >max_seethrough 台相機看穿成背景 → 幻影,刪除
        keep &= (ST <= args.max_seethrough)
    nkeep = int(keep.sum())
    if nkeep == 0:
        print(f"[skip] {scene}: carving 後無體素(keep_min={keep_min})")
        return

    # 5) 遮罩歸屬連通(union-find):相鄰且「跨 view 遮罩號一致比例 >= agree_frac」才連
    grid_keep = keep.reshape(nx, ny, nz)
    idx_grid = -np.ones((nx, ny, nz), dtype=np.int64)
    idx_grid[grid_keep] = np.arange(nkeep)
    Lk = L[keep]                       # (nkeep, n) 單標籤
    Bk = B[keep] if args.multi_label else None   # (nkeep, n) 多標籤 bitmask
    uf = UF(nkeep)

    def agree(a_ids, b_ids):
        if args.multi_label:
            Ba, Bb = Bk[a_ids], Bk[b_ids]
            both = (Ba != 0) & (Bb != 0)
            den = both.sum(1)
            eq = (((Ba & Bb) != 0) & both).sum(1)   # 共用至少一塊遮罩
        else:
            La, Lb = Lk[a_ids], Lk[b_ids]
            both = (La > 0) & (Lb > 0)
            den = both.sum(1)
            eq = ((La == Lb) & both).sum(1)          # 遮罩號相等
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(den > 0, eq / np.maximum(den, 1), 0.0)
        return (den > 0) & (frac >= args.agree_frac)

    # 沿三軸找相鄰 kept 體素對
    for axis in range(3):
        a = idx_grid
        sl_a = [slice(None)] * 3; sl_b = [slice(None)] * 3
        sl_a[axis] = slice(0, -1); sl_b[axis] = slice(1, None)
        ia = idx_grid[tuple(sl_a)].ravel(); ib = idx_grid[tuple(sl_b)].ravel()
        m = (ia >= 0) & (ib >= 0)
        ia, ib = ia[m], ib[m]
        if ia.size == 0:
            continue
        ok = agree(ia, ib)
        for x, y in zip(ia[ok], ib[ok]):
            uf.union(int(x), int(y))

    roots = np.array([uf.find(i) for i in range(nkeep)])
    uniq, inv, counts = np.unique(roots, return_inverse=True, return_counts=True)
    # 6) 過濾小元件,組 instance
    kept_P = P[keep]
    instances = []
    nsplit = 0
    order = np.argsort(-counts)
    for ci in order:
        if counts[ci] < args.min_vox:
            continue
        sel_idx = np.where(inv == ci)[0]
        # mask-split:把焊在一起的不同物體依「對應乾淨遮罩」切開(橋段丟掉)
        if args.mask_split:
            subs = split_component_by_mask(Lk[sel_idx], [v["adj"] for v in views],
                                           args.split_agree, 3, args.min_vox)
            if not subs:
                continue
            if len(subs) > 1:
                nsplit += 1
        else:
            subs = [np.ones(len(sel_idx), dtype=bool)]
        # 逐子物體建 instance
        for sub in subs:
            gidx = sel_idx[sub]
            comp_P = kept_P[gidx]
            comp_L = Lk[gidx]
            comp_B = Bk[gidx] if args.multi_label else None
            center = comp_P.mean(axis=0)
            thresh = max(2, int(0.05 * len(gidx)))
            _build_instance(views, comp_L, comp_B, center, len(gidx), thresh,
                            args, instances)
    if args.mask_split:
        print(f"mask-split 切開的元件數: {nsplit}")
    instances.sort(key=lambda a: -a["support"])
    if args.sig_merge:
        n0 = len(instances)
        instances = sig_merge(instances, args.sig_overlap)
        print(f"簽章合併: {n0} → {len(instances)}")
    print(f"{scene}: {n} views, 體素 {nx}x{ny}x{nz}, carved={nkeep}, "
          f"元件={len(uniq)} → instances={len(instances)}")

    # 7) GT 驗證
    gt = []
    mani = scene_dir / "scene_manifest.json"
    if mani.is_file():
        m = json.loads(mani.read_text(encoding="utf-8"))
        for o in m["actual"]["viewpoints"][0]["objects"]:
            gt.append((o["name"], np.array(o["position_m"], dtype=np.float64)))
    report = [f"scene: {scene}  (VOXEL 版)", f"views: {n}",
              f"voxel={vx} keep_frac={args.keep_frac}(keep_min={keep_min}) "
              f"agree_frac={args.agree_frac} min_vox={args.min_vox}",
              f"carved={nkeep}  instances: {len(instances)}  (GT 物體數: {len(gt)})", ""]
    for k, inst in enumerate(instances):
        c = np.array(inst["center"])
        line = (f"inst_{k:02d}: center=({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f}) "
                f"support={inst['support']}/{n} vox={inst['n_vox']}")
        if gt:
            name, dmin = min(((nm, float(np.linalg.norm(c - p))) for nm, p in gt),
                             key=lambda a: a[1])
            line += f"  最近GT={name} ({dmin*100:.1f}cm)"
        report.append(line)
    txt = "\n".join(report)
    print("\n" + txt)

    (out_dir / "instances.json").write_text(
        json.dumps({"scene": scene, "method": "voxel",
                    "centers": [i["center"] for i in instances],
                    "instances": instances}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "assoc_report.txt").write_text(txt + "\n", encoding="utf-8")
    print(f"\n→ {out_dir}/instances.json、assoc_report.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["n3_scene0001"], help="場景名或組號")
    ap.add_argument("--voxel", type=float, default=0.015, help="體素邊長(m)")
    ap.add_argument("--keep-frac", type=float, default=0.6, dest="keep_frac",
                    help="space carving:落在遮罩內的 view 比例門檻")
    ap.add_argument("--agree-frac", type=float, default=0.5, dest="agree_frac",
                    help="相鄰體素遮罩號一致比例門檻(連通)")
    ap.add_argument("--min-vox", type=int, default=8, dest="min_vox",
                    help="連通元件最小體素數(濾雜訊)")
    ap.add_argument("--sig-merge", action="store_true", dest="sig_merge",
                    help="開啟簽章合併(把同物體碎片用共用遮罩號黏回;預設關)")
    ap.add_argument("--sig-overlap", type=float, default=0.5, dest="sig_overlap",
                    help="簽章合併:共用遮罩的 view 比例門檻")
    ap.add_argument("--multi-label", action="store_true", dest="multi_label",
                    help="體素可同屬多遮罩(重疊不再最小者勝);連通改'共用遮罩有交集'。"
                         "輸出獨立目錄 instance_hull_voxel_ml(預設關)")
    ap.add_argument("--mask-split", action="store_true", dest="mask_split",
                    help="後處理:對每個元件依'重投影對應哪塊乾淨遮罩'全局切開(治幻影橋焊住的相鄰物體);"
                         "橋段體素丟掉。輸出加 _ms(預設關)")
    ap.add_argument("--split-agree", type=float, default=0.7, dest="split_agree",
                    help="mask-split:體素與種子在多數共同 view 對應到同一或相鄰遮罩的比例門檻")
    ap.add_argument("--split-gap", type=int, default=4, dest="split_gap",
                    help="mask-split:遮罩相鄰判定的像素間距容忍(<=此距離視為相鄰=同物體)")
    ap.add_argument("--max-seethrough", type=int, default=None, dest="max_seethrough",
                    help="see-through 過濾:被>此數台相機投影到背景的體素=幻影,刪除(預設 None=關;"
                         "建議 1~2)。輸出加 _st")
    args = ap.parse_args()
    scenes = resolve_scenes(args.scenes or ["n3_scene0001"])
    if not scenes:
        sys.exit("沒有場景")
    for i, scene in enumerate(scenes, 1):
        print(f"\n===== [{i}/{len(scenes)}] {scene} =====")
        try:
            process_scene(scene, args)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[error] {scene}: {e}")


if __name__ == "__main__":
    main()
