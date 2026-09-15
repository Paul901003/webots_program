#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""scene_graph_gt.py — 產生「物體級關係 GT」(乾淨真相,不碰遮罩/模型)。
每場景輸出 data/labels/<scene>/scene_graph_gt/ :
  gt.txt(文字鄰接矩陣) / gt.json(機器可讀) / global.png(支撐/前後/左右熱圖) / occlusion.png(12視角遮擋熱圖)

四種關係(物體級鄰接矩陣,有向 A→B):
  ★ 全部矩陣統一「列A 當主語」:讀「A [關係] B」(A擋B / A在B上 / A在B前 / A在B左)。
  ① 支撐 sup: on(A在B上)/under(A在B下)   來源 relations.json
  ② 前後 fb : 機器人在 x=-0.4 面朝+x,x小=近=前。A.x > B.x+THR → A→B「後」(A較遠);A.x < B.x−THR → 「前」(A較近)
  ③ 左右 lr : 機器人左=+y。A.y > B.y+THR → A→B「左」(A較+y);A.y < B.y−THR →「右」(A較−y)
  ④ 遮擋 occ: 12 個 A-3 挑選視角,各一個 N×N(A擋B=1)  來源 relations.json blocks_access
  方向質心 = GT 實心 mesh 佔據體素中心均值(solid_mesh_occ),死區 THR=0.03m。
用法: ./scene_graph_gt.py [scene|group|(空=全部)]
"""
import argparse, sys, json, glob, re
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "srp" / "io"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage2_instances"))
import eval_mesh as EM
import viewpoints as VP
from labels import label_dir

REPO = Path(__file__).resolve().parents[2]
import os as _os  # HULL_ROOT 可覆寫(僅取 grid，各 hull 法一致)；預設維持 srp_hull_v12
HULL = Path(_os.environ.get("HULL_ROOT", str(REPO / "data" / "eval" / "srp_hull_v12")))
DIR_THR = 0.03
sh = lambda s: s.split("_")[-1]

SUP = {0: "·", 1: "on", 2: "under"}     # A→B: 1=A在B上, 2=A在B下
FB = {0: "·", 1: "前", 2: "後"}          # A→B: 1=B較近(前), 2=B較遠(後)
LR = {0: "·", 1: "左", 2: "右"}          # A→B: 1=B較左, 2=B較右


def build(sc):
    z = np.load(HULL / sc / "hull.npz"); gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = z["occupancy"].shape
    gtocc = EM.solid_mesh_occ(sc, gm, vs, shape)
    objs = [n for n, o in gtocc.items() if int(o.sum()) > 50]
    cen = {n: (gm + (np.array(np.nonzero(gtocc[n])).T + 0.5) * vs).mean(0) for n in objs}
    N = len(objs); idx = {n: i for i, n in enumerate(objs)}

    # relations.json
    on_rel = set(); blocks = defaultdict(set)   # blocks[view] = {(x,y)}
    rp = label_dir(sc) / "relations.json"
    if rp.is_file():
        for r in json.load(open(rp)).get("relations", []):
            if r["type"] == "on": on_rel.add((r["x"], r["y"]))
            elif r["type"] == "blocks_access": blocks[r["view"]].add((r["x"], r["y"]))

    # ① 支撐矩陣
    Msup = np.zeros((N, N), int)
    for a in objs:
        for b in objs:
            if a == b: continue
            if (a, b) in on_rel: Msup[idx[a], idx[b]] = 1     # a on b
            elif (b, a) in on_rel: Msup[idx[a], idx[b]] = 2   # a under b
    # ② 前後 ③ 左右(GT 質心,機器人錨定)
    Mfb = np.zeros((N, N), int); Mlr = np.zeros((N, N), int)
    for a in objs:
        for b in objs:
            if a == b: continue
            # 列A 當主語(與遮擋/支撐一致):值描述 A 相對 B。A 在 B 前/後、A 在 B 左/右。
            dx = cen[a][0] - cen[b][0]; dy = cen[a][1] - cen[b][1]
            if dx > DIR_THR: Mfb[idx[a], idx[b]] = 2          # A較遠 → A在B後
            elif dx < -DIR_THR: Mfb[idx[a], idx[b]] = 1       # A較近 → A在B前
            if dy > DIR_THR: Mlr[idx[a], idx[b]] = 1          # A較+y → A在B左
            elif dy < -DIR_THR: Mlr[idx[a], idx[b]] = 2       # A較−y → A在B右
    # ④ 遮擋(12 視角,各 N×N;A擋B=1)
    views = sorted(VP.selected_view_names(12))
    Mocc = {}
    for v in views:
        M = np.zeros((N, N), int)
        for (x, y) in blocks.get(v, set()):
            if x in idx and y in idx: M[idx[x], idx[y]] = 1
        Mocc[v] = M
    return objs, cen, Msup, Mfb, Mlr, Mocc, views


def txt_matrix(objs, M, mapping):
    hdr = "        " + " ".join(f"{sh(o)[:6]:>6}" for o in objs)
    rows = [hdr]
    for a, oa in enumerate(objs):
        rows.append(f"{sh(oa)[:6]:>6}  " + " ".join(f"{mapping[M[a, b]]:>6}" for b in range(len(objs))))
    return "\n".join(rows)


def save_all(sc, objs, cen, Msup, Mfb, Mlr, Mocc, views):
    out = label_dir(sc) / "scene_graph_gt"; out.mkdir(parents=True, exist_ok=True)
    # ── 文字 ──
    L = [f"場景 {sc}", f"物體({len(objs)}): " + ", ".join(sh(o) for o in objs),
         "GT 質心(x,y,z):"]
    for o in objs: L.append(f"  {sh(o):14} ({cen[o][0]:.3f}, {cen[o][1]:.3f}, {cen[o][2]:.3f})")
    L += ["", "① 支撐 on/under (列A→欄B, A在B上=on / A在B下=under)", txt_matrix(objs, Msup, SUP),
          "", "② 前後 (列A→欄B, A在B前=前 / A在B後=後;機器人x=-0.4面朝+x,x小=前)", txt_matrix(objs, Mfb, FB),
          "", "③ 左右 (列A→欄B, A在B左=左 / A在B右=右;機器人左=+y)", txt_matrix(objs, Mlr, LR),
          "", "④ 遮擋 per-view (A擋B=遮),12 挑選視角:"]
    for v in views:
        has = Mocc[v].sum() > 0
        L.append(f"  [{v}]" + ("" if has else "  (無遮擋)"))
        if has:
            for a, oa in enumerate(objs):
                for b, ob in enumerate(objs):
                    if Mocc[v][a, b]: L.append(f"      {sh(oa)} 擋 {sh(ob)}")
    (out / "gt.txt").write_text("\n".join(L), encoding="utf-8")

    # ── json ──
    J = {"scene": sc, "objects": [sh(o) for o in objs],
         "objects_full": objs, "centroids": {sh(o): cen[o].tolist() for o in objs},
         "support": Msup.tolist(), "front_back": Mfb.tolist(), "left_right": Mlr.tolist(),
         "support_legend": SUP, "fb_legend": FB, "lr_legend": LR,
         "occlusion": {v: Mocc[v].tolist() for v in views}}
    (out / "gt.json").write_text(json.dumps(J, ensure_ascii=False, indent=1), encoding="utf-8")

    # ── 熱圖 ──
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try: plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    except Exception: pass
    names = [sh(o)[:6] for o in objs]

    def heat(ax, M, title, vmax, xtop=False):
        Mm = np.ma.masked_equal(np.asarray(M), 0)          # 0(對角線+無關)→留白
        cmap = plt.cm.viridis.copy(); cmap.set_bad("white")
        ax.imshow(Mm, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(len(objs)))
        ax.set_yticks(range(len(objs))); ax.set_yticklabels(names, fontsize=7)
        ax.set_xticks(np.arange(-.5, len(objs), 1), minor=True)   # 格線讓白格看得出邊界
        ax.set_yticks(np.arange(-.5, len(objs), 1), minor=True)
        ax.grid(which="minor", color="lightgray", linewidth=.5); ax.tick_params(which="minor", length=0)
        ax.set_ylabel("A")
        if xtop:                                            # B(欄)物體名移到矩陣上方
            ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
            ax.set_xticklabels(names, rotation=45, ha="left", fontsize=7); ax.set_xlabel("B")
            if title: ax.text(0.5, -0.16, title, transform=ax.transAxes, ha="center", va="top", fontsize=9)  # 標題移到底部,不撞上方B名
        else:
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
            ax.set_title(title, fontsize=9); ax.set_xlabel("B")
        for a in range(len(objs)):
            for b in range(len(objs)):
                if M[a, b]: ax.text(b, a, int(M[a, b]), ha="center", va="center", color="white", fontsize=8)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    heat(ax[0], Msup, "support(1=on,2=under)", 2, xtop=True)
    heat(ax[1], Mfb, "front-back(1=front,2=back)", 2, xtop=True)
    heat(ax[2], Mlr, "left-right(1=left,2=right)", 2, xtop=True)
    fig.suptitle(f"{sc}  物體級關係(A→B)"); fig.tight_layout()
    fig.savefig(out / "global.png", dpi=100); plt.close(fig)

    # 每視角:實際影像(captures_fast)+ 遮擋矩陣 並排,免自己從 34 視角挑對照
    grp = sc.split("_")[0]
    capdir = REPO / "data" / "captures_fast" / f"multi_{grp}" / sc
    nv = len(views); per_row = 2                     # 每列放 2 視角,每視角占「影像+矩陣」2 格
    nrows = (nv + per_row - 1) // per_row
    fig2 = plt.figure(figsize=(per_row * 5.2, nrows * 2.4))
    gs = fig2.add_gridspec(nrows, per_row * 2, width_ratios=[3, 1] * per_row)
    for k, v in enumerate(views):
        r, c = k // per_row, (k % per_row) * 2
        axi = fig2.add_subplot(gs[r, c]); axm = fig2.add_subplot(gs[r, c + 1])
        ip = capdir / f"{v}.png"
        if ip.is_file(): axi.imshow(plt.imread(ip))
        else: axi.text(0.5, 0.5, "(無影像)", ha="center", va="center", transform=axi.transAxes)
        axi.set_title(v.replace("view_", ""), fontsize=8); axi.axis("off")
        heat(axm, Mocc[v], "", 1, xtop=True)
    fig2.suptitle(f"{sc}  遮擋 per-view:影像 + 矩陣(A擋B=1),12 挑選視角")
    fig2.tight_layout()
    fig2.savefig(out / "occlusion.png", dpi=95); plt.close(fig2)
    return out


def resolve(t):
    if not t:  # 全跑排除單物體(n1);單物體無物體間關係、不參與模型
        scenes = [Path(p).parent.name for p in glob.glob(str(HULL / "*_scene*/hull.npz"))]
        return sorted(s for s in scenes if not re.match(r"^[a-z]+1_scene", s))
    out = []
    for a in t:
        if "scene" in a: out.append(a)
        else: out += [Path(p).parent.name for p in glob.glob(str(HULL / f"{a}_scene*/hull.npz"))]
    return sorted(set(out))


def _process_one(sc):
    """單場:build + save_all。回傳 (場名, 成功?, 錯誤訊息)。供多進程 Pool 用。"""
    try:
        r = build(sc); save_all(sc, *r)
        return (sc, True, "")
    except Exception:
        import traceback
        return (sc, False, traceback.format_exc())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("targets", nargs="*"); args = ap.parse_args()
    scenes = resolve(args.targets)
    # 每場獨立 → 多進程併發(matplotlib Agg 需分進程,不能用執行緒)。SG_JOBS 可覆寫;預設併發。
    jobs = int(_os.environ.get("SG_JOBS", str(min(6, (_os.cpu_count() or 4)))))
    n = 0
    if jobs > 1 and len(scenes) > 1:
        import multiprocessing as mp
        with mp.Pool(jobs) as pool:
            for i, (sc, ok, err) in enumerate(pool.imap_unordered(_process_one, scenes)):
                if ok: n += 1
                else: print(f"[err] {sc}: {err.strip().splitlines()[-1]}")
                if (i + 1) % 60 == 0: print(f"...{i+1}/{len(scenes)}", flush=True)
    else:
        for i, sc in enumerate(scenes):
            sc, ok, err = _process_one(sc)
            if ok:
                n += 1
                if len(scenes) <= 2: print((label_dir(sc) / "scene_graph_gt" / "gt.txt").read_text(encoding="utf-8"))
            else: print(f"[err] {sc}: {err.strip().splitlines()[-1]}")
            if (i + 1) % 60 == 0: print(f"...{i+1}/{len(scenes)}", flush=True)
    print(f"\n完成 {n} 場 → data/labels/<scene>/scene_graph_gt/(gt.txt / gt.json / global.png / occlusion.png)  [SG_JOBS={jobs}]")


if __name__ == "__main__":
    main()
