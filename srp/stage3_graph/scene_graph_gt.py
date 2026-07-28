#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""scene_graph_gt.py — 產生「物體級關係 GT」(乾淨真相,不碰遮罩/模型)。
每場景輸出 data/labels/<scene>/scene_graph_gt/ :
  gt.txt(文字鄰接矩陣) / gt.json(機器可讀) / global.png(支撐/前後/左右熱圖) / occlusion.png(12視角遮擋熱圖)

四種關係(物體級鄰接矩陣,有向 A→B):
  ① 支撐 sup: on(A在B上)/under(A在B下)   來源 relations.json
  ② 前後 fb : 機器人在 x=-0.4 面朝+x;物體離機器人越近(x小)=前。
              B.x > A.x+THR → A→B「後」(B較遠);B.x < A.x−THR → 「前」
  ③ 左右 lr : 機器人左=+y。B.y > A.y+THR → A→B「左」;B.y < A.y−THR →「右」
  ④ 遮擋 occ: 12 個 A-3 挑選視角,各一個 N×N(A擋B=1)  來源 relations.json blocks_access
  方向質心 = GT 實心 mesh 佔據體素中心均值(solid_mesh_occ),死區 THR=0.03m。
用法: ./scene_graph_gt.py [scene|group|(空=全部)]
"""
import argparse, sys, json, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "srp" / "io"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage2_instances"))
import eval_mesh as EM
import viewpoints as VP
from labels import label_dir

REPO = Path(__file__).resolve().parents[2]
HULL = REPO / "data" / "eval" / "srp_hull_v12"
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
            dx = cen[b][0] - cen[a][0]; dy = cen[b][1] - cen[a][1]
            if dx > DIR_THR: Mfb[idx[a], idx[b]] = 2          # B較遠 → 後
            elif dx < -DIR_THR: Mfb[idx[a], idx[b]] = 1       # B較近 → 前
            if dy > DIR_THR: Mlr[idx[a], idx[b]] = 1          # B較+y → 左
            elif dy < -DIR_THR: Mlr[idx[a], idx[b]] = 2       # B較−y → 右
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
          "", "② 前後 (機器人x=-0.4面朝+x;B較近=前 B較遠=後)", txt_matrix(objs, Mfb, FB),
          "", "③ 左右 (機器人左=+y;B較左=左 B較右=右)", txt_matrix(objs, Mlr, LR),
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

    def heat(ax, M, title, vmax):
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(objs))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(objs))); ax.set_yticklabels(names, fontsize=7)
        ax.set_title(title, fontsize=9); ax.set_xlabel("B"); ax.set_ylabel("A")
        for a in range(len(objs)):
            for b in range(len(objs)):
                if M[a, b]: ax.text(b, a, int(M[a, b]), ha="center", va="center", color="white", fontsize=8)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    heat(ax[0], Msup, "support(1=on,2=under)", 2)
    heat(ax[1], Mfb, "front-back(1=front,2=back)", 2)
    heat(ax[2], Mlr, "left-right(1=left,2=right)", 2)
    fig.suptitle(f"{sc}  物體級關係(A→B)"); fig.tight_layout()
    fig.savefig(out / "global.png", dpi=100); plt.close(fig)

    nv = len(views); cols = 4; rows = (nv + cols - 1) // cols
    fig2, ax2 = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    ax2 = np.array(ax2).reshape(-1)
    for k, v in enumerate(views):
        heat(ax2[k], Mocc[v], v.replace("view_", ""), 1)
    for k in range(nv, len(ax2)): ax2[k].axis("off")
    fig2.suptitle(f"{sc}  遮擋 per-view(A擋B=1),12 挑選視角"); fig2.tight_layout()
    fig2.savefig(out / "occlusion.png", dpi=90); plt.close(fig2)
    return out


def resolve(t):
    if not t: return sorted(Path(p).parent.name for p in glob.glob(str(HULL / "*_scene*/hull.npz")))
    out = []
    for a in t:
        if "scene" in a: out.append(a)
        else: out += [Path(p).parent.name for p in glob.glob(str(HULL / f"{a}_scene*/hull.npz"))]
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("targets", nargs="*"); args = ap.parse_args()
    scenes = resolve(args.targets); n = 0
    for i, sc in enumerate(scenes):
        try:
            r = build(sc); out = save_all(sc, *r); n += 1
            if len(scenes) <= 2: print((out / "gt.txt").read_text(encoding="utf-8"))
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")
        if (i + 1) % 60 == 0: print(f"...{i+1}/{len(scenes)}", flush=True)
    print(f"\n完成 {n} 場 → data/labels/<scene>/scene_graph_gt/(gt.txt / gt.json / global.png / occlusion.png)")


if __name__ == "__main__":
    main()
