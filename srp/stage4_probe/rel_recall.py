#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""rel_recall.py — 幾何判斷下,各關係「存在時被找到的比例(recall)」。

關係(都用幾何/hull 判斷):
  前後左右(方向):由質心 x(前後)、y(左右)+ 死區門檻;GT 用 GT mesh 質心,預測用 hull instance 質心。
  on / blocks_access:GT 取 relations.json;預測用 a1_rule 的規則作用在 hull instance 上。
指標:recall = 找到數 / GT 存在數(三元組 (type,x,y) 精確配對)。
範圍:sam_only 內除 n1 外(n3/n4/n5 + stack/occ)。
用法: ./srp/stage4_probe/rel_recall.py [groups...]   預設 n3 n4 n5 stack3 stack4 stack5 occ3 occ4 occ5
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
sys.path.insert(0, str(REPO / "srp" / "stage4_probe"))
import eval_mesh as EM          # noqa: E402
import a1_rule as A1            # noqa: E402  (entity_geom, rule_on, rule_blocks, iou3)

HULL = REPO / "data" / "eval" / "srp_hull"
LABELS = REPO / "data" / "labels"
DIR_THR = 0.03   # 方向死區(m):質心差小於此不算該軸關係


def centroid(occ, gm, vs):
    idx = np.array(np.nonzero(occ)).T
    return gm + (idx.mean(0) + 0.5) * vs


def dir_triples(cents):
    """cents: {name:(x,y,z)} → 方向三元組集合(right/left/front/back)。"""
    out = set()
    for A in cents:
        for B in cents:
            if A == B:
                continue
            ax, ay = cents[A][0], cents[A][1]; bx, by = cents[B][0], cents[B][1]
            if ay - by > DIR_THR:
                out.add(("right", A, B))
            elif by - ay > DIR_THR:
                out.add(("left", A, B))
            if ax - bx > DIR_THR:
                out.add(("front", A, B))
            elif bx - ax > DIR_THR:
                out.add(("back", A, B))
    return out


def hull_entities_and_cents(scene, gm, vs, shape):
    """回傳 (G:{name:geom}, cents:{name:(x,y,z)});instance→GT 名以 3D IoU 配對。"""
    ip = HULL / scene / "instances.npz"
    if not ip.is_file():
        return None, None
    labels = np.load(ip)["labels"]
    gt = EM.solid_mesh_occ(scene, gm, vs, shape)
    if not gt:
        return None, None
    names = list(gt); meshes = [gt[n] for n in names]
    insts = [k for k in range(1, int(labels.max()) + 1) if (labels == k).any()]
    occs = [labels == k for k in insts]
    M = np.array([[A1.EM.iou3(o, m) for m in meshes] for o in occs]) if occs else np.zeros((0, 0))
    G, cents = {}, {}
    if len(occs) and len(names):
        ri, cj = linear_sum_assignment(-M)
        for i, j in zip(ri, cj):
            if M[i, j] > 0:
                G[names[j]] = A1.entity_geom(occs[i], gm, vs)
                cents[names[j]] = centroid(occs[i], gm, vs)
    return G, cents


def process(scene, acc):
    hp = HULL / scene / "hull.npz"; rp = LABELS / scene / "relations.json"
    if not hp.is_file():
        return
    z = np.load(hp); gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = z["occupancy"].shape
    gt_occ = EM.solid_mesh_occ(scene, gm, vs, shape)
    if not gt_occ:
        return
    gt_cents = {n: centroid(o, gm, vs) for n, o in gt_occ.items()}
    G, pred_cents = hull_entities_and_cents(scene, gm, vs, shape)
    if G is None:
        return
    matched = set(pred_cents)                       # hull 有找到(配對到 instance)的物體
    def both(tr):                                    # 關係兩端物體是否都被找到
        return tr[1] in matched and tr[2] in matched
    # acc[type] = [GT全部, 找到(TP), GT(雙方都找到), 預測數]
    # 方向(前後左右)
    gt_dir = dir_triples(gt_cents); pred_dir = dir_triples(pred_cents)
    acc["dir"][0] += len(gt_dir); acc["dir"][1] += len(gt_dir & pred_dir)
    acc["dir"][2] += sum(both(t) for t in gt_dir); acc["dir"][3] += len(pred_dir)
    # on / block:GT 取 relations.json,預測用規則作用 hull
    if rp.is_file():
        rels = json.loads(rp.read_text())["relations"]
        gt_on = {("on", r["x"], r["y"]) for r in rels if r["type"] == "on"}
        gt_blk = {("blocks_access", r["x"], r["y"]) for r in rels if r["type"] == "blocks_access"}
        pred_on = set(A1.rule_on(G)); pred_blk = set(A1.rule_blocks(G, scene))
        acc["on"][0] += len(gt_on); acc["on"][1] += len(gt_on & pred_on)
        acc["on"][2] += sum(both(t) for t in gt_on); acc["on"][3] += len(pred_on)
        acc["blk"][0] += len(gt_blk); acc["blk"][1] += len(gt_blk & pred_blk)
        acc["blk"][2] += sum(both(t) for t in gt_blk); acc["blk"][3] += len(pred_blk)


def main():
    groups = sys.argv[1:] or ["n3", "n4", "n5", "stack3", "stack4", "stack5", "occ3", "occ4", "occ5"]
    scenes = []
    for g in groups:
        scenes += sorted(Path(p).name for p in glob.glob(str(HULL / f"{g}_scene*")))
    acc = {"dir": [0, 0, 0, 0], "on": [0, 0, 0, 0], "blk": [0, 0, 0, 0]}
    for i, sc in enumerate(scenes, 1):
        try:
            process(sc, acc)
        except Exception as e:
            print(f"[err] {sc}: {e}")
        if i % 40 == 0:
            print(f"  ...{i}/{len(scenes)}")
    print(f"\n範圍:{len(scenes)} 場景({' '.join(groups)})")
    print(f"{'關係':<18}{'GT':>6}{'預測':>6}{'找到':>6}{'recall':>8}{'prec':>7}{'F1':>7}{'recall|雙方':>11}")
    name = {"dir": "前後左右(方向)", "on": "on(支撐)", "blk": "blocks_access(遮擋)"}
    for k in ("dir", "on", "blk"):
        tot, found, both, pred = acc[k]
        r = found / tot if tot else float("nan")
        p = found / pred if pred else float("nan")
        f1 = 2 * p * r / (p + r) if (p == p and r == r and p + r > 0) else float("nan")
        rc = found / both if both else float("nan")
        print(f"{name[k]:<16}{tot:>6}{pred:>6}{found:>6}{r:>8.3f}{p:>7.3f}{f1:>7.3f}{rc:>11.3f}")


if __name__ == "__main__":
    main()
