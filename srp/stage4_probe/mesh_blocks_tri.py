#!/home/cho/.pyenv/versions/3.10.10/bin/python3
"""mesh_blocks_tri.py — [已棄用] 純三角面 mesh 的 blocks_access(pyrender 深度 z-buffer,不體素化)。

⚠ 已棄用:blocks 預測改用「重投影遮罩 − IoU最高 SAM 遮罩」(見 a1_rule.rule_blocks),與 GT 的
   amodal−modal 同套遮罩差。本檔的深度 z-buffer 與 GT 定義不對齊,僅留作歷史對照,勿再用於評估。


對照 a1_rule.py --geom mesh(mesh 先體素化 → 體素中心 z-buffer),看**體素化吃掉多少符合度**。
- 幾何:每物體 mesh 三角面(manifest 位姿,與 GT amodal 同源),full-res 渲染深度(1280×720)。
- 遮擋規則:與 a1_rule.rule_blocks 同義 —— 每視角對 (X 遮擋者, Y 被遮者)
  `front = (Y可見) ∧ (X可見) ∧ (depth_X < depth_Y)`;`occ_frac = |front|/|Y可見|`;取最大且 ≥OCC_MIN。
- 每視角逐條 → 四元組 (blocks_access, X, Y, view),與 GT relations.json 四元組精確配對。

GT blocks 用 amodal−modal 遮罩差,本版用三角面深度;幾何同源(full-res 三角面)→ 差異僅「遮擋定義」;
與體素版的差 = 純體素化影響。需 pyrender 環境(3.10.10)。
用法: ./srp/stage4_probe/mesh_blocks_tri.py [groups...]  預設 n3 n4 n5 occ3 occ4 occ5 stack3 stack4 stack5
env: CAPTURES_ROOT(預設 data/captures)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyrender

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import generate_labels as GL          # noqa: E402  (相機/mesh 數學,與 amodal 同源)
from generate_amodal_masks import load_pose  # noqa: E402

CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures")))
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "srp" / "io")); from labels import LABELS  # data/labels 分層(類別/數量/場景)
ASSETS = str(REPO / "urdfs" / "ycb_assets")
OCC_MIN = 0.10   # 與 gt_relations / a1_rule 一致


def render_depth(renderer, mesh, tf, cam_pose, K):
    """單物體三角面深度圖(可見>0,背景 0)。"""
    scene = pyrender.Scene(bg_color=[0, 0, 0, 255])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), pose=tf)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    scene.add(pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=10.0), pose=cam_pose)
    _, depth = renderer.render(scene)
    return depth


def scene_pred(scene, renderer):
    """回傳該場景預測 blocks 四元組 set (blocks_access, X, Y, view)。"""
    g = scene.split("_")[0]
    sdir = CAPTURES / f"multi_{g}" / scene
    mani = sdir / "scene_manifest.json"
    if not mani.is_file():
        return set()
    objs = json.loads(mani.read_text())["actual"]["viewpoints"][0]["objects"]
    views = sorted(sdir.glob("view_*_pose.json"))
    K = GL.camera_intrinsics()
    names = sorted({o["name"] for o in objs})
    mesh_cache = {nm: GL.load_ycb_mesh(ASSETS, nm) for nm in names}
    # 每物體世界位姿 tf(name→tf;同名多物體取其一,與 amodal by-name 一致)
    tfs = {}
    for o in objs:
        nm = o["name"]
        p = o["position_m"]
        pos = np.array([p[0], p[1], p[2]] if isinstance(p, list) else [p["x"], p["y"], p["z"]], float)
        aa = o.get("rotation_axis_angle", [0, 1, 0, 0])
        R = GL._axis_angle_to_mat(np.array(aa[:3], float), aa[3])
        tf = np.eye(4); tf[:3, :3] = R; tf[:3, 3] = pos - R @ GL.ycb_center(nm)
        tfs[nm] = tf

    rels = set()
    for vp in views:
        vname = vp.stem.replace("_pose", "")
        cam_pos, rpy = load_pose(vp)
        cam_pose = GL.webots_camera_pose(cam_pos, rpy)
        depth = {}
        for nm in names:
            m = mesh_cache.get(nm)
            if m is None:
                continue
            d = render_depth(renderer, m, tfs[nm], cam_pose, K).astype(np.float32)
            d[d <= 0] = np.inf          # 背景/不可見 → inf
            depth[nm] = d
        for Y in depth:
            visY = depth[Y] < np.inf
            ay = int(visY.sum())
            if ay == 0:
                continue
            best_x, best_f = None, 0.0
            for X in depth:
                if X == Y:
                    continue
                front = visY & (depth[X] < depth[Y])   # X 可見(dX<inf)且比 Y 近
                f = int(front.sum()) / ay
                if f > best_f:
                    best_f, best_x = f, X
            if best_x is not None and best_f >= OCC_MIN:
                rels.add(("blocks_access", best_x, Y, vname))
    return rels


def gt_blocks(scene):
    f = LABELS / scene / "relations.json"
    if not f.is_file():
        return None
    return {("blocks_access", r["x"], r["y"], r["view"])
            for r in json.loads(f.read_text())["relations"] if r["type"] == "blocks_access"}


def main():
    groups = sys.argv[1:] or ["n3", "n4", "n5", "occ3", "occ4", "occ5", "stack3", "stack4", "stack5"]
    renderer = pyrender.OffscreenRenderer(GL.CAM_WIDTH, GL.CAM_HEIGHT)
    print(f"{'組':<8}{'TP':>6}{'FP':>6}{'FN':>6}{'P':>8}{'R':>8}{'F1':>8}")
    tot = [0, 0, 0]
    try:
        for g in groups:
            scenes = sorted(d.parent.name for d in (LABELS).glob(f"{g}_scene*/relations.json"))
            tp = fp = fn = 0
            for sc in scenes:
                gt = gt_blocks(sc)
                if gt is None:
                    continue
                try:
                    pred = scene_pred(sc, renderer)
                except Exception as e:
                    print(f"[err] {sc}: {e}"); continue
                tp += len(pred & gt); fp += len(pred - gt); fn += len(gt - pred)
            P = tp / (tp + fp) if tp + fp else 0.0
            R = tp / (tp + fn) if tp + fn else 0.0
            F = 2 * P * R / (P + R) if P + R else 0.0
            print(f"{g:<8}{tp:>6}{fp:>6}{fn:>6}{P:>8.3f}{R:>8.3f}{F:>8.3f}", flush=True)
            tot[0] += tp; tot[1] += fp; tot[2] += fn
    finally:
        renderer.delete()
    tp, fp, fn = tot
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    print("-" * 52)
    print(f"{'總計':<8}{tp:>6}{fp:>6}{fn:>6}{P:>8.3f}{R:>8.3f}{F:>8.3f}")


if __name__ == "__main__":
    main()
