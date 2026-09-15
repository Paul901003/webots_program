"""srp_colmap_viz — 依 controllerArgs 指定場景,把 COLMAP MVS 點雲(PointSet)+ 真實 YCB GT 載入 Webots 對照。

架構比照 srp_hull_viz,只把 hull mesh 換成 COLMAP dense 點雲(以 Webots PointSet 顯示原始點)。
參數(worlds/srp_colmap_viz.wbt 的 controllerArgs,或 SRP_COLMAP_VIZ_ARGS 環境變數優先):
  controllerArgs [ "<scene>" "<show_gt:1/0>" ]
  例: controllerArgs [ "stack3_scene0001" "1" ]
流程:用 webots_visual_hull python 跑 srp/stage2_instances/gen_colmap_viz.py 產 ascii ply + manifest
      → importMFNodeFromString 把點雲(PointSet)+ 真實 YCB 模型(半透明 mesh)插入。
前置:先跑 srp/stage2_instances/colmap_mvs.py 產 data/eval/colmap_mvs/<scene>/fused.ply。
改場景:改 controllerArgs 後重載世界(Ctrl+Shift+R)。
"""
from controller import Supervisor
import json
import os
import subprocess
import sys
from pathlib import Path

CUR = Path(__file__).resolve().parent
REPO = CUR.parent.parent
OBJ_DIR = CUR / "objs"
GEN_PY = REPO / "srp" / "stage2_instances" / "gen_colmap_viz.py"
PYTHON = "/home/cho/.pyenv/versions/webots_visual_hull/bin/python3"


def parse_args():
    _env = os.environ.get("SRP_COLMAP_VIZ_ARGS")
    a = _env.split() if _env else sys.argv[1:]
    scene = a[0] if len(a) >= 1 and a[0] else "stack3_scene0001"
    show_gt = (a[1] != "0") if len(a) >= 2 else True
    tag = a[2] if len(a) >= 3 and a[2] else ""
    return scene, show_gt, tag


def generate(scene, show_gt, tag):
    OBJ_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, str(GEN_PY), scene, "--out", str(OBJ_DIR)]
    if tag:
        cmd += ["--tag", tag]
    if not show_gt:
        cmd.append("--no-gt")
    print("[colmap_viz] 產生點雲+manifest:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print("[colmap_viz] 產生失敗:\n", r.stderr[-2000:]); return None
    mani = OBJ_DIR / "manifest.json"
    return json.loads(mani.read_text(encoding="utf-8")) if mani.is_file() else None


def pointcloud_vrml(ply_name):
    """COLMAP 點雲(ascii ply)→ Webots PointSet(每點帶顏色)。"""
    lines = (OBJ_DIR / ply_name).read_text().splitlines()
    hi = next(i for i, l in enumerate(lines) if l.strip() == "end_header")
    pts, cols = [], []
    for l in lines[hi + 1:]:
        p = l.split()
        if len(p) < 6:
            continue
        pts.append(f"{p[0]} {p[1]} {p[2]}")
        cols.append(f"{int(p[3])/255:.3f} {int(p[4])/255:.3f} {int(p[5])/255:.3f}")
    pt_str = ", ".join(pts); col_str = ", ".join(cols)
    return (f'Solid {{ translation 0 0 0 name "colmap_pointcloud" children [ '
            f'Shape {{ geometry PointSet {{ '
            f'coord Coordinate {{ point [ {pt_str} ] }} '
            f'color Color {{ color [ {col_str} ] }} }} }} ] }}')


def ycb_vrml(item):
    """真實 YCB 模型:擺 GT 位姿(同 srp_hull_viz)。"""
    r, g, b = item["color"]; t = item["transparency"]
    tx, ty, tz = item["translation"]; ax, ay, az, ang = item["rotation"]
    if abs(ax) + abs(ay) + abs(az) < 1e-9:
        ax, ay, az, ang = 0, 1, 0, 0
    return (f'Solid {{ translation {tx} {ty} {tz} rotation {ax} {ay} {az} {ang} '
            f'name "{item["name"]}" children [ '
            f'Shape {{ appearance PBRAppearance {{ baseColor {r} {g} {b} '
            f'metalness 0 roughness 1 transparency {t} }} '
            f'geometry Mesh {{ url [ "{item["mesh"]}" ] }} }} ] }}')


def main():
    sv = Supervisor()
    scene, show_gt, tag = parse_args()
    print(f"[colmap_viz] scene={scene} show_gt={show_gt} tag={tag}")
    mani = generate(scene, show_gt, tag)
    if mani:
        children = sv.getRoot().getField("children")
        if mani.get("pointcloud"):
            children.importMFNodeFromString(-1, pointcloud_vrml(mani["pointcloud"]))
        for it in mani.get("ycb_items", []):
            children.importMFNodeFromString(-1, ycb_vrml(it))
        print(f"[colmap_viz] 已載入 COLMAP 點雲 {mani['n_points']} 點 + 真實YCB {mani.get('ycb', 0)}(場景 {mani['source']})")
    step = int(sv.getBasicTimeStep())
    while sv.step(step) != -1:
        pass


if __name__ == "__main__":
    main()
