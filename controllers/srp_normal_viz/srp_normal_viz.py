"""srp_normal_viz — 表面法向量線段(每群一個 IndexedLineSet + Material.emissiveColor 上色)插進 Webots。

參數(SRP_VIZ_ARGS 優先,否則 controllerArgs): [ "<scene>" "<root>" ]
每支箭頭 = 線段(表面點 → 表面點+法向*長度),方向由實際世界座標端點決定,不碰 primitive 軸慣例。
每群一個 Shape,用舊式 Appearance{material Material{emissiveColor 群色}}(VRML 線段標準上色,PBRAppearance 對線不吃色)。
流程:webots_visual_hull python 跑 gen_normals_viz.py 產 shapes → 每群一個 IndexedLineSet 插入。
"""
from controller import Supervisor
import json
import os
import subprocess
import sys
from pathlib import Path

CUR = Path(__file__).resolve().parent
REPO = CUR.parent.parent
GEN = REPO / "srp" / "stage2_instances" / "gen_normals_viz.py"
PYTHON = "/home/cho/.pyenv/versions/webots_visual_hull/bin/python3"
JSON = CUR / "normals.json"


def parse_args():
    env = os.environ.get("SRP_VIZ_ARGS")
    a = env.split() if env else sys.argv[1:]
    scene = a[0] if len(a) >= 1 and a[0] else "stack3_scene0001"
    root = a[1] if len(a) >= 2 and a[1] else "srp_hull_semcluster_surf_am1photo"
    return scene, root


def main():
    sup = Supervisor()
    scene, root = parse_args()
    cmd = [PYTHON, str(GEN), scene, "--root", root]
    print("[normal_viz] 產生:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print("[normal_viz] 失敗:\n", r.stderr[-2000:]); return
    d = json.loads(JSON.read_text())
    children = sup.getRoot().getField("children")
    for sh in d["shapes"]:
        cr, cg, cb = sh["color"]
        pts = ", ".join(f"{x} {y} {z}" for x, y, z in sh["points"])
        cidx = " ".join(str(i) for i in sh["index"])
        vrml = (f'Pose {{ children [ Shape {{ '
                f'appearance Appearance {{ material Material {{ emissiveColor {cr} {cg} {cb} }} }} '
                f'geometry IndexedLineSet {{ '
                f'coord Coordinate {{ point [ {pts} ] }} '
                f'coordIndex [ {cidx} ] }} }} ] }}')
        children.importMFNodeFromString(-1, vrml)
    total = sum(len(s["index"]) // 3 for s in d["shapes"])
    print(f"[normal_viz] 插入 {len(d['shapes'])} 群 / {total} 支法向線段(場景 {scene})")
    while sup.step(32) != -1:
        pass


if __name__ == "__main__":
    main()
