"""hull_viz_supervisor — 依 controllerArgs(場景+方法)動態載入該場景的 hull(+GT)到單一世界。

參數由 .wbt 的 controllerArgs 指定(指令輸入):
  controllerArgs [ "<scene>" "<root>" "<show_gt:1/0>" ]
  例: controllerArgs [ "n3_scene0030" "v3/instance_hull_voxel" "1" ]
流程:用 webots_visual_hull python 跑 instance_hull/gen_hull_objs.py 產 obj+manifest
      → 依 manifest importMFNodeFromString 把每個 hull/GT 當 Solid+Mesh 插入。
改場景:改 .wbt 的 controllerArgs 後重載世界(Ctrl+Shift+R)。
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
GEN_PY = REPO / "instance_hull" / "gen_hull_objs.py"
PYTHON = "/home/cho/.pyenv/versions/webots_visual_hull/bin/python3"


def parse_args():
    # HULL_VIZ_ARGS 環境變數優先於 .wbt controllerArgs(同 CAPTURE_ARGS 慣例)
    _env = os.environ.get("HULL_VIZ_ARGS")
    a = _env.split() if _env else sys.argv[1:]
    scene = a[0] if len(a) >= 1 and a[0] else "n3_scene0030"
    root = a[1] if len(a) >= 2 and a[1] else "v3/instance_hull_voxel"
    show_gt = (a[2] != "0") if len(a) >= 3 else True
    return scene, root, show_gt


def generate(scene, root, show_gt):
    cmd = [PYTHON, str(GEN_PY), scene, "--root", root, "--out", str(OBJ_DIR)]
    if not show_gt:
        cmd.append("--no-gt")
    print("[hull_viz] 產生 obj:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print("[hull_viz] 產生失敗:\n", r.stderr[-2000:]); return None
    mani = OBJ_DIR / "manifest.json"
    return json.loads(mani.read_text(encoding="utf-8")) if mani.is_file() else None


def solid_vrml(item):
    """hull / GT 視覺殼:obj 在 OBJ_DIR,放原點。"""
    r, g, b = item["color"]; t = item["transparency"]
    url = str((OBJ_DIR / item["file"]).resolve())
    return (f'Solid {{ translation 0 0 0 name "{item["name"]}" children [ '
            f'Shape {{ appearance PBRAppearance {{ baseColor {r} {g} {b} '
            f'metalness 0 roughness 1 transparency {t} }} '
            f'geometry Mesh {{ url [ "{url}" ] }} }} ] }}')


def ycb_vrml(item):
    """真實 YCB 模型:擺在 GT 位姿(translation = pos - R@center,rotation = axis-angle)。"""
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
    scene, root, show_gt = parse_args()
    print(f"[hull_viz] scene={scene} root={root} show_gt={show_gt}")
    mani = generate(scene, root, show_gt)
    if mani:
        children = sv.getRoot().getField("children")
        for it in mani.get("items", []):
            children.importMFNodeFromString(-1, solid_vrml(it))
        for it in mani.get("ycb_items", []):
            children.importMFNodeFromString(-1, ycb_vrml(it))
        print(f"[hull_viz] 已載入 hull {mani['hull']} + GT殼 {mani['gt']} + 真實YCB {mani.get('ycb',0)}(來源 {mani['source']})")
    step = int(sv.getBasicTimeStep())
    while sv.step(step) != -1:
        pass


if __name__ == "__main__":
    main()
