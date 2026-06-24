"""srp_hull_viz — 依 controllerArgs 指定場景,把 srp hull(每 instance 上色)+ 真實 YCB GT 載入 Webots 對照。

參數(worlds/hull_viz.wbt 的 controllerArgs,或 SRP_VIZ_ARGS 環境變數優先):
  controllerArgs [ "<scene>" "<show_gt:1/0>" "<root>" ]   # root 預設 srp_hull
  例: controllerArgs [ "n3_scene0030" "1" ]               # baseline
      controllerArgs [ "n3_scene0030" "1" "srp_hull_am1" ] # 看 allow_miss=1 那組
流程:用 webots_visual_hull python 跑 srp/stage2_instances/gen_viz_objs.py 產 obj+manifest
      → importMFNodeFromString 把每個 instance hull(半透明上色)+ 真實 YCB 模型插入。
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
GEN_PY = REPO / "srp" / "stage2_instances" / "gen_viz_objs.py"
PYTHON = "/home/cho/.pyenv/versions/webots_visual_hull/bin/python3"


def parse_args():
    _env = os.environ.get("SRP_VIZ_ARGS")
    a = _env.split() if _env else sys.argv[1:]
    scene = a[0] if len(a) >= 1 and a[0] else "n3_scene0030"
    show_gt = (a[1] != "0") if len(a) >= 2 else True
    root = a[2] if len(a) >= 3 and a[2] else "srp_hull"
    tag = a[3] if len(a) >= 4 and a[3] else ""
    return scene, show_gt, root, tag


def generate(scene, show_gt, root, tag):
    OBJ_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, str(GEN_PY), scene, "--out", str(OBJ_DIR), "--root", root]
    if tag:
        cmd += ["--tag", tag]
    if not show_gt:
        cmd.append("--no-gt")
    print("[srp_viz] 產生 obj:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print("[srp_viz] 產生失敗:\n", r.stderr[-2000:]); return None
    mani = OBJ_DIR / "manifest.json"
    return json.loads(mani.read_text(encoding="utf-8")) if mani.is_file() else None


def solid_vrml(item):
    """instance hull 殼:obj 世界座標,放原點。"""
    r, g, b = item["color"]; t = item["transparency"]
    url = str((OBJ_DIR / item["file"]).resolve())
    return (f'Solid {{ translation 0 0 0 name "{item["name"]}" children [ '
            f'Shape {{ appearance PBRAppearance {{ baseColor {r} {g} {b} '
            f'metalness 0 roughness 1 transparency {t} }} '
            f'geometry Mesh {{ url [ "{url}" ] }} }} ] }}')


def ycb_vrml(item):
    """真實 YCB 模型:擺 GT 位姿(translation = pos - R@center,rotation = axis-angle)。"""
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
    scene, show_gt, root, tag = parse_args()
    print(f"[srp_viz] scene={scene} show_gt={show_gt} root={root} tag={tag}")
    mani = generate(scene, show_gt, root, tag)
    if mani:
        children = sv.getRoot().getField("children")
        for it in mani.get("items", []):
            children.importMFNodeFromString(-1, solid_vrml(it))
        for it in mani.get("ycb_items", []):
            children.importMFNodeFromString(-1, ycb_vrml(it))
        print(f"[srp_viz] 已載入 instance hull {mani['hull']} + 真實YCB {mani.get('ycb', 0)}(場景 {mani['source']})")
    step = int(sv.getBasicTimeStep())
    while sv.step(step) != -1:
        pass


if __name__ == "__main__":
    main()
