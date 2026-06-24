"""visual_hull_viewer.py

檢視用 Supervisor(掛在 UR5e 上):
  手臂擺到 Home + 夾爪開 → 依外部指定場景,擺上 YCB 物體(實際位置, 靜態)
  + 疊上各物體算出的 visual hull(半透明紅),方便目視比對殼與物體。

指定場景(擇一):
  - 環境變數 VH_SCENE="n3_scene0002"
  - .wbt controllerArgs ["n3_scene0002"]

hull 來源 VH_SOURCE:
  - class(預設):per-class hull,data/eval/<VH_MASKDIR>/multi_n{N}/<scene>/visual_hull_<class>.obj
  - foreground :前景合併→連通元件分離,data/eval/foreground/<scene>/components/obj_*.obj
  - instance   :B 方法(幾何關聯→per-object 雕殼),data/eval/instance_hull/<scene>/visual_hull_inst_*.obj

讀:
  data/captures/multi_n{N}/<scene>/scene_manifest.json   物體名稱 + 實際位姿(actual)
"""

from controller import Supervisor
import json
import math
import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
YCB_SUP_DIR = REPO_ROOT / "controllers" / "ycb_supervisor"
if str(YCB_SUP_DIR) not in sys.path:
    sys.path.insert(0, str(YCB_SUP_DIR))

from config import ASSET_BASE, MASS_TABLE, DEFAULT_SHAPE, SHAPE_TABLE, PROMPT_TABLE  # noqa: E402

GEO = json.loads((YCB_SUP_DIR / "ycb_geometries.json").read_text(encoding="utf-8"))
CAPTURES = REPO_ROOT / "data" / "captures"
EVAL = REPO_ROOT / "data" / "eval"

ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
HOME_POSE_DEG = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]
GRIPPER = "ROBOTIQ 2F-140 Gripper"
# hull 與 mask 所在的子目錄(由 evaluate_masks 存出、build_torchhull 建殼);可用 VH_MASKDIR 覆寫
MASK_SUBDIR = os.environ.get("VH_MASKDIR", "grounded_sam_0.25_0.25_0.8")


# ── 名稱處理(與 build_torchhull / evaluate_masks 一致)─────────────────────────
def ycb_name_to_class(name: str) -> str:
    if name in PROMPT_TABLE:
        return PROMPT_TABLE[name]
    parts = name.split("_")
    start = 1 if parts[0].isdigit() else 0
    return " ".join(parts[start:])


def sanitize(value: str) -> str:
    out = []
    for ch in value.strip().lower():
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
    return "".join(out).strip("_")


def hull_filename(ycb_name: str) -> str:
    return f"visual_hull_{sanitize(ycb_name_to_class(ycb_name))}.obj"


# ── VRML 產生 ─────────────────────────────────────────────────────────────────
def make_object_vrml(name, pos, rot):
    """靜態(無 physics)YCB 物體,固定在實際位姿,方便和 hull 比對。"""
    geo = GEO.get(name, {"center": {"x": 0, "y": 0, "z": 0}})
    cx, cy, cz = geo["center"]["x"], geo["center"]["y"], geo["center"]["z"]
    base = f"{ASSET_BASE}/{name}/google_16k"
    rx, ry, rz, ra = (rot if rot and len(rot) == 4 else [0, 1, 0, 0])
    return f"""Solid {{
  translation {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}
  rotation {rx:.6f} {ry:.6f} {rz:.6f} {ra:.6f}
  children [
    Transform {{
      translation {-cx:.6f} {-cy:.6f} {-cz:.6f}
      children [
        Shape {{
          appearance PBRAppearance {{
            baseColorMap ImageTexture {{ url [ "{base}/texture_map.png" ] }}
            roughness 1
            metalness 0
          }}
          geometry Mesh {{ url [ "{base}/textured.obj" ] }}
        }}
      ]
    }}
  ]
  name "vh_obj_{name}"
}}"""


# 各物體 hull 用不同顏色區分(半透明)
HULL_COLORS = [
    (1.0, 0.0, 0.0),   # 紅
    (0.0, 0.6, 1.0),   # 藍
    (0.0, 0.8, 0.2),   # 綠
    (1.0, 0.8, 0.0),   # 黃
    (0.8, 0.0, 1.0),   # 紫
    (1.0, 0.4, 0.0),   # 橙
    (0.0, 0.9, 0.9),   # 青
    (1.0, 0.0, 0.6),   # 洋紅
]


def make_hull_vrml(mesh_path: Path, node_name: str, color):
    """visual hull(世界座標, translation 0),半透明、依物體不同色。"""
    url = str(mesh_path.resolve())
    r, g, b = color
    return f"""Solid {{
  translation 0 0 0
  children [
    Shape {{
      castShadows FALSE
      appearance PBRAppearance {{
        baseColor {r:.3f} {g:.3f} {b:.3f}
        transparency 0.5
        roughness 1
        metalness 0
      }}
      geometry Mesh {{ url [ "{url}" ] }}
    }}
  ]
  name "{node_name}"
}}"""


# ── 主流程 ────────────────────────────────────────────────────────────────────
def resolve_scene():
    scene = os.environ.get("VH_SCENE")
    if not scene and len(sys.argv) > 1:
        scene = sys.argv[1]
    if not scene:
        raise SystemExit("請用 VH_SCENE 環境變數或 controllerArgs 指定場景，例如 VH_SCENE=n3_scene0002")
    return scene.strip()


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    # 手臂 → Home，夾爪 → 開
    for jname, deg in zip(ARM_JOINTS, HOME_POSE_DEG):
        m = robot.getDevice(jname)
        if m:
            m.setPosition(math.radians(deg))
    for fj in (f"{GRIPPER}::left finger joint", f"{GRIPPER}::right finger joint"):
        m = robot.getDevice(fj)
        if m:
            m.setPosition(0.0)

    scene = resolve_scene()
    group = scene.split("_")[0]                      # n3
    scene_dir = CAPTURES / f"multi_{group}" / scene
    manifest_path = scene_dir / "scene_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"找不到 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    objects = manifest["actual"]["viewpoints"][0]["objects"]   # 實際位姿

    # hull 來源:VH_SOURCE=class(預設,per-class)或 foreground(前景合併→連通元件分離)
    source = os.environ.get("VH_SOURCE", "class")
    root = robot.getRoot().getField("children")
    print(f"[VH-Viewer] 場景 {scene}: {len(objects)} 物體  (hull 來源: {source})")

    # 1) 擺 YCB 物體(實際位姿)
    n_obj = 0
    for o in objects:
        name = o["name"]
        if name not in MASS_TABLE:
            continue
        root.importMFNodeFromString(-1, make_object_vrml(
            name, o["position_m"], o.get("rotation_axis_angle")))
        n_obj += 1

    # 2) 疊 hull
    n_hull = 0
    if source == "instance":
        inst_dir = EVAL / "instance_hull" / scene
        hulls = sorted(inst_dir.glob("visual_hull_inst_*.obj"))
        for i, hp in enumerate(hulls):
            color = HULL_COLORS[i % len(HULL_COLORS)]
            root.importMFNodeFromString(-1, make_hull_vrml(hp, f"vh_inst_{hp.stem}", color))
            n_hull += 1
            print(f"[VH-Viewer] {hp.name}: 色 RGB{tuple(round(c,1) for c in color)}")
        if not hulls:
            print(f"[VH-Viewer] 無 instance hull: {inst_dir}（先跑 associate.py→carve_instances.py）")
    elif source == "foreground":
        comp_dir = EVAL / "foreground" / scene / "components"
        comps = sorted(comp_dir.glob("obj_*.obj"))
        for i, hp in enumerate(comps):
            color = HULL_COLORS[i % len(HULL_COLORS)]
            root.importMFNodeFromString(-1, make_hull_vrml(hp, f"vh_fg_{hp.stem}", color))
            n_hull += 1
            print(f"[VH-Viewer] {hp.name}: 色 RGB{tuple(round(c,1) for c in color)}")
        if not comps:
            print(f"[VH-Viewer] 無 foreground 元件: {comp_dir}（先跑 make_foreground→build_torchhull→split_hull）")
    else:
        mask_dir = EVAL / MASK_SUBDIR / f"multi_{group}" / scene
        for i, o in enumerate(objects):
            name = o["name"]
            if name not in MASS_TABLE:
                continue
            hp = mask_dir / hull_filename(name)
            if hp.exists():
                color = HULL_COLORS[i % len(HULL_COLORS)]
                root.importMFNodeFromString(-1, make_hull_vrml(hp, f"vh_hull_{sanitize(name)}", color))
                n_hull += 1
                print(f"[VH-Viewer] {name}: hull 色 RGB{tuple(round(c,1) for c in color)}")
            else:
                print(f"[VH-Viewer] 無 hull: {hp.name}")
    print(f"[VH-Viewer] 已擺物體 {n_obj}，疊 hull {n_hull}。手臂在 Home。")

    while robot.step(timestep) != -1:
        pass


if __name__ == "__main__":
    main()
