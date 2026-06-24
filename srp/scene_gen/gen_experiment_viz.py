#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""gen_experiment_viz.py — 產生實驗場景可視化世界 worlds/experiment_viz.wbt(純幾何、無 controller)。

內容:手臂(UR5e@[-0.4,0,0])+ 桌面 + look-at 中心 + 12 相機中心(取自 pose.json)+ 視線 +
  拍攝半月(半徑0.65、方位135–225°、仰角20–90°,紅半透明)+ 工作空間球(r0.35@中心,藍半透明)+
  voxel 工作空間(0.7×0.7×0.35,白半透明)。
用法: ./srp/scene_gen/gen_experiment_viz.py
"""
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
import camera as cam  # noqa: E402

CENTER = (0.35, 0.0, 0.0)
R = 0.65               # 拍攝半徑
WS_R = 0.35            # 工作空間球半徑
BOX_MIN = (0.0, -0.35, 0.0)
BOX_MAX = (0.7, 0.35, 0.35)
SCENE = REPO / "data" / "captures" / "multi_stack3" / "stack3_scene0001"

HEADER = '''#VRML_SIM R2025a utf8

EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "../protos/blank_floor.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/robots/universal_robots/protos/UR5e.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/devices/robotiq/protos/Robotiq2f140Gripper.proto"
EXTERNPROTO "../protos/IntelRealsenseD455.proto"

WorldInfo {
  basicTimeStep 16
}
Viewpoint {
  orientation 1 0 -1 3.14
  position 0 0 5.5
  followType "None"
}
TexturedBackground {
}
TexturedBackgroundLight {
  luminosity 1.2
  castShadows FALSE
}
blank_floor {
}
DEF UR5E UR5e {
  translation -0.4 0 0
  controller "arm_apex"
  toolSlot [
    DEF UR5E_CAMERA IntelRealsenseD455 {
      translation 0 -0.03 0.05
      rotation 0 0 1 1.5708
      controller "<none>"
      resolution "HD"
    }
    Pose {
      rotation 0 1 0 1.5707996938995747
      children [
        Robotiq2f140Gripper {
          rotation 1 0 0 -1.5707996938995747
        }
      ]
    }
  ]
}
'''


def marker(name, pos, color, r=0.012, emis=None):
    e = emis or color
    return f'''Solid {{
  translation {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}
  children [ Shape {{
    appearance Appearance {{ material Material {{ diffuseColor {color[0]} {color[1]} {color[2]}  emissiveColor {e[0]} {e[1]} {e[2]} }} }}
    geometry Sphere {{ radius {r} subdivision 2 }}
  }} ]
  name "{name}"
}}
'''


def transp_shape_solid(name, translation, geometry, color, transparency):
    return f'''Solid {{
  translation {translation[0]:.4f} {translation[1]:.4f} {translation[2]:.4f}
  children [ Shape {{
    appearance Appearance {{ material Material {{ diffuseColor {color[0]} {color[1]} {color[2]}  transparency {transparency}  emissiveColor {color[0]*0.3:.2f} {color[1]*0.3:.2f} {color[2]*0.3:.2f} }} }}
    geometry {geometry}
  }} ]
  name "{name}"
}}
'''


def crescent_geometry():
    az = np.deg2rad(np.arange(135, 225.1, 5))
    el = np.deg2rad(np.arange(20, 90.1, 5))
    nel = len(el)
    pts = []
    for a in az:
        for e in el:
            pts.append((R * math.cos(e) * math.cos(a),
                        R * math.cos(e) * math.sin(a),
                        R * math.sin(e)))
    faces = []
    for i in range(len(az) - 1):
        for j in range(nel - 1):
            a = i * nel + j; b = (i + 1) * nel + j
            c = (i + 1) * nel + j + 1; d = i * nel + j + 1
            faces.append((a, b, c, d))
    pstr = ", ".join(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" for p in pts)
    fstr = " ".join(f"{f[0]} {f[1]} {f[2]} {f[3]} -1" for f in faces)
    return (f"IndexedFaceSet {{ coord Coordinate {{ point [ {pstr} ] }} "
            f"coordIndex [ {fstr} ] creaseAngle 1.5 }}")


def main():
    cams = []
    for pf in sorted(SCENE.glob("view_*_pose.json")):
        C, _ = cam.load_pose(pf)
        cams.append(tuple(float(x) for x in C))

    out = [HEADER]
    # 工作空間半球中心(藍)
    out.append(marker("center", CENTER, (0, 0, 1), r=0.022))
    # 12 相機中心(紅)
    for i, c in enumerate(cams, 1):
        out.append(marker(f"cam_{i:02d}", c, (1, 0, 0), r=0.014))
    # 視線(中心→各相機,細灰線)
    lp = [CENTER] + cams
    pstr = ", ".join(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" for p in lp)
    istr = " ".join(f"0 {k} -1" for k in range(1, len(cams) + 1))
    out.append(f'''Shape {{
  appearance Appearance {{ material Material {{ emissiveColor 0.4 0.4 0.4 }} }}
  geometry IndexedLineSet {{ coord Coordinate {{ point [ {pstr} ] }} coordIndex [ {istr} ] }}
}}
''')
    # 拍攝半月(紅半透明,Solid 平移到中心,mesh 點相對中心)
    out.append(transp_shape_solid("capture_crescent", CENTER, crescent_geometry(),
                                  (1, 0, 0), 0.6))
    # 工作空間球(藍半透明)
    out.append(transp_shape_solid("workspace_sphere", CENTER,
                                  f"Sphere {{ radius {WS_R} subdivision 3 }}", (0, 0, 1), 0.6))
    # voxel 工作空間(白半透明盒)
    bc = ((BOX_MIN[0] + BOX_MAX[0]) / 2, (BOX_MIN[1] + BOX_MAX[1]) / 2, (BOX_MIN[2] + BOX_MAX[2]) / 2)
    bs = (BOX_MAX[0] - BOX_MIN[0], BOX_MAX[1] - BOX_MIN[1], BOX_MAX[2] - BOX_MIN[2])
    out.append(transp_shape_solid("voxel_workspace", bc,
                                  f"Box {{ size {bs[0]} {bs[1]} {bs[2]} }}", (1, 1, 1), 0.7))

    wbt = REPO / "worlds" / "experiment_viz.wbt"
    wbt.write_text("\n".join(out), encoding="utf-8")
    print(f"→ {wbt}  (相機 {len(cams)} 個)")


if __name__ == "__main__":
    main()
