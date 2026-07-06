#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""gen_workspace_viz.py — 把 compute_workspace_mc 的可達點雲生成 Webots 世界檔。

讀 data/viewpoints/workspace_mc.npz → 以 PointSet 渲染相機可達點雲(依高度上色),
加上 UR5e(於 [-0.4,0,0])、桌面、物體中心標記當參考。輸出 worlds/workspace_viz.wbt。
用法: ./gen_workspace_viz.py [which=camera|tool0] [max_points=20000]
"""
import os
import sys

import numpy as np
import matplotlib.cm as cm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
NPZ = os.path.join(REPO, "data", "viewpoints", "workspace_mc.npz")
OUT = os.path.join(REPO, "worlds", "workspace_viz.wbt")

HEADER = """#VRML_SIM R2025a utf8

EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackgroundLight.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/backgrounds/protos/TexturedBackground.proto"
EXTERNPROTO "../protos/blank_floor.proto"
EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/robots/universal_robots/protos/UR5e.proto"

WorldInfo {
  basicTimeStep 16
}
Viewpoint {
  orientation -0.3 0.4 0.86 1.9
  position 1.8 -1.8 1.6
  followType "None"
}
TexturedBackground {
}
TexturedBackgroundLight {
}
blank_floor {
}
DEF UR5E UR5e {
  translation -0.4 0 0
}
DEF OBJ_CENTER Solid {
  translation 0.35 0 0.02
  children [
    Shape {
      appearance PBRAppearance { baseColor 1 0.5 0  emissiveColor 1 0.5 0 }
      geometry Sphere { radius 0.025 }
    }
  ]
  name "object_center"
}
"""

POINTSET_TMPL = """DEF WORKSPACE Solid {{
  children [
    Shape {{
      geometry PointSet {{
        coord Coordinate {{
          point [
{points}
          ]
        }}
        color Color {{
          color [
{colors}
          ]
        }}
      }}
    }}
  ]
  name "workspace_pointcloud"
}}
"""


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "camera"
    maxp = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    z = np.load(NPZ)
    key = "camera_pts" if which == "camera" else "tool0_pts"
    P = z[key]
    if len(P) == 0:
        sys.exit("點雲為空")
    if len(P) > maxp:
        idx = np.random.default_rng(0).choice(len(P), maxp, replace=False)
        P = P[idx]
    # 依高度 z 上色
    zc = P[:, 2]
    t = (zc - zc.min()) / max(1e-9, float(np.ptp(zc)))
    rgb = cm.get_cmap("viridis")(t)[:, :3]

    pts = "\n".join(f"            {x:.4f} {y:.4f} {zz:.4f}" for x, y, zz in P)
    cols = "\n".join(f"            {r:.3f} {g:.3f} {b:.3f}" for r, g, b in rgb)
    world = HEADER + "\n" + POINTSET_TMPL.format(points=pts, colors=cols)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(world)
    print(f"[viz] {len(P)} 點({which})依高度上色 → {OUT}")
    print(f"      開啟: webots {os.path.relpath(OUT, REPO)}")


if __name__ == "__main__":
    main()
