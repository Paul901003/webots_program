#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""compute_workspace_mc.py — 正向 Monte Carlo 算 UR5e+相機+夾爪 的可達工作空間(mesh 自碰撞)。

與 .wbt 拍攝幾何一致:
  - 連桿/夾爪 mesh = URDF(ur5e_with_140gripper.urdf,其 package 路徑解析到 meshes/,即 Webots proto 同源 mesh);
  - 相機 = meshes/cameras 的 D455,用「tool0→相機」固定轉換掛上(由已驗證視角的 Webots 相機位姿導出,
    跨視角標準差=0:平移(0,-0.05,-0.025)m + 固定旋轉,與 toolSlot 掛載一致);
  - 機器人 base 在世界 [-0.4,0,0]。
做法:在關節極限(candidate_viewpoint_config.JOINT_LIMITS_DEG)內隨機抽 6 軸 → yourdfpy FK 擺所有 mesh →
  fcl(trimesh CollisionManager)查自碰撞(排除 SRDF disable_collisions + 相機掛點鄰接) → 無撞則記錄
  相機光心 + tool0(夾爪參考)位置。輸出點雲 + 體素體積 + 三視圖。
用法: ./compute_workspace_mc.py [N取樣數=50000] [體素cm=2]
"""
import math
import os
import re
import sys

import numpy as np
import trimesh
import yourdfpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib import font_manager   # noqa: E402

_FP = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
try:
    font_manager.fontManager.addfont(_FP)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_FP).get_name()
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
URDF = os.path.join(REPO, "urdfs", "ur5e_with_140gripper.urdf")
SRDF = os.path.join(REPO, "ros2_ws", "src", "ur5e_2f140_planning", "config", "ur5e_2f140.srdf")
CAM_MESH = os.path.join(REPO, "meshes", "cameras", "collison", "IntelRealsenseD455.stl")
OUT_DIR = os.path.join(REPO, "data", "viewpoints")
PKG_PREFIX = "package://ur5e_webots_planning/"

sys.path.insert(0, HERE)
from candidate_viewpoint_config import JOINT_LIMITS_DEG, ROBOT_BASE_M  # noqa: E402

ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
BASE = np.array(ROBOT_BASE_M, dtype=float)

# tool0→相機(已由 webots_camera_transform_world 跨視角導出,固定)
T_TOOL0_CAM = np.array([
    [0.0, -1.0,  0.0,  0.0],
    [0.0,  0.0, -1.0, -0.10],   # 相機沿 D455 Z 由 5cm→10cm(對應 toolSlot +Z);URDF camera_mount 同步
    [1.0,  0.0,  0.0, -0.025],
    [0.0,  0.0,  0.0,  1.0],
])


def resolve(path):
    return os.path.join(REPO, path.replace(PKG_PREFIX, ""))


def load_collision_geoms(urdf):
    """回傳 [(link_name, trimesh_mesh, origin_local 4x4)]。"""
    geoms = []
    for name, link in urdf.link_map.items():
        for col in getattr(link, "collisions", []):
            g = col.geometry
            if g.mesh is not None:                          # 三角網格
                m = trimesh.load(resolve(g.mesh.filename), force="mesh")
                if g.mesh.scale is not None:
                    m = m.copy(); m.apply_scale(g.mesh.scale)
            elif g.box is not None:                         # box primitive(如指墊)
                m = trimesh.creation.box(extents=np.asarray(g.box.size, float))
            elif g.cylinder is not None:
                m = trimesh.creation.cylinder(radius=g.cylinder.radius, height=g.cylinder.length)
            elif g.sphere is not None:
                m = trimesh.creation.icosphere(radius=g.sphere.radius)
            else:
                continue
            origin = col.origin if col.origin is not None else np.eye(4)
            geoms.append((name, m, np.asarray(origin, float)))
    return geoms


def adjacency_pairs(urdf):
    """URDF 關節父子(恆相觸)——保底排除。"""
    return {frozenset((j.parent, j.child)) for j in urdf.joint_map.values()}


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    vox_cm = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    vox = vox_cm / 100.0

    print(f"[ws] 載入 URDF + 碰撞 mesh ...")
    urdf = yourdfpy.URDF.load(URDF, load_meshes=False, build_collision_scene_graph=False)
    geoms = load_collision_geoms(urdf)
    cam_mesh = trimesh.load(CAM_MESH, force="mesh")
    if cam_mesh.extents.max() > 1.0:        # D455 STL 為 mm 單位(extent~124) → 換算成 m
        cam_mesh.apply_scale(0.001)
    print(f"[ws] 碰撞幾何 {len(geoms)} + 相機(extent={np.round(cam_mesh.extents,3).tolist()}m);取樣 {N}")

    mgr = trimesh.collision.CollisionManager()
    for i, (link, m, _) in enumerate(geoms):
        mgr.add_object(f"{i}:{link}", m)
    mgr.add_object("camera", cam_mesh)
    idx2link = {f"{i}:{link}": link for i, (link, _, _) in enumerate(geoms)}
    idx2link["camera"] = "camera"

    lo = np.array([math.radians(a) for a, _ in JOINT_LIMITS_DEG])
    hi = np.array([math.radians(b) for _, b in JOINT_LIMITS_DEG])

    def place(q):
        """擺好所有 mesh,回傳 (T_tool0, T_cam, 碰撞 link-pair 集合)。"""
        urdf.update_cfg({n: v for n, v in zip(ARM, q)})
        for i, (link, _, origin) in enumerate(geoms):
            Tw = (urdf.get_transform(link, "base_link") @ origin).copy()
            Tw[:3, 3] += BASE
            mgr.set_transform(f"{i}:{link}", Tw)
        T_tool0 = urdf.get_transform("tool0", "base_link").copy(); T_tool0[:3, 3] += BASE
        T_cam = T_tool0 @ T_TOOL0_CAM
        mgr.set_transform("camera", T_cam)
        hit, names = mgr.in_collision_internal(return_names=True)
        pairs = {frozenset((idx2link[a], idx2link[b])) for a, b in names} if hit else set()
        return T_tool0, T_cam, pairs

    # 自動校準:抽 K 個隨機姿態,恆撞(≥thresh)的對 = 結構/剛性重疊 → 排除(名稱無關,同 MoveIt SRDF 產生器)
    from collections import Counter
    K, thresh = 400, 0.95
    rng = np.random.default_rng(0)
    cnt = Counter()
    for _ in range(K):
        cnt.update(place(lo + rng.random(6) * (hi - lo))[2])
    disabled = adjacency_pairs(urdf) | {p for p, c in cnt.items() if c >= thresh * K}
    print(f"[ws] 自動校準排除 {len(disabled)} 對(恆撞≥{thresh:.0%} of {K})")

    cam_pts, tool_pts = [], []
    n_free = 0
    rng = np.random.default_rng(1)
    for s in range(N):
        T_tool0, T_cam, pairs = place(lo + rng.random(6) * (hi - lo))
        if not (pairs - disabled):
            n_free += 1
            cam_pts.append(T_cam[:3, 3].copy())
            tool_pts.append(T_tool0[:3, 3].copy())
        if (s + 1) % 5000 == 0:
            print(f"  {s+1}/{N}  無撞 {n_free} ({n_free/(s+1):.1%})")

    cam_pts = np.array(cam_pts); tool_pts = np.array(tool_pts)
    # 體素體積
    def vol(pts):
        if len(pts) == 0:
            return 0, 0.0
        keys = set(map(tuple, np.floor(pts / vox).astype(int)))
        return len(keys), len(keys) * vox**3

    os.makedirs(OUT_DIR, exist_ok=True)
    npz = os.path.join(OUT_DIR, "workspace_mc.npz")
    np.savez_compressed(npz, camera_pts=cam_pts, tool0_pts=tool_pts,
                        n_samples=N, n_free=n_free, voxel_m=vox, robot_base=BASE)
    nv_c, vol_c = vol(cam_pts); nv_t, vol_t = vol(tool_pts)
    print(f"\n[ws] 無自碰撞 {n_free}/{N} ({n_free/N:.1%})")
    print(f"[ws] 相機可達:體素 {nv_c} 個 → 體積 ≈ {vol_c*1000:.1f} L")
    print(f"[ws] 夾爪tool0 可達:體素 {nv_t} 個 → 體積 ≈ {vol_t*1000:.1f} L")
    print(f"[ws] 點雲 → {npz}")

    # 三視圖(相機可達點雲)
    if len(cam_pts):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        P = cam_pts
        for a, (i, j, lab) in zip(ax, [(0, 1, "XY(俯視)"), (0, 2, "XZ(側視)"), (1, 2, "YZ(前視)")]):
            a.scatter(P[:, i], P[:, j], s=1, c=P[:, 2], cmap="viridis", alpha=0.3)
            a.scatter([BASE[i]], [BASE[j]], c="red", marker="s", s=60, label="base")
            a.scatter([0.35 if i == 0 else 0.0], [0.0 if j != 2 else 0.0],
                      c="orange", marker="*", s=120, label="物體中心")
            a.set_title(lab); a.set_aspect("equal"); a.legend(fontsize=7); a.grid(alpha=0.3)
        fig.suptitle(f"相機可達工作空間(mesh自碰撞,N={N},無撞{n_free})  體積≈{vol_c*1000:.0f}L")
        png = os.path.join(REPO, "data", "eval", "_diag", "workspace_mc.png")
        os.makedirs(os.path.dirname(png), exist_ok=True)
        fig.tight_layout(); fig.savefig(png, dpi=110); plt.close(fig)
        print(f"[ws] 三視圖 → {png}")


if __name__ == "__main__":
    main()
