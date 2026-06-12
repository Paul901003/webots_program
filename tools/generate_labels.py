#!/home/cho/.pyenv/versions/3.10.10/bin/python3
"""
generate_labels.py

使用 pyrender 渲染 UR5e + Robotiq 2f-140 夾爪 + YCB 物體的 segmentation mask，
輸出 COCO JSON。

輸入格式（擇一）：

  1. scene_manifest.json（由 ycb_supervisor_four_view_single/multi 產生）：
     python generate_labels.py \
       --manifest captures_single/024_bowl_20260514/scene_manifest.json \
       --output labels/024_bowl_20260514/

  2. validated_viewpoints.json（由 ycb_viewpoint_validator 產生，須含 ycb_object_* 欄位）：
     python generate_labels.py \
       --viewpoints controllers/ycb_supervisor_four_view/validated_viewpoints.json \
       --output labels/024_bowl/
"""

import argparse
import json
import math
import os

import numpy as np
import pyrender
import trimesh
from PIL import Image
from pycocotools import mask as mask_utils

os.environ["PYOPENGL_PLATFORM"] = "egl"  # headless rendering

# ── 路徑預設值 ────────────────────────────────────────────────────────────────
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR       = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
PROTO_MESH_DIR    = os.path.join(PROJECT_DIR, "urdfs", "webots_proto_meshes",
                                 "UR5e", "meshes")
GRIPPER_MESH_DIR  = os.path.join(PROJECT_DIR, "urdfs", "webots_proto_meshes",
                                 "robotiq_2f140", "meshes")
DEFAULT_ASSETS    = os.path.join(PROJECT_DIR, "urdfs", "ycb_assets")

# YCB 幾何中心（Webots _make_vrml 用 Transform[-center] 將 mesh 置中，渲染須一致）
_YCB_GEO_PATH = os.path.join(PROJECT_DIR, "controllers", "ycb_supervisor", "ycb_geometries.json")
try:
    with open(_YCB_GEO_PATH, encoding="utf-8") as _f:
        _YCB_GEO = json.load(_f)
except FileNotFoundError:
    _YCB_GEO = {}


def ycb_center(name: str) -> np.ndarray:
    c = _YCB_GEO.get(name, {}).get("center", {"x": 0, "y": 0, "z": 0})
    return np.array([c["x"], c["y"], c["z"]])

# ── 相機規格（IntelRealsenseD455, HD 模式）────────────────────────────────────
CAM_FOV_H  = 1.4746   # 水平 FOV (radians)，來自 proto
CAM_WIDTH  = 1280
CAM_HEIGHT = 720

# ── UR5e 機器人底座在世界座標的位姿（來自 .wbt）────────────────────────────────
ROBOT_BASE_XYZ = np.array([-0.4, 0.0, 0.0])

# ── Segmentation 顏色 ID ───────────────────────────────────────────────────
SEG_ROBOT = (1, 0, 0)   # red channel = 1 → robot
SEG_YCB   = (0, 1, 0)   # green channel = 1 → ycb


# ── 基本幾何工具 ──────────────────────────────────────────────────────────────

def _axis_angle_to_mat(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues axis-angle → 3×3 rotation matrix."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])


def _tf(xyz=(0.0, 0.0, 0.0), axis=None, angle=0.0) -> np.ndarray:
    """Build 4×4 transform: translate then (optionally) rotate."""
    m = np.eye(4)
    m[:3, 3] = xyz
    if axis is not None:
        m[:3, :3] = _axis_angle_to_mat(np.array(axis, dtype=float), angle)
    return m


def _load_mesh(path: str, transform: np.ndarray, nodes: list) -> None:
    """Load mesh file and append (trimesh, transform) to nodes list."""
    if not os.path.exists(path):
        print(f"  [warn] mesh 不存在: {path}")
        return
    try:
        m = trimesh.load(path, force="mesh")
        nodes.append((m, transform.copy()))
    except Exception as e:
        print(f"  [warn] mesh 載入失敗 {os.path.basename(path)}: {e}")


# ── 相機 ──────────────────────────────────────────────────────────────────────

def camera_intrinsics() -> np.ndarray:
    fx = (CAM_WIDTH / 2.0) / math.tan(CAM_FOV_H / 2.0)
    return np.array([[fx, 0, CAM_WIDTH / 2.0],
                     [0, fx, CAM_HEIGHT / 2.0],
                     [0,  0, 1.0]], dtype=np.float64)


def webots_camera_pose(cam_pos: np.ndarray, rpy) -> np.ndarray:
    """從 Webots 相機(position + rpy, ZYX 內旋)建 pyrender camera-to-world pose。

    對齊 Webots 拍攝相機:
      - rpy → R(camera→world) = Rz(yaw)·Ry(pitch)·Rx(roll)（與 supervisor rot_mat_to_rpy 互逆）
      - Webots 相機視線 = local +X(= R[:,0])；pyrender 相機視線 = local -Z
      - up 取 Webots local +Z(= R[:,2])
    """
    roll, pitch, yaw = rpy
    def Rx(a): return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    def Ry(a): return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    def Rz(a): return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    forward, up_guess = R[:, 0], R[:, 2]
    backward = -forward
    right = np.cross(up_guess, backward); right /= np.linalg.norm(right)
    up = np.cross(backward, right); up /= np.linalg.norm(up)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = backward
    pose[:3, 3] = cam_pos
    return pose


def camera_pose_matrix(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera-to-world 4×4（Webots Y-up）。"""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward   # OpenGL: camera looks along -Z
    pose[:3, 3] = eye
    return pose


# ── UR5e FK（來自 Webots UR5e.proto）─────────────────────────────────────────

def _ur5e_link_transforms(joint_rad: list) -> tuple[list, np.ndarray]:
    """
    計算 UR5e 各 solid 的世界轉換矩陣。
    返回 (transforms_per_link, T_wrist3)。

    Proto joint chain（anchor 和 axis 均在父 solid 的局部座標系）:
      shoulder_pan_joint : anchor [0,0,0.163],      axis [0,0,1]
      shoulder_lift_joint: anchor [0,0.138,0],      axis [0,1,0], child init rot [0,1,0, π/2]
      elbow_joint        : anchor [0,-0.131,0.425],  axis [0,1,0]
      wrist_1_joint      : anchor [0,0,0.392],      axis [0,1,0], child init rot [0,1,0, π/2]
      wrist_2_joint      : anchor [0,0.127,0],      axis [0,0,1]
      wrist_3_joint      : anchor [0,0,0.1],        axis [0,1,0]
    """
    j = joint_rad
    T_base    = _tf(ROBOT_BASE_XYZ)
    T_shoulder = T_base     @ _tf([0, 0, 0.163])    @ _tf(axis=[0, 0, 1], angle=j[0])
    T_upper    = T_shoulder  @ _tf([0, 0.138, 0])   @ _tf(axis=[0, 1, 0], angle=j[1]) \
                                                     @ _tf(axis=[0, 1, 0], angle=math.pi / 2)
    T_forearm  = T_upper     @ _tf([0, -0.131, 0.425]) @ _tf(axis=[0, 1, 0], angle=j[2])
    T_w1       = T_forearm   @ _tf([0, 0, 0.392])   @ _tf(axis=[0, 1, 0], angle=j[3]) \
                                                     @ _tf(axis=[0, 1, 0], angle=math.pi / 2)
    T_w2       = T_w1        @ _tf([0, 0.127, 0])   @ _tf(axis=[0, 0, 1], angle=j[4])
    T_w3       = T_w2        @ _tf([0, 0, 0.1])     @ _tf(axis=[0, 1, 0], angle=j[5])

    return {
        "base":     T_base,
        "shoulder": T_shoulder,
        "upper":    T_upper,
        "forearm":  T_forearm,
        "wrist_1":  T_w1,
        "wrist_2":  T_w2,
        "wrist_3":  T_w3,
    }


def load_robot_scene_nodes(arm_mesh_dir: str, joint_rad: list) -> list:
    """
    返回 UR5e 手臂的 list of (trimesh.Trimesh, world_transform_4x4)。
    """
    tf = _ur5e_link_transforms(joint_rad)
    nodes = []
    d = arm_mesh_dir

    for fname in ["base_link_0.obj", "base_link_1.obj"]:
        _load_mesh(os.path.join(d, fname), tf["base"], nodes)
    for i in range(4):
        _load_mesh(os.path.join(d, f"shoulder_link_{i}.obj"), tf["shoulder"], nodes)
    for i in range(10):
        _load_mesh(os.path.join(d, f"upper_arm_link_{i}.obj"), tf["upper"], nodes)
    for i in range(8):
        _load_mesh(os.path.join(d, f"forearm_link_{i}.obj"), tf["forearm"], nodes)
    for i in range(4):
        _load_mesh(os.path.join(d, f"wrist_1_link_{i}.obj"), tf["wrist_1"], nodes)
    for i in range(4):
        _load_mesh(os.path.join(d, f"wrist_2_link_{i}.obj"), tf["wrist_2"], nodes)
    _load_mesh(os.path.join(d, "wrist_3_link_0.obj"), tf["wrist_3"], nodes)

    return nodes


# ── Robotiq 2f-140 FK（來自 Webots Robotiq2f140Gripper.proto）────────────────

def load_gripper_nodes(gripper_mesh_dir: str, joint_rad: list,
                       finger_angle: float = 0.0) -> list:
    """
    計算 Robotiq 2f-140 夾爪的 FK 並返回 (mesh, world_tf) 列表。

    夾爪附著在 UR5e wrist_3 toolSlot，wbt 定義：
      Pose { rotation [0,1,0, π/2] }
        Robotiq2f140Gripper { rotation [1,0,0, -π/2] }

    finger_angle: 夾爪主動關節角度 (rad)，0 = 完全開啟，最大 ≈ 0.7。
    被動關節在此為近似值（適用於小角度）。
    """
    tf = _ur5e_link_transforms(joint_rad)
    T_w3 = tf["wrist_3"]

    # 夾爪基座在世界座標
    T_grip = T_w3 @ _tf(axis=[0, 1, 0], angle=math.pi / 2) \
                  @ _tf(axis=[1, 0, 0], angle=-math.pi / 2)

    nodes = []
    d = gripper_mesh_dir

    def stl(fname, t):
        _load_mesh(os.path.join(d, fname), t, nodes)

    # base_link
    stl("robotiq_base_link.stl", T_grip)

    # ── 左側 ──────────────────────────────────────────────────────────────────
    # left outer knuckle: axis [-1,0,0], anchor [0,-0.030601,0.054905]
    #   initial rot [1,0,0, 2.295796]
    T_Lk = (T_grip
            @ _tf([0, -0.030601, 0.054905])
            @ _tf(axis=[-1, 0, 0], angle=finger_angle)
            @ _tf(axis=[1, 0, 0], angle=2.295796))
    stl("robotiq_2f140_outer_knuckle.stl", T_Lk)

    # left outer finger: child solid of left outer knuckle at T([0,0.01822,0.026002])
    T_Lof = T_Lk @ _tf([0, 0.01822, 0.026002])
    stl("robotiq_2f140_outer_finger.stl", T_Lof)

    # left inner finger: HingeJoint at anchor [0,0.081755,-0.02822]
    #   initial rot [-1,0,0,0.725] = R([1,0,0],-0.725)，passive ≈ -finger_angle
    T_Lif = (T_Lof
             @ _tf([0, 0.081755, -0.02822])
             @ _tf(axis=[1, 0, 0], angle=-finger_angle)
             @ _tf(axis=[1, 0, 0], angle=-0.725))
    stl("robotiq_2f140_inner_finger.stl", T_Lif)

    # left inner knuckle: axis [1,0,0] (default), anchor [0,-0.0127,0.06142]
    #   initial rot [1,0,0, 2.295796]，passive ≈ finger_angle
    T_Lik = (T_grip
             @ _tf([0, -0.0127, 0.06142])
             @ _tf(axis=[1, 0, 0], angle=finger_angle)
             @ _tf(axis=[1, 0, 0], angle=2.295796))
    stl("robotiq_2f140_inner_knuckle.stl", T_Lik)

    # ── 右側 ──────────────────────────────────────────────────────────────────
    # right outer knuckle: axis [1,0,0], anchor [0,0.030601,0.054905]
    #   initial rot [0,0.9119,0.4104, π]
    _right_ax = np.array([0.0, 0.911903298450496, 0.41040513431864556])
    T_Rk = (T_grip
            @ _tf([0, 0.030601, 0.054905])
            @ _tf(axis=[1, 0, 0], angle=finger_angle)
            @ _tf(axis=_right_ax, angle=math.pi))
    stl("robotiq_2f140_outer_knuckle.stl", T_Rk)

    T_Rof = T_Rk @ _tf([0, 0.01822, 0.026002])
    stl("robotiq_2f140_outer_finger.stl", T_Rof)

    T_Rif = (T_Rof
             @ _tf([0, 0.081755, -0.02822])
             @ _tf(axis=[1, 0, 0], angle=-finger_angle)
             @ _tf(axis=[1, 0, 0], angle=-0.725))
    stl("robotiq_2f140_inner_finger.stl", T_Rif)

    # right inner knuckle: axis [-1,0,0], anchor [0,0.0127,0.06142]
    #   initial rot [0,-0.9119,-0.4104, π]
    _right_ik_ax = np.array([0.0, -0.911903298450496, -0.41040513431864556])
    T_Rik = (T_grip
             @ _tf([0, 0.0127, 0.06142])
             @ _tf(axis=[-1, 0, 0], angle=finger_angle)
             @ _tf(axis=_right_ik_ax, angle=math.pi))
    stl("robotiq_2f140_inner_knuckle.stl", T_Rik)

    return nodes


# ── YCB mesh ──────────────────────────────────────────────────────────────────

def load_ycb_mesh(assets_dir: str, object_name: str) -> trimesh.Trimesh:
    for fname in ("textured.obj", "nontextured.ply", "nontextured.stl"):
        path = os.path.join(assets_dir, object_name, "google_16k", fname)
        if os.path.exists(path):
            return trimesh.load(path, force="mesh")
    raise FileNotFoundError(f"找不到 YCB mesh: {object_name}")


# ── 渲染 ──────────────────────────────────────────────────────────────────────


def render_color(robot_nodes: list, ycb_mesh, ycb_tf: np.ndarray,
                  cam_pose: np.ndarray, K: np.ndarray,
                  extra_ycb: list = None) -> np.ndarray:
    """渲染彩色影像（灰色背景），支援多個 YCB 物體。"""
    scene = pyrender.Scene(bg_color=[180, 180, 180, 255],
                            ambient_light=[0.3, 0.3, 0.3])

    if ycb_mesh is not None:
        scene.add(pyrender.Mesh.from_trimesh(ycb_mesh, smooth=True), pose=ycb_tf)
    if extra_ycb:
        for mesh, tf, *_ in extra_ycb:
            scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True), pose=tf)
    for mesh, tf in robot_nodes:
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True), pose=tf)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    scene.add(pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy,
                                         znear=0.05, zfar=10.0), pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0),
               pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(CAM_WIDTH, CAM_HEIGHT)
    try:
        color, _ = renderer.render(scene)
    finally:
        renderer.delete()
    return color


# ── COCO 工具 ─────────────────────────────────────────────────────────────────

def mask_to_coco(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


# ── 主流程 ────────────────────────────────────────────────────────────────────

def render_labels(actual_viewpoints, planned_objects, obj_name_to_cat_id,
                  mesh_cache, coco_categories, scene_id, pose_source,
                  out_dir, args):
    """渲染並輸出一組標籤。pose_source='actual'|'planned'"""
    images_dir = os.path.join(out_dir, "images")
    masks_dir  = os.path.join(out_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir,  exist_ok=True)

    K = camera_intrinsics()

    # planned 模式：所有視角共用同一組理論位姿
    if pose_source == "planned":
        planned_vp_objects = [
            {
                "name":                 o["name"],
                "position_m":           o["spawn_position_m"],
                "rotation_axis_angle":  o.get("spawn_rotation_axis_angle", [0, 1, 0, 0]),
            }
            for o in planned_objects
        ]

    coco = {
        "info": {
            "scene_id":          scene_id,
            "pose_source":       pose_source,
            "gripper_angle_rad": args.gripper_angle,
            "planned_objects":   planned_objects,
        },
        "categories": coco_categories,
        "images":      [],
        "annotations": [],
    }
    ann_id = 1

    for vp in actual_viewpoints:
        vp_id    = vp["id"]
        print(f"  [{vp_id:3d}] ({pose_source}) 渲染中...", end=" ", flush=True)

        joint_rad  = [math.radians(d) for d in vp["joint_deg"]]
        cam_pos_m  = np.array(vp["camera"]["position_m"])

        vp_objects = planned_vp_objects if pose_source == "planned" else vp["objects"]

        rpy = vp.get("camera", {}).get("rotation_rpy_rad")
        if rpy is not None:
            cam_pose = webots_camera_pose(cam_pos_m, rpy)   # 對齊 Webots 拍攝相機
        else:
            # fallback（validated_viewpoints 無 rpy）：看向物體平均
            target_pos = np.mean([o["position_m"] for o in vp_objects], axis=0) if vp_objects else np.zeros(3)
            cam_pose = camera_pose_matrix(cam_pos_m, target_pos)

        ycb_entries = []
        for obj in vp_objects:
            name   = obj["name"]
            pos    = obj["position_m"]
            rot    = obj.get("rotation_axis_angle", [0, 1, 0, 0])
            cat_id = obj_name_to_cat_id.get(name, 2)
            mesh   = mesh_cache.get(name)
            if mesh is None:
                continue
            Raa = np.eye(3)
            if len(rot) == 4 and rot[3] != 0.0:
                Raa = _axis_angle_to_mat(np.array(rot[:3]), rot[3])
            tf = np.eye(4)
            tf[:3, :3] = Raa
            tf[:3, 3] = np.array(pos) - Raa @ ycb_center(name)   # 對齊 Webots 置中擺法 T(pos)@R@T(-center)
            ycb_entries.append((mesh, tf, cat_id, name))

        arm_nodes     = load_robot_scene_nodes(args.arm_mesh_dir, joint_rad)
        gripper_nodes = load_gripper_nodes(args.gripper_mesh_dir, joint_rad,
                                           args.gripper_angle)
        all_robot_nodes = arm_nodes + gripper_nodes

        scene = pyrender.Scene(bg_color=[0, 0, 0, 255])
        seg_node_map = {}
        for mesh, tf in all_robot_nodes:
            node = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), pose=tf)
            seg_node_map[node] = SEG_ROBOT

        ycb_seg_ids = {}
        for i, (mesh, tf, cat_id, obj_name) in enumerate(ycb_entries):
            seg_val = (i + 1) & 0xFF
            color   = (0, seg_val, 0)
            node    = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), pose=tf)
            seg_node_map[node] = color
            ycb_seg_ids[color] = (cat_id, obj_name)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        scene.add(pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy,
                                             znear=0.05, zfar=10.0), pose=cam_pose)
        renderer = pyrender.OffscreenRenderer(CAM_WIDTH, CAM_HEIGHT)
        try:
            seg_color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SEG,
                                            seg_node_map=seg_node_map)
        finally:
            renderer.delete()

        robot_mask = (seg_color[:, :, 0] == 1).astype(np.uint8)

        color_img = render_color(all_robot_nodes,
                                 ycb_entries[0][0] if ycb_entries else None,
                                 ycb_entries[0][1] if ycb_entries else np.eye(4),
                                 cam_pose, K,
                                 extra_ycb=ycb_entries[1:])

        img_filename = f"viewpoint_{vp_id:04d}.png"
        Image.fromarray(color_img).save(os.path.join(images_dir, img_filename))

        mask_vis = np.zeros((CAM_HEIGHT, CAM_WIDTH), dtype=np.uint8)
        mask_vis[robot_mask == 1] = 128

        coco["images"].append({
            "id":           vp_id,
            "file_name":    f"images/{img_filename}",
            "width":        CAM_WIDTH,
            "height":       CAM_HEIGHT,
            "joint_deg":    vp["joint_deg"],
            "camera_pos_m": cam_pos_m.tolist(),
            "objects":      [{"name": o["name"], "position_m": o["position_m"],
                              "rotation_axis_angle": o.get("rotation_axis_angle", [0,1,0,0])}
                             for o in vp_objects],
        })

        if robot_mask.sum() > 0:
            rle = mask_to_coco(robot_mask)
            coco["annotations"].append({
                "id": ann_id, "image_id": vp_id, "category_id": 1,
                "segmentation": rle,
                "area": float(mask_utils.area(rle)),
                "bbox": mask_utils.toBbox(rle).tolist(),
                "iscrowd": 0,
            })
            ann_id += 1

        total_ycb_px = 0
        for (color, (cat_id, obj_name)) in ycb_seg_ids.items():
            ycb_mask = (seg_color[:, :, 1] == color[1]).astype(np.uint8)
            mask_vis[ycb_mask == 1] = 255
            if ycb_mask.sum() == 0:
                continue
            total_ycb_px += int(ycb_mask.sum())
            rle = mask_to_coco(ycb_mask)
            coco["annotations"].append({
                "id": ann_id, "image_id": vp_id, "category_id": cat_id,
                "segmentation": rle,
                "area": float(mask_utils.area(rle)),
                "bbox": mask_utils.toBbox(rle).tolist(),
                "iscrowd": 0,
            })
            ann_id += 1

        Image.fromarray(mask_vis).save(
            os.path.join(masks_dir, f"viewpoint_{vp_id:04d}_mask.png"))
        print(f"robot={int(robot_mask.sum())}px  ycb={total_ycb_px}px")

    ann_path = os.path.join(out_dir, "annotations.json")
    with open(ann_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"完成({pose_source})：{len(actual_viewpoints)} 張影像，{ann_id - 1} 個標注")
    print(f"輸出目錄：{out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest",   help="scene_manifest.json 路徑（single/multi supervisor 產生）")
    src.add_argument("--viewpoints", help="validated_viewpoints.json 路徑（viewpoint validator 產生）")
    parser.add_argument("--output", default=os.path.join(PROJECT_DIR, "data", "labels"), help="輸出根目錄")
    parser.add_argument("--mode", choices=["actual", "planned", "both"], default="both",
                        help="標籤使用實際位姿(actual)、理論位姿(planned)或兩者(both)")
    parser.add_argument("--arm-mesh-dir", default=PROTO_MESH_DIR)
    parser.add_argument("--gripper-mesh-dir", default=GRIPPER_MESH_DIR)
    parser.add_argument("--ycb-assets", default=DEFAULT_ASSETS)
    parser.add_argument("--gripper-angle", type=float, default=0.0,
                        help="夾爪開合角度 (rad)，0=完全開啟，最大≈0.7")
    args = parser.parse_args()

    # ── 讀取場景資訊 ─────────────────────────────────────────────────────────
    if args.manifest:
        with open(args.manifest, encoding="utf-8") as f:
            data = json.load(f)
        planned_objects  = data["planned"]["objects"]   # [{"name", "spawn_position_m", ...}]
        actual_viewpoints = data["actual"]["viewpoints"] # [{"id", "joint_deg", "camera", "objects", "files"}]
        print(f"輸入: scene_manifest  物體: {len(planned_objects)} 個  視角: {len(actual_viewpoints)} 個")
    else:
        with open(args.viewpoints, encoding="utf-8") as f:
            data = json.load(f)
        for key in ("ycb_object_name", "ycb_object_pos_m", "ycb_object_rotation_axis_angle"):
            if key not in data:
                raise KeyError(f"validated_viewpoints.json 缺少欄位 '{key}'，"
                               f"請重新執行 ycb_viewpoint_validator 產生新版 JSON")
        obj_name = data["ycb_object_name"]
        planned_objects = [{"name": obj_name}]
        # validated_viewpoints 格式：物體位姿固定，包進 actual_viewpoints
        actual_viewpoints = [
            {
                "id":        v.get("id", i + 1),
                "joint_deg": v["joint_deg"],
                "camera":    {"position_m": v["ray"]["ray_origin_m"]},
                "objects":   [{
                    "name":                data["ycb_object_name"],
                    "position_m":          data["ycb_object_pos_m"],
                    "rotation_axis_angle": data["ycb_object_rotation_axis_angle"],
                }],
            }
            for i, v in enumerate(data["validated"]) if v.get("ok")
        ]
        print(f"輸入: validated_viewpoints  物體: {obj_name}  視角: {len(actual_viewpoints)} 個")

    # 建立物體名稱 → category_id 對應（從 planned_objects 確定順序）
    coco_categories = [{"id": 1, "name": "ur5e", "supercategory": "robot"}]
    obj_name_to_cat_id = {}
    for cat_offset, obj in enumerate(planned_objects):
        cat_id = cat_offset + 2
        obj_name_to_cat_id[obj["name"]] = cat_id
        coco_categories.append({"id": cat_id, "name": obj["name"], "supercategory": "ycb"})

    # 預先載入所有物體 mesh
    mesh_cache = {}
    for obj in planned_objects:
        name = obj["name"]
        if name not in mesh_cache:
            mesh_cache[name] = load_ycb_mesh(args.ycb_assets, name)
            print(f"載入 mesh: {name}")

    scene_id = data.get("scene_id", "scene")
    base_out  = os.path.join(args.output, scene_id)

    modes = ["actual", "planned"] if args.mode == "both" else [args.mode]
    for mode in modes:
        out_dir = os.path.join(base_out, mode)
        render_labels(actual_viewpoints, planned_objects, obj_name_to_cat_id,
                      mesh_cache, coco_categories, scene_id, mode,
                      out_dir, args)


if __name__ == "__main__":
    main()
