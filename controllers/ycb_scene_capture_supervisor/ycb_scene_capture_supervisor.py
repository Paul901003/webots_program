"""
ycb_scene_capture_supervisor.py

從 scene_config.json 讀取場景描述，重建場景並拍攝。
支援任意數量的物體（含位置/旋轉）和任意數量的視角（含手臂關節角度）。

scene_config.json 格式：
{
  "capture_root": "captures_config",
  "scene_label": "my_scene",
  "objects": [
    {
      "name": "024_bowl",
      "position_m": [x, y, z],
      "rotation_axis_angle": [ax, ay, az, angle]   ← 選填，預設 [0,1,0,0]
    },
    ...
  ],
  "viewpoints": [
    {"id": 1, "joint_deg": [j1, j2, j3, j4, j5, j6]},
    ...
  ]
}
"""

import json
import math
import os
import sys

from controller import Supervisor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR  = os.path.join(os.path.dirname(CURRENT_DIR), "ycb_supervisor")
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

from config import (  # noqa: E402
    ASSET_BASE,
    MASS_TABLE,
    ALL_OBJECTS,
    DEFAULT_SHAPE,
    SHAPE_TABLE,
    SPAWN_CLEARANCE,
    ARM_MOTOR_VELOCITY_RAD_PER_SEC,
    ARM_SETTLE_TIME_BUFFER_SEC,
    ARM_SETTLE_TIME_SEC,
    POST_ARRIVAL_PAUSE_SEC,
)

JSON_PATH = os.path.join(SOURCE_DIR, "ycb_geometries.json")
with open(JSON_PATH, "r", encoding="utf-8") as _f:
    YCB_GEO_DATA = json.load(_f)

CONFIG_PATH = os.path.join(CURRENT_DIR, "scene_config.json")

UR5E_DEF           = "UR5E"
CAMERA_DEF         = "UR5E_CAMERA"
ARM_EMITTER_NAME   = "arm_command_emitter"
ARM_RECEIVER_NAME  = "arm_status_receiver"
SCENE_POSE_FILE    = "scene_objects_pose.json"
CAPTURE_WAIT_SEC   = 1.0
SCENE_SETTLE_SEC   = 2.0
HOME_POSE_RAD      = [0.0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0.0]

REPO_ROOT       = os.path.dirname(os.path.dirname(CURRENT_DIR))
TEST_IMAGES_DIR = os.path.join(REPO_ROOT, "Grounded-Segment-Anything", "test_images")


# ── 幾何輔助 ──────────────────────────────────────────────────────────────────

def get_geometry(name: str) -> dict:
    return YCB_GEO_DATA.get(name, {
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "size":   {"x": 0.1, "y": 0.1, "z": 0.1},
    })


def collision_half_height(name: str) -> float:
    shape = SHAPE_TABLE.get(name, DEFAULT_SHAPE)
    s = get_geometry(name)["size"]
    if shape == "Sphere":
        return (s["x"] + s["y"] + s["z"]) / 6.0
    return s["z"] / 2.0


def make_bounding_object(name: str, sx, sy, sz) -> str:
    shape = SHAPE_TABLE.get(name, DEFAULT_SHAPE)
    if shape == "Sphere":
        r = (sx + sy + sz) / 6.0
        return f"boundingObject Sphere {{ radius {r:.6f} }}"
    if shape == "Cylinder":
        r = (sx + sy) / 4.0
        return f"boundingObject Cylinder {{ radius {r:.6f} height {sz:.6f} }}"
    return f"boundingObject Box {{ size {sx:.6f} {sy:.6f} {sz:.6f} }}"


def make_vrml(name: str, pos: list, rot: list) -> str:
    """
    pos: [x, y, z]
    rot: [ax, ay, az, angle]  (Webots axis-angle)
    """
    x, y, z          = pos
    ax, ay, az, angle = rot
    mass = MASS_TABLE[name]
    base = f"{ASSET_BASE}/{name}/google_16k"
    geo  = get_geometry(name)
    cx, cy, cz = geo["center"]["x"], geo["center"]["y"], geo["center"]["z"]
    sx, sy, sz = geo["size"]["x"],   geo["size"]["y"],   geo["size"]["z"]
    bounding   = make_bounding_object(name, sx, sy, sz)

    return f"""Solid {{
  translation {x:.6f} {y:.6f} {z:.6f}
  rotation {ax:.6f} {ay:.6f} {az:.6f} {angle:.6f}
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
  name "{name}"
  {bounding}
  physics Physics {{
    density -1
    mass {mass}
  }}
}}"""


# ── 場景管理 ──────────────────────────────────────────────────────────────────

def clear_ycb_objects(supervisor: Supervisor):
    root_children = supervisor.getRoot().getField("children")
    i = root_children.getCount() - 1
    while i >= 0:
        node = root_children.getMFNode(i)
        if node is not None:
            nf = node.getField("name")
            if nf is not None and nf.getSFString() in ALL_OBJECTS:
                node.remove()
        i -= 1


def spawn_objects(supervisor: Supervisor, objects: list):
    """
    objects: list of dict with keys name, position_m, rotation_axis_angle
    """
    root_children = supervisor.getRoot().getField("children")
    for obj in objects:
        name = obj["name"]
        if name not in MASS_TABLE:
            print(f"[Supervisor] 未知物體跳過: {name}")
            continue
        pos = list(obj.get("position_m", [0, 0, 0]))
        rot = list(obj.get("rotation_axis_angle", [0, 1, 0, 0]))

        # 確保 y（高度）至少高於碰撞半徑 + 間隙，避免卡入地板
        min_y = collision_half_height(name) + SPAWN_CLEARANCE
        pos[1] = max(pos[1], min_y)

        root_children.importMFNodeFromString(-1, make_vrml(name, pos, rot))
    print(f"[Supervisor] 已生成 {len(objects)} 個物體")


def get_node_by_name(supervisor: Supervisor, name: str):
    root_children = supervisor.getRoot().getField("children")
    for i in range(root_children.getCount() - 1, -1, -1):
        node = root_children.getMFNode(i)
        if node is None:
            continue
        nf = node.getField("name")
        if nf is not None and nf.getSFString() == name:
            return node
    return None


# ── 姿態工具 ──────────────────────────────────────────────────────────────────

def rotation_matrix_to_rpy(m):
    sy = math.sqrt(m[0][0]**2 + m[1][0]**2)
    if sy >= 1e-9:
        roll  = math.atan2(m[2][1], m[2][2])
        pitch = math.atan2(-m[2][0], sy)
        yaw   = math.atan2(m[1][0], m[0][0])
    else:
        roll  = math.atan2(-m[1][2], m[1][1])
        pitch = math.atan2(-m[2][0], sy)
        yaw   = 0.0
    return roll, pitch, yaw


def rotation_matrix_to_axis_angle(m):
    trace = m[0][0] + m[1][1] + m[2][2]
    cos_a = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    angle = math.acos(cos_a)
    if angle < 1e-9:
        return 0.0, 1.0, 0.0, 0.0
    if abs(math.pi - angle) < 1e-6:
        xx = max(0.0, (m[0][0]+1)/2); yy = max(0.0, (m[1][1]+1)/2); zz = max(0.0, (m[2][2]+1)/2)
        xy = (m[0][1]+m[1][0])/4;     xz = (m[0][2]+m[2][0])/4;     yz = (m[1][2]+m[2][1])/4
        if xx >= yy and xx >= zz:
            ax = math.sqrt(xx); ay = 0 if ax<1e-9 else xy/ax; az = 0 if ax<1e-9 else xz/ax
        elif yy >= zz:
            ay = math.sqrt(yy); ax = 0 if ay<1e-9 else xy/ay; az = 0 if ay<1e-9 else yz/ay
        else:
            az = math.sqrt(zz); ax = 0 if az<1e-9 else xz/az; ay = 0 if az<1e-9 else yz/az
    else:
        d = 2.0 * math.sin(angle)
        ax = (m[2][1]-m[1][2])/d; ay = (m[0][2]-m[2][0])/d; az = (m[1][0]-m[0][1])/d
    n = math.sqrt(ax**2 + ay**2 + az**2)
    if n < 1e-9:
        return 0.0, 1.0, 0.0, 0.0
    return ax/n, ay/n, az/n, angle


def build_object_pose_record(index: int, name: str, node) -> dict:
    pos = node.getPosition()
    ori = node.getOrientation()
    m = [[float(ori[r*3+c]) for c in range(3)] for r in range(3)]
    roll, pitch, yaw = rotation_matrix_to_rpy(m)
    ax, ay, az, angle = rotation_matrix_to_axis_angle(m)
    return {
        "index": index,
        "name": name,
        "position_m": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
        "rotation_axis_angle": {"x": float(ax), "y": float(ay), "z": float(az), "angle": float(angle)},
        "rotation_matrix": m,
        "rotation_rpy_rad": {"roll": float(roll), "pitch": float(pitch), "yaw": float(yaw)},
        "rotation_rpy_deg": {"roll": math.degrees(roll), "pitch": math.degrees(pitch), "yaw": math.degrees(yaw)},
    }


def save_scene_poses(supervisor: Supervisor, objects: list,
                     scene_dir: str, capture_root: str, scene_label: str) -> dict:
    os.makedirs(scene_dir, exist_ok=True)
    records = []
    for i, obj in enumerate(objects, start=1):
        node = get_node_by_name(supervisor, obj["name"])
        if node is None:
            print(f"[Supervisor] 找不到物體節點: {obj['name']}")
            continue
        records.append(build_object_pose_record(i, obj["name"], node))

    payload = {
        "scene_label": scene_label,
        "capture_root": capture_root,
        "scene_dir": scene_dir,
        "saved_at_sim_time_sec": float(supervisor.getTime()),
        "coordinate_frame": "webots_world",
        "object_count": len(records),
        "objects": records,
    }
    path = os.path.join(scene_dir, SCENE_POSE_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Supervisor] 物體位姿已儲存: {path}")
    return payload


# ── 手臂控制 ──────────────────────────────────────────────────────────────────

def send_waypoint(emitter, joints_rad: list, cid: str) -> bool:
    msg = json.dumps({"type": "waypoint", "joints": joints_rad, "gripper": 0.0, "id": cid})
    emitter.send(msg.encode())
    return True


def wait_seconds(supervisor: Supervisor, timestep: int, seconds: float) -> bool:
    steps = max(0, int(seconds * 1000 / max(1, timestep)))
    for _ in range(steps):
        if supervisor.step(timestep) == -1:
            return False
    return True


def clear_receiver(receiver):
    if receiver is None:
        return
    while receiver.getQueueLength() > 0:
        receiver.nextPacket()


def wait_for_arm_arrival(supervisor: Supervisor, timestep: int,
                          receiver, cid: str, timeout_sec: float) -> bool:
    if receiver is None:
        return wait_seconds(supervisor, timestep, timeout_sec)
    t0 = supervisor.getTime()
    while supervisor.getTime() - t0 <= timeout_sec:
        if supervisor.step(timestep) == -1:
            return False
        while receiver.getQueueLength() > 0:
            msg = receiver.getString()
            receiver.nextPacket()
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue
            if data.get("status") == "arrived" and str(data.get("command_id")) == str(cid):
                print(f"[Supervisor] 手臂到達 (max_error={float(data.get('max_error_rad',0)):.4f} rad)")
                return True
    print(f"[Supervisor] 手臂到達逾時 (id={cid})")
    return False


def estimate_travel_time(current: list, target: list) -> float:
    max_delta = max(abs(t-c) for t, c in zip(target, current))
    motion = max_delta / max(ARM_MOTOR_VELOCITY_RAD_PER_SEC, 1e-6)
    return max(ARM_SETTLE_TIME_SEC, motion + ARM_SETTLE_TIME_BUFFER_SEC)


# ── 相機觸發 ──────────────────────────────────────────────────────────────────

def set_custom_data(node, data: str):
    if node is None:
        return
    f = node.getField("customData")
    if f is not None:
        f.setSFString(data)


def trigger_capture(supervisor: Supervisor, camera_node,
                    view_id, scene_label: str, capture_root: str,
                    joint_deg: list) -> str:
    token = f"{view_id}_{int(supervisor.getTime() * 1000)}"
    joint_str = ",".join(f"{v:.6f}" for v in joint_deg)
    # 不送 num_views，使相機 controller 不加 _Nviews 後綴，
    # 確保影像存放路徑與 scene_objects_pose.json 相同目錄
    camera_data = (
        f"capture_token={token};"
        f"view={view_id};"
        f"label={scene_label};"
        f"capture_root={capture_root};"
        f"joint_deg={joint_str}"
    )
    set_custom_data(camera_node, camera_data)
    return token


# ── 主流程 ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    objects    = cfg.get("objects", [])
    viewpoints = cfg.get("viewpoints", [])

    if not objects:
        raise ValueError("scene_config.json 缺少 objects")
    if not viewpoints:
        raise ValueError("scene_config.json 缺少 viewpoints")
    for obj in objects:
        if "name" not in obj:
            raise ValueError(f"物體缺少 name 欄位: {obj}")
        obj.setdefault("position_m",         [0.0, 0.0, 0.0])
        obj.setdefault("rotation_axis_angle", [0.0, 1.0, 0.0, 0.0])
    for vp in viewpoints:
        if "joint_deg" not in vp or len(vp["joint_deg"]) != 6:
            raise ValueError(f"視角 {vp.get('id','?')} 缺少 6 個關節角度")

    return cfg


def main():
    supervisor = Supervisor()
    timestep   = int(supervisor.getBasicTimeStep())

    # 讀取設定
    try:
        cfg = load_config(CONFIG_PATH)
    except Exception as e:
        print(f"[Supervisor] 無法讀取 scene_config.json: {e}")
        return

    objects       = cfg["objects"]
    viewpoints    = cfg["viewpoints"]
    capture_root  = cfg.get("capture_root", "captures_config")
    scene_label   = cfg.get("scene_label", "scene")
    scene_dir     = os.path.join(TEST_IMAGES_DIR, capture_root, scene_label)

    print(f"[Supervisor] 場景: {scene_label}")
    print(f"[Supervisor] 物體數: {len(objects)}  視角數: {len(viewpoints)}")

    # 裝置初始化
    arm_emitter  = supervisor.getDevice(ARM_EMITTER_NAME)
    arm_receiver = supervisor.getDevice(ARM_RECEIVER_NAME)
    camera_node  = supervisor.getFromDef(CAMERA_DEF)

    if arm_receiver is not None:
        arm_receiver.enable(timestep)
    if arm_emitter is None:
        print("[Supervisor] 找不到 arm_command_emitter")
        return

    # 移回 HOME
    supervisor.step(timestep)
    clear_receiver(arm_receiver)
    send_waypoint(arm_emitter, HOME_POSE_RAD, "home")
    wait_for_arm_arrival(supervisor, timestep, arm_receiver, "home", 30.0)

    # 清除舊物體，生成新物體
    clear_ycb_objects(supervisor)
    spawn_objects(supervisor, objects)

    # 等待物理穩定
    print(f"[Supervisor] 等待場景穩定 ({SCENE_SETTLE_SEC}s)...")
    if not wait_seconds(supervisor, timestep, SCENE_SETTLE_SEC):
        return

    # 記錄實際物體位姿
    scene_pose = save_scene_poses(supervisor, objects, scene_dir, capture_root, scene_label)

    # 依序移動手臂到每個視角並拍攝
    current_joints = HOME_POSE_RAD[:]
    total_views    = len(viewpoints)

    for vp in viewpoints:
        view_id   = vp["id"]
        joint_deg = vp["joint_deg"]
        joint_rad = [math.radians(d) for d in joint_deg]

        travel = estimate_travel_time(current_joints, joint_rad)
        print(f"[Supervisor] 視角 {view_id}/{total_views}，預估移動 {travel:.1f}s")

        clear_receiver(arm_receiver)
        send_waypoint(arm_emitter, joint_rad, str(view_id))
        if not wait_for_arm_arrival(supervisor, timestep, arm_receiver, str(view_id), travel):
            print(f"[Supervisor] 視角 {view_id} 手臂逾時，跳過")
            continue
        current_joints = joint_rad

        if POST_ARRIVAL_PAUSE_SEC > 0:
            wait_seconds(supervisor, timestep, POST_ARRIVAL_PAUSE_SEC)

        trigger_capture(supervisor, camera_node, view_id, scene_label,
                        capture_root, joint_deg)
        wait_seconds(supervisor, timestep, CAPTURE_WAIT_SEC)

    print(f"[Supervisor] 完成，共拍攝 {total_views} 個視角")
    print(f"[Supervisor] 輸出: {scene_dir}")


if __name__ == "__main__":
    main()
