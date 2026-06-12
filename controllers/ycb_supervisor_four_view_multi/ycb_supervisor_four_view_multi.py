from controller import Supervisor
from datetime import datetime
import json
import math
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_CONTROLLER_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "ycb_supervisor")
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if SOURCE_CONTROLLER_DIR not in sys.path:
    sys.path.append(SOURCE_CONTROLLER_DIR)

from config import (
    GRID_COLS, SPACING, SPAWN_HEIGHT,
    REFERENCE_X, REFERENCE_Y, X_OFFSET, Z_OFFSET,
    ASSET_BASE, MASS_TABLE, ALL_OBJECTS,
    DEFAULT_SHAPE, SHAPE_TABLE, SPAWN_CLEARANCE, SPACING_MARGIN,
    ARM_SETTLE_TIME_SEC, POST_ARRIVAL_PAUSE_SEC,
    ARM_MOTOR_VELOCITY_RAD_PER_SEC, ARM_SETTLE_TIME_BUFFER_SEC,
)

JSON_PATH = os.path.join(SOURCE_CONTROLLER_DIR, "ycb_geometries.json")
with open(JSON_PATH, "r", encoding="utf-8") as _f:
    YCB_GEO_DATA = json.load(_f)

REPO_ROOT           = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_DIR            = os.path.join(REPO_ROOT, "data")
SCENE_PLAN_PATH        = os.path.join(DATA_DIR, "scene_plans", "multi_scene_plan.json")
SINGLE_SCENE_PLAN_PATH = os.path.join(DATA_DIR, "scene_plans", "single_scene_plan.json")
PLANNED_PATHS_PATH  = os.path.join(DATA_DIR, "viewpoints", "planned_paths.json")
CAPTURES_DIR        = os.path.join(DATA_DIR, "captures", "multi")
UR5E_DEF            = "UR5E"
CAMERA_DEF          = "UR5E_CAMERA"
ARM_COMMAND_EMITTER = "arm_command_emitter"
ARM_STATUS_RECEIVER = "arm_status_receiver"
CAPTURE_WAIT_SEC    = 1.5
SCENE_SETTLE_SEC    = 1.5
HOME_POSE_DEG       = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]

CAMERA_SPEC = {
    "model":       "IntelRealsenseD455",
    "resolution":  "HD",
    "width":       1280,
    "height":      720,
    "fov_h_rad":   1.4746,
    "min_range_m": 0.3,
    "max_range_m": 3.0,
}


# ── geometry helpers ─────────────────────────────────────────────────────────

def _geo(name):
    return YCB_GEO_DATA.get(name, {"center": {"x":0,"y":0,"z":0},
                                    "size":   {"x":.1,"y":.1,"z":.1}})

def _half_height(name):
    sz    = _geo(name)["size"]
    shape = SHAPE_TABLE.get(name, DEFAULT_SHAPE)
    if shape == "Sphere":
        return (sz["x"] + sz["y"] + sz["z"]) / 6.0
    return sz["z"] / 2.0

def _footprint(name):
    sz = _geo(name)["size"]
    return max(sz["x"], sz["y"])

def _bounding(name):
    sz = _geo(name)["size"]
    sx, sy, sz_ = sz["x"], sz["y"], sz["z"]
    shape = SHAPE_TABLE.get(name, DEFAULT_SHAPE)
    if shape == "Sphere":
        r = (sx + sy + sz_) / 6.0
        return f"boundingObject Sphere {{ radius {r:.6f} }}"
    if shape == "Cylinder":
        r = (sx + sy) / 4.0
        return f"boundingObject Cylinder {{ radius {r:.6f} height {sz_:.6f} }}"
    return f"boundingObject Box {{ size {sx:.6f} {sy:.6f} {sz_:.6f} }}"

def _make_vrml(name, x, y, z):
    mass = MASS_TABLE[name]
    base = f"{ASSET_BASE}/{name}/google_16k"
    geo  = _geo(name)
    cx, cy, cz = geo["center"]["x"], geo["center"]["y"], geo["center"]["z"]
    return (
        f'Solid {{\n'
        f'  translation {x:.6f} {y:.6f} {z:.6f}\n'
        f'  children [\n'
        f'    Transform {{\n'
        f'      translation {-cx:.6f} {-cy:.6f} {-cz:.6f}\n'
        f'      children [\n'
        f'        Shape {{\n'
        f'          appearance PBRAppearance {{\n'
        f'            baseColorMap ImageTexture {{ url [ "{base}/texture_map.png" ] }}\n'
        f'            roughness 1  metalness 0\n'
        f'          }}\n'
        f'          geometry Mesh {{ url [ "{base}/textured.obj" ] }}\n'
        f'        }}\n'
        f'      ]\n'
        f'    }}\n'
        f'  ]\n'
        f'  name "{name}"\n'
        f'  {_bounding(name)}\n'
        f'  physics Physics {{ density -1  mass {mass} }}\n'
        f'}}'
    )


# ── scene helpers ─────────────────────────────────────────────────────────────

def clear_ycb_objects(supervisor):
    root = supervisor.getRoot().getField("children")
    for i in range(root.getCount() - 1, -1, -1):
        node = root.getMFNode(i)
        if node is None:
            continue
        nf = node.getField("name")
        if nf and nf.getSFString() in ALL_OBJECTS:
            node.remove()


def spawn_objects(supervisor, objects):
    """生成物體並回傳各物體的理論生成位置 {name: [x, y, z]}。
    objects: list of dict，含 name 與可選 position_m [x, y, z]。
    有 position_m 時直接使用，z 自動調整為物體半高+間隙。
    """
    if not objects:
        return {}
    root = supervisor.getRoot().getField("children")
    spawn_positions = {}
    for obj in objects:
        name = obj if isinstance(obj, str) else obj["name"]
        if name not in MASS_TABLE:
            continue
        pos_m = None if isinstance(obj, str) else obj.get("position_m")
        if pos_m is not None:
            x, y = float(pos_m[0]), float(pos_m[1])
        else:
            x = REFERENCE_X + X_OFFSET
            y = REFERENCE_Y + Z_OFFSET
        z = max(SPAWN_HEIGHT, _half_height(name) + SPAWN_CLEARANCE)
        root.importMFNodeFromString(-1, _make_vrml(name, x, y, z))
        spawn_positions[name] = [x, y, z]
    return spawn_positions


def get_node_by_name(supervisor, name):
    root = supervisor.getRoot().getField("children")
    for i in range(root.getCount() - 1, -1, -1):
        node = root.getMFNode(i)
        if node is None:
            continue
        nf = node.getField("name")
        if nf and nf.getSFString() == name:
            return node
    return None


def rot_mat_to_rpy(m):
    sy = math.sqrt(m[0][0]**2 + m[1][0]**2)
    if sy > 1e-9:
        roll  = math.atan2(m[2][1], m[2][2])
        pitch = math.atan2(-m[2][0], sy)
        yaw   = math.atan2(m[1][0], m[0][0])
    else:
        roll  = math.atan2(-m[1][2], m[1][1])
        pitch = math.atan2(-m[2][0], sy)
        yaw   = 0.0
    return roll, pitch, yaw


def rot_mat_to_axis_angle(m):
    trace = m[0][0] + m[1][1] + m[2][2]
    cos_a = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    angle = math.acos(cos_a)
    if angle < 1e-9:
        return [0.0, 1.0, 0.0, 0.0]
    if abs(math.pi - angle) < 1e-6:
        xx = max(0.0, (m[0][0] + 1.0) / 2.0)
        yy = max(0.0, (m[1][1] + 1.0) / 2.0)
        zz = max(0.0, (m[2][2] + 1.0) / 2.0)
        xy = (m[0][1] + m[1][0]) / 4.0
        xz = (m[0][2] + m[2][0]) / 4.0
        yz = (m[1][2] + m[2][1]) / 4.0
        if xx >= yy and xx >= zz:
            ax = math.sqrt(xx); ay = xy/ax if ax>1e-9 else 0; az = xz/ax if ax>1e-9 else 0
        elif yy >= zz:
            ay = math.sqrt(yy); ax = xy/ay if ay>1e-9 else 0; az = yz/ay if ay>1e-9 else 0
        else:
            az = math.sqrt(zz); ax = xz/az if az>1e-9 else 0; ay = yz/az if az>1e-9 else 0
    else:
        d  = 2.0 * math.sin(angle)
        ax = (m[2][1] - m[1][2]) / d
        ay = (m[0][2] - m[2][0]) / d
        az = (m[1][0] - m[0][1]) / d
    n = math.sqrt(ax**2 + ay**2 + az**2)
    if n < 1e-9:
        return [0.0, 1.0, 0.0, 0.0]
    return [ax/n, ay/n, az/n, angle]


def read_object_poses(supervisor, names):
    result = []
    for name in names:
        node = get_node_by_name(supervisor, name)
        if node is None:
            print(f"[Supervisor] 找不到物體節點: {name}")
            continue
        pos = list(node.getPosition())
        ori = list(node.getOrientation())
        m   = [[ori[r*3+c] for c in range(3)] for r in range(3)]
        aa  = rot_mat_to_axis_angle(m)
        result.append({
            "name":                "name",
            "position_m":          pos,
            "rotation_axis_angle": aa,
        })
        result[-1]["name"] = name
    return result


def read_camera_pose(camera_node):
    pos = list(camera_node.getPosition())
    ori = list(camera_node.getOrientation())
    m   = [[ori[r*3+c] for c in range(3)] for r in range(3)]
    roll, pitch, yaw = rot_mat_to_rpy(m)
    return pos, [roll, pitch, yaw]


# ── arm control ───────────────────────────────────────────────────────────────

def send_waypoint(emitter, joints_rad, command_id):
    payload = {"type": "waypoint", "joints": list(joints_rad), "gripper": 0.0, "id": str(command_id)}
    emitter.send(json.dumps(payload).encode("utf-8"))


def send_path(emitter, waypoints_rad, command_id):
    payload = {"type": "path", "waypoints": [list(w) for w in waypoints_rad], "gripper": 0.0, "id": str(command_id)}
    emitter.send(json.dumps(payload).encode("utf-8"))


def load_planned_paths():
    if not os.path.exists(PLANNED_PATHS_PATH):
        return None, []
    with open(PLANNED_PATHS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    path_dict = {}
    for entry in data.get("paths", []):
        key = (str(entry["from_id"]), str(entry["to_id"]))
        if "waypoints_rad" in entry:
            path_dict[key] = [list(wp) for wp in entry["waypoints_rad"]]
        else:
            path_dict[key] = [wp["positions"] for wp in entry["waypoints"]]
    visit_order = []
    cur = "home"
    visited = set()
    while True:
        found = False
        for (from_id, to_id) in path_dict:
            if from_id == cur and to_id != "home" and to_id not in visited:
                visit_order.append(to_id)
                visited.add(to_id)
                cur = to_id
                found = True
                break
        if not found:
            break
    return path_dict, visit_order


def wait_seconds(supervisor, timestep, seconds):
    steps = max(0, int(seconds * 1000 / max(1, timestep)))
    for _ in range(steps):
        if supervisor.step(timestep) == -1:
            return False
    return True


def wait_for_arrival(supervisor, timestep, receiver, command_id, timeout_sec):
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
            if data.get("status") == "arrived" and str(data.get("command_id")) == str(command_id):
                print(f"[Supervisor] 到達 (max_err={float(data.get('max_error_rad',0)):.4f} rad)")
                return True
    print(f"[Supervisor] 等待超時: command {command_id}")
    return False


def ensure_home(supervisor, timestep, emitter, receiver, timeout_sec):
    """送手臂回 Home 並等到位。

    重送 "home" 直到控制器回報 "moving"（ack）為止——涵蓋每場景重開 webots 時、
    手臂控制器尚在啟動而漏接早期指令的競態。ack 後即停止重送，避免一直重置
    控制器的到達穩定計時（ARRIVAL_HOLD_SEC）導致永遠等不到 arrived。
    """
    home_rad = [math.radians(d) for d in HOME_POSE_DEG]
    t0 = supervisor.getTime()
    acked = False
    last_send = -1e9
    while supervisor.getTime() - t0 <= timeout_sec:
        if not acked and supervisor.getTime() - last_send >= 1.0:
            while receiver and receiver.getQueueLength() > 0:
                receiver.nextPacket()
            send_waypoint(emitter, home_rad, "home")
            last_send = supervisor.getTime()
        if supervisor.step(timestep) == -1:
            return False
        while receiver and receiver.getQueueLength() > 0:
            msg = receiver.getString()
            receiver.nextPacket()
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue
            if str(data.get("command_id")) != "home":
                continue
            if data.get("status") == "moving":
                acked = True
            elif data.get("status") == "arrived":
                print(f"[Supervisor] spawn 前已回到 Home (max_err={float(data.get('max_error_rad', 0)):.4f} rad)")
                return True
    print("[Supervisor] spawn 前回 Home 未確認（逾時），繼續。")
    return False


# ── main scene loop ───────────────────────────────────────────────────────────

def run_scene(supervisor, timestep, emitter, receiver, camera_node, scene, path_dict=None, visit_order=None, captures_dir=None):
    scene_objects = scene["objects"]
    names      = [obj["name"] for obj in scene_objects]
    label      = "+".join(names)
    n_views    = len(scene["viewpoints"])
    scene_name = scene.get("scene_name", label)
    scene_id   = scene_name
    n_objs     = len(scene_objects)
    base_dir   = captures_dir or os.path.join(DATA_DIR, "captures", f"multi_n{n_objs}")
    scene_dir  = os.path.join(base_dir, scene_name)
    os.makedirs(scene_dir, exist_ok=True)
    print(f"[Supervisor] 場景目錄: {scene_dir}")

    clear_ycb_objects(supervisor)

    # 先確保手臂回到 Home（桌面已清空、尚未 spawn 時），才生成物體。
    # 涵蓋：第一場景手臂初始姿態未必安全、每場景重開 webots 時手臂從初始姿態啟動。
    # 重送 home 直到控制器 ack（fresh webots 啟動有非同步競態，早期指令會漏接）。
    ensure_home(supervisor, timestep, emitter, receiver, 30.0)

    spawn_positions = spawn_objects(supervisor, scene_objects)
    if not wait_seconds(supervisor, timestep, SCENE_SETTLE_SEC):
        return False

    current_id  = "home"
    current_deg = HOME_POSE_DEG[:]
    actual_viewpoints = []

    viewpoints = scene["viewpoints"]
    if visit_order:
        vp_by_id = {str(vp["id"]): vp for vp in viewpoints}
        ordered = [vp_by_id[vid] for vid in visit_order if vid in vp_by_id]
        remaining = [vp for vp in viewpoints if str(vp["id"]) not in set(visit_order)]
        viewpoints = ordered + remaining

    for vp in viewpoints:
        vp_id     = vp["id"]
        joint_deg = vp["joint_deg"]
        joint_rad = [math.radians(d) for d in joint_deg]
        curr_rad  = [math.radians(d) for d in current_deg]

        path_key = (str(current_id), str(vp_id))
        planned  = path_dict.get(path_key) if path_dict else None

        max_delta = max(abs(t - c) for t, c in zip(joint_rad, curr_rad))
        timeout   = max(ARM_SETTLE_TIME_SEC,
                        max_delta / max(ARM_MOTOR_VELOCITY_RAD_PER_SEC, 1e-6)
                        + ARM_SETTLE_TIME_BUFFER_SEC)
        if planned:
            timeout += len(planned) * 0.5

        print(f"[Supervisor] 移動到視角 {vp_id}{'（規劃路徑）' if planned else '（直接）'}...")
        while receiver and receiver.getQueueLength() > 0:
            receiver.nextPacket()
        if planned:
            send_path(emitter, planned, vp_id)
        else:
            send_waypoint(emitter, joint_rad, vp_id)
        if not wait_for_arrival(supervisor, timestep, receiver, vp_id, timeout):
            print(f"[Supervisor] 視角 {vp_id} 超時，跳過。")
            continue
        current_id  = str(vp_id)
        current_deg = joint_deg[:]

        if POST_ARRIVAL_PAUSE_SEC > 0:
            if not wait_seconds(supervisor, timestep, POST_ARRIVAL_PAUSE_SEC):
                return False

        cam_pos, cam_rpy = read_camera_pose(camera_node)
        cam_rpy_deg = [math.degrees(r) for r in cam_rpy]

        # 手臂到位後讀取物體實際位姿
        actual_objects = read_object_poses(supervisor, names)

        view_name     = f"view_{vp_id:02d}"
        capture_token = f"{vp_id}_{int(supervisor.getTime() * 1000)}"
        joint_str     = ",".join(f"{d:.6f}" for d in joint_deg)
        camera_node.getField("customData").setSFString(
            f"capture_token={capture_token};"
            f"view={view_name};"
            f"label={scene_id};"
            f"scene_dir={scene_dir};"
            f"joint_deg={joint_str}"
        )
        print(f"[Supervisor] 拍攝視角 {vp_id}")
        if not wait_seconds(supervisor, timestep, CAPTURE_WAIT_SEC):
            return False

        actual_viewpoints.append({
            "id":        vp_id,
            "joint_deg": joint_deg,
            "camera": {
                "position_m":       cam_pos,
                "rotation_rpy_rad": cam_rpy,
                "rotation_rpy_deg": cam_rpy_deg,
            },
            "objects": actual_objects,
            "files": {
                "rgb":       f"{view_name}.png",
                "depth_npy": f"{view_name}_depth.npy",
                "depth_vis": f"{view_name}_depth.png",
            },
        })

    # 所有視角完成後，手臂回 Home，穩定 1 秒後重置物理
    home_rad = [math.radians(d) for d in HOME_POSE_DEG]
    home_path_key = (str(current_id), "home")
    home_planned = path_dict.get(home_path_key) if path_dict else None
    while receiver and receiver.getQueueLength() > 0:
        receiver.nextPacket()
    if home_planned:
        send_path(emitter, home_planned, "home")
    else:
        send_waypoint(emitter, home_rad, "home")
    curr_home_rad = [math.radians(d) for d in current_deg]
    max_delta = max(abs(t - c) for t, c in zip(home_rad, curr_home_rad))
    home_timeout = max(ARM_SETTLE_TIME_SEC,
                       max_delta / max(ARM_MOTOR_VELOCITY_RAD_PER_SEC, 1e-6)
                       + ARM_SETTLE_TIME_BUFFER_SEC)
    if home_planned:
        home_timeout += len(home_planned) * 0.5
    print("[Supervisor] 手臂返回 Home...")
    if not wait_for_arrival(supervisor, timestep, receiver, "home", home_timeout):
        print("[Supervisor] 返回 Home 超時，繼續。")
    if not wait_seconds(supervisor, timestep, 1.0):
        return True
    supervisor.simulationResetPhysics()

    manifest = {
        "scene_id":    scene_id,
        "scene_dir":   scene_dir,
        "camera_spec": CAMERA_SPEC,
        "planned": {
            "objects": [
                {
                    "name":                      n,
                    "spawn_position_m":          spawn_positions.get(n, [0, 0, 0]),
                    "spawn_rotation_axis_angle": [0, 1, 0, 0],
                }
                for n in names
            ],
            "viewpoints": [{"id": vp["id"], "joint_deg": vp["joint_deg"]}
                           for vp in scene["viewpoints"]],
        },
        "actual": {
            "viewpoints": actual_viewpoints,
        },
    }
    manifest_path = os.path.join(scene_dir, "scene_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Supervisor] Manifest 已寫入: {manifest_path}")
    return True


def main():
    supervisor = Supervisor()
    timestep   = int(supervisor.getBasicTimeStep())

    # 解析 --1/--3/--4/--5 與 --<場景編號> 參數
    # CAPTURE_ARGS 環境變數優先於 .wbt 的 controllerArgs（同 VALIDATOR_ARGS 慣例），
    # 讓 run_capture_multi.sh 能逐場景傳參而不必改 .wbt。
    _env_args = os.environ.get("CAPTURE_ARGS")
    cli_args  = _env_args.split() if _env_args else sys.argv[1:]
    n_filter  = None
    scene_num = None
    for arg in cli_args:
        if n_filter is None and arg in ("--1", "--3", "--4", "--5"):
            n_filter = int(arg[2:])
        elif n_filter is not None and arg.startswith("--") and arg[2:].isdigit():
            scene_num = int(arg[2:])

    if n_filter == 1:
        with open(SINGLE_SCENE_PLAN_PATH, encoding="utf-8") as f:
            plan = json.load(f)
    else:
        with open(SCENE_PLAN_PATH, encoding="utf-8") as f:
            plan = json.load(f)
    all_scenes = plan["scenes"]

    if n_filter is not None:
        prefix = f"n{n_filter}_"
        scenes = [s for s in all_scenes if s.get("scene_name", "").startswith(prefix)]
        if scene_num is not None:
            target = f"n{n_filter}_scene{scene_num:04d}"
            scenes = [s for s in scenes if s.get("scene_name") == target]
            print(f"[Supervisor] 單一場景：{target}")
        else:
            print(f"[Supervisor] n{n_filter} 全部 {len(scenes)} 個場景")
    else:
        scenes = all_scenes
        print(f"[Supervisor] 全部 {len(scenes)} 個場景")

    camera_node = supervisor.getFromDef(CAMERA_DEF)
    emitter     = supervisor.getDevice(ARM_COMMAND_EMITTER)
    receiver    = supervisor.getDevice(ARM_STATUS_RECEIVER)
    if receiver:
        receiver.enable(timestep)
    else:
        print(f"[Supervisor] 找不到 {ARM_STATUS_RECEIVER}，改用時間等待")

    path_dict, visit_order = load_planned_paths()
    if path_dict:
        print(f"[Supervisor] 載入規劃路徑：{len(path_dict)} 條路段")
        if visit_order:
            print(f"[Supervisor] 規劃遍歷順序: home → {' → '.join(visit_order)} → home")
    else:
        print("[Supervisor] 未找到 planned_paths.json，使用直接 joint 控制")

    print(f"[Supervisor] 共 {len(scenes)} 個場景")
    for i, scene in enumerate(scenes, 1):
        print(f"\n[Supervisor] ── 場景 {i}/{len(scenes)} ──")
        if not run_scene(supervisor, timestep, emitter, receiver,
                         camera_node, scene, path_dict=path_dict, visit_order=visit_order):
            print("[Supervisor] 場景失敗，中止。")
            return

    print("\n[Supervisor] 所有場景完成。")
    supervisor.simulationQuit(0)


if __name__ == "__main__":
    main()
