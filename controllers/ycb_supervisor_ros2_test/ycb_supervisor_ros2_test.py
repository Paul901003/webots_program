"""
ycb_supervisor_ros2_test.py

Webots Supervisor + ROS2/MoveIt 路徑規劃整合控制器。

Webots controller 使用 Python 3.8（pyenv），rclpy 需要 Python 3.12。
解法：以 subprocess 啟動 ros2_bridge_subprocess.py（python3.12），
透過 stdin/stdout JSON 進行 IPC。

流程：
  1. 動態 spawn YCB 物件
  2. 等物件穩定
  3. 對每個視角：
     a. 從 Webots 讀取 YCB 物件當前位置 → 建立碰撞物體清單
     b. 透過 subprocess 送規劃請求 → 等 waypoints
     c. 逐一執行 waypoints（送給手臂控制器 → 等 arrived）
     d. 觸發相機拍照

使用前須先啟動 ROS2 規劃橋接器：
  source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash
  ros2 launch ur5e_webots_planning planning_bridge_launch.py
"""

import importlib.util
import json
import math
import os
import random
import sys
import time

from controller import Supervisor

# ── 路徑設定 ───────────────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YCB_SUPERVISOR_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "ycb_supervisor")
if YCB_SUPERVISOR_DIR not in sys.path:
    sys.path.insert(0, YCB_SUPERVISOR_DIR)

from config import (  # noqa: E402
    NUM_OBJECTS, GRID_COLS, SPACING, SPAWN_HEIGHT, X_OFFSET, Z_OFFSET,
    ASSET_BASE, TARGET_OBJECTS, MASS_TABLE, ALL_OBJECTS,
    DEFAULT_SHAPE, SHAPE_TABLE, SPAWN_CLEARANCE, SPACING_MARGIN,
    ARM_MOTOR_VELOCITY_RAD_PER_SEC, ARM_SETTLE_TIME_BUFFER_SEC,
    POST_ARRIVAL_PAUSE_SEC,
)

# ── 常數 ───────────────────────────────────────────────────────────────────────
ARM_COMMAND_EMITTER = "arm_command_emitter"
ARM_STATUS_RECEIVER = "arm_status_receiver"
UR5E_DEF = "UR5E"
UR5E_CAMERA_DEF = "UR5E_CAMERA"
FALLBACK_VIEW_SEQUENCE = (1, 2, 3, 4)
HOME_POSE_RAD = [0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0]

BRIDGE_STARTUP_TIMEOUT_SEC = 30.0
PHYSICS_SETTLE_SEC = 2.0
CAPTURE_WARMUP_STEPS = 5


# ── YCB 幾何資料 ───────────────────────────────────────────────────────────────
def _load_ycb_geo() -> dict:
    path = os.path.join(YCB_SUPERVISOR_DIR, "ycb_geometries.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Supervisor] 無法載入 ycb_geometries.json: {e}")
        return {}


YCB_GEO_DATA = _load_ycb_geo()


def get_geometry(name: str) -> dict:
    return YCB_GEO_DATA.get(name, {
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "size": {"x": 0.1, "y": 0.1, "z": 0.1},
    })


# ── spawn 邏輯 ──────────────────────────────────────────────────────────────────
def get_collision_half_height(name: str) -> float:
    shape = SHAPE_TABLE.get(name, DEFAULT_SHAPE)
    size = get_geometry(name)["size"]
    if shape == "Sphere":
        return (size["x"] + size["y"] + size["z"]) / 6.0
    return size["z"] / 2.0


def get_collision_footprint(name: str) -> float:
    size = get_geometry(name)["size"]
    return max(size["x"], size["y"])


def compute_grid_positions(count: int, cols: int, spacing: float) -> list:
    rows = math.ceil(count / cols)
    return [
        ((i % cols - (cols - 1) / 2.0) * spacing,
         (i // cols - (rows - 1) / 2.0) * spacing)
        for i in range(count)
    ]


def make_bounding_object(name: str, sx: float, sy: float, sz: float) -> str:
    shape = SHAPE_TABLE.get(name, DEFAULT_SHAPE)
    if shape == "Sphere":
        r = (sx + sy + sz) / 6.0
        return f"boundingObject Sphere {{ radius {r:.6f} }}"
    if shape == "Cylinder":
        r = (sx + sy) / 4.0
        return f"boundingObject Cylinder {{ radius {r:.6f} height {sz:.6f} }}"
    return f"boundingObject Box {{ size {sx:.6f} {sy:.6f} {sz:.6f} }}"


def make_vrml(name: str, x: float, y: float, z: float) -> str:
    mass = MASS_TABLE[name]
    base = f"{ASSET_BASE}/{name}/google_16k"
    geo = get_geometry(name)
    cx, cy, cz = geo["center"]["x"], geo["center"]["y"], geo["center"]["z"]
    sx, sy, sz = geo["size"]["x"], geo["size"]["y"], geo["size"]["z"]
    bounding = make_bounding_object(name, sx, sy, sz)
    return f"""Solid {{
  translation {x:.6f} {y:.6f} {z:.6f}
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
  physics Physics {{ mass {mass} }}
}}"""


def clear_ycb_objects(supervisor: Supervisor):
    root_children = supervisor.getRoot().getField("children")
    i = root_children.getCount() - 1
    while i >= 0:
        node = root_children.getMFNode(i)
        if node is not None:
            name_field = node.getField("name")
            if name_field and name_field.getSFString() in ALL_OBJECTS:
                node.remove()
        i -= 1


def spawn_objects(supervisor: Supervisor, object_list: list):
    if not object_list:
        return
    largest = max(get_collision_footprint(n) for n in object_list)
    safe_spacing = max(SPACING, largest + SPACING_MARGIN)
    positions = compute_grid_positions(len(object_list), GRID_COLS, safe_spacing)
    root_children = supervisor.getRoot().getField("children")
    for name, (gx, gy) in zip(object_list, positions):
        if name not in MASS_TABLE:
            continue
        fz = max(SPAWN_HEIGHT, get_collision_half_height(name) + SPAWN_CLEARANCE)
        root_children.importMFNodeFromString(-1, make_vrml(name, gx + X_OFFSET, gy + Z_OFFSET, fz))


# ── 手臂姿態讀取 ───────────────────────────────────────────────────────────────
def load_camera_poses(supervisor: Supervisor):
    ur5e_node = supervisor.getFromDef(UR5E_DEF)
    if ur5e_node is None:
        return {}, FALLBACK_VIEW_SEQUENCE
    ctrl_field = ur5e_node.getField("controller")
    if ctrl_field is None:
        return {}, FALLBACK_VIEW_SEQUENCE
    controller_name = ctrl_field.getSFString()
    ctrl_path = os.path.join(
        os.path.dirname(CURRENT_DIR), controller_name, f"{controller_name}.py"
    )
    try:
        spec = importlib.util.spec_from_file_location(controller_name, ctrl_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        camera_poses = getattr(mod, "CAMERA_POSES", {})
        sequence = tuple(sorted(int(k) for k in camera_poses.keys()))
        return camera_poses, sequence or FALLBACK_VIEW_SEQUENCE
    except Exception as e:
        print(f"[Supervisor] 無法讀取 CAMERA_POSES: {e}")
        return {}, FALLBACK_VIEW_SEQUENCE


def pose_to_joints_rad(camera_poses: dict, view_index: int):
    pose = camera_poses.get(view_index)
    if not isinstance(pose, dict):
        return None
    joint_deg = pose.get("joint_deg")
    if not isinstance(joint_deg, list) or len(joint_deg) != 6:
        return None
    return [math.radians(float(v)) for v in joint_deg]


def estimate_travel_time(current: list, target: list) -> float:
    max_delta = max(abs(t - c) for t, c in zip(target, current))
    motion_time = max_delta / max(ARM_MOTOR_VELOCITY_RAD_PER_SEC, 1e-6)
    return max(5.0, motion_time + ARM_SETTLE_TIME_BUFFER_SEC)


# ── ROS2 subprocess 橋接（從共用模組引入）────────────────────────────────────
from ros2_bridge_utils import (  # noqa: E402
    launch_ros2_bridge,
    wait_for_bridge_ready,
    request_plan,
    stop_ros2_bridge,
)


# ── 碰撞物件建立 ───────────────────────────────────────────────────────────────
def build_collision_objects(supervisor: Supervisor, spawned_names: list, ur5e_world_pos: list) -> list:
    root_children = supervisor.getRoot().getField("children")
    node_map = {}
    for i in range(root_children.getCount()):
        node = root_children.getMFNode(i)
        if node is None:
            continue
        name_field = node.getField("name")
        if name_field and name_field.getSFString() in spawned_names:
            node_map[name_field.getSFString()] = node

    objects = []
    for name in spawned_names:
        node = node_map.get(name)
        if node is None:
            continue
        wp = node.getPosition()
        bx = wp[0] - ur5e_world_pos[0]
        by = wp[1] - ur5e_world_pos[1]
        bz = wp[2] - ur5e_world_pos[2]
        geo = get_geometry(name)
        sx, sy, sz = geo["size"]["x"], geo["size"]["y"], geo["size"]["z"]
        objects.append({
            "id": name,
            "position": [round(bx, 4), round(by, 4), round(bz, 4)],
            "size": [round(sx, 4), round(sy, 4), round(sz, 4)],
            "shape": SHAPE_TABLE.get(name, DEFAULT_SHAPE).lower(),
        })

    # 地板碰撞物件
    floor_z = 0.0 - ur5e_world_pos[2]
    objects.append({
        "id": "floor",
        "position": [0.0, 0.0, round(floor_z - 0.05, 4)],
        "size": [4.0, 4.0, 0.1],
        "shape": "box",
    })
    return objects


# ── Webots 通訊輔助 ────────────────────────────────────────────────────────────
def clear_receiver(receiver):
    if receiver is None:
        return
    while receiver.getQueueLength() > 0:
        receiver.nextPacket()


def send_waypoint(emitter, joints: list, command_id: str) -> bool:
    if emitter is None:
        return False
    payload = {"type": "waypoint", "joints": [float(v) for v in joints], "id": command_id}
    emitter.send(json.dumps(payload).encode("utf-8"))
    return True


def interpolate_trajectory(waypoints: list, timestep_ms: int, travel_time_sec: float) -> list:
    """MoveIt 稀疏 waypoints → 每個 simulation step 一個點的密集軌跡（線性內插）。"""
    n_steps = max(len(waypoints), int(travel_time_sec * 1000 / max(1, timestep_ms)))
    if len(waypoints) == 1:
        return [waypoints[0][:] for _ in range(n_steps)]
    n_seg = len(waypoints) - 1
    result = []
    for i in range(n_steps):
        t = i / (n_steps - 1)
        seg_f = t * n_seg
        seg = min(int(seg_f), n_seg - 1)
        lt = seg_f - seg
        a, b = waypoints[seg], waypoints[seg + 1]
        result.append([a[j] + lt * (b[j] - a[j]) for j in range(len(a))])
    return result


def wait_steps(supervisor: Supervisor, timestep: int, seconds: float) -> bool:
    steps = max(0, int(seconds * 1000 / max(1, timestep)))
    for _ in range(steps):
        if supervisor.step(timestep) == -1:
            return False
    return True


def wait_for_arrival(
    supervisor: Supervisor, timestep: int, receiver, command_id: str, timeout_sec: float
) -> bool:
    if receiver is None:
        return wait_steps(supervisor, timestep, timeout_sec)
    start = supervisor.getTime()
    while supervisor.getTime() - start <= timeout_sec:
        if supervisor.step(timestep) == -1:
            return False
        while receiver.getQueueLength() > 0:
            message = receiver.getString()
            receiver.nextPacket()
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue
            if (
                data.get("status") == "arrived"
                and str(data.get("command_id")) == str(command_id)
            ):
                print(
                    f"[Supervisor] Arm arrived '{command_id}' "
                    f"(max_error={float(data.get('max_error_rad', 0.0)):.4f} rad)"
                )
                return True
    print(f"[Supervisor] Arrival timeout '{command_id}'")
    return False


def trigger_capture(camera_node, view_index: int, label: str, capture_token: str):
    if camera_node is None:
        return
    custom = f"capture_token={capture_token};view={view_index};label={label}"
    camera_node.getField("customData").setSFString(custom)


# ── 主程式 ─────────────────────────────────────────────────────────────────────
def main():
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    arm_emitter = supervisor.getDevice(ARM_COMMAND_EMITTER)
    arm_receiver = supervisor.getDevice(ARM_STATUS_RECEIVER)
    if arm_receiver is not None:
        arm_receiver.enable(timestep)
    else:
        print(f"[Supervisor] 找不到 {ARM_STATUS_RECEIVER}，改用時間等待")

    camera_node = supervisor.getFromDef(UR5E_CAMERA_DEF)

    ur5e_node = supervisor.getFromDef(UR5E_DEF)
    ur5e_world_pos = list(ur5e_node.getPosition()) if ur5e_node else [0.0, 0.0, 0.0]
    print(f"[Supervisor] UR5E 世界座標: {ur5e_world_pos}")

    camera_poses, view_sequence = load_camera_poses(supervisor)
    print(f"[Supervisor] 視角序列: {view_sequence}")

    # 決定 spawn 的物件
    if TARGET_OBJECTS:
        object_list = list(TARGET_OBJECTS)
    else:
        object_list = random.sample(ALL_OBJECTS, k=min(NUM_OBJECTS, len(ALL_OBJECTS)))
    print(f"[Supervisor] 生成物件: {object_list}")

    clear_ycb_objects(supervisor)
    supervisor.step(timestep)
    spawn_objects(supervisor, object_list)

    # 先啟動 bridge 子行程（非阻塞），讓它在物理穩定期間完成初始化
    bridge_proc_raw, bridge_line_queue = launch_ros2_bridge()

    # 物理穩定 2 秒（同時輪詢 bridge，若提早就緒不會多等）
    print(f"[Supervisor] 等待物理穩定 {PHYSICS_SETTLE_SEC:.1f}s ...")
    bridge_proc = wait_for_bridge_ready(
        supervisor, timestep, bridge_proc_raw, bridge_line_queue, PHYSICS_SETTLE_SEC
    )

    # 物理穩定結束後若 bridge 還沒好，再多等 BRIDGE_STARTUP_TIMEOUT_SEC
    if bridge_proc_raw is not None and bridge_proc is None:
        print(f"[Supervisor] Bridge 尚未就緒，繼續等候（最多 {BRIDGE_STARTUP_TIMEOUT_SEC:.0f}s）...")
        bridge_proc = wait_for_bridge_ready(
            supervisor, timestep, bridge_proc_raw, bridge_line_queue, BRIDGE_STARTUP_TIMEOUT_SEC
        )
    if bridge_proc is None:
        print("[Supervisor] 無 ROS2 bridge，使用直接移動模式")

    current_joints = HOME_POSE_RAD[:]
    capture_label = "_".join(n.split("_")[0] for n in object_list[:3])

    # 結果記錄
    view_results = []

    try:
        for view_index in view_sequence:
            target_joints = pose_to_joints_rad(camera_poses, view_index)
            if target_joints is None:
                print(f"[Supervisor] 視角 {view_index} 無效，跳過")
                view_results.append({
                    "view": view_index, "status": "skipped", "reason": "invalid pose",
                    "plan_success": None, "waypoints": 0, "arm_arrived": False, "captured": False,
                })
                continue

            print(f"\n[Supervisor] ═══ 視角 {view_index} ═══")
            travel_time = estimate_travel_time(current_joints, target_joints)

            plan_success = None
            plan_error = ""
            n_waypoints = 1

            if bridge_proc is not None:
                collision_objects = build_collision_objects(supervisor, object_list, ur5e_world_pos)
                print(f"[Supervisor] 碰撞物件數: {len(collision_objects)}，請求 MoveIt 規劃...")
                result = request_plan(bridge_proc, bridge_line_queue, current_joints, target_joints, collision_objects,
                                     supervisor=supervisor, timestep=timestep)

                if result is None or not result.get("success"):
                    plan_error = result.get("error", "unknown") if result else "no response"
                    print(f"[Supervisor] 規劃失敗: {plan_error}，改用直接移動")
                    waypoints = [target_joints]
                    plan_success = False
                else:
                    waypoints = result.get("waypoints") or [target_joints]
                    n_waypoints = len(waypoints)
                    print(f"[Supervisor] 規劃成功，{n_waypoints} 個 waypoints")
                    plan_success = True
            else:
                waypoints = [target_joints]

            # 平滑軌跡執行：內插成密集軌跡，每個 simulation timestep 送一個點
            dense = interpolate_trajectory(waypoints, timestep, travel_time)
            final_cid = f"v{view_index}_final"
            clear_receiver(arm_receiver)
            arm_arrived = True

            for step_i, wp_joints in enumerate(dense):
                is_last  = (step_i == len(dense) - 1)
                cmd_id   = final_cid if is_last else f"v{view_index}_s{step_i}"

                if not send_waypoint(arm_emitter, wp_joints, cmd_id):
                    print("[Supervisor] 找不到 arm emitter，停止")
                    return

                if supervisor.step(timestep) == -1:
                    arm_arrived = False
                    break

            if arm_arrived:
                if not wait_for_arrival(supervisor, timestep, arm_receiver, final_cid, travel_time):
                    print(f"[Supervisor] 視角 {view_index} 最終位置 timeout，停止")
                    arm_arrived = False
                    view_results.append({
                        "view": view_index, "status": "arm_timeout",
                        "plan_success": plan_success, "plan_error": plan_error,
                        "waypoints": n_waypoints, "arm_arrived": False, "captured": False,
                    })
                    return

            current_joints = list(dense[-1]) if dense else list(target_joints)

            # 拍照
            captured = False
            if arm_arrived:
                if POST_ARRIVAL_PAUSE_SEC > 0:
                    wait_steps(supervisor, timestep, POST_ARRIVAL_PAUSE_SEC)
                capture_token = f"v{view_index}_{int(supervisor.getTime() * 1000)}"
                trigger_capture(camera_node, view_index, capture_label, capture_token)
                print(f"[Supervisor] 觸發拍照 (token={capture_token})")
                captured = True
                for _ in range(CAPTURE_WARMUP_STEPS):
                    if supervisor.step(timestep) == -1:
                        break

            view_results.append({
                "view": view_index,
                "status": "ok" if arm_arrived else "arm_timeout",
                "plan_success": plan_success,
                "plan_error": plan_error,
                "waypoints": n_waypoints,
                "arm_arrived": arm_arrived,
                "captured": captured,
            })

    finally:
        stop_ros2_bridge(bridge_proc)

    # ── 結果摘要 ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 52)
    print("  執行結果摘要")
    print("═" * 52)
    print(f"  物件:   {', '.join(object_list)}")
    print(f"  視角數: {len(view_sequence)}  |  完成: {sum(1 for r in view_results if r['captured'])}")
    print(f"  MoveIt: {'已啟用' if bridge_proc is not None else '未啟用（直接移動）'}")
    print("─" * 52)
    print(f"  {'視角':^4}  {'規劃':^6}  {'Waypoints':^9}  {'到達':^4}  {'拍照':^4}  備註")
    print("─" * 52)
    for r in view_results:
        plan_str = {True: "成功", False: "失敗", None: "跳過"}.get(r["plan_success"], "-")
        arr_str  = "✓" if r["arm_arrived"] else "✗"
        cap_str  = "✓" if r["captured"]    else "✗"
        note     = r.get("plan_error", "") or r.get("reason", "")
        note     = f"  ({note})" if note else ""
        print(f"  {r['view']:^4}  {plan_str:^6}  {r['waypoints']:^9}  {arr_str:^4}  {cap_str:^4}{note}")
    print("═" * 52)
    ok_count  = sum(1 for r in view_results if r["captured"])
    fail_count = len(view_results) - ok_count
    print(f"  完成 {ok_count} 個視角，失敗/跳過 {fail_count} 個")
    print("═" * 52 + "\n")


if __name__ == "__main__":
    main()
