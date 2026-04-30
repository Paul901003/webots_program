from controller import Supervisor
import importlib.util
import math
import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_CONTROLLER_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "ycb_supervisor")
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if SOURCE_CONTROLLER_DIR not in sys.path:
    sys.path.append(SOURCE_CONTROLLER_DIR)

from config import (  # noqa: E402
    NUM_OBJECTS,
    GRID_COLS,
    SPACING,
    SPAWN_HEIGHT,
    REFERENCE_X,
    REFERENCE_Y,
    X_OFFSET,
    Z_OFFSET,
    ASSET_BASE,
    TARGET_OBJECTS,
    MASS_TABLE,
    ALL_OBJECTS,
    DEFAULT_SHAPE,
    SHAPE_TABLE,
    SPAWN_CLEARANCE,
    SPACING_MARGIN,
    ARM_SETTLE_TIME_SEC,
    POST_ARRIVAL_PAUSE_SEC,
    ARM_MOTOR_VELOCITY_RAD_PER_SEC,
    ARM_SETTLE_TIME_BUFFER_SEC,
    MULTI_SCENE_FILE,
)

JSON_PATH = os.path.join(SOURCE_CONTROLLER_DIR, "ycb_geometries.json")
with open(JSON_PATH, "r", encoding="utf-8") as file:
    YCB_GEO_DATA = json.load(file)

UR5E_DEF = "UR5E"
CAMERA_DEF = "UR5E_CAMERA"
ARM_COMMAND_EMITTER = "arm_command_emitter"
ARM_STATUS_RECEIVER = "arm_status_receiver"
CAPTURE_WAIT_SEC = 1.0
VIEW_SEQUENCE = None  # None = 依 CAMERA_POSES 鍵值自動決定
CAPTURE_ROOT = "captures_multi"
SCENE_SETTLE_TIME_SEC = 1.0
SCENE_POSE_FILENAME = "scene_objects_pose.json"
HOME_POSE_RAD = [0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0]
REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
TEST_IMAGES_DIR = os.path.join(
    REPO_ROOT,
    "Grounded-Segment-Anything",
    "test_images",
)
SCENE_PLAN_PATH = os.path.join(CURRENT_DIR, MULTI_SCENE_FILE)
ACTIVE_CAPTURE_ROOT = CAPTURE_ROOT


def get_geometry(name: str):
    return YCB_GEO_DATA.get(
        name,
        {
            "center": {"x": 0.0, "y": 0.0, "z": 0.0},
            "size": {"x": 0.1, "y": 0.1, "z": 0.1},
        },
    )


def get_collision_half_height(name: str) -> float:
    shape = SHAPE_TABLE.get(name, DEFAULT_SHAPE)
    size = get_geometry(name)["size"]
    sx, sy, sz = size["x"], size["y"], size["z"]
    if shape == "Sphere":
        return (sx + sy + sz) / 6.0
    return sz / 2.0


def get_collision_footprint(name: str) -> float:
    size = get_geometry(name)["size"]
    return max(size["x"], size["y"])


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
  physics Physics {{
    density -1
    mass {mass}
  }}
}}"""


def choose_grid_cols(n: int, max_cols: int) -> int:
    if n <= 0:
        return 1
    square_like_cols = math.ceil(math.sqrt(n))
    return max(1, min(max_cols, square_like_cols))


def compute_grid_positions(n: int, max_cols: int, spacing: float):
    cols = choose_grid_cols(n, max_cols)
    rows = math.ceil(n / cols)
    positions = []
    for i in range(n):
        col = i % cols
        row = i // cols
        x = (col - (cols - 1) / 2.0) * spacing
        y = (row - (rows - 1) / 2.0) * spacing
        positions.append((x, y))
    return positions


def clear_ycb_objects(supervisor: Supervisor):
    root_children = supervisor.getRoot().getField("children")
    index = root_children.getCount() - 1
    while index >= 0:
        node = root_children.getMFNode(index)
        if node is not None:
            name_field = node.getField("name")
            if name_field is not None:
                node_name = name_field.getSFString()
                if node_name in ALL_OBJECTS:
                    node.remove()
        index -= 1


def spawn_objects(supervisor: Supervisor, object_list: list):
    if not object_list:
        print("[Supervisor] Warning: Object list is empty.")
        return

    largest_footprint = max(get_collision_footprint(name) for name in object_list)
    safe_spacing = max(SPACING, largest_footprint + SPACING_MARGIN)
    positions = compute_grid_positions(len(object_list), GRID_COLS, safe_spacing)
    root_children = supervisor.getRoot().getField("children")

    for name, (grid_x, grid_y) in zip(object_list, positions):
        if name not in MASS_TABLE:
            continue

        final_x = REFERENCE_X + X_OFFSET + grid_x
        final_y = REFERENCE_Y + Z_OFFSET + grid_y
        safe_spawn_height = max(
            SPAWN_HEIGHT,
            get_collision_half_height(name) + SPAWN_CLEARANCE,
        )
        root_children.importMFNodeFromString(
            -1,
            make_vrml(name, final_x, final_y, safe_spawn_height),
        )


def wait_seconds(supervisor: Supervisor, timestep: int, seconds: float):
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


def wait_for_arm_arrival(supervisor: Supervisor, timestep: int, receiver, command_id: str, timeout_sec: float):
    if receiver is None:
        return wait_seconds(supervisor, timestep, timeout_sec)
    start_time = supervisor.getTime()
    while supervisor.getTime() - start_time <= timeout_sec:
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
                    "[Supervisor] Arm arrived "
                    f"(max_error={float(data.get('max_error_rad', 0.0)):.4f} rad)"
                )
                return True
    print(f"[Supervisor] Arm arrival timeout for command {command_id}; stopping sequence.")
    return False


def rotation_matrix_to_rpy(matrix):
    sy = math.sqrt(matrix[0][0] ** 2 + matrix[1][0] ** 2)
    singular = sy < 1e-9

    if not singular:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        pitch = math.atan2(-matrix[2][0], sy)
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        roll = math.atan2(-matrix[1][2], matrix[1][1])
        pitch = math.atan2(-matrix[2][0], sy)
        yaw = 0.0

    return [roll, pitch, yaw]


def rotation_matrix_to_axis_angle(matrix):
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    angle = math.acos(cos_angle)

    if angle < 1e-9:
        return [0.0, 1.0, 0.0, 0.0]

    if abs(math.pi - angle) < 1e-6:
        xx = max(0.0, (matrix[0][0] + 1.0) / 2.0)
        yy = max(0.0, (matrix[1][1] + 1.0) / 2.0)
        zz = max(0.0, (matrix[2][2] + 1.0) / 2.0)
        xy = (matrix[0][1] + matrix[1][0]) / 4.0
        xz = (matrix[0][2] + matrix[2][0]) / 4.0
        yz = (matrix[1][2] + matrix[2][1]) / 4.0

        if xx >= yy and xx >= zz:
            axis_x = math.sqrt(xx)
            axis_y = 0.0 if axis_x < 1e-9 else xy / axis_x
            axis_z = 0.0 if axis_x < 1e-9 else xz / axis_x
        elif yy >= zz:
            axis_y = math.sqrt(yy)
            axis_x = 0.0 if axis_y < 1e-9 else xy / axis_y
            axis_z = 0.0 if axis_y < 1e-9 else yz / axis_y
        else:
            axis_z = math.sqrt(zz)
            axis_x = 0.0 if axis_z < 1e-9 else xz / axis_z
            axis_y = 0.0 if axis_z < 1e-9 else yz / axis_z
    else:
        denom = 2.0 * math.sin(angle)
        axis_x = (matrix[2][1] - matrix[1][2]) / denom
        axis_y = (matrix[0][2] - matrix[2][0]) / denom
        axis_z = (matrix[1][0] - matrix[0][1]) / denom

    axis_norm = math.sqrt(axis_x ** 2 + axis_y ** 2 + axis_z ** 2)
    if axis_norm < 1e-9:
        return [0.0, 1.0, 0.0, 0.0]

    return [axis_x / axis_norm, axis_y / axis_norm, axis_z / axis_norm, angle]


def get_scene_capture_dir(content_label: str):
    return os.path.join(TEST_IMAGES_DIR, ACTIVE_CAPTURE_ROOT, content_label)


def sanitize_path_part(value: str) -> str:
    cleaned = []
    for char in value.strip():
        if char.isalnum() or char in ("_", "-", "+"):
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "scene"


def build_capture_root_from_scene_plan(scene_plan_path: str) -> str:
    scene_plan_name = os.path.splitext(os.path.basename(scene_plan_path))[0]
    return f"{CAPTURE_ROOT}__{sanitize_path_part(scene_plan_name)}"


def get_object_node_by_name(supervisor: Supervisor, object_name: str):
    root_children = supervisor.getRoot().getField("children")
    for index in range(root_children.getCount() - 1, -1, -1):
        node = root_children.getMFNode(index)
        if node is None:
            continue
        name_field = node.getField("name")
        if name_field is None:
            continue
        if name_field.getSFString() == object_name:
            return node
    return None


def build_object_pose_record(index: int, name: str, node):
    position = node.getPosition()
    orientation = node.getOrientation()
    rotation_matrix = [
        [float(orientation[0]), float(orientation[1]), float(orientation[2])],
        [float(orientation[3]), float(orientation[4]), float(orientation[5])],
        [float(orientation[6]), float(orientation[7]), float(orientation[8])],
    ]
    roll, pitch, yaw = rotation_matrix_to_rpy(rotation_matrix)
    axis_x, axis_y, axis_z, angle = rotation_matrix_to_axis_angle(rotation_matrix)

    return {
        "index": index,
        "name": name,
        "position_m": {
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
        },
        "rotation_axis_angle": {
            "x": float(axis_x),
            "y": float(axis_y),
            "z": float(axis_z),
            "angle": float(angle),
        },
        "rotation_matrix": rotation_matrix,
        "rotation_rpy_rad": {
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
        },
        "rotation_rpy_deg": {
            "roll": float(math.degrees(roll)),
            "pitch": float(math.degrees(pitch)),
            "yaw": float(math.degrees(yaw)),
        },
    }


def save_scene_object_poses(supervisor: Supervisor, object_list, content_label: str):
    scene_dir = get_scene_capture_dir(content_label)
    os.makedirs(scene_dir, exist_ok=True)

    objects = []
    for index, name in enumerate(object_list, start=1):
        node = get_object_node_by_name(supervisor, name)
        if node is None:
            raise RuntimeError(f"找不到物體節點: {name}")
        objects.append(build_object_pose_record(index, name, node))

    payload = {
        "scene_label": content_label,
        "capture_root": ACTIVE_CAPTURE_ROOT,
        "scene_dir": scene_dir,
        "saved_at_sim_time_sec": float(supervisor.getTime()),
        "coordinate_frame": "webots_world",
        "object_count": len(objects),
        "objects": objects,
    }

    output_path = os.path.join(scene_dir, SCENE_POSE_FILENAME)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print(f"[Supervisor] Saved scene object poses: {output_path}")
    return output_path


def set_custom_data(node, data: str):
    if node is None:
        return
    field = node.getField("customData")
    if field is not None:
        field.setSFString(data)


def send_arm_pose_command(emitter, view_index: int):
    if emitter is None:
        print("[Supervisor] 找不到手臂 emitter，無法送出移動指令")
        return False
    emitter.send(str(view_index).encode("utf-8"))
    return True


def get_arm_controller_name(supervisor: Supervisor):
    ur5e_node = supervisor.getFromDef(UR5E_DEF)
    if ur5e_node is None:
        return None
    controller_field = ur5e_node.getField("controller")
    if controller_field is None:
        return None
    return controller_field.getSFString()


def load_camera_poses_from_arm_controller(supervisor: Supervisor):
    controller_name = get_arm_controller_name(supervisor)
    if not controller_name:
        return {}
    controller_path = os.path.join(
        os.path.dirname(CURRENT_DIR),
        controller_name,
        f"{controller_name}.py",
    )
    try:
        spec = importlib.util.spec_from_file_location(controller_name, controller_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "CAMERA_POSES", {})
    except Exception as error:
        print(f"[Supervisor] 無法讀取 {controller_path} 的 CAMERA_POSES: {error}")
        return {}


def pose_joints_rad(camera_poses: dict, view_index: int):
    pose = camera_poses.get(view_index)
    if not isinstance(pose, dict):
        return None
    joint_deg = pose.get("joint_deg")
    if not isinstance(joint_deg, list) or len(joint_deg) != 6:
        return None
    return [math.radians(float(value)) for value in joint_deg]


def estimate_settle_time(current_joints_rad, target_joints_rad):
    if current_joints_rad is None or target_joints_rad is None:
        return ARM_SETTLE_TIME_SEC
    max_delta = max(abs(target - current) for target, current in zip(target_joints_rad, current_joints_rad))
    motion_time = max_delta / max(ARM_MOTOR_VELOCITY_RAD_PER_SEC, 1e-6)
    return max(ARM_SETTLE_TIME_SEC, motion_time + ARM_SETTLE_TIME_BUFFER_SEC)


def build_content_label(object_list):
    if not object_list:
        return "empty_scene"

    cleaned_names = []
    for name in object_list:
        parts = name.split("_", 1)
        readable = parts[1] if len(parts) == 2 else parts[0]
        readable = readable.replace("-", "_")
        cleaned_names.append(readable)
    return "+".join(cleaned_names)


def normalize_scene_entry(scene_entry, scene_index: int):
    if not isinstance(scene_entry, list):
        raise ValueError(f"場景 {scene_index} 必須是 list，收到 {type(scene_entry).__name__}")

    normalized_objects = []
    for object_name in scene_entry:
        if not isinstance(object_name, str):
            raise ValueError(f"場景 {scene_index} 內的物體名稱必須是字串")
        if object_name not in MASS_TABLE:
            raise ValueError(f"場景 {scene_index} 含有未知物體: {object_name}")
        normalized_objects.append(object_name)

    if not normalized_objects:
        raise ValueError(f"場景 {scene_index} 不能是空的")

    return normalized_objects


def resolve_scene_plan_path(scene_plan_path: str):
    directory = os.path.dirname(scene_plan_path)
    filename = os.path.basename(scene_plan_path)
    stem, ext = os.path.splitext(filename)
    prefix = f"{stem}_"

    candidate_names = []
    for entry in os.listdir(directory):
        if entry.startswith(prefix) and entry.endswith(ext):
            candidate_names.append(entry)

    if not candidate_names:
        return scene_plan_path

    latest_name = max(candidate_names)
    return os.path.join(directory, latest_name)


def load_scene_plan_from_file(scene_plan_path: str):
    resolved_path = resolve_scene_plan_path(scene_plan_path)
    if not os.path.exists(resolved_path):
        return None

    with open(resolved_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        scenes = payload.get("scenes")
    else:
        scenes = payload

    if scenes is None:
        raise ValueError(f"場景檔缺少 scenes 欄位: {resolved_path}")
    if not isinstance(scenes, list):
        raise ValueError(f"場景檔內容必須是 list: {resolved_path}")

    normalized_scenes = []
    for scene_index, scene_entry in enumerate(scenes, start=1):
        normalized_scenes.append(normalize_scene_entry(scene_entry, scene_index))
    return normalized_scenes, resolved_path


def build_capture_plan():
    global ACTIVE_CAPTURE_ROOT

    scene_plan = load_scene_plan_from_file(SCENE_PLAN_PATH)
    if not scene_plan:
        raise RuntimeError(
            "[Supervisor] 找不到可用場景。請先執行 generate_multi_object_scenes.py 產生場景檔。"
        )
    file_scenes, resolved_path = scene_plan
    ACTIVE_CAPTURE_ROOT = build_capture_root_from_scene_plan(resolved_path)
    print(f"[Supervisor] Loaded {len(file_scenes)} scenes from {resolved_path}")
    print(f"[Supervisor] Capture root: {ACTIVE_CAPTURE_ROOT}")
    return file_scenes


def run_capture_sequence(supervisor: Supervisor, timestep: int, object_list):
    ur5e_node = supervisor.getFromDef(UR5E_DEF)
    camera_node = supervisor.getFromDef(CAMERA_DEF)
    arm_emitter = supervisor.getDevice(ARM_COMMAND_EMITTER)
    arm_status_receiver = supervisor.getDevice(ARM_STATUS_RECEIVER)
    if arm_status_receiver is not None:
        arm_status_receiver.enable(timestep)
    else:
        print(f"[Supervisor] 找不到 {ARM_STATUS_RECEIVER}，改用時間等待。")
    content_label = build_content_label(object_list)
    camera_poses = load_camera_poses_from_arm_controller(supervisor)
    view_sequence = sorted(camera_poses.keys()) if VIEW_SEQUENCE is None else VIEW_SEQUENCE
    current_joints_rad = getattr(run_capture_sequence, "current_joints_rad", HOME_POSE_RAD[:])

    if ur5e_node is None:
        print(f"[Supervisor] 找不到 DEF {UR5E_DEF}")
        return False
    if camera_node is None:
        print(f"[Supervisor] 找不到 DEF {CAMERA_DEF}")
        return False

    for view_index in view_sequence:
        target_joints_rad = pose_joints_rad(camera_poses, view_index)
        settle_time = estimate_settle_time(current_joints_rad, target_joints_rad)
        print(f"[Supervisor] Moving arm to view {view_index}...")
        clear_receiver(arm_status_receiver)
        if not send_arm_pose_command(arm_emitter, view_index):
            return False
        print(f"[Supervisor] Waiting for arm arrival (timeout {settle_time:.2f}s)...")
        if not wait_for_arm_arrival(
            supervisor,
            timestep,
            arm_status_receiver,
            str(view_index),
            settle_time,
        ):
            return False
        current_joints_rad = target_joints_rad or current_joints_rad
        run_capture_sequence.current_joints_rad = current_joints_rad
        if POST_ARRIVAL_PAUSE_SEC > 0.0:
            print(f"[Supervisor] Pausing {POST_ARRIVAL_PAUSE_SEC:.2f}s after arrival...")
            if not wait_seconds(supervisor, timestep, POST_ARRIVAL_PAUSE_SEC):
                return False

        capture_token = f"{view_index}_{int(supervisor.getTime() * 1000)}"
        camera_data = (
            f"capture_token={capture_token};"
            f"view={view_index};"
            f"label={content_label};"
            f"capture_root={ACTIVE_CAPTURE_ROOT};"
            f"num_views={len(camera_poses)}"
        )
        print(f"[Supervisor] Triggering capture {view_index}_{content_label}")
        set_custom_data(camera_node, camera_data)

        if not wait_seconds(supervisor, timestep, CAPTURE_WAIT_SEC):
            return False

    return True


def run_scene(supervisor: Supervisor, timestep: int, object_list, scene_index: int, total_scenes: int):
    content_label = build_content_label(object_list)
    print(f"[Supervisor] Scene {scene_index}/{total_scenes}: {content_label}")
    print("[Supervisor] Clearing existing YCB objects...")
    clear_ycb_objects(supervisor)
    spawn_objects(supervisor, object_list)
    if not wait_seconds(supervisor, timestep, SCENE_SETTLE_TIME_SEC):
        return False
    save_scene_object_poses(supervisor, object_list, content_label)
    return run_capture_sequence(supervisor, timestep, object_list)


def main():
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    capture_plan = build_capture_plan()

    print("[Supervisor] Dataset mode: multi")
    print(f"[Supervisor] Total scenes to capture: {len(capture_plan)}")

    for scene_index, object_list in enumerate(capture_plan, start=1):
        if not run_scene(supervisor, timestep, object_list, scene_index, len(capture_plan)):
            return

    print("[Supervisor] All dataset scenes captured.")


if __name__ == "__main__":
    main()
