from controller import Supervisor
import math
import random
import json
import os
import sys
import itertools

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_CONTROLLER_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "ycb_supervisor")
if SOURCE_CONTROLLER_DIR not in sys.path:
    sys.path.insert(0, SOURCE_CONTROLLER_DIR)

from config import (  # noqa: E402
    NUM_OBJECTS,
    GRID_COLS,
    SPACING,
    SPAWN_HEIGHT,
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
    DATASET_CAPTURE_MODE,
    MULTI_OBJECT_COUNT,
    MULTI_MIN_APPEARANCES,
)

JSON_PATH = os.path.join(SOURCE_CONTROLLER_DIR, "ycb_geometries.json")
with open(JSON_PATH, "r", encoding="utf-8") as file:
    YCB_GEO_DATA = json.load(file)

UR5E_DEF = "UR5E"
CAMERA_DEF = "UR5E_CAMERA"
ARM_COMMAND_EMITTER = "arm_command_emitter"
CAPTURE_WAIT_SEC = 1.0
VIEW_SEQUENCE = (1, 2, 3)


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


def compute_grid_positions(n: int, cols: int, spacing: float):
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

        final_x = grid_x + X_OFFSET
        final_y = grid_y + Z_OFFSET
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


def get_capture_object_pool():
    return TARGET_OBJECTS[:] if TARGET_OBJECTS else ALL_OBJECTS[:]


def build_single_object_scenes(object_pool):
    return [[name] for name in object_pool]


def build_multi_object_scenes(object_pool, group_size, min_appearances):
    if len(object_pool) < group_size:
        return []

    combos = list(itertools.combinations(sorted(object_pool), group_size))
    appearance_counts = {name: 0 for name in object_pool}
    scenes = []
    used_combos = set()

    while min(appearance_counts.values()) < min_appearances:
        best_combo = None
        best_score = None

        for combo in combos:
            if combo in used_combos:
                continue

            deficits = [max(0, min_appearances - appearance_counts[name]) for name in combo]
            score = (
                sum(deficits),
                min(deficits),
                -sum(appearance_counts[name] for name in combo),
                random.random(),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_combo = combo

        if best_combo is None or best_score[0] <= 0:
            break

        scenes.append(list(best_combo))
        used_combos.add(best_combo)
        for name in best_combo:
            appearance_counts[name] += 1

    return scenes


def build_capture_plan():
    object_pool = get_capture_object_pool()

    if DATASET_CAPTURE_MODE == "single_and_multi":
        single_scenes = build_single_object_scenes(object_pool)
        multi_scenes = build_multi_object_scenes(
            object_pool,
            MULTI_OBJECT_COUNT,
            MULTI_MIN_APPEARANCES,
        )
        return single_scenes + multi_scenes

    if TARGET_OBJECTS:
        return [TARGET_OBJECTS[:]]

    return [random.sample(object_pool, k=min(NUM_OBJECTS, len(object_pool)))]


def run_capture_sequence(supervisor: Supervisor, timestep: int, object_list):
    ur5e_node = supervisor.getFromDef(UR5E_DEF)
    camera_node = supervisor.getFromDef(CAMERA_DEF)
    arm_emitter = supervisor.getDevice(ARM_COMMAND_EMITTER)
    content_label = build_content_label(object_list)

    if ur5e_node is None:
        print(f"[Supervisor] 找不到 DEF {UR5E_DEF}")
        return False
    if camera_node is None:
        print(f"[Supervisor] 找不到 DEF {CAMERA_DEF}")
        return False

    for view_index in VIEW_SEQUENCE:
        print(f"[Supervisor] Moving arm to view {view_index}...")
        if not send_arm_pose_command(arm_emitter, view_index):
            return False
        if not wait_seconds(supervisor, timestep, ARM_SETTLE_TIME_SEC):
            return False

        capture_token = f"{view_index}_{int(supervisor.getTime() * 1000)}"
        camera_data = (
            f"capture_token={capture_token};"
            f"view={view_index};"
            f"label={content_label}"
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
    return run_capture_sequence(supervisor, timestep, object_list)


def main():
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    capture_plan = build_capture_plan()

    print(f"[Supervisor] Dataset mode: {DATASET_CAPTURE_MODE}")
    print(f"[Supervisor] Total scenes to capture: {len(capture_plan)}")

    for scene_index, object_list in enumerate(capture_plan, start=1):
        if not run_scene(supervisor, timestep, object_list, scene_index, len(capture_plan)):
            return

    print("[Supervisor] All dataset scenes captured.")


if __name__ == "__main__":
    main()
