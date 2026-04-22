from controller import Supervisor
import json
import math
import os
from pathlib import Path
from typing import List


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
YCB_SUPERVISOR_DIR = REPO_ROOT / "controllers" / "ycb_supervisor"

import sys

if str(YCB_SUPERVISOR_DIR) not in sys.path:
    sys.path.insert(0, str(YCB_SUPERVISOR_DIR))

from config import (  # noqa: E402
    GRID_COLS,
    SPACING,
    SPAWN_HEIGHT,
    X_OFFSET,
    Z_OFFSET,
    ASSET_BASE,
    MASS_TABLE,
    ALL_OBJECTS,
    DEFAULT_SHAPE,
    SHAPE_TABLE,
    SPAWN_CLEARANCE,
    SPACING_MARGIN,
)


JSON_PATH = YCB_SUPERVISOR_DIR / "ycb_geometries.json"
with JSON_PATH.open("r", encoding="utf-8") as file:
    YCB_GEO_DATA = json.load(file)

SCENE_POSE_FILENAME = "scene_objects_pose.json"


def parse_custom_data(raw_text: str):
    data = {}
    for item in raw_text.split(";"):
        key, sep, value = item.partition("=")
        if sep:
            data[key.strip().lower()] = value.strip()
    return data


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
        radius = (sx + sy + sz) / 6.0
        return f"boundingObject Sphere {{ radius {radius:.6f} }}"
    if shape == "Cylinder":
        radius = (sx + sy) / 4.0
        return f"boundingObject Cylinder {{ radius {radius:.6f} height {sz:.6f} }}"
    return f"boundingObject Box {{ size {sx:.6f} {sy:.6f} {sz:.6f} }}"


def make_ycb_vrml(
    name: str,
    x: float,
    y: float,
    z: float,
    rotation_axis_angle=None,
) -> str:
    mass = MASS_TABLE[name]
    base = f"{ASSET_BASE}/{name}/google_16k"

    geo = get_geometry(name)
    cx, cy, cz = geo["center"]["x"], geo["center"]["y"], geo["center"]["z"]
    sx, sy, sz = geo["size"]["x"], geo["size"]["y"], geo["size"]["z"]
    bounding = make_bounding_object(name, sx, sy, sz)
    if rotation_axis_angle is None:
        rotation_axis_angle = [0.0, 1.0, 0.0, 0.0]

    rot_x, rot_y, rot_z, rot_angle = rotation_axis_angle

    return f"""Solid {{
  translation {x:.6f} {y:.6f} {z:.6f}
  rotation {rot_x:.6f} {rot_y:.6f} {rot_z:.6f} {rot_angle:.6f}
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


def make_visual_hull_vrml(mesh_path: Path) -> str:
    mesh_url = os.path.relpath(mesh_path, REPO_ROOT).replace(os.sep, "/")
    return (
        "Solid {\n"
        "  translation 0 0 0\n"
        "  rotation 0 1 0 0\n"
        "  children [\n"
        "    Transform {\n"
        "      scale 1 1 1\n"
        "      children [\n"
        "        Shape {\n"
        "          appearance PBRAppearance {\n"
        "            baseColor 1 0 0\n"
        "            transparency 0.45\n"
        "            roughness 1\n"
        "            metalness 0\n"
        "          }\n"
        "          geometry Mesh {\n"
        f'            url [ "../{mesh_url}" ]\n'
        "          }\n"
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        '  name "visual_hull_overlay"\n'
        "}\n"
    )


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


def build_readable_label_map():
    label_map = {}
    for name in ALL_OBJECTS:
        parts = name.split("_", 1)
        readable = parts[1] if len(parts) == 2 else parts[0]
        readable = readable.replace("-", "_")
        label_map.setdefault(readable, []).append(name)
    return label_map


READABLE_LABEL_MAP = build_readable_label_map()


def resolve_object_names(scene_name: str):
    requested_labels = [part for part in scene_name.split("+") if part]
    resolved_names = []
    unknown_labels = []

    for label in requested_labels:
        candidates = READABLE_LABEL_MAP.get(label, [])
        if not candidates:
            unknown_labels.append(label)
            continue

        # Reuse sequential variants if the same readable label appears multiple times,
        # e.g. "cups+cups" can map to different cup assets when available.
        same_label_count = sum(1 for name in resolved_names if label in READABLE_LABEL_MAP and name in READABLE_LABEL_MAP[label])
        candidate_index = min(same_label_count, len(candidates) - 1)
        resolved_names.append(candidates[candidate_index])

    return resolved_names, unknown_labels


def resolve_rotation_axis_angle(rotation_axis_angle_data):
    if not isinstance(rotation_axis_angle_data, dict):
        return [0.0, 1.0, 0.0, 0.0]

    return [
        float(rotation_axis_angle_data.get("x", 0.0)),
        float(rotation_axis_angle_data.get("y", 1.0)),
        float(rotation_axis_angle_data.get("z", 0.0)),
        float(rotation_axis_angle_data.get("angle", 0.0)),
    ]


def load_scene_pose_records(scene_dir: Path):
    pose_path = scene_dir / SCENE_POSE_FILENAME
    if not pose_path.is_file():
        return None, pose_path

    with pose_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    objects = payload.get("objects", [])
    if not isinstance(objects, list):
        raise ValueError(f"Invalid scene pose format: {pose_path}")

    normalized_records = []
    for index, record in enumerate(objects, start=1):
        if not isinstance(record, dict):
            continue

        name = str(record.get("name", "")).strip()
        position = record.get("position_m", {})
        if not name or name not in MASS_TABLE:
            continue
        if not isinstance(position, dict):
            continue

        normalized_records.append(
            {
                "index": int(record.get("index", index)),
                "name": name,
                "position_m": {
                    "x": float(position.get("x", 0.0)),
                    "y": float(position.get("y", 0.0)),
                    "z": float(position.get("z", 0.0)),
                },
                "rotation_axis_angle": resolve_rotation_axis_angle(
                    record.get("rotation_axis_angle", {})
                ),
            }
        )

    return normalized_records, pose_path


def clear_generated_nodes(supervisor: Supervisor):
    root_children = supervisor.getRoot().getField("children")
    index = root_children.getCount() - 1
    while index >= 0:
        node = root_children.getMFNode(index)
        if node is not None:
            name_field = node.getField("name")
            if name_field is not None:
                node_name = name_field.getSFString()
                if node_name in ALL_OBJECTS or node_name == "visual_hull_overlay":
                    node.remove()
        index -= 1


def spawn_objects(supervisor: Supervisor, object_list: List[str]):
    if not object_list:
        return

    largest_footprint = max(get_collision_footprint(name) for name in object_list)
    safe_spacing = max(SPACING, largest_footprint + SPACING_MARGIN)
    positions = compute_grid_positions(len(object_list), GRID_COLS, safe_spacing)
    root_children = supervisor.getRoot().getField("children")

    for name, (grid_x, grid_y) in zip(object_list, positions):
        final_x = grid_x + X_OFFSET
        final_y = grid_y + Z_OFFSET
        safe_spawn_height = max(
            SPAWN_HEIGHT,
            get_collision_half_height(name) + SPAWN_CLEARANCE,
        )
        root_children.importMFNodeFromString(
            -1,
            make_ycb_vrml(name, final_x, final_y, safe_spawn_height),
        )


def spawn_objects_from_scene_pose(supervisor: Supervisor, scene_pose_records):
    if not scene_pose_records:
        return

    root_children = supervisor.getRoot().getField("children")
    for record in scene_pose_records:
        name = record["name"]
        position = record["position_m"]
        root_children.importMFNodeFromString(
            -1,
            make_ycb_vrml(
                name,
                position["x"],
                position["y"],
                position["z"],
                record["rotation_axis_angle"],
            ),
        )


def spawn_visual_hull(supervisor: Supervisor, mesh_path: Path):
    root_children = supervisor.getRoot().getField("children")
    root_children.importMFNodeFromString(-1, make_visual_hull_vrml(mesh_path))


def resolve_scene_dir(raw_scene: str) -> Path:
    scene_dir = Path(raw_scene)
    if not scene_dir.is_absolute():
        scene_dir = REPO_ROOT / scene_dir
    return scene_dir.resolve()


def main():
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    raw_custom_data = supervisor.getCustomData().strip()
    data = parse_custom_data(raw_custom_data)

    raw_scene = data.get("scene", "")
    weight_name = data.get("weight", "")
    mesh_name = data.get("mesh", "visual_hull.obj")

    if not raw_scene or not weight_name:
        print("[visual_hull_check] Missing scene or weight in customData.")
        print(
            "[visual_hull_check] Example: "
            "scene=Grounded-Segment-Anything/test_images/captures_single/apple;"
            "weight=grounded_sam_0.3_0.3_0.7"
        )
        while supervisor.step(timestep) != -1:
            pass
        return

    scene_dir = resolve_scene_dir(raw_scene)
    if not scene_dir.is_dir():
        print(f"[visual_hull_check] Scene directory not found: {scene_dir}")
        while supervisor.step(timestep) != -1:
            pass
        return

    weight_dir = scene_dir / weight_name
    if not weight_dir.is_dir():
        print(f"[visual_hull_check] Weight directory not found: {weight_dir}")
        while supervisor.step(timestep) != -1:
            pass
        return

    mesh_path = weight_dir / mesh_name
    if not mesh_path.is_file():
        print(f"[visual_hull_check] Mesh not found: {mesh_path}")
        while supervisor.step(timestep) != -1:
            pass
        return

    scene_pose_records, scene_pose_path = load_scene_pose_records(scene_dir)
    scene_name = scene_dir.name
    object_names, unknown_labels = resolve_object_names(scene_name)

    clear_generated_nodes(supervisor)
    if scene_pose_records:
        spawn_objects_from_scene_pose(supervisor, scene_pose_records)
    else:
        spawn_objects(supervisor, object_names)
    spawn_visual_hull(supervisor, mesh_path)

    print(f"[visual_hull_check] Scene: {scene_dir}")
    print(f"[visual_hull_check] Weight: {weight_dir}")
    if scene_pose_records:
        print(f"[visual_hull_check] Reconstructed from saved pose: {scene_pose_path}")
        print(
            f"[visual_hull_check] Reconstructed objects: "
            f"{[record['name'] for record in scene_pose_records]}"
        )
    else:
        print(f"[visual_hull_check] Reconstructed objects: {object_names}")
        print(
            f"[visual_hull_check] Warning: saved pose file not found, fallback to label inference: "
            f"{scene_pose_path}"
        )
    print(f"[visual_hull_check] Visual hull: {mesh_path}")
    if unknown_labels and not scene_pose_records:
        print(f"[visual_hull_check] Warning: unresolved labels: {unknown_labels}")

    while supervisor.step(timestep) != -1:
        pass


if __name__ == "__main__":
    main()
