from controller import Supervisor
import json
import os
import sys
from pathlib import Path
from typing import List


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
YCB_SUPERVISOR_DIR = REPO_ROOT / "controllers" / "ycb_supervisor"

if str(YCB_SUPERVISOR_DIR) not in sys.path:
    sys.path.insert(0, str(YCB_SUPERVISOR_DIR))

from config import (  # noqa: E402
    ASSET_BASE,
    MASS_TABLE,
    ALL_OBJECTS,
    DEFAULT_SHAPE,
    SHAPE_TABLE,
    ARM_SETTLE_TIME_SEC,
)


JSON_PATH = YCB_SUPERVISOR_DIR / "ycb_geometries.json"
with JSON_PATH.open("r", encoding="utf-8") as file:
    YCB_GEO_DATA = json.load(file)

SCENE_POSE_FILENAME = "scene_objects_pose.json"
DEFAULT_CONFIG_PATH = CURRENT_DIR / "visual_hull_check.json"
VISUAL_HULL_OVERLAY_NAME = "visual_hull_overlay"


def load_runtime_config():
    if DEFAULT_CONFIG_PATH.is_file():
        with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid config format: {DEFAULT_CONFIG_PATH}")
        normalized = {}
        for key in ("scene", "weight", "mesh"):
            value = payload.get(key, "")
            normalized[key] = str(value).strip() if value is not None else ""
        return normalized, str(DEFAULT_CONFIG_PATH)

    return {}, str(DEFAULT_CONFIG_PATH)


def get_geometry(name: str):
    return YCB_GEO_DATA.get(
        name,
        {
            "center": {"x": 0.0, "y": 0.0, "z": 0.0},
            "size": {"x": 0.1, "y": 0.1, "z": 0.1},
        },
    )


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
    return make_named_visual_hull_vrml(mesh_path, VISUAL_HULL_OVERLAY_NAME)


def make_named_visual_hull_vrml(mesh_path: Path, node_name: str) -> str:
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
        f'  name "{node_name}"\n'
        "}\n"
    )


def resolve_rotation_axis_angle(rotation_axis_angle_data):
    if not isinstance(rotation_axis_angle_data, dict):
        return [0.0, 1.0, 0.0, 0.0]

    return [
        float(rotation_axis_angle_data.get("x", 0.0)),
        float(rotation_axis_angle_data.get("y", 1.0)),
        float(rotation_axis_angle_data.get("z", 0.0)),
        float(rotation_axis_angle_data.get("angle", 0.0)),
    ]


def normalize_scene_pose_record(record, index: int):
    if not isinstance(record, dict):
        return None

    name = str(record.get("name", "")).strip()
    position = record.get("position_m", {})
    if not name or name not in MASS_TABLE or not isinstance(position, dict):
        return None

    return {
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
        normalized_record = normalize_scene_pose_record(record, index)
        if normalized_record is not None:
            normalized_records.append(normalized_record)

    return normalized_records, pose_path


def get_root_children(supervisor: Supervisor):
    return supervisor.getRoot().getField("children")


def clear_generated_nodes(supervisor: Supervisor):
    root_children = get_root_children(supervisor)
    index = root_children.getCount() - 1
    while index >= 0:
        node = root_children.getMFNode(index)
        if node is not None:
            name_field = node.getField("name")
            if name_field is not None:
                node_name = name_field.getSFString()
                if (
                    node_name in ALL_OBJECTS
                    or node_name == VISUAL_HULL_OVERLAY_NAME
                    or node_name.startswith(f"{VISUAL_HULL_OVERLAY_NAME}_")
                ):
                    node.remove()
        index -= 1


def spawn_objects_from_scene_pose(supervisor: Supervisor, scene_pose_records):
    if not scene_pose_records:
        return

    root_children = get_root_children(supervisor)
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
    root_children = get_root_children(supervisor)
    root_children.importMFNodeFromString(-1, make_visual_hull_vrml(mesh_path))


def spawn_visual_hulls(supervisor: Supervisor, mesh_paths: List[Path]):
    root_children = get_root_children(supervisor)
    for mesh_path in mesh_paths:
        suffix = mesh_path.stem.removeprefix("visual_hull_") or mesh_path.stem
        node_name = f"{VISUAL_HULL_OVERLAY_NAME}_{suffix}"
        root_children.importMFNodeFromString(
            -1,
            make_named_visual_hull_vrml(mesh_path, node_name),
        )


def resolve_multi_mesh_paths(weight_dir: Path, scene_name: str):
    scene_labels = [part.strip().replace("-", "_") for part in scene_name.split("+") if part.strip()]
    mesh_paths = []
    missing_labels = []

    for label in scene_labels:
        mesh_path = weight_dir / f"visual_hull_{label}.obj"
        if mesh_path.is_file():
            mesh_paths.append(mesh_path)
        else:
            missing_labels.append(label)

    return mesh_paths, missing_labels


def resolve_scene_dir(raw_scene: str) -> Path:
    scene_dir = Path(raw_scene)
    if not scene_dir.is_absolute():
        scene_dir = REPO_ROOT / scene_dir
    return scene_dir.resolve()


def select_mesh_paths(weight_dir: Path, scene_name: str, mesh_name: str):
    auto_mesh_paths, missing_mesh_labels = resolve_multi_mesh_paths(weight_dir, scene_name)
    if mesh_name:
        return [weight_dir / mesh_name], False, missing_mesh_labels
    if len(auto_mesh_paths) > 1:
        return auto_mesh_paths, True, missing_mesh_labels
    return [weight_dir / "visual_hull.obj"], False, missing_mesh_labels


def fail_and_idle(supervisor: Supervisor, timestep: int, *messages: str):
    for message in messages:
        print(message)
    while supervisor.step(timestep) != -1:
        pass


def wait_seconds(supervisor: Supervisor, timestep: int, seconds: float):
    steps = max(0, int(seconds * 1000 / max(1, timestep)))
    for _ in range(steps):
        if supervisor.step(timestep) == -1:
            return False
    return True


def main():
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    data, config_source = load_runtime_config()

    raw_scene = data.get("scene", "")
    weight_name = data.get("weight", "")
    mesh_name = data.get("mesh", "").strip()

    if not raw_scene or not weight_name:
        fail_and_idle(
            supervisor,
            timestep,
            "[visual_hull_check] Missing scene or weight in runtime config.",
            f"[visual_hull_check] Create config file: {DEFAULT_CONFIG_PATH}",
            '[visual_hull_check] Example JSON: {"scene": "Grounded-Segment-Anything/test_images/captures_single/apple", '
            '"weight": "grounded_sam_0.25_0.25_0.8", "mesh": ""}',
        )
        return

    scene_dir = resolve_scene_dir(raw_scene)
    if not scene_dir.is_dir():
        fail_and_idle(supervisor, timestep, f"[visual_hull_check] Scene directory not found: {scene_dir}")
        return

    weight_dir = scene_dir / weight_name
    if not weight_dir.is_dir():
        fail_and_idle(supervisor, timestep, f"[visual_hull_check] Weight directory not found: {weight_dir}")
        return

    scene_pose_records, scene_pose_path = load_scene_pose_records(scene_dir)
    if scene_pose_records is None:
        fail_and_idle(supervisor, timestep, f"[visual_hull_check] Scene pose file not found: {scene_pose_path}")
        return

    scene_name = scene_dir.name
    mesh_paths, auto_multi_mesh, missing_mesh_labels = select_mesh_paths(
        weight_dir, scene_name, mesh_name
    )

    missing_mesh_paths = [mesh_path for mesh_path in mesh_paths if not mesh_path.is_file()]
    if missing_mesh_paths:
        fail_and_idle(supervisor, timestep, f"[visual_hull_check] Mesh not found: {missing_mesh_paths[0]}")
        return

    clear_generated_nodes(supervisor)
    print(f"[visual_hull_check] Waiting {ARM_SETTLE_TIME_SEC:.1f}s for arm to reach target pose...")
    if not wait_seconds(supervisor, timestep, ARM_SETTLE_TIME_SEC):
        return
    spawn_objects_from_scene_pose(supervisor, scene_pose_records)
    if auto_multi_mesh:
        spawn_visual_hulls(supervisor, mesh_paths)
    else:
        spawn_visual_hull(supervisor, mesh_paths[0])

    print(f"[visual_hull_check] Config source: {config_source}")
    print(f"[visual_hull_check] Scene: {scene_dir}")
    print(f"[visual_hull_check] Weight: {weight_dir}")
    print(f"[visual_hull_check] Reconstructed from saved pose: {scene_pose_path}")
    print(
        f"[visual_hull_check] Reconstructed objects: "
        f"{[record['name'] for record in scene_pose_records]}"
    )
    if auto_multi_mesh:
        print(f"[visual_hull_check] Visual hulls: {mesh_paths}")
        if missing_mesh_labels:
            print(f"[visual_hull_check] Missing class meshes: {missing_mesh_labels}")
    else:
        print(f"[visual_hull_check] Visual hull: {mesh_paths[0]}")
    while supervisor.step(timestep) != -1:
        pass


if __name__ == "__main__":
    main()
